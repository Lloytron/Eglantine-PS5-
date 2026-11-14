#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""The translation id for EDEN_UP_SHADER_PREP_SOURCE_STAMP (core patch 0596).

Based on waveydude's tools/gen-translation-id.py (fork waveydude/Eglantine-PS5-, branch
w11-two-frame-guard, commit 2ca0fdb: the idea, the header name and the macro), with our own
framing and fail-closed rule, reworked per our notes (12 Nov 2026). Runs at build time
(a Ninja custom command in headless/vulkan.cmake), after configure has written every rewritten
source, so it hashes the bytes that are compiled.

What it hashes (SHA-256, framed: u64 path length, logical path, u64 content length, content;
logical paths, so the build folder's place does not matter; the first 32 hex are the id):
  - src/shader_recompiler/** and every quoted #include they reach, followed recursively and
    resolved as the compiler does (the including file's folder, then the include folders named
    in the context file, in order);
  - the single files in SINGLE (the translation's inputs outside the translator), each resolved
    build override first, includes not followed;
  - the sirit sources (or, when the target is IMPORTED, its library file) and the SPIR-V headers;
  - the context file (compiler, flags, definitions; absolute roots replaced by tokens) and this
    script.
Anything it cannot account for (a missing input, a symlink, a macro include, no sirit) writes an
empty id: the core then keeps today's stamp (exe<size>.<mtime>).

Writes the header only when its text changes, a depfile listing every hashed file, and an
inventory (sha256 and logical path per file) for comparing two builds.
"""
import argparse
import hashlib
import os
import re
import struct
import sys
from pathlib import Path

VERSION_TAG = b"eden-translation-id-1"
SINGLE = (
    "<build>/shader_environment.cpp",
    "src/video_core/shader_environment.h",
    "src/video_core/renderer_vulkan/vk_pipeline_prep.h",
    "src/video_core/renderer_vulkan/vk_pipeline_prep.cpp",
    "<build>/vk_pipeline_cache_cost.cpp",
    "src/video_core/renderer_vulkan/vk_pipeline_cache.h",
    "video_core/renderer_vulkan/pipeline_helper.h",  # build override first
    "src/video_core/renderer_vulkan/fixed_pipeline_state.h",
    "src/video_core/renderer_vulkan/fixed_pipeline_state.cpp",
    "src/video_core/renderer_vulkan/maxwell_to_vk.h",
    "src/video_core/renderer_vulkan/maxwell_to_vk.cpp",
    "src/video_core/surface.h",
    "src/video_core/surface.cpp",
    "src/video_core/transform_feedback.h",
    "src/video_core/transform_feedback.cpp",
)
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*(.*?)\s*(?://.*)?$')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="the Eden core root (holds src/)")
    ap.add_argument("--port-build", required=True, help="PORT_BUILD_DIR (rewritten sources)")
    ap.add_argument("--context", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depfile", required=True)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--test-force-id", default="",
                    help="TEST ONLY (build D of the design's test): write this id instead. "
                         "Never in a build that is shared.")
    a = ap.parse_args()
    source, build = Path(a.source).resolve(), Path(a.port_build).resolve()
    reasons, files = [], {}  # logical path -> real path

    def add(path, logical):
        if path.is_symlink():
            reasons.append(f"symlink: {logical}")
        elif not path.is_file():
            reasons.append(f"missing: {logical}")
        else:
            files.setdefault(logical, path)

    def logical_of(path):
        # Not resolved first: a symlink is named by its own path.
        for candidate in (Path(os.path.abspath(path)), path.resolve()):
            for root, name in ((build, "<build>"), (source, "<source>")):
                try:
                    return f"{name}/{candidate.relative_to(root).as_posix()}"
                except ValueError:
                    pass
        return f"<abs>{path.resolve().as_posix()}"

    # Context: key=value lines, written by CMake's file(GENERATE).
    context_path = Path(a.context)
    context = {}
    if context_path.is_file():
        for line in context_path.read_text().splitlines():
            key, _, value = line.partition("=")
            context[key.strip()] = value.strip()
    else:
        reasons.append("missing: context file")
    include_dirs = [build / "include"]
    for entry in context.get("include_dirs", "").split(";"):
        if entry and Path(entry).is_dir() and Path(entry) not in include_dirs:
            include_dirs.append(Path(entry))
    for fallback in (build / "vulkan-cache", source / "src", build):
        if fallback not in include_dirs:
            include_dirs.append(fallback)

    # Followed seeds: the translator and every quoted include it reaches.
    unresolved = set()
    queue = []
    translator = source / "src/shader_recompiler"
    if not translator.is_dir():
        reasons.append("missing: src/shader_recompiler")
    else:
        for folder, dirs, names in os.walk(translator, followlinks=False):
            for d in list(dirs):
                if (Path(folder) / d).is_symlink():
                    reasons.append(f"symlink: {logical_of(Path(folder) / d)}")
            dirs[:] = sorted(d for d in dirs if d != ".git")
            for name in sorted(names):
                queue.append(Path(folder) / name)
    seen = set()
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        add(path, logical_of(path))
        if path.suffix not in (".h", ".hpp", ".cpp", ".inc", ".def", ".cc") or not path.is_file():
            continue
        for line in path.read_text(errors="replace").splitlines():
            m = INCLUDE_RE.match(line)
            if not m:
                continue
            target = m.group(1)
            if target.startswith("<"):
                continue  # angle includes: system and external libraries (not followed)
            if not target.startswith('"'):
                reasons.append(f"macro include in {logical_of(path)}: {target[:60]}")
                continue
            name = target.strip('"')
            for folder in [path.parent] + include_dirs:
                candidate = folder / name
                if candidate.is_file():
                    queue.append(candidate)
                    break
            else:
                unresolved.add(f"{logical_of(path)} -> {name}")

    # Single files, build override first.
    for entry in SINGLE:
        if entry.startswith("<build>/"):
            add(build / entry[len("<build>/"):], entry)
        elif entry.startswith("src/"):
            override = build / "include" / entry[len("src/"):]
            add(override if override.is_file() else source / entry, entry)
        else:
            override = build / "include" / entry
            add(override if override.is_file() else source / "src" / entry, entry)

    # sirit and SPIR-V headers.
    sirit_source = context.get("sirit_source", "")
    if context.get("sirit_imported", "").upper() in ("TRUE", "1", "ON"):
        location = Path(context.get("sirit_location", ""))
        if location.is_file():
            add(location, "sirit/" + location.name)
        else:
            reasons.append("sirit is IMPORTED and its library file was not found")
    elif sirit_source and Path(sirit_source).is_dir():
        root = Path(sirit_source)
        for folder, dirs, names in os.walk(root, followlinks=False):
            for d in list(dirs):
                if (Path(folder) / d).is_symlink():
                    reasons.append(f"symlink in sirit: {d}")
            dirs[:] = sorted(d for d in dirs if d != ".git")
            for name in sorted(names):
                if name == ".git":
                    continue
                p = Path(folder) / name
                add(p, "sirit/" + p.relative_to(root).as_posix())
    else:
        reasons.append("sirit not resolved (no source folder in the context)")
    spirv = context.get("spirv_headers", "")
    if spirv and Path(spirv).is_dir():
        root = Path(spirv)
        for folder, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d != ".git")
            for name in sorted(names):
                p = Path(folder) / name
                add(p, "spirv-headers/" + p.relative_to(root).as_posix())
    else:
        reasons.append("SPIR-V headers not resolved")

    # Context (roots replaced by tokens) and the generator itself.
    context_text = context_path.read_bytes() if context_path.is_file() else b""
    roots = [(str(build), "<build>"), (str(source), "<source>")]
    if sirit_source:
        roots.append((str(Path(sirit_source).resolve()), "<sirit>"))
        roots.append((sirit_source, "<sirit>"))
    if spirv:
        roots.append((str(Path(spirv).resolve()), "<spirv-headers>"))
        roots.append((spirv, "<spirv-headers>"))
    # Longest first, so a root inside another is replaced whole.
    for root, token in sorted(roots, key=lambda r: -len(r[0])):
        context_text = context_text.replace(root.encode(), token.encode())
    script = Path(__file__).resolve()

    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", len(VERSION_TAG)) + VERSION_TAG)
    inventory = []
    for logical in sorted(files):
        data = files[logical].read_bytes()
        name = logical.encode()
        digest.update(struct.pack("<Q", len(name)) + name + struct.pack("<Q", len(data)) + data)
        inventory.append(f"{hashlib.sha256(data).hexdigest()} {logical}")
    for logical, data in (("<context>", context_text), ("<generator>", script.read_bytes())):
        name = logical.encode()
        digest.update(struct.pack("<Q", len(name)) + name + struct.pack("<Q", len(data)) + data)
        inventory.append(f"{hashlib.sha256(data).hexdigest()} {logical}")
    full = digest.hexdigest()
    ident = "" if reasons else full[:32]
    if a.test_force_id:
        ident = a.test_force_id
        reasons.append("TEST ONLY: --test-force-id")

    text = ("// Generated at build time by tools/gen-translation-id.py; never copy it into a tree.\n"
            "#pragma once\n"
            f'#define EDEN_TRANSLATION_ID "{ident}"\n'
            f"#define EDEN_TRANSLATION_FILE_COUNT {len(files) + 2}\n")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.is_file() or out.read_text() != text:
        out.write_text(text)
    deps = sorted({str(p.resolve()) for p in files.values()} | {str(script)} |
                  ({str(context_path.resolve())} if context_path.is_file() else set()))
    esc = lambda s: s.replace("\\", "\\\\").replace(" ", "\\ ").replace("#", "\\#").replace("$", "$$")
    dep_text = esc(str(out.resolve())) + ": " + " \\\n  ".join(esc(d) for d in deps) + "\n"
    Path(a.depfile).write_text(dep_text)
    head = [f"id {ident or '(none)'}", f"sha256 {full}", f"files {len(files) + 2}"]
    head += [f"fallback {r}" for r in sorted(set(reasons))]
    head += [f"unresolved {u}" for u in sorted(unresolved)]
    inv_text = "\n".join(head + inventory) + "\n"
    inv = Path(a.inventory)
    if not inv.is_file() or inv.read_text() != inv_text:
        inv.write_text(inv_text)
    state = f"src{ident}" if ident else "none (today's stamp)"
    print(f"EDEN translation id: {state}, files {len(files) + 2}, unresolved quoted includes "
          f"{len(unresolved)}" + "".join(f"; fallback: {r}" for r in sorted(set(reasons))[:5]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
