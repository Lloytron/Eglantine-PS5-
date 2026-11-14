#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pipeline lists (EDEN_UP_PIPELINE_LIST_IMPORT / _EXPORT, patch 0720) on the Mac. Data only:
reads and writes files, never runs or loads anything from them.

A pipeline list is a console's pipeline records without the game's code. Each record keeps its
kind, its key (six program hashes and 1,008 bytes of drawing state) and, per program, the
CityHash64 of its code plus the environment's other bytes (sizes, answers, program header).

  pipeline-list.py count FILE...            what record files (vulkan.bin, vulkan_mesh.bin) and
                                            lists hold: records, unique keys, kinds, drivers
  pipeline-list.py export OUT FILE...       record files -> one list (no code); the same bytes
                                            the console's EDEN_UP_PIPELINE_LIST_EXPORT=1 writes
  pipeline-list.py merge OUT INPUT...       lists and/or record files -> one list, each key once
                                            (the first input that has it wins), with counts
  pipeline-list.py selftest                 checks the CityHash64 port against fixed values

A record file whose name contains "mesh" is read with mesh state (vulkan_mesh.bin); prefix a
path with "mesh:" or "plain:" to say it. On the console the merged list goes to
<shader folder>/pipeline_import.list (or any path given to EDEN_UP_PIPELINE_LIST_IMPORT).
"""
import collections
import os
import struct
import sys

M = (1 << 64) - 1
K0, K1, K2 = 0xC3A5C85C97CB3127, 0xB492B66FBE98F273, 0x9AE16A3B2F90404F

# --- CityHash64 (common/cityhash.cpp, little endian) --------------------------------------


def _f64(s, i):
    return int.from_bytes(s[i:i + 8], "little")


def _f32(s, i):
    return int.from_bytes(s[i:i + 4], "little")


def _rot(v, n):
    return v if n == 0 else ((v >> n) | (v << (64 - n))) & M


def _smix(v):
    return v ^ (v >> 47)


def _swap64(v):
    return int.from_bytes(v.to_bytes(8, "little"), "big")


def _h16(u, v, mul=None):
    if mul is None:
        mul = 0x9DDFEA08EB382D69
        a = ((u ^ v) * mul) & M
        a ^= a >> 47
        b = ((v ^ a) * mul) & M
        b ^= b >> 47
        return (b * mul) & M
    a = ((u ^ v) * mul) & M
    a ^= a >> 47
    b = ((v ^ a) * mul) & M
    b ^= b >> 47
    return (b * mul) & M


def _h0to16(s, n):
    if n >= 8:
        mul = (K2 + n * 2) & M
        a = (_f64(s, 0) + K2) & M
        b = _f64(s, n - 8)
        c = (_rot(b, 37) * mul + a) & M
        d = ((_rot(a, 25) + b) * mul) & M
        return _h16(c, d, mul)
    if n >= 4:
        mul = (K2 + n * 2) & M
        a = _f32(s, 0)
        return _h16((n + (a << 3)) & M, _f32(s, n - 4), mul)
    if n > 0:
        a, b, c = s[0], s[n >> 1], s[n - 1]
        y = (a + (b << 8)) & 0xFFFFFFFF
        z = (n + (c << 2)) & 0xFFFFFFFF
        return (_smix(((y * K2) ^ (z * K0)) & M) * K2) & M
    return K2


def _h17to32(s, n):
    mul = (K2 + n * 2) & M
    a = (_f64(s, 0) * K1) & M
    b = _f64(s, 8)
    c = (_f64(s, n - 8) * mul) & M
    d = (_f64(s, n - 16) * K2) & M
    return _h16((_rot((a + b) & M, 43) + _rot(c, 30) + d) & M,
                (a + _rot((b + K2) & M, 18) + c) & M, mul)


def _weak(w, x, y, z, a, b):
    a = (a + w) & M
    b = _rot((b + a + z) & M, 21)
    c = a
    a = (a + x) & M
    a = (a + y) & M
    b = (b + _rot(a, 44)) & M
    return (a + z) & M, (b + c) & M


def _weak_s(s, i, a, b):
    return _weak(_f64(s, i), _f64(s, i + 8), _f64(s, i + 16), _f64(s, i + 24), a, b)


def _h33to64(s, n):
    mul = (K2 + n * 2) & M
    a = (_f64(s, 0) * K2) & M
    b = _f64(s, 8)
    c = _f64(s, n - 24)
    d = _f64(s, n - 32)
    e = (_f64(s, 16) * K2) & M
    f = (_f64(s, 24) * 9) & M
    g = _f64(s, n - 8)
    h = (_f64(s, n - 16) * mul) & M
    u = (_rot((a + g) & M, 43) + ((_rot(b, 30) + c) * 9)) & M
    v = (((a + g) & M ^ d) + f + 1) & M
    w = (_swap64(((u + v) * mul) & M) + h) & M
    x = (_rot((e + f) & M, 42) + c) & M
    y = ((_swap64(((v + w) * mul) & M) + g) * mul) & M
    z = (e + f + c) & M
    a = (_swap64(((x + z) * mul + y) & M) + b) & M
    b = (_smix(((z + a) * mul + d + h) & M) * mul) & M
    return (b + x) & M


def cityhash64(s):
    s = bytes(s)
    n = len(s)
    if n <= 32:
        return _h0to16(s, n) if n <= 16 else _h17to32(s, n)
    if n <= 64:
        return _h33to64(s, n)
    x = _f64(s, n - 40)
    y = (_f64(s, n - 16) + _f64(s, n - 56)) & M
    z = _h16((_f64(s, n - 48) + n) & M, _f64(s, n - 24))
    v = _weak_s(s, n - 64, n, z)
    w = _weak_s(s, n - 32, (y + K1) & M, x)
    x = (x * K1 + _f64(s, 0)) & M
    left = (n - 1) & ~63
    i = 0
    while True:
        x = (_rot((x + y + v[0] + _f64(s, i + 8)) & M, 37) * K1) & M
        y = (_rot((y + v[1] + _f64(s, i + 48)) & M, 42) * K1) & M
        x ^= w[1]
        y = (y + v[0] + _f64(s, i + 40)) & M
        z = (_rot((z + w[0]) & M, 33) * K1) & M
        v = _weak_s(s, i, (v[1] * K1) & M, (x + w[0]) & M)
        w = _weak_s(s, i + 32, (z + w[1]) & M, (y + _f64(s, i + 16)) & M)
        z, x = x, z
        i += 64
        left -= 64
        if left == 0:
            break
    return _h16((_h16(v[0], w[0]) + _smix(y) * K1 + z) & M, (_h16(v[1], w[1]) + x) & M)


# --- record layout (src/video_core/shader_environment.cpp, cache version 23) --------------

CACHE_MAGIC = b"yuzucach"
CACHE_VERSION = 23
LIST_MAGIC = b"EGPLIST1"
LIST_FORMAT = 1
HEAD = 76                       # six u64 counts + seven u32 fields (the last is the stage)
GRAPHICS_KEY = 1056             # 6 program hashes + FixedPipelineState (1,008 bytes)
COMPUTE_KEY = 24
STATE = 48                      # where FixedPipelineState starts in the key
# FixedPipelineState offsets (the T154 layout probe, PS5 compiler): raw1 0, driver_id 16,
# driver_version 20, attachments 80, attributes 112, vertex_strides 368, xfb_state 432.
OFF_DRIVER, OFF_ATTACH, OFF_ATTR, OFF_STRIDES, OFF_XFB = 16, 80, 112, 368, 432
STAGES = ["VertexB", "TessCtl", "TessEval", "Geometry", "Fragment", "Compute", "VertexA",
          "Task", "Mesh"]
COMPUTE, GEOMETRY, TASK, MESH = 5, 3, 7, 8
KIND_COMPUTE, KIND_GRAPHICS, KIND_MESH = 0, 1, 2
KIND_NAMES = {0: "compute", 1: "graphics", 2: "task/mesh"}
MOST_CODE = 16 << 20


def tail_size(head, mesh_state, limit):
    """FileEnvironment::RecordTailSize: the environment's bytes after its code."""
    counts = struct.unpack_from("<6Q", head, 0)
    stage = struct.unpack_from("<I", head, HEAD - 4)[0]
    if stage > MESH or counts[0] > limit:
        return None
    size = 0
    for count, entry in zip(counts[1:], (8, 8, 12, 9, 12)):
        if count > limit // entry:
            return None
        size += count * entry
    if stage == COMPUTE:
        size += 16
    else:
        size += 80
        if stage == GEOMETRY:
            size += 32
        if mesh_state and stage in (TASK, MESH):
            size += 12 + 80
    return size if size <= limit else None


def state_size(key):
    raw1 = struct.unpack_from("<I", key, STATE)[0]
    eds, dvi, xfb, eds3_blend = raw1 & 1, (raw1 >> 5) & 1, (raw1 >> 6) & 1, (raw1 >> 3) & 1
    if xfb:
        return 1008
    if dvi and eds3_blend:
        return OFF_ATTACH
    if dvi:
        return OFF_ATTR
    if eds:
        return OFF_STRIDES
    return OFF_XFB


def key_hash(kind, key):
    """GraphicsPipelineCacheKey::Hash / ComputePipelineCacheKey::Hash (CityHash64 of Size())."""
    if kind == KIND_COMPUTE:
        return cityhash64(key)
    return cityhash64(key[:48 + state_size(key)])


class Record:
    __slots__ = ("kind", "envs", "key")

    def __init__(self, kind, envs, key):
        self.kind = kind      # KIND_*
        self.envs = envs      # [(code_hash, head bytes, tail bytes)]
        self.key = key

    def stages(self):
        return tuple(struct.unpack_from("<I", head, HEAD - 4)[0] for _, head, _ in self.envs)

    def list_bytes(self):
        out = [struct.pack("<BBH", self.kind, len(self.envs), 0)]
        for code_hash, head, tail in self.envs:
            out.append(struct.pack("<QI", code_hash, len(head) + len(tail)))
            out.append(head)
            out.append(tail)
        out.append(self.key)
        return b"".join(out)


def mesh_state_for(path):
    if path.startswith("mesh:"):
        return path[5:], True
    if path.startswith("plain:"):
        return path[6:], False
    return path, "mesh" in os.path.basename(path)


def read_record_file(path, mesh_state, hash_cache):
    """Every record of a vulkan.bin-type file. Returns (records, whole, version)."""
    data = open(path, "rb").read()
    if len(data) < 12 or data[:8] != CACHE_MAGIC:
        raise SystemExit(f"{path}: not a pipeline cache file")
    version = struct.unpack_from("<I", data, 8)[0]
    if version != CACHE_VERSION:
        return [], True, version
    at, end, records = 12, len(data), []
    mv = memoryview(data)
    while at != end:
        if at + 4 > end:
            return records, False, version
        count = struct.unpack_from("<I", data, at)[0]
        at += 4
        if count == 0 or count > 6:
            return records, False, version
        envs = []
        for _ in range(count):
            if at + HEAD > end:
                return records, False, version
            head = bytes(mv[at:at + HEAD])
            code_size = struct.unpack_from("<Q", head, 0)[0]
            tail = tail_size(head, mesh_state, end - at - HEAD)
            if tail is None or code_size > MOST_CODE or at + HEAD + code_size + tail > end:
                return records, False, version
            code = bytes(mv[at + HEAD:at + HEAD + code_size])
            code_hash = hash_cache.get(code)
            if code_hash is None:
                code_hash = hash_cache[code] = cityhash64(code)
            envs.append((code_hash, head, bytes(mv[at + HEAD + code_size:at + HEAD + code_size + tail])))
            at += HEAD + code_size + tail
        compute = struct.unpack_from("<I", envs[0][1], HEAD - 4)[0] == COMPUTE
        size = COMPUTE_KEY if compute else GRAPHICS_KEY
        if at + size > end:
            return records, False, version
        key = bytes(mv[at:at + size])
        at += size
        records.append(Record(KIND_COMPUTE if compute else (KIND_MESH if mesh_state else KIND_GRAPHICS), envs, key))
    return records, True, version


def read_list(path):
    data = open(path, "rb").read()
    if len(data) < 32 or data[:8] != LIST_MAGIC:
        raise SystemExit(f"{path}: not a pipeline list")
    fmt_, version, gks, cks, count, _ = struct.unpack_from("<6I", data, 8)
    if fmt_ != LIST_FORMAT or version != CACHE_VERSION or gks != GRAPHICS_KEY or cks != COMPUTE_KEY:
        raise SystemExit(f"{path}: list format {fmt_}, cache {version}, key sizes {gks}/{cks}: "
                         f"not this tool's ({LIST_FORMAT}, {CACHE_VERSION}, {GRAPHICS_KEY}/{COMPUTE_KEY})")
    at, records, whole = 32, [], True
    for _ in range(count):
        if at + 4 > len(data):
            whole = False
            break
        kind, n, _ = struct.unpack_from("<BBH", data, at)
        at += 4
        if kind > KIND_MESH or not 1 <= n <= 6:
            whole = False
            break
        envs = []
        for _ in range(n):
            code_hash, size = struct.unpack_from("<QI", data, at)
            at += 12
            envs.append((code_hash, data[at:at + HEAD], data[at + HEAD:at + size]))
            at += size
        ks = COMPUTE_KEY if kind == KIND_COMPUTE else GRAPHICS_KEY
        records.append(Record(kind, envs, data[at:at + ks]))
        at += ks
    whole &= at == len(data)
    return records, whole


def read_any(path, hash_cache):
    real, mesh = mesh_state_for(path)
    with open(real, "rb") as f:
        magic = f.read(8)
    if magic == LIST_MAGIC:
        records, whole = read_list(real)
        return records, whole, "list"
    records, whole, version = read_record_file(real, mesh, hash_cache)
    what = f"record file{' (mesh state)' if mesh else ''}"
    if version != CACHE_VERSION:
        what += f", cache version {version}: nothing read"
    return records, whole, what


def usable(record):
    """What the console's export leaves out: task/mesh records of a file without mesh state."""
    return not (record.kind == KIND_GRAPHICS and any(s in (TASK, MESH) for s in record.stages()))


def write_list(path, records):
    body = b"".join(r.list_bytes() for r in records)
    header = LIST_MAGIC + struct.pack("<6I", LIST_FORMAT, CACHE_VERSION, GRAPHICS_KEY, COMPUTE_KEY,
                                      len(records), 0)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(header + body)
    os.replace(tmp, path)
    return len(header) + len(body)


def describe(records):
    kinds = collections.Counter(r.kind for r in records)
    drivers = collections.Counter()
    state_bits = collections.Counter()
    for r in records:
        if r.kind == KIND_COMPUTE:
            continue
        did, dver = struct.unpack_from("<II", r.key, STATE + OFF_DRIVER)
        drivers[f"{did:x}.{dver:x}"] += 1
        raw1 = struct.unpack_from("<I", r.key, STATE)[0]
        state_bits[f"eds{raw1 & 1}{(raw1 >> 1) & 1}{(raw1 >> 2) & 1}{(raw1 >> 3) & 1}{(raw1 >> 4) & 1}"
                   f" dvi{(raw1 >> 5) & 1} cwe{(raw1 >> 20) & 1} mask{(raw1 >> 22) & 1}"] += 1
    programs = len({h for r in records for h, _, _ in r.envs})
    return kinds, drivers, state_bits, programs


def print_summary(label, records):
    kinds, drivers, state_bits, programs = describe(records)
    uniq = len({(r.kind == KIND_COMPUTE, key_hash(r.kind, r.key)) for r in records})
    print(f"  {label}: {len(records)} records ({kinds[KIND_GRAPHICS]} graphics, {kinds[KIND_MESH]} "
          f"task/mesh, {kinds[KIND_COMPUTE]} compute), {uniq} unique keys, {programs} distinct programs")
    for d, n in drivers.most_common():
        print(f"    driver {d}: {n} graphics/mesh records")
    for b, n in state_bits.most_common():
        print(f"    state {b}: {n}")


def cmd_count(paths):
    cache = {}
    everything = []
    for path in paths:
        records, whole, what = read_any(path, cache)
        print(f"{path} [{what}{'' if whole else ', ENDS INSIDE A RECORD'}]")
        print_summary("holds", records)
        everything += records
    if len(paths) > 1:
        print("all inputs together")
        print_summary("hold", everything)


def cmd_export(out, paths):
    cache = {}
    records = []
    for path in paths:
        got, whole, what = read_any(path, cache)
        if what == "list":
            raise SystemExit(f"{path}: export reads record files; use merge for lists")
        left = [r for r in got if usable(r)]
        print(f"{path}: {len(got)} records, {len(got) - len(left)} task/mesh without mesh state left out"
              f"{'' if whole else ', ENDS INSIDE A RECORD'}")
        records += left
    size = write_list(out, records)
    print(f"wrote {out}: {len(records)} records, {size} bytes, no program code")


def cmd_merge(out, paths):
    cache = {}
    seen = set()
    merged = []
    for path in paths:
        got, whole, what = read_any(path, cache)
        new = dup = skipped = 0
        for r in got:
            if not usable(r):
                skipped += 1
                continue
            k = (r.kind == KIND_COMPUTE, key_hash(r.kind, r.key))
            if k in seen:
                dup += 1
                continue
            seen.add(k)
            merged.append(r)
            new += 1
        print(f"{path} [{what}{'' if whole else ', ENDS INSIDE A RECORD'}]: {len(got)} records, "
              f"{new} new, {dup} already in an earlier input, {skipped} task/mesh without mesh state left out")
    size = write_list(out, merged)
    print(f"wrote {out}: {size} bytes, no program code")
    print_summary("merged", merged)


def selftest_bytes(n):
    # A fixed byte pattern (no random module: the same on every machine).
    x = 0x12345678
    out = bytearray()
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append((x >> 16) & 0xFF)
    return bytes(out)


SELFTEST_LENGTHS = [0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129,
                    1000, 4096, 12345, 80000]
# Common::CityHash64 of selftest_bytes(n), from a host build of common/cityhash.cpp (13 Nov 2026).
SELFTEST_VALUES = [
    0x9ae16a3b2f90404f,
    0xda6bbb6b7aa6594d,
    0xa90e71fdf3036a6e,
    0x5bfbe5d4bde4fabf,
    0xcdf8678a74af37b8,
    0x526c135df07bf001,
    0x1eab80883a5edeb5,
    0xec7816b02adb2736,
    0x732355676f985bb8,
    0x3f73c77386cc78cb,
    0x4e52aa5f44e5f2bc,
    0x8fe7864ae350a74e,
    0xfa5c5e3a268b04fb,
    0x4374f2e6dad0f0b7,
    0xe69cfcb491cf31de,
    0xaf9378f811d9d4d2,
    0xb5884d3a281d9a13,
    0x6bcb5da1b12473fa,
    0xb17cd862f5a768ef,
    0xfccdc94727b5bb99,
    0xdca8110c3f72f84b,
    0x347adcc93164ebb0,
    0x0492d485a5fcf8eb,
]


def cmd_selftest():
    bad = 0
    for n, want in zip(SELFTEST_LENGTHS, SELFTEST_VALUES):
        got = cityhash64(selftest_bytes(n))
        if got != want:
            print(f"length {n}: {got:016x}, want {want:016x}")
            bad += 1
    print(f"selftest: {len(SELFTEST_LENGTHS) - bad} of {len(SELFTEST_LENGTHS)} CityHash64 values right")
    return 1 if bad else 0


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "count" and len(argv) > 2:
        cmd_count(argv[2:])
    elif cmd == "export" and len(argv) > 3:
        cmd_export(argv[2], argv[3:])
    elif cmd == "merge" and len(argv) > 3:
        cmd_merge(argv[2], argv[3:])
    elif cmd == "selftest":
        return cmd_selftest()
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
