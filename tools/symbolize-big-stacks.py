#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Names the call chains of EDEN_UP_BIG_STACKS (headless/heap_arenas.inc, eden_big_*).

  python3 -I tools/symbolize-big-stacks.py LOG ELF [--all | --best] [--symbolizer PATH]

LOG is a stderr.log (EDEN_BIG_STACKS / EDEN_BIG_TABLE / EDEN_BIG_STACK lines) or a crash report
(its "big requests" section). ELF is the unstripped executable of the same build: build/headless-native/
llvm-pie.elf right after a build, or the package's llvm-pie.elf.zst unpacked with
`zstd -dc <package>/llvm-pie.elf.zst > llvm-pie.elf`. The code size the log names (`code=0x0-0x<size>` on the
EDEN_BIG_STACKS on line, or "(code size 0x...)" in a crash report) must be the ELF's .text size; the
script stops when it is not (an ELF of another build gives confident wrong names).

The pcs are offsets from the code's start: the ELF's .text address plus the offset is the address
in the ELF (.text is at 0 in llvm-pie.elf). Before the '|': the frame-pointer chain from the
allocator's caller up (exact; a function without a frame pointer is missing from it), its first 12
links and its outermost 4; '..' after the 12th link (and chain_total= above 16) means links between
them were left out, printed here as '...'. After the '|': the stack scan's first code words (every
return address, and stale words). Every address whose previous instruction is not a call is dropped
here (as tools/symbolize-crash.py does: that removes the stale words), and every kept one is named
one byte back (inside its call). All links and scan words are printed.

Rows are merged when thread, build bit and the first three named frames that are not allocation
wrappers (operator new, __wrap_*, ralloc/rzalloc/linear_*, vk_alloc, std:: allocation and vector
growth, ...) agree, so different driver sites behind the same wrappers stay apart. Each row gets a
class from its first frame that is not a wrapper: driver-cache (vk_pipeline_cache, blob_, disk_cache,
radv_*serializ*), driver-compile (aco, nir_, ac_nir/ac_, radv_*compile*/nir_to_asm/radv_shader*,
spirv/vtn), emu (Vulkan::, Shader::, Sirit::, VideoCore::) or other; "also:" lists the other classes
further up the chain.

By default every wave / after / short table and the crash report's rows are shown (a wave's bytes
are split across them). --all adds the steady tables; --best shows only the table with the largest
window peak.
"""
import re
import shutil
import struct
import subprocess
import sys


def sections(data):
    assert data[:4] == b'\x7fELF' and data[4] == 2, 'not a 64-bit ELF file'
    shoff, = struct.unpack_from('<Q', data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from('<HHH', data, 0x3A)
    headers = [struct.unpack_from('<IIQQQQIIQQ', data, shoff + i * shentsize) for i in range(shnum)]
    names_at = headers[shstrndx][4]
    found = {}
    for name, _kind, _flags, address, offset, size, *_ in headers:
        end = data.index(b'\0', names_at + name)
        found[data[names_at + name:end].decode()] = (address, offset, size)
    return found


def after_call(code):
    """Whether the eight bytes that end at an address end with a call (as crash_report.cpp checks)."""
    if len(code) != 8:
        return False
    if code[3] == 0xE8:
        return True
    for length in (2, 3, 4, 6, 7):
        at = code[8 - length:]
        if at[0] == 0xFF and at[1] & 0x38 == 0x10:
            return True
        if length >= 3 and at[0] & 0xF0 == 0x40 and at[1] == 0xFF and at[2] & 0x38 == 0x10:
            return True
    return False


def tool(given):
    if given:
        return given
    for name in ('llvm-symbolizer-18', 'llvm-symbolizer', '/opt/homebrew/opt/llvm/bin/llvm-symbolizer'):
        if shutil.which(name):
            return name
    sys.exit('needs llvm-symbolizer (--symbolizer PATH)')


ROW = re.compile(r'EDEN_BIG_STACK n=(\d+) row=(\d+) ')
TABLE = re.compile(r'EDEN_BIG_TABLE n=(\d+) why=(\S+) steady_ms=(\d+) class_live_mib=([\d.]+) win_peak_mib=([\d.]+)')
CRASH = re.compile(r'  big row (\d+) ')
FIELD = re.compile(r'(\w+)=(\S*)')
GAP = -1  # '..' in a chain: links left out between the first 12 and the outermost 4

WRAPPERS = re.compile(r'^(operator new|operator delete|__wrap_|__real_|malloc|calloc|realloc|free\b|posix_memalign|'
                      r'aligned_alloc|ralloc|rzalloc|reralloc|rerzalloc|linear_|gc_|vk_alloc|vk_zalloc|vk_realloc|'
                      r'vk_default_|vk_free|util_dynarray|u_vector|eden_|ps5_heap|std::(__1::)?(allocator|__allocate|'
                      r'__libcpp_allocate|vector<.*>::(__push_back_slow_path|__emplace_back_slow_path|__append|reserve|'
                      r'__vallocate|resize|__recommend)|__split_buffer|basic_string|__hash_table|unique_ptr|make_unique|'
                      r'make_shared|__shared_ptr|deque)|boost::container|absl::)')
CLASSES = (
    ('driver-cache', re.compile(r'vk_pipeline_cache|\bblob_|disk_cache|radv_\w*serializ|radv_pipeline_cache|'
                                r'vk_pipeline_cache_object|radv_\w*cache')),
    ('driver-compile', re.compile(r'^aco|aco::|\bnir_|^ac_|\bac_nir|radv_\w*compile|nir_to_asm|radv_shader|'
                                  r'radv_postprocess|spirv_to_nir|\bvtn_|radv_\w*_nir|radv_optimize')),
    ('emu', re.compile(r'Vulkan::|Shader::|Sirit::|VideoCore::|Tegra::|VideoCommon::')),
)


def split_pcs(text):
    """(chain, scan) offsets of a pcs= field; a '..' in the chain becomes GAP."""
    chain, _, scan = text.partition('|')
    near = [GAP if x == '..' else int(x, 16) for x in chain.split(',') if x]
    return near, [int(x, 16) for x in scan.split(',') if x and x != '..']


def fields(line):
    return {k: v for k, v in FIELD.findall(line)}


def mib(text):
    """'+12.3/45.6' -> (12.3, 45.6); '12.3' -> (12.3, 12.3); '-1.0' -> -1.0."""
    text = text.lstrip('+')
    if '/' in text:
        a, b = text.split('/')
        return float(a.lstrip('+')), float(b)
    return float(text), float(text)


def main():
    args = [a for a in sys.argv[1:]]
    show_all = '--all' in args
    best = '--best' in args
    symbolizer = None
    if '--symbolizer' in args:
        at = args.index('--symbolizer')
        symbolizer = args[at + 1]
        del args[at:at + 2]
    args = [a for a in args if a not in ('--all', '--best')]
    if len(args) != 2:
        sys.exit(__doc__)
    text = open(args[0], encoding='utf-8', errors='replace').read()
    data = open(args[1], 'rb').read()
    start, at, size = sections(data)['.text']
    said = re.search(r'EDEN_BIG_STACKS on .*? code=0x([0-9a-f]+)-0x([0-9a-f]+)', text)
    crash = re.search(r'\(code size 0x([0-9a-f]+)\)', text)
    if said:
        log_size = int(said.group(2), 16) - int(said.group(1), 16)
    elif crash:
        log_size = int(crash.group(1), 16)
    else:
        sys.exit('no "EDEN_BIG_STACKS on" line and no crash report header: no code size to check')
    if log_size != size:
        sys.exit(f'the log is from a build with 0x{log_size:x} bytes of code; this ELF has 0x{size:x}. Wrong ELF.')

    tables, current = [], None
    for line in text.splitlines():
        m = TABLE.search(line)
        if m:
            current = {'n': int(m.group(1)), 'why': m.group(2), 'ms': int(m.group(3)), 'live': float(m.group(4)),
                       'peak': float(m.group(5)), 'line': line.strip(), 'rows': []}
            tables.append(current)
            continue
        m = ROW.search(line)
        if m and current is not None and int(m.group(1)) == current['n']:
            f = fields(line[m.start():])
            near, far = split_pcs(f.get('pcs', ''))
            rise = f.get('rise_mib', '+0')
            current['rows'].append({'thread': f['thread'], 'build': f['build'], 'at_peak': float(f['at_peak_mib']),
                                    'rise': float(rise.lstrip('+')) if not rise.startswith('-') else -float(rise[1:]),
                                    'live': float(f['live_mib']), 'calls': int(f['calls'].split('/')[1]),
                                    'req': mib(f['req_mib'])[1], 'grew': int(f['grew_mib'].split('/')[1]),
                                    'grew_new': int(f['grew_mib'].split('/')[0].lstrip('+')),
                                    'refused': int(f['refused'].split('/')[1]), 'young': 0.0, 'near': near, 'far': far,
                                    'depth': f.get('depth', '?'), 'total': f.get('chain_total', '?'),
                                    'deep': f.get('deep', '?')})
    crash_rows = []
    for line in text.splitlines():
        m = CRASH.match(line)
        if m:
            f = fields(line)
            near, far = split_pcs(f.get('pcs', ''))
            crash_rows.append({'thread': f['thread'], 'build': f['build'], 'at_peak': float(f['live_mib']), 'rise': 0.0,
                               'live': float(f['live_mib']), 'young': float(f.get('young_mib', 0)),
                               'calls': int(f['calls']), 'req': 0.0, 'grew': int(f['grew_mib']), 'grew_new': 0,
                               'refused': int(f['refused']), 'near': near, 'far': far, 'depth': f.get('depth', '?'),
                               'total': f.get('chain_total', '?'), 'deep': '?'})

    # The bytes of the executable segment: the hooks count code up to the 16 KiB page after .text
    # (operator new is in __lcxx_override, right behind it).
    phoff, = struct.unpack_from('<Q', data, 0x20)
    phentsize, phnum = struct.unpack_from('<HH', data, 0x36)
    loads = []
    for i in range(phnum):
        p_type, p_flags, p_offset, p_vaddr, _paddr, p_filesz, _memsz, _align = struct.unpack_from(
            '<IIQQQQQQ', data, phoff + i * phentsize)
        if p_type == 1 and p_flags & 1:
            loads.append((p_vaddr, p_offset, p_filesz))

    def code_bytes(address, count):
        for vaddr, offset, filesz in loads:
            if vaddr + count <= address <= vaddr + filesz:
                return data[offset + address - vaddr - count:offset + address - vaddr]
        return b''

    def kept(offset):
        return after_call(code_bytes(start + offset, 8))

    wanted = sorted({o for t in tables for r in t['rows'] for o in r['near'] + r['far'] if o != GAP and kept(o)} |
                    {o for r in crash_rows for o in r['near'] + r['far'] if o != GAP and kept(o)})
    names = {}
    if wanted:
        query = ''.join(f'0x{start + o - 1:x}\n' for o in wanted)
        out = subprocess.run([tool(symbolizer), f'--obj={args[1]}', '--functions=linkage', '--demangle', '--inlines'],
                             input=query, capture_output=True, text=True, check=True).stdout
        blocks = out.strip('\n').split('\n\n')
        assert len(blocks) == len(wanted), 'unexpected answer from llvm-symbolizer'
        for offset, block in zip(wanted, blocks):
            lines = block.splitlines()
            funcs = lines[0::2]
            inner, outer = funcs[0][:90], funcs[-1][:90]
            names[offset] = inner if inner == outer else f'{inner} [in {outer}]'

    def chain(offsets):
        """Named links in order; GAP stays as '...'."""
        out = []
        for o in offsets:
            if o == GAP:
                out.append('...')
            elif o in names:
                out.append(names[o])
        return out

    def is_wrapper(name):
        inner = name.split(' [in ')[0]
        return bool(WRAPPERS.match(inner))

    def kind(name):
        for label, pattern in CLASSES:
            if pattern.search(name):
                return label
        return 'other'

    def classify(named):
        real = [n for n in named if n != '...' and not is_wrapper(n)]
        if not real:
            return 'wrapper-only', []
        first = kind(real[0])
        also = []
        for n in real[1:]:
            k = kind(n)
            if k not in ('other', first) and k not in also:
                also.append(k)
        return first, also

    def show(title, rows, key):
        merged = {}
        for r in rows:
            named = chain(r['near']) or chain(r['far'])
            real = [n for n in named if n != '...' and not is_wrapper(n)]
            group = (r['thread'], r['build'], tuple(real[:3]))
            m = merged.setdefault(group, {'at_peak': 0.0, 'rise': 0.0, 'live': 0.0, 'young': 0.0, 'calls': 0, 'req': 0.0,
                                          'grew': 0, 'grew_new': 0, 'refused': 0, 'rows': 0, 'near': chain(r['near']),
                                          'far': chain(r['far']), 'depth': r['depth'], 'total': r['total'],
                                          'deep': r['deep'], 'named': named})
            for k in ('at_peak', 'rise', 'live', 'young', 'calls', 'req', 'grew', 'grew_new', 'refused'):
                m[k] += r[k]
            m['rows'] += 1
            if len(chain(r['near'])) > len(m['near']):
                m['near'], m['far'], m['depth'], m['total'], m['deep'], m['named'] = (
                    chain(r['near']), chain(r['far']), r['depth'], r['total'], r['deep'], named)
        total = sum(max(m[key], 0.0) for m in merged.values()) or 1.0
        print(title)
        for (thread, build, _first), m in sorted(merged.items(), key=lambda kv: -kv[1][key]):
            label, also = classify(m['named'])
            print(f"  {m[key]:8.1f} MiB {100.0 * m[key] / total:5.1f}% ({key})  class={label}"
                  f"{' also:' + ','.join(also) if also else ''}  thread={thread} build={build} rows={m['rows']} "
                  f"at_peak={m['at_peak']:.1f} rise={m['rise']:+.1f} live={m['live']:.1f} young={m['young']:.1f} "
                  f"calls={m['calls']} grew_mib={m['grew']}(+{m['grew_new']}) refused={m['refused']} "
                  f"depth={m['depth']} chain_total={m['total']} deep={m['deep']}")
            print('      chain: ' + ('  <-  '.join(m['near']) if m['near'] else '(none)'))
            if m['far']:
                print('      scan:  ' + '  <-  '.join(m['far']))

    if best:
        picked = sorted(tables, key=lambda t: -t['peak'])[:1]
    else:
        picked = [t for t in tables if show_all or t['why'] != 'steady']
    for t in sorted(picked, key=lambda t: t['n']):
        show(t['line'], t['rows'], 'rise' if t['why'] in ('wave', 'after') else 'at_peak' if t['why'] != 'steady' else 'req')
    if crash_rows:
        show('Crash report, big requests (young = live in blocks under 10 s old):', crash_rows,
             'young' if any(r['young'] > 0 for r in crash_rows) else 'live')
    if not tables and not crash_rows:
        print('no EDEN_BIG_TABLE or crash report rows in ' + args[0])


if __name__ == '__main__':
    main()
