// SPDX-License-Identifier: GPL-3.0-or-later
// Where the app's long-lived memory goes in the pool the CPU shares with the GPU.
//
// The system gives out direct memory from the bottom of the pool: the lowest room that fits.
// A game's own memory (4 GiB) and its graphics come and go with each session, while two kinds
// of blocks stay for good: the pieces the heap grows by (headless/heap_arenas.inc) and the
// stacks of threads (the Payload SDK takes 2 MiB of direct memory for each, and keeps freed
// ones for the next threads). Taken from the bottom like the rest, those landed between a
// running game's graphics memory, and when the game ended they were left spread over the
// middle of the pool. The next game's graphics get the largest free stretch, which was then
// much smaller than what was free in total, and smaller again after every game.
//
// So blocks that stay are taken from the top of the pool downwards, and everything a session
// takes and gives back is left to the system as before. The link wraps the calls
// (tools/link-headless-native.sh): the heap asks for the top itself (eden_direct_high), and
// the other blocks that stay are known by their size: a thread's stack is 2 MiB, and so is the
// one block the graphics driver keeps for good (its zero block, taken in the middle of a game
// and then left where it was). The driver's command ring and marker page are known by who
// asks. The app's own 2 MiB blocks, the slots of the guest page tables, go back when a game
// ends: they bypass the wrap; EDEN_GUEST_BAND2 optionally prefers a fixed bottom band.
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

int __real_sceKernelAllocateDirectMemory(int64_t, int64_t, size_t, size_t, int, int64_t *);
int __real_sceKernelReleaseDirectMemory(int64_t, size_t);
int64_t sceKernelGetDirectMemorySize(void);

#define EDEN_DIRECT_STEP ((int64_t)2 << 20)       /* blocks at the top are placed in 2 MiB steps */
#define EDEN_DIRECT_SMALL ((size_t)2 << 20)       /* a thread's stack; the driver's zero block */

/* What is out of the pool, by size: under 2 MiB, 2 MiB, up to 32 MiB, more. Every request of the
 * program and the driver passes here, so the reports can say what the owners they name leave out
 * (at a refusal in run 53 they named 10.8 GiB of 12.1). */
static atomic_ullong eden_direct_bytes[4], eden_direct_blocks[4];
static atomic_ullong eden_band_hits, eden_band_fallbacks;

static unsigned eden_direct_class(size_t size) {
  return size < EDEN_DIRECT_SMALL ? 0 : size == EDEN_DIRECT_SMALL ? 1 : size <= ((size_t)32 << 20) ? 2 : 3;
}
/* EDEN_POOL_LEDGER_EXACT=1 (exact "1"; off by default): the counters above are taken off by the
 * size each release names, but a block is often given back in pieces of another size class. The
 * guest's 2 MiB slots (taken in class "2 MiB") go back as two halves (the in-place check), as a
 * tail after packing in place, or as a packed block whose kept size changed since it was taken;
 * every such piece is taken off "under 2 MiB", which wraps below zero (waveydude's four 13.60
 * runs, 12 Nov: "17592186038714 MiB in 18446744073709546170 blocks"; 5,446 blocks too many).
 * With the switch every block taken is noted (start, end, the block it came from, its class) and
 * a release takes its bytes off the class of the block they came from; the block count drops when
 * the last piece of that block is back. A release (or the part of it) that matches no noted block
 * is taken off by the old rule and counted as "unmatched". Off: one relaxed load of a cached answer per take and per release. */
#define EDEN_LEDGER_MAX 8192 /* 256 KiB of zeroes in .bss, written only with the switch */
struct eden_ledger_piece {
  int64_t start, end; /* what is still out of this piece of a block */
  int64_t block;      /* where the block it came from began; its class in the low 2 bits */
  int64_t block_end;  /* where that block ended: its pieces all lie in [block, block_end) */
};
static struct eden_ledger_piece eden_ledger_pieces[EDEN_LEDGER_MAX]; /* by start */
static unsigned eden_ledger_count;
static atomic_flag eden_ledger_lock = ATOMIC_FLAG_INIT;
static atomic_int eden_ledger_state;         /* 0 not decided yet (off), 1 on, 2 off */
static atomic_ullong eden_ledger_unmatched;  /* releases (or parts) that matched no noted block */
static atomic_ullong eden_ledger_unmatched_bytes;
static atomic_ullong eden_ledger_untracked;  /* blocks or pieces not noted: the table was full */

/* The switch is decided by main right after env.txt is applied (eden_direct_ledger_exact_init):
 * the heap's first blocks and the launcher's thread stacks are taken before that, and a getenv
 * here would have read the switch too early and kept it off for the whole run. Until then: off,
 * nothing cached. Blocks taken before it is on are not noted; their releases count "unmatched". */
static int eden_ledger_exact(void) {
  return atomic_load_explicit(&eden_ledger_state, memory_order_relaxed) == 1;
}
void eden_direct_ledger_exact_init(void) {
  const char *value = getenv("EDEN_POOL_LEDGER_EXACT");
  atomic_store_explicit(&eden_ledger_state, value && value[0] == '1' && value[1] == '\0' ? 1 : 2,
                        memory_order_relaxed);
}
static void eden_ledger_acquire(void) {
  while (atomic_flag_test_and_set_explicit(&eden_ledger_lock, memory_order_acquire)) {}
}
static void eden_ledger_release(void) { atomic_flag_clear_explicit(&eden_ledger_lock, memory_order_release); }
/* Under eden_ledger_lock: the first piece whose end is past `at`. */
static unsigned eden_ledger_find(int64_t at) {
  unsigned low = 0, high = eden_ledger_count;
  while (low < high) {
    const unsigned mid = low + (high - low) / 2;
    if (eden_ledger_pieces[mid].end <= at) low = mid + 1;
    else high = mid;
  }
  return low;
}
/* Under eden_ledger_lock. 0: the table is full. */
static int eden_ledger_insert(int64_t start, int64_t end, int64_t block, int64_t block_end) {
  if (eden_ledger_count == EDEN_LEDGER_MAX) {
    atomic_fetch_add_explicit(&eden_ledger_untracked, 1, memory_order_relaxed);
    return 0;
  }
  const unsigned at = eden_ledger_find(start);
  for (unsigned i = eden_ledger_count; i > at; --i) eden_ledger_pieces[i] = eden_ledger_pieces[i - 1];
  eden_ledger_pieces[at] = (struct eden_ledger_piece){start, end, block, block_end};
  ++eden_ledger_count;
  return 1;
}
/* Under eden_ledger_lock (taken around the system call, so that a range given back cannot be
 * given out again and noted before its release is noted). */
static void eden_ledger_note_take(int64_t start, size_t size) {
  (void)eden_ledger_insert(start, start + (int64_t)size, start | (int64_t)eden_direct_class(size),
                           start + (int64_t)size);
}
/* Under eden_ledger_lock, like the take: the release off the class of the block each part of it
 * came from. */
static void eden_ledger_note_release(int64_t start, size_t size) {
  const int64_t end = start + (int64_t)size;
  uint64_t matched = 0;
  unsigned i = eden_ledger_find(start);
  while (i < eden_ledger_count && eden_ledger_pieces[i].start < end) {
    struct eden_ledger_piece piece = eden_ledger_pieces[i];
    const int64_t low = piece.start > start ? piece.start : start;
    const int64_t high = piece.end < end ? piece.end : end;
    const unsigned cls = (unsigned)(piece.block & 3);
    const int64_t block = piece.block & ~(int64_t)3;
    matched += (uint64_t)(high - low);
    atomic_fetch_sub_explicit(&eden_direct_bytes[cls], (unsigned long long)(high - low), memory_order_relaxed);
    if (low > piece.start && high < piece.end) {
      /* A hole in the middle: the piece becomes two. */
      eden_ledger_pieces[i].end = low;
      (void)eden_ledger_insert(high, piece.end, piece.block, piece.block_end);
      break;
    }
    if (low > piece.start) {
      eden_ledger_pieces[i].end = low;
      ++i;
      continue;
    }
    if (high < piece.end) {
      eden_ledger_pieces[i].start = high;
      break;
    }
    /* The whole piece is back: out of the table, and the block's count too if it was its last. */
    for (unsigned j = i + 1; j < eden_ledger_count; ++j) eden_ledger_pieces[j - 1] = eden_ledger_pieces[j];
    --eden_ledger_count;
    int last = 1;
    for (unsigned j = eden_ledger_find(block); j < eden_ledger_count && eden_ledger_pieces[j].start < piece.block_end; ++j) {
      if (eden_ledger_pieces[j].block == piece.block) {
        last = 0;
        break;
      }
    }
    if (last) atomic_fetch_sub_explicit(&eden_direct_blocks[cls], 1, memory_order_relaxed);
  }
  if (matched < (uint64_t)size) {
    /* Not noted: taken before main turned the switch on (the launcher's stacks, the heap's first
     * blocks) or while the table was full. Those were counted by the old rule, so the rest is
     * taken off by the old rule too (and a block, when nothing at all matched). */
    const unsigned cls = eden_direct_class(size);
    atomic_fetch_sub_explicit(&eden_direct_bytes[cls], (unsigned long long)((uint64_t)size - matched),
                              memory_order_relaxed);
    if (matched == 0) atomic_fetch_sub_explicit(&eden_direct_blocks[cls], 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&eden_ledger_unmatched, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&eden_ledger_unmatched_bytes, (uint64_t)size - matched, memory_order_relaxed);
  }
}
/* 0: EDEN_POOL_LEDGER_EXACT is off (nothing written). headless/performance.cpp prints it. */
int eden_direct_ledger_exact(unsigned long long *pieces, unsigned long long *unmatched,
                             unsigned long long *unmatched_bytes, unsigned long long *untracked) {
  if (!eden_ledger_exact()) return 0;
  eden_ledger_acquire();
  *pieces = eden_ledger_count;
  eden_ledger_release();
  *unmatched = atomic_load_explicit(&eden_ledger_unmatched, memory_order_relaxed);
  *unmatched_bytes = atomic_load_explicit(&eden_ledger_unmatched_bytes, memory_order_relaxed);
  *untracked = atomic_load_explicit(&eden_ledger_untracked, memory_order_relaxed);
  return 1;
}

static int eden_direct_take(int64_t from, int64_t to, size_t size, size_t alignment, int type, int64_t *start) {
  const int exact = eden_ledger_exact();
  if (exact) eden_ledger_acquire();
  const int result = __real_sceKernelAllocateDirectMemory(from, to, size, alignment, type, start);
  if (result == 0) {
    atomic_fetch_add_explicit(&eden_direct_bytes[eden_direct_class(size)], size, memory_order_relaxed);
    atomic_fetch_add_explicit(&eden_direct_blocks[eden_direct_class(size)], 1, memory_order_relaxed);
    if (exact) eden_ledger_note_take(*start, size);
  }
  if (exact) eden_ledger_release();
  return result;
}
void eden_direct_ledger(unsigned long long *bytes, unsigned long long *blocks,
                        unsigned long long *hits, unsigned long long *fallbacks) {
  *hits = atomic_load_explicit(&eden_band_hits, memory_order_relaxed);
  *fallbacks = atomic_load_explicit(&eden_band_fallbacks, memory_order_relaxed);
  for (unsigned i = 0; i < 4; ++i) {
    bytes[i] = atomic_load_explicit(&eden_direct_bytes[i], memory_order_relaxed);
    blocks[i] = atomic_load_explicit(&eden_direct_blocks[i], memory_order_relaxed);
  }
}

static atomic_flag eden_direct_lock = ATOMIC_FLAG_INIT;
static int64_t eden_direct_floor;                 /* the lowest block taken at the top; 0: none yet */
static int64_t eden_direct_holes[1024];           /* 2 MiB blocks given back above the floor */
static unsigned eden_direct_hole_count;

static void eden_direct_acquire(void) {
  while (atomic_flag_test_and_set_explicit(&eden_direct_lock, memory_order_acquire)) {}
}
static void eden_direct_release(void) { atomic_flag_clear_explicit(&eden_direct_lock, memory_order_release); }

/* Called under eden_direct_lock. Configuration is fixed before any placement preference.
 * Round down to whole guest slots; reject a band that leaves no outside slot. */
static int64_t eden_direct_band2(int64_t total) {
  static int64_t band = -1;
  if (band < 0) {
    const char *value = getenv("EDEN_GUEST_BAND2");
    char *end = NULL;
    const unsigned long long mib = value && value[0] >= '0' && value[0] <= '9'
                                    ? strtoull(value, &end, 10) : 0;
    band = end && *end == '\0' && total > EDEN_DIRECT_STEP &&
                   mib <= (unsigned long long)((total - EDEN_DIRECT_STEP) >> 20)
             ? (int64_t)(mib / 2 * 2 << 20) : 0;
  }
  return band;
}

int eden_direct_low(size_t size, size_t alignment, int type, int64_t *start);

/* Memory from as high in the pool as there is room; from anywhere when the top has none. */
int eden_direct_high(size_t size, size_t alignment, int type, int64_t *start) {
  const int64_t total = sceKernelGetDirectMemorySize();
  const int64_t bytes = (int64_t)size;
  if (bytes <= 0 || total <= 0) return eden_direct_take(0, total, size, alignment, type, start);
  eden_direct_acquire();
  const int64_t band = eden_direct_band2(total);
  /* A block given back earlier left a hole of its size. */
  while (size == EDEN_DIRECT_SMALL && eden_direct_hole_count != 0) {
    const int64_t hole = eden_direct_holes[--eden_direct_hole_count];
    if (hole >= band && eden_direct_take(hole, hole + bytes, size, alignment, type, start) == 0) {
      eden_direct_release();
      return 0;
    }
  }
  /* Below what is at the top already; past what something else has put there, for a while. */
  int64_t next = eden_direct_floor != 0 ? eden_direct_floor : total;
  for (unsigned tries = 0; tries < 512 && next >= bytes && next - bytes >= band; ++tries) {
    const int64_t at = (next - bytes) / EDEN_DIRECT_STEP * EDEN_DIRECT_STEP;
    next = at;
    if (eden_direct_take(at, at + bytes, size, alignment, type, start) == 0) {
      eden_direct_floor = at;
      eden_direct_release();
      return 0;
    }
  }
  eden_direct_release();
  return eden_direct_low(size, alignment, type, start);
}

/* Session/graphics allocations prefer outside the guest band; full-pool recovery stays. */
int eden_direct_low(size_t size, size_t alignment, int type, int64_t *start) {
  const int64_t total = sceKernelGetDirectMemorySize();
  eden_direct_acquire();
  const int64_t band = eden_direct_band2(total);
  eden_direct_release();
  if (band > 0 && eden_direct_take(band, total, size, alignment, type, start) == 0) return 0;
  return eden_direct_take(0, total, size, alignment, type, start);
}

/* Guest slots and packed blocks prefer [0, band). No placeholder or shared high floor.
 * Both owners may cross the boundary on failure: this is deliberately a soft partition. */
int eden_direct_guest(size_t size, size_t alignment, int type, int64_t *start) {
  const int64_t total = sceKernelGetDirectMemorySize();
  eden_direct_acquire();
  const int64_t band = eden_direct_band2(total);
  if (band > 0) {
    if (eden_direct_take(0, band, size, alignment, type, start) == 0) {
      atomic_fetch_add_explicit(&eden_band_hits, 1, memory_order_relaxed);
      eden_direct_release();
      return 0;
    }
    atomic_fetch_add_explicit(&eden_band_fallbacks, 1, memory_order_relaxed);
  }
  eden_direct_release();
  return eden_direct_take(0, total, size, alignment, type, start);
}

/* The Vulkan driver makes its command ring (16 MiB) and, right after it, the ring's marker page
 * once, with the first device, and keeps them until the app ends. Taken from the bottom like a
 * session's memory they landed above the first game's (the ring at 4.6 GiB), and every later
 * game found the pool's largest free stretch cut from 10.3 GiB to 5.7. They are known by who
 * asks (the driver's allocator; no such symbol in a build without that driver), by their sizes
 * and by being the first of those sizes in that order. */
extern void eden_radv_private_radv_ps5_memory_alloc(void) __attribute__((weak));
#define EDEN_DIRECT_RING ((size_t)16 << 20)
#define EDEN_DIRECT_PAGE ((size_t)16 << 10)
static atomic_int eden_direct_driver_kept; /* 0: neither yet; 1: the ring; 2: the marker page too */

static int eden_direct_driver_keeps(size_t size, const void *caller) {
  const char *const allocator = (const char *)(uintptr_t)&eden_radv_private_radv_ps5_memory_alloc;
  if (!allocator || (const char *)caller < allocator || (const char *)caller >= allocator + 2048) return 0;
  int kept = atomic_load_explicit(&eden_direct_driver_kept, memory_order_relaxed);
  if (kept == 0 && size == EDEN_DIRECT_RING)
    return atomic_compare_exchange_strong(&eden_direct_driver_kept, &kept, 1);
  if (kept == 1 && size == EDEN_DIRECT_PAGE)
    return atomic_compare_exchange_strong(&eden_direct_driver_kept, &kept, 2);
  return 0;
}

int __wrap_sceKernelAllocateDirectMemory(int64_t from, int64_t to, size_t size, size_t alignment, int type,
                                         int64_t *start) {
  /* A thread's stack or the driver's zero block, asked for anywhere in the pool; and the
   * driver's command ring and marker page. */
  if (from == 0 && start && to >= sceKernelGetDirectMemorySize() &&
      (size == EDEN_DIRECT_SMALL || eden_direct_driver_keeps(size, __builtin_return_address(0))))
    return eden_direct_high(size, alignment, type, start);
  if (from == 0 && start && to >= sceKernelGetDirectMemorySize())
    return eden_direct_low(size, alignment, type, start);
  return eden_direct_take(from, to, size, alignment, type, start);
}

int __wrap_sceKernelReleaseDirectMemory(int64_t start, size_t size) {
  int result;
  if (eden_ledger_exact()) {
    eden_ledger_acquire();
    result = __real_sceKernelReleaseDirectMemory(start, size);
    if (result == 0) eden_ledger_note_release(start, size);
    eden_ledger_release();
    if (result != 0) return result;
  } else {
    result = __real_sceKernelReleaseDirectMemory(start, size);
    if (result != 0) return result;
    atomic_fetch_sub_explicit(&eden_direct_bytes[eden_direct_class(size)], size, memory_order_relaxed);
    atomic_fetch_sub_explicit(&eden_direct_blocks[eden_direct_class(size)], 1, memory_order_relaxed);
  }
  eden_direct_acquire();
  if (size == EDEN_DIRECT_SMALL && eden_direct_floor != 0 && start >= eden_direct_floor &&
      eden_direct_hole_count < sizeof(eden_direct_holes) / sizeof(eden_direct_holes[0]))
    eden_direct_holes[eden_direct_hole_count++] = start;
  eden_direct_release();
  return result;
}
