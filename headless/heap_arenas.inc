/* Eden PS5 port: a heap that takes its memory as it grows, with per-thread arenas.
 *
 * Memory. PS5 direct memory is real from the first byte, and the CPU shares one 12 GiB pool with
 * the GPU. The heap used to take its whole 3 GiB at start; the largest game measured used
 * 1.16 GiB of it, and the rest was graphics memory that game then ran out of. So the heap now
 * reserves its range (6 GiB, PS5_OPENGL_HEAP_SIZE) as address space only and commits it a piece
 * (32 MiB) at a time, from the start of the range, when an allocation finds no room. Where no
 * range can be reserved, it takes 3 GiB at once, as it did before it grew. Every piece (or run of pieces, for one
 * request larger than a piece) is a Sony mspace of its own; an address tells its piece, and
 * the piece its mspace. Memory is not given back: a piece stays for the next game.
 *
 * Large blocks. A request of 32 MiB or more gets memory of its own, outside the range, and gives
 * it back when it is freed: a game file read whole, or a buffer that doubles as it grows, would
 * otherwise leave the heap larger for good and its pieces full of holes no other large block fits.
 *
 * Arenas. One Sony mspace serialised every allocation in the process. Its lock collapses
 * under contention: a malloc/free pair costs 225 ns alone but 33/67/138 us with
 * 2/4/8 contending threads (EDEN_PERF_ALLOC),
 * which made allocation-heavy shader compilation ~100-250x slower than on a
 * Zen 2 host. A fixed split of the heap into arenas (run 220741) exhausted a
 * partition at level load, so capacity is not partitioned here: each allocating thread (up to
 * EDEN_HEAP_FIRST_ARENAS) gets a lazily created 8 MiB arena allocated *from* the heap, aligned to
 * its own 8 MiB slot. Small requests use the thread's arena and fall back to the heap's own
 * spaces; frees and reallocs find the owner in O(1) from the address.
 *
 * More than one arena a thread. A thread that keeps more than 8 MiB of small blocks filled its
 * arena, and from then on every small request it made went to the heap's own spaces, under the
 * lock that collapses. The threads that build a game's shader cache keep what they build (a
 * pipeline and the driver's copy of its shaders, tens of KiB each): fifteen builders fill their
 * arenas after a few thousand pipelines, which is where Tears of the Kingdom's build of 16,000
 * fell from about 200 to 60 a second at every start, and where more CPUs stopped helping (that
 * this is the cause is read from the logs, not yet measured: EDEN_SHADER_START_HEAP counts it). A
 * thread whose arena is full now
 * moves to another of its own arenas with room, and makes a new one when none has (up to
 * EDEN_HEAP_THREAD_ARENAS, for requests under EDEN_HEAP_ARENA_GROW_REQUEST); a thread that
 * shares an arena because the first ones were taken makes one of its own the same way. A thread's
 * first arena is still one of EDEN_HEAP_FIRST_ARENAS: the many threads that allocate little do
 * not each take 8 MiB of the heap. Its arenas stay with
 * its record, so the next thread that takes the record uses them: they are made once for the
 * most a thread ever held, not again for each session. The small requests that still reach the
 * heap's own spaces, and the frees of blocks there, are counted with the time they took
 * (eden_heap_spill_stats; EDEN_SHADER_START_HEAP in stderr.log).
 *
 * Memory given back. The heap only grew: a Tears of the Kingdom session held 2.69 GB of pieces at
 * its end with 0.16 GB of them in use, and the next session grew from there. A space (the mspace
 * over one piece, or a run of pieces for a larger request) that holds no block is now given back
 * by eden_heap_trim: retired so no request uses it, its memory released to the system, its
 * address space kept, and its pieces made again (and the slot used again) by a later growth. A
 * space counts the blocks taken from it, and the threads inside it at the moment (so it is
 * retired only when none is); the first space, where the arenas are, stays. The app trims at the
 * end of a shader cache build, at the end of a session, when the pool the CPU shares with the GPU
 * is short, and before an allocation is refused (headless/performance.cpp TrimHeap).
 *
 * Large blocks the pool has no room for in one piece. A block of its own is one piece of direct
 * memory, and a growth of the heap was too: in a fragmented pool both failed for a 486 MiB request
 * with a gigabyte free in smaller pieces, and the session ended. A large block that finds no one
 * piece is now address space with memory behind it in pieces of whatever size the pool has
 * (eden_heap_split_pages), and a growth takes its pieces' memory the same way (src/memory_pages.cpp).
 *
 * Room in other threads' arenas. An arena is never given back, and its room is its thread's: what a
 * thread freed stays in pieces no other thread asks. Ocarina of Time reached the pool's limit with
 * 1097 MiB held and not in use, and the heap had grown 640 MiB for one scene load with 497 MiB of
 * that room already there. While the heap holds at least EDEN_HEAP_ADOPT_SLACK more than is in use,
 * a thread with no room in its own arenas now takes an arena no thread is using before it makes a
 * new one (EDEN_HEAP_NO_ADOPT in the environment turns this off). Arenas could already be shared:
 * mspaces take their own locks, and a block finds its arena by its address.
 *
 * Spaces for what stays and spaces for what goes. An arena is never freed, and a piece goes back
 * only with no block in it, so one arena in a piece keeps all 32 MiB: at nearly every refusal of
 * those sessions eden_heap_trim found no space to give back. With EDEN_HEAP_SCRATCH_SPACES in the
 * environment (a test, off by default) a new space holds either arenas and small blocks, or blocks
 * of 1 MiB and more, so that the latter empty and go back. Its price: the room beside a long-lived
 * block of 16 MiB is then no use to an arena. EDEN_HEAP_SPACES says which it is.
 *
 * What holds the room. EDEN_HEAP_GROWTH (every 20 s) says which requests made the heap grow and
 * who asked; at a refusal, eden_heap_arena_report asks every arena and space how much it has left.
 *
 * Empty arenas spared. An arena stayed with its record for good, full or empty: Ocarina of Time
 * (run 51) held 976 MiB it did not use, 589 MiB of it room in 458 arenas, 155 of them empty, and
 * the shader builders alone held 100 empty arenas of their 241; run 57 ended with 155 empty of
 * 296. An arena that holds no block now leaves its record when its thread ends
 * (eden_heap_thread_exit) and whenever the heap is trimmed (eden_heap_trim: the pool short, a
 * build or session over, an allocation about to be refused), from every record, the ended ones
 * too, and goes to the spares (eden_heap_spare): the next thread that needs an arena, its first
 * or one more, takes a spare before the heap makes a new one. A record keeps its first arena, so
 * the next thread on it starts with one; an arena a thread is on now, or one the record of an
 * ended thread still names, stays. Every arena counts its blocks for this (a block kept in a
 * thread's cache is one); the paths that read the list of arenas or a record's arenas (a thread
 * out of room, the reports) run under the creation lock, the lock a spare is taken and given
 * under. The arena's memory is not handed back to its space: an mspace cannot be destroyed with
 * this SDK (libSceLibcInternal has no sceLibcMspaceDestroy), and a 2 MiB block whose mspace libc
 * still knows went wrong in run 54. So the heap no longer grows for arenas past the most that
 * were ever full at once: run 57 grew 576 MiB for arenas with 141 in use at the end.
 * EDEN_HEAP_KEEP_ARENAS in the environment keeps every arena with its record, as before.
 */
#include <pthread.h>
#include <stdio.h>
#include <string.h>

#define EDEN_HEAP_ARENA_SHIFT 21u /* 2 MiB arenas and slots */
#define EDEN_HEAP_ARENA_SIZE (1u << EDEN_HEAP_ARENA_SHIFT)
/* Leave room for the parent mspace's chunk metadata before the next aligned slot.
 * A full-slot payload forces the next aligned allocation to skip a whole slot. */
#define EDEN_HEAP_ARENA_PAYLOAD (EDEN_HEAP_ARENA_SIZE - 16384u)
/* Absolute 8 MiB slots covering the heap (+1: the base need not be slot aligned). */
#define EDEN_HEAP_SLOTS ((PS5_OPENGL_HEAP_SIZE >> EDEN_HEAP_ARENA_SHIFT) + 1u)
/* Threads that get an arena of their own at their first small request; later ones share them. */
#define EDEN_HEAP_FIRST_ARENAS 32u
/* Every arena, with those made because a thread's were full: at most 2 GiB of the heap's 6 GiB. An
 * arena stays with its thread's record when its blocks are freed (the next thread on the record
 * uses it), so this is also the most the heap keeps for small blocks alone. */
#define EDEN_HEAP_MAX_ARENAS 1024u
/* One thread's arenas: at most 256 MiB of small blocks before they go to the heap's spaces. */
#define EDEN_HEAP_THREAD_ARENAS 128u
/* A request this large or larger that finds no room in the thread's arenas makes no new one. */
#define EDEN_HEAP_ARENA_GROW_REQUEST (64u * 1024u)
/* After a thread found all its arenas full and could make no other, the requests it makes before
 * it looks through them again (they take room in the heap's spaces meanwhile). */
#define EDEN_HEAP_ARENA_RETRY 256u
/* The heap holds this much more than is in use: a thread out of room looks in other arenas first. */
#define EDEN_HEAP_ADOPT_SLACK ((size_t)256 << 20)
/* Arenas a thread asks each time, and the room in one piece an arena must have to be taken: a
 * sliver would send the thread looking again after a few blocks. */
#define EDEN_HEAP_ADOPT_TRIES 16u
#define EDEN_HEAP_ADOPT_ROOM ((size_t)256 << 10)
/* The threads looked at, on the stack of whichever thread asks; past them two may share an arena. */
#define EDEN_HEAP_ADOPT_BUSY 128u
/* Larger requests (and all over-aligned ones) always use the heap's own spaces. */
#define EDEN_HEAP_SMALL_REQUEST (256u * 1024u)
/* The heap grows by pieces of this size. */
#define EDEN_HEAP_PIECE_SHIFT 25u /* 32 MiB: share remaining memory with graphics in smaller steps */
#define EDEN_HEAP_PIECE_SIZE ((size_t)1 << EDEN_HEAP_PIECE_SHIFT)
#define EDEN_HEAP_PIECES (PS5_OPENGL_HEAP_SIZE >> EDEN_HEAP_PIECE_SHIFT)

void *eden_heap_reserve(size_t size);             /* address space only; NULL when refused */
int eden_heap_commit(void *address, size_t size); /* zeroed memory for a part of it; 0 when done */
/* Its memory back to the system, the address space kept: 1 done, 0 nothing done, -1 in part. */
int eden_heap_decommit(void *address, size_t size);
void *eden_heap_split_pages(size_t size);         /* a large block in pieces of memory; NULL: none */
void eden_heap_split_pages_free(void *address, size_t size);
/* The block kept for a big image (EDEN_BIG_IMAGE_RESERVE_MIB, headless/heap_pages.cpp): for a large
 * block that found no single free range; NULL: no reserve, given out, or the wrong size. */
void *eden_heap_reserve_pages(size_t size);
void eden_heap_reserve_pages_free(void *address, size_t size);
/* 1: a block of this size is of the reserve's class and may not be taken in pieces
 * (EDEN_BIG_IMAGE_NO_SPLIT): the request fails instead. */
int eden_heap_big_image_refuses(size_t size);

static void *_Atomic eden_heap_slots[EDEN_HEAP_SLOTS]; /* arena mspace owning each slot */
static void *eden_heap_arena_list[EDEN_HEAP_MAX_ARENAS]; /* NULL: given back, free for a new one */
static void *_Atomic eden_heap_arena_blocks_probe[EDEN_HEAP_MAX_ARENAS];
static atomic_size_t eden_heap_arena_backing_probe;
static atomic_uint eden_heap_arena_count; /* entries of the list in use, those given back too */
static atomic_uint eden_heap_arena_holes; /* entries given back and not taken again */
static atomic_ullong eden_heap_arenas_returned; /* arenas that left a record for the spares */
static atomic_ullong eden_heap_arenas_taken_again; /* spares taken instead of a new arena */
/* Empty arenas no record holds, for the next thread that needs one (under the creation lock). */
static void *eden_heap_spare[EDEN_HEAP_MAX_ARENAS];
static atomic_uint eden_heap_spare_count;
/* The blocks alive in each slot's arena, one cache line each: the first arenas of threads made
 * together lie in neighbouring slots, and their every request would share one line otherwise. */
static struct {
  atomic_long blocks;
  char pad[64 - sizeof(atomic_long)];
} __attribute__((aligned(64))) eden_heap_arena_live[EDEN_HEAP_SLOTS];
static unsigned eden_heap_first_arenas; /* first arenas made (under the creation lock) */
static atomic_uint eden_heap_next_shared;
static pthread_key_t eden_heap_arena_key;
static int eden_heap_arena_key_valid;
static atomic_flag eden_heap_arena_create_lock = ATOMIC_FLAG_INIT;
/* Who holds the creation lock and how many threads wait for it (read by the development stall
 * watchdog, headless/stall_watchdog.h). */
static _Atomic unsigned long eden_heap_create_holder;
static atomic_uint eden_heap_create_waiters;

/* The heap's own mspaces, a slot each; each spans one or more pieces in a row. Slot 0 is the
 * first space and stays. A slot whose space was given back (eden_heap_trim) is retired, and a
 * later growth may use it again. */
static void *_Atomic eden_heap_spaces[EDEN_HEAP_PIECES];
static atomic_uint eden_heap_space_count;                    /* slots in use, retired ones too */
static atomic_uint eden_heap_space_users[EDEN_HEAP_PIECES];  /* threads in the slot's space now */
static atomic_int eden_heap_space_retired[EDEN_HEAP_PIECES]; /* 1: no request uses it */
static atomic_long eden_heap_space_blocks[EDEN_HEAP_PIECES]; /* blocks taken from it, not freed */
/* What a slot's space takes: anything, unless EDEN_HEAP_SCRATCH_SPACES made it one of the others. */
enum { EDEN_HEAP_CLASS_ANY = 0, EDEN_HEAP_CLASS_STAYS = 1, EDEN_HEAP_CLASS_SCRATCH = 2 };
static atomic_uchar eden_heap_space_class[EDEN_HEAP_PIECES];
/* A slot's pieces, [first, first + span); span 0: given back (both under the growth lock). */
static unsigned eden_heap_space_first[EDEN_HEAP_PIECES], eden_heap_space_span[EDEN_HEAP_PIECES];
/* Bumped by every growth: a request that found no room tries again before growing itself. */
static atomic_uint eden_heap_space_generation;
/* The mspace each piece belongs to, and its slot. */
static void *_Atomic eden_heap_piece_space[EDEN_HEAP_PIECES];
static atomic_uint eden_heap_piece_slot[EDEN_HEAP_PIECES];
/* A piece below `used`: 0 in a space, 1 given back (a growth may take it again), 2 given back in
 * part (never used again; not the heap's to free into). */
static atomic_uchar eden_heap_piece_state[EDEN_HEAP_PIECES];
/* What the pieces hold now, and what was given back since the process began. */
static atomic_size_t eden_heap_committed_bytes, eden_heap_returned_bytes;
/* Where no range can be reserved, the heap takes this much at once. Its 6 GiB taken whole would
 * leave graphics too little of the 12 GiB pool. */
#define EDEN_HEAP_WHOLE_SIZE ((size_t)3072 << 20)
/* Pieces committed so far: [0, used). All of them when the heap could not be reserved to grow. */
static atomic_uint eden_heap_pieces_used;
static int eden_heap_grows;
/* The heap's range: PS5_OPENGL_HEAP_SIZE when it grows, EDEN_HEAP_WHOLE_SIZE when taken whole. */
static size_t eden_heap_range_size;
static atomic_uint eden_heap_space_hint; /* where the last allocation from the spaces succeeded */
static atomic_flag eden_heap_grow_lock = ATOMIC_FLAG_INIT;

/* Requests of this size or more get memory of their own (eden_heap_pages), freed with them. */
#define EDEN_HEAP_LARGE_REQUEST ((size_t)32 << 20)
/* 8192 since T118, as waveydude's branch ocarina-t85-town has it (858ddae0c); 4096 in T116 and
 * T117, 1024 before (he raised it from 256): with EDEN_HEAP_LARGE_KIB=256 his 1024 entries were
 * all taken at the Castle Town gate (issue 21, 10 Nov 2026) and later requests went back into the
 * heap's pieces. An entry is 24 bytes; the table is only searched for an address outside the
 * heap's range; a full table is counted (eden_heap_large_full). */
#define EDEN_HEAP_LARGE_BLOCKS 8192u
/* EDEN_HEAP_LARGE_MIB=n in env.txt (1 to 32; without it 32, as before): requests of n MiB or more
 * get memory of their own. A block of 8 to 32 MiB took room in a 32 MiB piece, and a piece goes
 * back only with no block left in it: Ocarina of Time on a base console (crash-20261109-170104)
 * ended with the pool at 0 while the heap held 1728 MiB with 644 MiB in use, 768 MiB of its
 * growth asked by requests of 8 to 32 MiB. A block of its own goes back the moment it is freed.
 * Its price is a mapping and a release for each (counted: EDEN_HEAP_MID, every 20 s) and the
 * 2 MiB lead of a block of 2 MiB or more. Set from main after env.txt (eden_heap_set_large). */
static atomic_size_t eden_heap_large_from = EDEN_HEAP_LARGE_REQUEST;
/* Blocks under EDEN_HEAP_LARGE_REQUEST with memory of their own: taken, refused (those went to
 * the heap's spaces), alive now and at most, and the cycles their mappings and releases took. */
static atomic_ullong eden_heap_mid_taken, eden_heap_mid_refused, eden_heap_mid_cycles, eden_heap_mid_taken_bytes;
static atomic_size_t eden_heap_mid_bytes, eden_heap_mid_peak;
static atomic_uint eden_heap_mid_count;
/* Requests for memory of their own that found every entry of the table taken. */
static atomic_ullong eden_heap_large_full;
/* Requests that do not go to an arena, by size, whichever way they went: under 1 MiB, under 2,
 * under 8, under 32 (count and bytes). They say what a lower EDEN_HEAP_LARGE_KIB would cost. */
static atomic_ullong eden_heap_class_count[4], eden_heap_class_bytes[4];
static void eden_heap_class_note(size_t bytes) {
  const unsigned kind = bytes < ((size_t)1 << 20) ? 0u : bytes < ((size_t)2 << 20) ? 1u : bytes < ((size_t)8 << 20) ? 2u : 3u;
  atomic_fetch_add_explicit(&eden_heap_class_count[kind], 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&eden_heap_class_bytes[kind], bytes, memory_order_relaxed);
}
/* In KiB: 256 (the arenas take what is smaller) to 32768. EDEN_HEAP_LARGE_KIB, or
 * EDEN_HEAP_LARGE_MIB in MiB. A block of 2 MiB or more is whole 2 MiB pages of the pool; a
 * smaller one is 16 KiB pages, which leaves the pool in smaller pieces. */
void eden_heap_set_large(size_t kib) {
  if (kib < 256u)
    kib = 256u;
  if (kib > 32768u)
    kib = 32768u;
  atomic_store_explicit(&eden_heap_large_from, kib << 10, memory_order_relaxed);
}
/* The large blocks alive: address 0 marks a free entry, 1 one being filled in. */
static struct {
  _Atomic uintptr_t address;
  size_t size;
  int split; /* its memory in pieces (eden_heap_split_pages) */
} eden_heap_large[EDEN_HEAP_LARGE_BLOCKS];
static atomic_uint eden_heap_large_count;
static atomic_size_t eden_heap_large_bytes;

int sceKernelUsleep(unsigned int microseconds);

/* Arena creation is rare (once per thread) but runs under a spin flag, and its holder also takes
 * an mspace lock. The emulated cores and the GPU thread run at real-time priority pinned to
 * one CPU each; spinning there forever starves a preempted lower-priority holder queued on that
 * CPU, which hung game starts. After a short spin, sleep so the holder can run. */
static void eden_heap_create_acquire(void) {
  if (atomic_flag_test_and_set_explicit(&eden_heap_arena_create_lock, memory_order_acquire)) {
    atomic_fetch_add_explicit(&eden_heap_create_waiters, 1, memory_order_relaxed);
    for (unsigned spins = 0;
         atomic_flag_test_and_set_explicit(&eden_heap_arena_create_lock, memory_order_acquire); ++spins) {
      if (spins < 64)
        __builtin_ia32_pause();
      else
        sceKernelUsleep(50);
    }
    atomic_fetch_sub_explicit(&eden_heap_create_waiters, 1, memory_order_relaxed);
  }
  atomic_store_explicit(&eden_heap_create_holder, (unsigned long)pthread_self(), memory_order_relaxed);
}

static void eden_heap_create_release(void) {
  atomic_store_explicit(&eden_heap_create_holder, 0, memory_order_relaxed);
  atomic_flag_clear_explicit(&eden_heap_arena_create_lock, memory_order_release);
}

/* Development stall reports: the creation lock's holder (0: free) and its waiters. */
unsigned long eden_heap_create_lock_state(unsigned *waiters) {
  if (waiters)
    *waiters = atomic_load_explicit(&eden_heap_create_waiters, memory_order_relaxed);
  return atomic_load_explicit(&eden_heap_create_holder, memory_order_relaxed);
}

static void *eden_heap_owner(const void *address);
static void eden_heap_free_owned(void *address);
struct eden_heap_thread;
static void eden_heap_return_record(struct eden_heap_thread *record, int ended);
static unsigned eden_heap_return_arenas(void);
static void eden_heap_use_arena(struct eden_heap_thread *self, void *arena);

/* Empty arenas stay with their records, as they did before (EDEN_HEAP_KEEP_ARENAS). */
static int eden_heap_keeps_arenas(void) {
  return getenv("EDEN_HEAP_KEEP_ARENAS") != NULL;
}

/* Per-thread allocation statistics. The live/peak/block counters are process-wide atomics, and
 * updating them on every malloc and free made all allocating threads bounce the same cache
 * lines: EDEN_PERF_ALLOC measured 210 ns per malloc/free pair alone and 874/1590 ns with 4/8
 * threads, while guest cores compile JIT blocks (allocation-heavy) together during loads. Each
 * thread now accumulates its changes and applies them every 64 operations, beyond 1 MiB, and at
 * thread exit; snapshots lag by that much. The record also keeps the thread's arena, and a
 * record freed by an exiting thread is reused with its arena by the next thread. */
/* Freed small blocks are kept per thread and reused without calling into the mspace, whose
 * calls serialise across threads even with separate arenas (EDEN_PERF_ALLOC with per-thread
 * arenas and statistics: 232 ns per pair alone, 803/1423 ns with 4/8 threads). A bin holds blocks
 * whose usable size is in [16 * bin, 16 * bin + 15], linked through their first word, with the
 * usable size in the second; a request of size s takes from bin ceil(s / 16). Bins are bounded,
 * and an exiting thread returns its blocks to their mspaces. */
#define EDEN_TCACHE_BINS 128u
#define EDEN_TCACHE_LIMIT 64u
struct eden_heap_thread {
  void *arena;   /* the arena small requests try first: one of `own`, or one shared at the limit */
  void *own[EDEN_HEAP_THREAD_ARENAS]; /* the arenas this record made, [0, owned) */
  unsigned owned;
  unsigned retry; /* requests to make before looking through `own` again (EDEN_HEAP_ARENA_RETRY) */
  unsigned grown; /* pieces the heap took for this record's requests */
  char name[16];  /* what the thread on this record called itself, for the reports */
  /* EDEN_UP_HEAP_STEAL: 1 + the index in eden_heap_arena_list of the arena this thread's last
   * request about to be refused was served from (0: none); asked first next time, under the creation
   * lock. Its own thread's only; cleared when a thread claims the record. It sits in the 4 bytes of
   * padding before `live`, so no field moves and the record keeps its size. */
  unsigned steal;
  long live;
  long blocks;
  unsigned pending;
  unsigned starve_left; /* EDEN_TEST_HEAP_STARVE: this thread's next refusals */
  unsigned wait_scope;  /* EDEN_UP_HEAP_WAIT_GPU_MS: run-time pipeline builds this thread is inside */
  unsigned wait_spent;  /* ... and one of this build's requests has waited and got nothing */
  /* Small requests that went to the heap's spaces, frees there, and their cycles: added to the
   * process's totals with live and blocks (eden_heap_flush). */
  unsigned long long spills, spill_cycles, shared_frees, shared_free_cycles;
  void *bins[EDEN_TCACHE_BINS];
  unsigned char counts[EDEN_TCACHE_BINS];
};
#define EDEN_HEAP_THREADS 512u
static struct eden_heap_thread eden_heap_threads[EDEN_HEAP_THREADS];
static atomic_uchar eden_heap_thread_used[EDEN_HEAP_THREADS];

/* EDEN_UP_BUILD_BURST_LOG (core patch 0592): a sidecar, so eden_heap_thread's size and offsets
 * stay unchanged. It counts operations on the executing thread, NOT ownership of blocks freed on
 * another thread, and only while that thread is inside a build scope (depth > 0): a thread outside
 * one pays one more load. One cache line per record (512 x 64 = 32 KiB), so neighbouring threads'
 * records never share a line. Only the record's own thread writes it (every eden_heap_charge caller
 * passes eden_heap_self()), hence a relaxed load and store instead of a locked add; the reporter
 * reads with relaxed loads, a snapshot is approximate, not transactional. */
struct eden_build_heap_record {
  _Alignas(64) atomic_llong net;
  atomic_llong peak;
  atomic_ullong allocated;
  atomic_llong base;
  atomic_ullong allocated_base;
  unsigned depth;
};
_Static_assert(sizeof(struct eden_build_heap_record) == 64, "one cache line per build heap record");
static struct eden_build_heap_record eden_build_heap_records[EDEN_HEAP_THREADS];
static atomic_uint eden_build_heap_enabled;
void eden_build_heap_enable(void) {
  atomic_store_explicit(&eden_build_heap_enabled, 1, memory_order_release);
}

static void eden_build_heap_charge(struct eden_heap_thread *self, long bytes) {
  if (!atomic_load_explicit(&eden_build_heap_enabled, memory_order_relaxed))
    return;
  struct eden_build_heap_record *r = &eden_build_heap_records[self - eden_heap_threads];
  if (r->depth == 0)
    return;
  const long long net = atomic_load_explicit(&r->net, memory_order_relaxed) + bytes;
  atomic_store_explicit(&r->net, net, memory_order_relaxed);
  if (bytes > 0)
    atomic_store_explicit(&r->allocated,
                          atomic_load_explicit(&r->allocated, memory_order_relaxed) + (unsigned long long)bytes,
                          memory_order_relaxed);
  const long long delta = net - atomic_load_explicit(&r->base, memory_order_relaxed);
  if (delta > atomic_load_explicit(&r->peak, memory_order_relaxed))
    atomic_store_explicit(&r->peak, delta, memory_order_relaxed);
}

/* RADV_PS5_LONGLIVED_ARENA (T175b; our own, after heydemoura/EglantinePS5 PR 104, branch
 * perf/heap-memory, commit 0a5efdc, docs/HEAP-MEMORY-RESEARCH.md "next steps" 1, which proposes it
 * and does not write it). Read from main in eden_heap_pool_init (headless/heap_pages.cpp); unset: off,
 * nothing below runs but one relaxed load on the paths named.
 *   1  route: a small request made inside a long-lived scope (eden_heap_long_lived_enter/leave, which
 *      the driver calls around the objects it keeps: lead-mesh-42) goes to one shared set of arenas
 *      kept for such objects, not to the thread's own; no thread cache either way; and the counts below.
 *   0  count only: no routing, but the scopes are tracked and the requests (and bytes) made inside
 *      them counted (ll_requests, ll_bytes), with the boot-end lines and the 20 s line, so a run
 *      without routing gives the same numbers to compare.
 * Why: on a cold shader build the driver's eight compile threads churn tens of GB of compiler
 * memory (ralloc) through their own arenas, and between those short-lived blocks they make the
 * objects the driver keeps (radv_shader and its code copy and debug info, cache objects, pipelines,
 * descriptor set layouts). One kept object holds a whole 2 MiB arena: heydemoura measured 231 arenas
 * (462 MiB) holding 83 MiB after a cold build; ours, about 650 MiB of slack a trim cannot give back.
 * With the kept objects elsewhere the compile arenas empty at the end of each build: their 2 MiB pages
 * free whole ("pinned" falls). With EDEN_HEAP_KEEP_ARENAS unset they also leave their records at the
 * next trim and become spares that any thread takes before the heap grows; with it set (play env)
 * they stay their records' and only their own thread or an adopt reuses them.
 * Review (T175b fix): a thread with no arena of its own never takes one of the set as its arena
 * (eden_heap_thread_arena's round-robin); after a request the set could not serve, the thread's next
 * EDEN_HEAP_ARENA_RETRY kept-object requests try only the set's current arena, without the creation
 * lock (eden_heap_ll_retry, as eden_heap_arena_allocate's retry); no new arena is made for a request
 * of EDEN_HEAP_ARENA_GROW_REQUEST or more (it takes room in the set's arenas or the usual path). */
#define EDEN_HEAP_LL_ARENAS 128u /* 256 MiB of kept objects at most; past it the thread's arenas */
static atomic_int eden_heap_ll_mode = -1;      /* -1 unset, 0 count only, 1 route */
static atomic_int eden_heap_ll_on;             /* 1: route (mode 1) */
static atomic_int eden_heap_ll_track;          /* 1: scopes tracked and counted (mode 0 or 1) */
static void *_Atomic eden_heap_ll_arena[EDEN_HEAP_LL_ARENAS];
static atomic_uint eden_heap_ll_count;   /* entries made (under the creation lock); never removed */
static atomic_uint eden_heap_ll_current; /* the entry asked first */
static atomic_uchar eden_heap_ll_slot[EDEN_HEAP_SLOTS]; /* 1: the slot's arena is one of the set */
static unsigned eden_heap_ll_depth[EDEN_HEAP_THREADS];  /* per record: scopes its thread is inside */
static unsigned eden_heap_ll_retry[EDEN_HEAP_THREADS];  /* per record: requests before the set is searched again */
static atomic_ullong eden_heap_ll_requests, eden_heap_ll_fallbacks, eden_heap_ll_scopes, eden_heap_ll_bytes;

static void eden_heap_apply(long bytes, long blocks) {
  if (blocks)
    atomic_fetch_add_explicit(&ps5_heap_blocks, (size_t)blocks, memory_order_relaxed);
  if (bytes >= 0)
    ps5_heap_resize_stats(0, (size_t)bytes);
  else
    ps5_heap_resize_stats((size_t)-bytes, 0);
}

/* The process's totals of the small requests the arenas could not take and of the frees of blocks
 * in the heap's spaces (eden_heap_spill_stats), with their processor cycles. */
static atomic_ullong eden_heap_spills, eden_heap_spill_cycles, eden_heap_shared_frees,
    eden_heap_shared_free_cycles;

/* The processor's cycle counter, where there is one (the console and the x86-64 host). */
static unsigned long long eden_heap_cycles(void) {
#if defined(__x86_64__)
  return __builtin_ia32_rdtsc();
#else
  return 0;
#endif
}

static void eden_heap_flush(struct eden_heap_thread *self) {
  eden_heap_apply(self->live, self->blocks);
  self->live = 0;
  self->blocks = 0;
  self->pending = 0;
  if (self->spills | self->shared_frees) {
    atomic_fetch_add_explicit(&eden_heap_spills, self->spills, memory_order_relaxed);
    atomic_fetch_add_explicit(&eden_heap_spill_cycles, self->spill_cycles, memory_order_relaxed);
    atomic_fetch_add_explicit(&eden_heap_shared_frees, self->shared_frees, memory_order_relaxed);
    atomic_fetch_add_explicit(&eden_heap_shared_free_cycles, self->shared_free_cycles, memory_order_relaxed);
    self->spills = self->spill_cycles = self->shared_frees = self->shared_free_cycles = 0;
  }
}

/* Small requests that went to the heap's spaces and frees of blocks there since the process began,
 * and the cycles they took (the counter's rate is the caller's to find). A thread's last 64
 * operations may not be in them yet. */
void eden_heap_spill_stats(unsigned long long *spills, unsigned long long *spill_cycles,
                           unsigned long long *frees, unsigned long long *free_cycles) {
  if (spills)
    *spills = atomic_load_explicit(&eden_heap_spills, memory_order_relaxed);
  if (spill_cycles)
    *spill_cycles = atomic_load_explicit(&eden_heap_spill_cycles, memory_order_relaxed);
  if (frees)
    *frees = atomic_load_explicit(&eden_heap_shared_frees, memory_order_relaxed);
  if (free_cycles)
    *free_cycles = atomic_load_explicit(&eden_heap_shared_free_cycles, memory_order_relaxed);
}

static void eden_big_thread_ended(struct eden_heap_thread *self); /* EDEN_UP_BIG_STACKS */

static void eden_heap_thread_exit(void *value) {
  struct eden_heap_thread *self = value;
  eden_big_thread_ended(self);
  for (unsigned bin = 0; bin < EDEN_TCACHE_BINS; ++bin) {
    while (self->bins[bin] != NULL) {
      void *block = self->bins[bin];
      self->bins[bin] = *(void **)block;
      eden_heap_free_owned(block);
    }
    __atomic_store_n(&self->counts[bin], 0, __ATOMIC_RELAXED);
  }
  eden_heap_flush(self);
  /* Its empty arenas go back; the record still holds the thread's, so no other thread takes it. */
  if (!eden_heap_keeps_arenas()) {
    eden_heap_create_acquire();
    eden_heap_return_record(self, 1);
    eden_heap_create_release();
  }
  atomic_store_explicit(&eden_heap_thread_used[self - eden_heap_threads], 0, memory_order_release);
}

/* The arenas there are now (the list's entries given back are not counted). */
unsigned eden_heap_arenas_created(void) {
  return atomic_load_explicit(&eden_heap_arena_count, memory_order_relaxed) -
         atomic_load_explicit(&eden_heap_arena_holes, memory_order_relaxed);
}

/* Empty arenas that left their records for the spares since the process began. */
unsigned long long eden_heap_arenas_given_back(void) {
  return atomic_load_explicit(&eden_heap_arenas_returned, memory_order_relaxed);
}

/* Spare arenas now: empty, held by no record, for the next thread that needs one. */
unsigned eden_heap_arenas_spare(void) {
  return atomic_load_explicit(&eden_heap_spare_count, memory_order_relaxed);
}

/* Spares taken instead of a new arena since the process began. */
unsigned long long eden_heap_arenas_reused(void) {
  return atomic_load_explicit(&eden_heap_arenas_taken_again, memory_order_relaxed);
}

/* Whether `arena` is a spare (under the creation lock). */
static int eden_heap_is_spare(const void *arena) {
  const unsigned count = atomic_load_explicit(&eden_heap_spare_count, memory_order_relaxed);
  for (unsigned i = 0; i < count; ++i)
    if (eden_heap_spare[i] == arena)
      return 1;
  return 0;
}

/* Memory the heap's pieces hold now (pieces given back are not counted). */
size_t eden_heap_committed(void) {
  return atomic_load_explicit(&eden_heap_committed_bytes, memory_order_relaxed);
}

/* What the heap holds (its pieces and its large blocks), what of that its blocks use (its own
 * count, which lags each thread's last 64 operations), and what it has given back. */
void eden_heap_usage(size_t *held, size_t *in_use, size_t *returned) {
  if (held)
    *held = atomic_load_explicit(&eden_heap_committed_bytes, memory_order_relaxed) +
            atomic_load_explicit(&eden_heap_large_bytes, memory_order_relaxed);
  if (in_use)
    *in_use = atomic_load_explicit(&ps5_heap_live_bytes, memory_order_relaxed);
  if (returned)
    *returned = atomic_load_explicit(&eden_heap_returned_bytes, memory_order_relaxed);
}

/* Memory of the large blocks alive now, and how many there are. */
size_t eden_heap_large_held(unsigned *blocks) {
  if (blocks)
    *blocks = atomic_load_explicit(&eden_heap_large_count, memory_order_relaxed);
  return atomic_load_explicit(&eden_heap_large_bytes, memory_order_relaxed);
}

/* The entry of a large block, or NULL: most addresses are not one, and with none alive
 * nothing is searched. */
static int eden_heap_large_find(const void *address) {
  if (atomic_load_explicit(&eden_heap_large_count, memory_order_acquire) == 0 || address == NULL)
    return -1;
  for (unsigned i = 0; i < EDEN_HEAP_LARGE_BLOCKS; ++i)
    if (atomic_load_explicit(&eden_heap_large[i].address, memory_order_acquire) == (uintptr_t)address)
      return (int)i;
  return -1;
}

/* A large block of its own (zeroed), or NULL: no entry free, or no memory. */
static void *eden_heap_large_allocate_inner(size_t bytes);
static void *eden_heap_large_allocate(size_t bytes) {
  if (bytes >= EDEN_HEAP_LARGE_REQUEST)
    return eden_heap_large_allocate_inner(bytes);
  const unsigned long long start = __builtin_ia32_rdtsc();
  void *block = eden_heap_large_allocate_inner(bytes);
  atomic_fetch_add_explicit(&eden_heap_mid_cycles, __builtin_ia32_rdtsc() - start, memory_order_relaxed);
  if (block == NULL) {
    atomic_fetch_add_explicit(&eden_heap_mid_refused, 1, memory_order_relaxed);
    return NULL;
  }
  atomic_fetch_add_explicit(&eden_heap_mid_taken, 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&eden_heap_mid_taken_bytes, bytes, memory_order_relaxed);
  atomic_fetch_add_explicit(&eden_heap_mid_count, 1, memory_order_relaxed);
  const size_t now = atomic_fetch_add_explicit(&eden_heap_mid_bytes, bytes, memory_order_relaxed) + bytes;
  size_t peak = atomic_load_explicit(&eden_heap_mid_peak, memory_order_relaxed);
  while (now > peak &&
         !atomic_compare_exchange_weak_explicit(&eden_heap_mid_peak, &peak, now, memory_order_relaxed, memory_order_relaxed)) {
  }
  return block;
}

static void *eden_heap_large_allocate_inner(size_t bytes) {
  for (unsigned i = 0; i < EDEN_HEAP_LARGE_BLOCKS; ++i) {
    uintptr_t expected = 0;
    if (!atomic_compare_exchange_strong_explicit(&eden_heap_large[i].address, &expected, 1, memory_order_acquire,
                                                 memory_order_relaxed))
      continue;
    void *block = eden_heap_pages(bytes);
    int split = 0;
    if (block == NULL) {
      /* No free piece of the pool that large: the block kept for a big image, when there is one
       * and the request is its size (EDEN_BIG_IMAGE_RESERVE_MIB). */
      block = eden_heap_reserve_pages(bytes);
      if (block != NULL)
        split = 2;
    }
    if (block == NULL && !eden_heap_big_image_refuses(bytes)) {
      /* Memory in smaller pieces behind one range; not for a big image's block that the reserve
       * could not serve, with EDEN_BIG_IMAGE_NO_SPLIT (eden_heap_allocate gives up on it). */
      block = eden_heap_split_pages(bytes);
      split = 1;
    }
    if (block == NULL) {
      atomic_store_explicit(&eden_heap_large[i].address, 0, memory_order_release);
      return NULL;
    }
    eden_heap_large[i].size = bytes;
    eden_heap_large[i].split = split;
    atomic_fetch_add_explicit(&eden_heap_large_bytes, bytes, memory_order_relaxed);
    atomic_fetch_add_explicit(&eden_heap_large_count, 1, memory_order_release);
    atomic_store_explicit(&eden_heap_large[i].address, (uintptr_t)block, memory_order_release);
    return block;
  }
  atomic_fetch_add_explicit(&eden_heap_large_full, 1, memory_order_relaxed);
  return NULL;
}

static void eden_heap_large_free(int entry) {
  void *block = (void *)atomic_load_explicit(&eden_heap_large[entry].address, memory_order_relaxed);
  const size_t bytes = eden_heap_large[entry].size;
  /* The entry stops naming the address before the memory goes: the next large block may be
   * given the same address at once. */
  atomic_store_explicit(&eden_heap_large[entry].address, 1, memory_order_release);
  const int mid = bytes < EDEN_HEAP_LARGE_REQUEST;
  const unsigned long long start = mid ? __builtin_ia32_rdtsc() : 0;
  if (eden_heap_large[entry].split == 2)
    eden_heap_reserve_pages_free(block, bytes);
  else if (eden_heap_large[entry].split)
    eden_heap_split_pages_free(block, bytes);
  else
    eden_heap_pages_free(block, bytes);
  if (mid) {
    atomic_fetch_add_explicit(&eden_heap_mid_cycles, __builtin_ia32_rdtsc() - start, memory_order_relaxed);
    atomic_fetch_sub_explicit(&eden_heap_mid_bytes, bytes, memory_order_relaxed);
    atomic_fetch_sub_explicit(&eden_heap_mid_count, 1, memory_order_relaxed);
  }
  atomic_fetch_sub_explicit(&eden_heap_large_bytes, bytes, memory_order_relaxed);
  atomic_fetch_sub_explicit(&eden_heap_large_count, 1, memory_order_release);
  atomic_store_explicit(&eden_heap_large[entry].address, 0, memory_order_release);
}

static int ps5_heap_ready(void) {
  int state = atomic_load_explicit(&ps5_heap_state, memory_order_acquire);
  if (state == 2)
    return 1;
  if (state != 0)
    return 0;

  int expected = 0;
  if (!atomic_compare_exchange_strong_explicit(
          &ps5_heap_state, &expected, 1, memory_order_acq_rel,
          memory_order_acquire))
    return expected == 2;

  /* Address space for the whole heap and memory for its first piece; where a range cannot be
   * reserved, 3 GiB at once as before. */
  void *base = eden_heap_reserve(PS5_OPENGL_HEAP_SIZE);
  size_t first_size = EDEN_HEAP_PIECE_SIZE;
  unsigned first_pieces = 1;
  if (base != NULL && eden_heap_commit(base, EDEN_HEAP_PIECE_SIZE) == 0) {
    eden_heap_grows = 1;
  } else {
    base = eden_heap_pages(EDEN_HEAP_WHOLE_SIZE);
    first_size = EDEN_HEAP_WHOLE_SIZE;
    first_pieces = (unsigned)(EDEN_HEAP_WHOLE_SIZE >> EDEN_HEAP_PIECE_SHIFT);
  }
  if (base == NULL) {
    atomic_store_explicit(&ps5_heap_state, -1, memory_order_release);
    return 0;
  }

  ps5_heap_base = base;
  eden_heap_range_size = eden_heap_grows ? PS5_OPENGL_HEAP_SIZE : EDEN_HEAP_WHOLE_SIZE;
  ps5_heap_mspace = sceLibcMspaceCreate("Eden-headless", base, first_size, 0);
  if (ps5_heap_mspace == NULL) {
    ps5_heap_base = NULL;
    if (!eden_heap_grows)
      eden_heap_pages_free(base, EDEN_HEAP_WHOLE_SIZE);
    atomic_store_explicit(&ps5_heap_state, -1, memory_order_release);
    return 0;
  }
  atomic_store_explicit(&eden_heap_spaces[0], ps5_heap_mspace, memory_order_relaxed);
  eden_heap_space_first[0] = 0;
  eden_heap_space_span[0] = first_pieces;
  for (unsigned piece = 0; piece < first_pieces; ++piece)
    atomic_store_explicit(&eden_heap_piece_space[piece], ps5_heap_mspace, memory_order_relaxed);
  atomic_store_explicit(&eden_heap_pieces_used, first_pieces, memory_order_relaxed);
  atomic_store_explicit(&eden_heap_committed_bytes, first_size, memory_order_relaxed);
  atomic_store_explicit(&eden_heap_space_count, 1, memory_order_release);
  /* libthr keeps specific data in its own storage; no recursion into malloc. */
  eden_heap_arena_key_valid = pthread_key_create(&eden_heap_arena_key, eden_heap_thread_exit) == 0;

  atomic_store_explicit(&ps5_heap_state, 2, memory_order_release);
  return 1;
}

/* A block inside the heap's range: an arena's or a piece's. */
static int eden_heap_in_range(const void *address) {
  /* Acquire publication before reading non-atomic heap metadata. */
  if (atomic_load_explicit(&ps5_heap_state, memory_order_acquire) != 2)
    return 0;
  uintptr_t value = (uintptr_t)address;
  uintptr_t base = (uintptr_t)ps5_heap_base;
  if (value < base || value - base >= eden_heap_range_size)
    return 0;
  /* A piece given back in part is not the heap's any more. */
  return atomic_load_explicit(&eden_heap_piece_state[(value - base) >> EDEN_HEAP_PIECE_SHIFT],
                              memory_order_relaxed) != 2;
}

/* Any block of this heap: one in its range, or a large block with memory of its own. */
static int ps5_heap_owns(const void *address) {
  return eden_heap_in_range(address) || eden_heap_large_find(address) >= 0;
}

/* The usable size of a block this heap owns. */
static size_t eden_heap_usable(const void *address) {
  if (eden_heap_in_range(address))
    return sceLibcMspaceMallocUsableSize(address);
  const int entry = eden_heap_large_find(address);
  return entry >= 0 ? eden_heap_large[entry].size : 0;
}

static uintptr_t eden_heap_slot(const void *address) {
  return ((uintptr_t)address >> EDEN_HEAP_ARENA_SHIFT) -
         ((uintptr_t)ps5_heap_base >> EDEN_HEAP_ARENA_SHIFT);
}

/* A block taken from the arena of its slot, and one given back to it. The count is the mspace's:
 * a block in a thread's cache is still taken. It is read, for an arena no thread is on, under the
 * creation lock (eden_heap_return_record): a block is counted after it is taken and after it is
 * freed, so a count read early says the arena is fuller than it is, never emptier. A realloc to
 * size 0 that frees the block leaves it counted (the console does not say); that arena stays. */
static void eden_heap_arena_took(const void *address) {
  atomic_fetch_add_explicit(&eden_heap_arena_live[eden_heap_slot(address)].blocks, 1, memory_order_relaxed);
}

static void eden_heap_arena_gave(const void *address) {
  atomic_fetch_sub_explicit(&eden_heap_arena_live[eden_heap_slot(address)].blocks, 1, memory_order_relaxed);
}

/* The slot tail belongs to the parent mspace and can hold ordinary allocations. */
static void *eden_heap_arena_owner(const void *address) {
  if (((uintptr_t)address & (EDEN_HEAP_ARENA_SIZE - 1u)) >= EDEN_HEAP_ARENA_PAYLOAD)
    return NULL;
  return atomic_load_explicit(&eden_heap_slots[eden_heap_slot(address)], memory_order_acquire);
}

/* The mspace owning an owned block: its slot's arena, else the space of its piece. */
static void *eden_heap_owner(const void *address) {
  void *arena = eden_heap_arena_owner(address);
  if (arena != NULL)
    return arena;
  const uintptr_t piece = ((uintptr_t)address - (uintptr_t)ps5_heap_base) >> EDEN_HEAP_PIECE_SHIFT;
  return atomic_load_explicit(&eden_heap_piece_space[piece], memory_order_acquire);
}

/* ---- Whole free pages (a measurement, reported as EDEN_HEAP_PAGES) ----
 * The bytes of the blocks handed out, per 2 MiB page of the range; a block kept in a thread's cache
 * counts as free. A heap that gives whole free pages back could return the committed pages that hold
 * none (an idle arena's page among them). This one cannot: Sony's mspaces keep their free lists in
 * the free memory itself. Asked for before replacing them (Mario Kart World's heap held 1952 MiB with
 * 1574 MiB of it counted in use). */
static _Atomic long eden_page_live[EDEN_HEAP_SLOTS];

static void eden_pages_count(const void *address, size_t bytes, int sign) {
  if (address == NULL || bytes == 0 || !eden_heap_in_range(address))
    return;
  uintptr_t at = (uintptr_t)address;
  const uintptr_t end = at + bytes;
  while (at < end) {
    const uintptr_t page_end = (at | (uintptr_t)(EDEN_HEAP_ARENA_SIZE - 1u)) + 1u;
    const uintptr_t part = (page_end < end ? page_end : end) - at;
    const uintptr_t slot = eden_heap_slot((const void *)at);
    if (slot < EDEN_HEAP_SLOTS)
      atomic_fetch_add_explicit(&eden_page_live[slot], sign > 0 ? (long)part : -(long)part,
                                memory_order_relaxed);
    at += part;
  }
}

/* The committed pages of the heap's pieces that hold no block in use, those of them that are an
 * arena's, and the bytes in use in the other pages. */
void eden_heap_page_usage(size_t *free_pages, size_t *idle_arenas, size_t *live) {
  size_t pages = 0, arenas = 0;
  long total = 0;
  if (atomic_load_explicit(&ps5_heap_state, memory_order_acquire) == 2) {
    const uintptr_t base = (uintptr_t)ps5_heap_base;
    const unsigned used = atomic_load_explicit(&eden_heap_pieces_used, memory_order_acquire);
    for (unsigned piece = 0; piece < used && piece < EDEN_HEAP_PIECES; ++piece) {
      if (atomic_load_explicit(&eden_heap_piece_state[piece], memory_order_relaxed) != 0)
        continue;
      const uintptr_t start = base + ((uintptr_t)piece << EDEN_HEAP_PIECE_SHIFT);
      const uintptr_t end = start + EDEN_HEAP_PIECE_SIZE;
      for (uintptr_t page = (start + EDEN_HEAP_ARENA_SIZE - 1u) & ~(uintptr_t)(EDEN_HEAP_ARENA_SIZE - 1u);
           page + EDEN_HEAP_ARENA_SIZE <= end; page += EDEN_HEAP_ARENA_SIZE) {
        const uintptr_t slot = eden_heap_slot((const void *)page);
        if (slot >= EDEN_HEAP_SLOTS)
          continue;
        const long bytes = atomic_load_explicit(&eden_page_live[slot], memory_order_relaxed);
        if (bytes > 0) {
          total += bytes;
          continue;
        }
        ++pages;
        if (atomic_load_explicit(&eden_heap_slots[slot], memory_order_relaxed) != NULL)
          ++arenas;
      }
    }
  }
  *free_pages = pages << EDEN_HEAP_ARENA_SHIFT;
  *idle_arenas = arenas << EDEN_HEAP_ARENA_SHIFT;
  *live = total > 0 ? (size_t)total : 0;
}

/* ---- the heap's own spaces ---- */

enum { EDEN_HEAP_MALLOC, EDEN_HEAP_CALLOC, EDEN_HEAP_ALIGNED };

/* Frees a block of a space (not an arena's), counting it off its space. */
static void eden_heap_space_free(void *address) {
  const uintptr_t piece = ((uintptr_t)address - (uintptr_t)ps5_heap_base) >> EDEN_HEAP_PIECE_SHIFT;
  const unsigned slot = atomic_load_explicit(&eden_heap_piece_slot[piece], memory_order_acquire);
  sceLibcMspaceFree(atomic_load_explicit(&eden_heap_piece_space[piece], memory_order_acquire), address);
  atomic_fetch_sub_explicit(&eden_heap_space_blocks[slot], 1, memory_order_release);
}

/* Frees an owned block (not a large one) to its arena or its space. */
static void eden_heap_free_owned(void *address) {
  void *arena = eden_heap_arena_owner(address);
  if (arena != NULL) {
    sceLibcMspaceFree(arena, address);
    eden_heap_arena_gave(address);
  } else
    eden_heap_space_free(address);
}

/* One request to one mspace: malloc(a), calloc(a, b) or an a-aligned block of b bytes. */
static void *eden_heap_space_try(void *space, int kind, size_t a, size_t b) {
  if (kind == EDEN_HEAP_MALLOC)
    return sceLibcMspaceMalloc(space, a);
  if (kind == EDEN_HEAP_CALLOC)
    return sceLibcMspaceCalloc(space, a, b);
  void *address = NULL;
  return sceLibcMspacePosixMemalign(space, &address, a, b) == 0 ? address : NULL;
}

int sceKernelDebugOutText(int channel, const char *text);

/* Why the heap could not grow, for the kernel log: the allocation that asked is about to fail.
 * No stdio here (it allocates), and only the first few times. */
static void eden_heap_grow_refused(const char *line) {
  static atomic_uint reported;
  if (atomic_fetch_add_explicit(&reported, 1, memory_order_relaxed) < 8)
    sceKernelDebugOutText(0, line);
}

/* The last allocation the heap could not make, for the crash report (headless/crash_report.cpp):
 * a std::bad_alloc names neither the size nor the reason, and the report's memory line is read
 * after the failure, when other threads may have given memory back. `why` is the last reason the
 * heap gave for not growing (EDEN_HEAP_WHY_*), on whichever thread asked last. */
enum {
  EDEN_HEAP_WHY_NONE = 0,     /* it did not need to grow, or it grew */
  EDEN_HEAP_WHY_NO_SPACE = 1, /* memory, but no allocator over it */
  EDEN_HEAP_WHY_NO_MEMORY = 2,
  EDEN_HEAP_WHY_AT_LIMIT = 3,
  EDEN_HEAP_WHY_OVERFLOW = 4  /* calloc's count * size does not fit */
};
static atomic_uint eden_heap_grow_why;
static atomic_size_t eden_heap_failed_bytes;
static atomic_uint eden_heap_failed_why;

static void eden_heap_note_failure(size_t bytes, unsigned why) {
  atomic_store_explicit(&eden_heap_failed_bytes, bytes, memory_order_relaxed);
  atomic_store_explicit(&eden_heap_failed_why, why, memory_order_relaxed);
}

static void eden_heap_failed(size_t bytes, unsigned why) {
  eden_heap_note_failure(bytes, why);
  atomic_fetch_add_explicit(&ps5_heap_failures, 1, memory_order_relaxed);
}

/* How many allocations failed, and the size and reason of the last one (0: none). */
size_t eden_heap_last_failure(unsigned *why, size_t *count) {
  if (why)
    *why = atomic_load_explicit(&eden_heap_failed_why, memory_order_relaxed);
  if (count)
    *count = atomic_load_explicit(&ps5_heap_failures, memory_order_relaxed);
  return atomic_load_explicit(&eden_heap_failed_bytes, memory_order_relaxed);
}

static void eden_heap_grow_acquire(void) {
  for (unsigned spins = 0; atomic_flag_test_and_set_explicit(&eden_heap_grow_lock, memory_order_acquire); ++spins) {
    if (spins < 64)
      __builtin_ia32_pause();
    else
      sceKernelUsleep(50);
  }
}

/* EDEN_UP_HEAP_STEAL (eden_heap_steal, after eden_heap_arena_rescue): its mode, 0 off. */
static atomic_int eden_heap_steal_mode;
/* Mode 3: the process time (us, | 1) at which the pool last refused a growth; eden_heap_main asks
 * the idle arenas before the pool for EDEN_HEAP_STEAL_BEFORE_GROW_US after it. Written only while
 * the mode is 3. */
static atomic_ullong eden_heap_steal_pool_no_us;
#define EDEN_HEAP_STEAL_BEFORE_GROW_US 1000000ull
/* EDEN_TEST_HEAP_NO_GROW=<every>[,<after_s>] (a test, off unless set): from <after_s> seconds after
 * main read it, every <every>-th growth of the heap is refused as if the pool had no memory (reason
 * 2, "the console has no memory for a new piece"), so the heap's brink is reached on any route
 * without taking memory away from anyone. Without EDEN_UP_HEAP_STEAL a refused growth fails its
 * request, as a real one does. */
static atomic_uint eden_heap_no_grow_every;
static atomic_ullong eden_heap_no_grow_from_us, eden_heap_no_grow_seen, eden_heap_no_grow_refused;
uint64_t sceKernelGetProcessTime(void);

static int eden_heap_test_no_grow(void) {
  const unsigned every = atomic_load_explicit(&eden_heap_no_grow_every, memory_order_relaxed);
  if (every == 0)
    return 0;
  const unsigned long long from = atomic_load_explicit(&eden_heap_no_grow_from_us, memory_order_relaxed);
  if (from != 0 && (unsigned long long)sceKernelGetProcessTime() < from)
    return 0;
  if ((atomic_fetch_add_explicit(&eden_heap_no_grow_seen, 1, memory_order_relaxed) + 1) % every != 0)
    return 0;
  atomic_fetch_add_explicit(&eden_heap_no_grow_refused, 1, memory_order_relaxed);
  return 1;
}

void eden_heap_set_no_grow(unsigned every, unsigned after_s) {
  atomic_store_explicit(&eden_heap_no_grow_from_us,
                        after_s != 0 ? (unsigned long long)sceKernelGetProcessTime() + (unsigned long long)after_s * 1000000ull : 0,
                        memory_order_relaxed);
  atomic_store_explicit(&eden_heap_no_grow_every, every, memory_order_relaxed);
}

/* What made the heap grow: the pieces each kind of request took, and the largest of them. */
enum {
  EDEN_HEAP_GREW_ARENA, EDEN_HEAP_GREW_SMALL, EDEN_HEAP_GREW_1M, EDEN_HEAP_GREW_8M, EDEN_HEAP_GREW_32M,
  EDEN_HEAP_GREW_LARGE, EDEN_HEAP_GREW_KINDS
};
static atomic_uint eden_heap_grew_pieces[EDEN_HEAP_GREW_KINDS];
static atomic_size_t eden_heap_grew_largest[EDEN_HEAP_GREW_KINDS];

static struct eden_heap_thread *eden_heap_self(void);

/* The spaces a request may use: an arena's (its size and alignment together) and anything under
 * 1 MiB in the spaces that stay, the rest in the others. */
static unsigned eden_heap_request_class(size_t bytes) {
  return bytes >= ((size_t)1 << 20) && bytes != (size_t)EDEN_HEAP_ARENA_SIZE + EDEN_HEAP_ARENA_PAYLOAD
             ? EDEN_HEAP_CLASS_SCRATCH
             : EDEN_HEAP_CLASS_STAYS;
}

/* Called under the growth lock. */
static void eden_heap_note_growth(size_t bytes, unsigned pieces) {
  const unsigned kind = bytes == (size_t)EDEN_HEAP_ARENA_SIZE + EDEN_HEAP_ARENA_PAYLOAD ? EDEN_HEAP_GREW_ARENA
                        : bytes < EDEN_HEAP_SMALL_REQUEST                               ? EDEN_HEAP_GREW_SMALL
                        : bytes < ((size_t)1 << 20)                                     ? EDEN_HEAP_GREW_1M
                        : bytes < ((size_t)8 << 20)                                     ? EDEN_HEAP_GREW_8M
                        : bytes < EDEN_HEAP_LARGE_REQUEST                               ? EDEN_HEAP_GREW_32M
                                                                                        : EDEN_HEAP_GREW_LARGE;
  atomic_fetch_add_explicit(&eden_heap_grew_pieces[kind], pieces, memory_order_relaxed);
  if (bytes > atomic_load_explicit(&eden_heap_grew_largest[kind], memory_order_relaxed))
    atomic_store_explicit(&eden_heap_grew_largest[kind], bytes, memory_order_relaxed);
  struct eden_heap_thread *self = eden_heap_self();
  if (self != NULL)
    __atomic_fetch_add(&self->grown, pieces, __ATOMIC_RELAXED);
}

/* ---- EDEN_UP_HEAP_HOLD_AT_LOAD=<s> (t174a, our own; 1 to 600, digits only; unset, 0 or anything
 * else: off; read from main in eden_heap_pool_init, headless/heap_pages.cpp) ----
 * Why (13 Nov 2026, T166, the load into the area past the field): run1395's heap gave a 32 MiB piece
 * back to the pool ("gave back 32 MiB: the pool is short", stderr 983) 2 s into the load, and 9 s
 * later the image table's doubling needed 10,354,696 bytes with the pool at 15-27 MiB free in pieces
 * of 2 MiB or less: EDEN_NEW_WAIT gave up after 2000 ms and the program ended (stderr 2010-2017). The
 * heap grew 64-96 MiB at that load in every start of the batch (1248 -> 1344, 1280 -> 1344 MiB).
 * What it does:
 *  - A hold: from each call of eden_heap_hold_area (the texture cache, when AreaEvict's window opens
 *    or is extended by a burst of new images: core patch 0893) or eden_heap_hold_begin (the driver
 *    refused a request of 64 MiB or more: headless/performance.cpp, eden_radv_out_of_memory; an
 *    area's big image comes before its burst of images: run1399, its crash log
 *    975-1012: the driver refused 304,132 KiB, the trim gave it the heap's 32 MiB, and 1.6 s later a
 *    4 MiB heap request waited 1800 ms and failed; the burst was at line 1605) until <s> seconds after
 *    the last one,
 *    eden_heap_trim gives no space back (empty arenas still go to the spares; nothing leaves the
 *    heap). Held: TrimHeap ("the pool is short", "the driver found no memory") and the builders'
 *    give-back; operator new's and the door's waits, the JIT's and the guest's waits are let
 *    through (eden_heap_trim_unheld, see Bounds below).
 *  - A spare: EDEN_UP_HEAP_HOLD_SPARE_MIB (32 or 64; unset: 32; 0: none) of pieces committed while the
 *    pool has room (at main's start, and from the memory report every 2 s at most while the pool's
 *    largest free block is 256 MiB or more and no hold runs), no space over them.
 *    When the pool refuses a growth (in a hold or not), the spare becomes that space. Not counted in
 *    the heap's "holds" figure while it is a spare (the census's heap= and the adopt slack gate are
 *    as without it); its line says what it holds.
 * Bounds (review, 13 Nov): a hold runs at most 3 x <s> s however often it is extended (the texture
 * cache's window, driver refusals), and after one that ran that long none begins for <s> s. The hold
 * stops only the give-backs to others: TrimHeap ("the pool is short", "the driver found no memory")
 * and the builders' give-back (EDEN_UP_BUILD_GIVE_BACK). These trim through eden_heap_trim_unheld,
 * which a hold does not stop: the guest's block wait (EDEN_UP_BLOCK_WAIT_TRIM,
 * src/guest_memory_service.inc; a game write with no 2 MiB block is the old crash, and waveydude's
 * W-05 died with 26 MiB of empty heap pages and the pool at 1 MiB), the JIT code cache's waits
 * (headless/jit-allocator.h; it aborts when the wait finds nothing) and operator new's own waits
 * (headless/main.cpp: give the empty spaces back, then grow one large enough; run1402's 9,529,344
 * byte request was served that way after 480 ms and 2 trims, its crash log 1145-1146).
 * Its price: the spare's 32-64 MiB are out of the pool for good while the switch is on, and during a
 * hold an empty piece stays in the heap where graphics might have used it. Off: one relaxed load in
 * eden_heap_trim and in a growth the pool refused, and eden_heap_hold_area returns at once.
 * Counted: EDEN_UP_HEAP_HOLD line every 10 s while on (headless/performance.cpp). */
static atomic_uint eden_heap_hold_seconds;           /* 0: off */
static atomic_uint eden_heap_hold_spare_pieces_want; /* pieces the spare should hold */
static atomic_ullong eden_heap_hold_until_us;        /* process time (us) the hold ends; 0: none yet */
static atomic_ullong eden_heap_hold_since_us;        /* when the running hold began */
static atomic_ullong eden_heap_hold_quiet_until_us;  /* no hold begins before this (after a capped one) */
static atomic_ullong eden_heap_hold_capped, eden_heap_hold_unheld_trims;
/* ---- T178b EDEN_UP_HEAP_HOLD_REFILL_MIB=<n> (32 to 4096, digits only; unset or anything else: off, the
 * spare as T177 keeps it). Why: T175's F3 (13 Nov, with the stand) had its spare serve one refused growth
 * and never get it back (the refill waited for 256 MiB in one block and no hold), and C3 died 2 s later
 * on a plain 8,294,400-byte operator new nobody caught. With the switch:
 *  - a refused growth takes only the pieces it needs from the spare; the rest stays a spare for the next
 *    one (with the 32 MiB pieces this matters for a 64 MiB spare: two growths of up to 31 MiB each);
 *  - the memory report (every 100 ms at most) tops the spare up to its size as soon as the pool's largest
 *    free block is <n> MiB or more, during a hold too (a hold keeps the heap's empty spaces; it never
 *    stopped the heap from growing), one try every 250 ms at most;
 *  - the spare is up to EDEN_HEAP_HOLD_RUNS runs of pieces (a run is a range in address: a growth is one
 *    space, so it is served by one run that is large enough).
 * Its own line: EDEN_UP_HEAP_HOLD_REFILL every 10 s while its counts move (eden_heap_hold_refill, outside
 * every lock). The EDEN_UP_HEAP_HOLD line keeps its fields: "spare ... MiB ready" is all runs together,
 * "taken" counts each run committed, "served" each growth the spare made. */
#define EDEN_HEAP_HOLD_RUNS 4u
static atomic_ullong eden_heap_hold_refill_bytes;    /* 0: off */
static atomic_ullong eden_heap_hold_refill_last_us;  /* the last try of the refill */
static atomic_ullong eden_heap_hold_refills, eden_heap_hold_refills_in_hold, eden_heap_hold_refill_tries,
    eden_heap_hold_served_partial, eden_heap_hold_refill_reported_us, eden_heap_hold_refill_reported_sum;
/* The spare's runs [first, first + span), under the growth lock; span 0: an empty run. Off: run 0 only. */
static unsigned eden_heap_hold_run_first[EDEN_HEAP_HOLD_RUNS], eden_heap_hold_run_span[EDEN_HEAP_HOLD_RUNS];
static atomic_uint eden_heap_hold_spare_now; /* pieces in all runs, for the reports */
/* Under the growth lock: the pieces the runs hold. */
static unsigned eden_heap_hold_spare_total_locked(void) {
  unsigned total = 0;
  for (unsigned run = 0; run < EDEN_HEAP_HOLD_RUNS; ++run)
    total += eden_heap_hold_run_span[run];
  return total;
}
/* Under the growth lock: a run that can serve `pieces` (the smallest that can), or EDEN_HEAP_HOLD_RUNS. */
static unsigned eden_heap_hold_run_for_locked(size_t pieces) {
  unsigned best = EDEN_HEAP_HOLD_RUNS;
  for (unsigned run = 0; run < EDEN_HEAP_HOLD_RUNS; ++run)
    if (eden_heap_hold_run_span[run] != 0 && pieces <= eden_heap_hold_run_span[run] &&
        (best == EDEN_HEAP_HOLD_RUNS || eden_heap_hold_run_span[run] < eden_heap_hold_run_span[best]))
      best = run;
  return best;
}
static atomic_ullong eden_heap_hold_last_fill_us;
/* Counts: holds begun, trims that kept the spaces, spares taken (and their MiB), takes the pool
 * refused, growths the spare served (and the bytes asked), of them inside a hold. */
static atomic_ullong eden_heap_hold_begun, eden_heap_hold_begun_area, eden_heap_hold_trims_kept, eden_heap_hold_spare_taken,
    eden_heap_hold_spare_taken_mib, eden_heap_hold_spare_refused, eden_heap_hold_served,
    eden_heap_hold_served_bytes, eden_heap_hold_served_in_hold;

static int eden_heap_hold_active(unsigned long long now_us) {
  if (atomic_load_explicit(&eden_heap_hold_seconds, memory_order_relaxed) == 0)
    return 0;
  const unsigned long long until = atomic_load_explicit(&eden_heap_hold_until_us, memory_order_relaxed);
  return until != 0 && now_us < until;
}

/* The spare, under the growth lock: pieces given back earlier in a row first, else the next ones,
 * committed now. 1: there is a spare (now or already). */
static int eden_heap_hold_fill_locked(void) {
  /* T178b refill: the pieces missing, into an empty run; off: only when the spare is empty (as T177). */
  const int refill = atomic_load_explicit(&eden_heap_hold_refill_bytes, memory_order_relaxed) != 0;
  const unsigned held = eden_heap_hold_spare_total_locked();
  const unsigned full = atomic_load_explicit(&eden_heap_hold_spare_pieces_want, memory_order_relaxed);
  if (held != 0 && (!refill || held >= full))
    return 1;
  unsigned slot_run = EDEN_HEAP_HOLD_RUNS;
  for (unsigned run = 0; run < (refill ? EDEN_HEAP_HOLD_RUNS : 1u); ++run)
    if (eden_heap_hold_run_span[run] == 0) {
      slot_run = run;
      break;
    }
  if (slot_run == EDEN_HEAP_HOLD_RUNS)
    return held != 0;
  const unsigned want = full - held;
  if (want == 0 || !eden_heap_grows)
    return held != 0;
  const unsigned used = atomic_load_explicit(&eden_heap_pieces_used, memory_order_relaxed);
  unsigned first = used;
  for (unsigned at = 0, run = 0; at < used; ++at) {
    run = atomic_load_explicit(&eden_heap_piece_state[at], memory_order_relaxed) == 1 ? run + 1 : 0;
    if (run == want) {
      first = at + 1 - want;
      break;
    }
  }
  if (first == used && want > EDEN_HEAP_PIECES - used)
    return held != 0;
  void *start = (char *)ps5_heap_base + ((size_t)first << EDEN_HEAP_PIECE_SHIFT);
  if (eden_heap_commit(start, (size_t)want << EDEN_HEAP_PIECE_SHIFT) != 0) {
    atomic_fetch_add_explicit(&eden_heap_hold_spare_refused, 1, memory_order_relaxed);
    return held != 0;
  }
  /* 3: the spare's (committed, no space over it; nothing is ever freed into it). */
  for (unsigned piece = first; piece < first + want; ++piece)
    atomic_store_explicit(&eden_heap_piece_state[piece], 3, memory_order_relaxed);
  if (first == used)
    atomic_store_explicit(&eden_heap_pieces_used, used + want, memory_order_relaxed);
  eden_heap_hold_run_first[slot_run] = first;
  eden_heap_hold_run_span[slot_run] = want;
  atomic_store_explicit(&eden_heap_hold_spare_now, held + want, memory_order_relaxed);
  atomic_fetch_add_explicit(&eden_heap_hold_spare_taken, 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&eden_heap_hold_spare_taken_mib, (unsigned long long)want << (EDEN_HEAP_PIECE_SHIFT - 20),
                            memory_order_relaxed);
  return 1;
}

static void eden_heap_hold_fill(void) {
  const unsigned want = atomic_load_explicit(&eden_heap_hold_spare_pieces_want, memory_order_relaxed);
  const unsigned now = atomic_load_explicit(&eden_heap_hold_spare_now, memory_order_relaxed);
  /* Off: only an empty spare is filled (T177); T178b refill: one that holds less than its size too. */
  const int refill = atomic_load_explicit(&eden_heap_hold_refill_bytes, memory_order_relaxed) != 0;
  if (atomic_load_explicit(&ps5_heap_state, memory_order_acquire) != 2 || want == 0 ||
      (refill ? now >= want : now != 0))
    return;
  atomic_store_explicit(&eden_heap_hold_last_fill_us, (unsigned long long)sceKernelGetProcessTime(),
                        memory_order_relaxed);
  eden_heap_grow_acquire();
  (void)eden_heap_hold_fill_locked();
  atomic_flag_clear_explicit(&eden_heap_grow_lock, memory_order_release);
}

/* For eden_heap_grow's test, under the growth lock: could the spare serve `pieces`? */
static int eden_heap_hold_spare_fits(size_t pieces) {
  return eden_heap_hold_run_for_locked(pieces) != EDEN_HEAP_HOLD_RUNS;
}

/* For eden_heap_grow, under the growth lock, after the pool refused `pieces`: the spare's range when
 * it is that large. Off: a run is taken whole, its pieces become the new space (T177). T178b refill: only
 * `pieces` of the smallest run that has them (its first ones); the rest stays a spare. */
static int eden_heap_hold_spare_take(size_t pieces, size_t bytes, unsigned *first, size_t *span) {
  const unsigned run = eden_heap_hold_run_for_locked(pieces);
  if (run == EDEN_HEAP_HOLD_RUNS)
    return 0;
  const int partial = atomic_load_explicit(&eden_heap_hold_refill_bytes, memory_order_relaxed) != 0;
  *first = eden_heap_hold_run_first[run];
  *span = partial ? pieces : eden_heap_hold_run_span[run];
  eden_heap_hold_run_first[run] += (unsigned)*span;
  eden_heap_hold_run_span[run] -= (unsigned)*span;
  if (eden_heap_hold_run_span[run] == 0)
    eden_heap_hold_run_first[run] = 0;
  const unsigned left = eden_heap_hold_spare_total_locked();
  atomic_store_explicit(&eden_heap_hold_spare_now, left, memory_order_relaxed);
  if (left != 0)
    atomic_fetch_add_explicit(&eden_heap_hold_served_partial, 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&eden_heap_hold_served, 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&eden_heap_hold_served_bytes, bytes, memory_order_relaxed);
  if (eden_heap_hold_active((unsigned long long)sceKernelGetProcessTime()))
    atomic_fetch_add_explicit(&eden_heap_hold_served_in_hold, 1, memory_order_relaxed);
  return 1;
}

/* main (heap_pages.cpp): the switch, and the spare's size in MiB (32 or 64; 0: none). */
void eden_heap_set_hold(unsigned seconds, unsigned spare_mib) {
  atomic_store_explicit(&eden_heap_hold_spare_pieces_want, spare_mib >> (EDEN_HEAP_PIECE_SHIFT - 20),
                        memory_order_relaxed);
  /* T178b EDEN_UP_HEAP_HOLD_REFILL_MIB (above): read here, with the hold's own switches (main, before
   * any game thread). Digits only, 32 to 4096; anything else: off. */
  unsigned long long refill_mib = 0;
  const char *refill = getenv("EDEN_UP_HEAP_HOLD_REFILL_MIB");
  if (seconds != 0 && refill != NULL && *refill != '\0') {
    const char *at = refill;
    while (*at >= '0' && *at <= '9' && refill_mib <= 4096)
      refill_mib = refill_mib * 10 + (unsigned long long)(*at++ - '0');
    if (*at != '\0' || refill_mib < 32 || refill_mib > 4096)
      refill_mib = 0;
    if (refill_mib != 0)
      fprintf(stderr,
              "EDEN_UP_HEAP_HOLD_REFILL_MIB=%s: a refused growth takes only the spare's pieces it needs, and the "
              "spare is topped up whenever the pool has a free block of that many MiB, in a hold too\n",
              refill);
    else
      fprintf(stderr, "EDEN_UP_HEAP_HOLD_REFILL_MIB=%s is not 32 to 4096: off\n", refill);
  }
  atomic_store_explicit(&eden_heap_hold_refill_bytes, refill_mib << 20, memory_order_relaxed);
  atomic_store_explicit(&eden_heap_hold_seconds, seconds, memory_order_relaxed);
  if (seconds != 0)
    eden_heap_hold_fill();
}

/* A hold begins or goes on for <s> s from now. 1: it began now (none was running); 0: it went on,
 * or the switch is off. */
int eden_heap_hold_begin(void) {
  const unsigned seconds = atomic_load_explicit(&eden_heap_hold_seconds, memory_order_relaxed);
  if (seconds == 0)
    return 0;
  const unsigned long long now = (unsigned long long)sceKernelGetProcessTime();
  const unsigned long long span_us = (unsigned long long)seconds * 1000000ull;
  const int began = !eden_heap_hold_active(now);
  unsigned long long since = atomic_load_explicit(&eden_heap_hold_since_us, memory_order_relaxed);
  if (began) {
    /* After a hold that ran its longest, <s> s with none. */
    if (now < atomic_load_explicit(&eden_heap_hold_quiet_until_us, memory_order_relaxed))
      return 0;
    since = now;
    atomic_store_explicit(&eden_heap_hold_since_us, since, memory_order_relaxed);
  }
  /* At most 3 x <s> s in one hold, however often it is extended. */
  unsigned long long until = now + span_us;
  if (until > since + 3 * span_us) {
    until = since + 3 * span_us;
    if (atomic_exchange_explicit(&eden_heap_hold_quiet_until_us, until + span_us, memory_order_relaxed) !=
        until + span_us)
      atomic_fetch_add_explicit(&eden_heap_hold_capped, 1, memory_order_relaxed);
  }
  atomic_store_explicit(&eden_heap_hold_until_us, until, memory_order_relaxed);
  /* No spare taken here: a hold begins when the pool is already short, and the driver may be waiting
   * for that memory. The spare is taken while the pool has room (eden_heap_set_hold, refill). */
  if (began)
    atomic_fetch_add_explicit(&eden_heap_hold_begun, 1, memory_order_relaxed);
  return began;
}

/* The texture cache (0893): an area change began or goes on. */
void eden_heap_hold_area(void) {
  if (eden_heap_hold_begin())
    atomic_fetch_add_explicit(&eden_heap_hold_begun_area, 1, memory_order_relaxed);
}

/* The memory report (performance.cpp), with the pool's largest free block: the spare again, at
 * most every 2 s, while the pool has 256 MiB in one block and no hold runs. */
static void eden_heap_hold_refill_report(unsigned long long now);
void eden_heap_hold_refill(unsigned long long largest_free) {
  if (atomic_load_explicit(&eden_heap_hold_seconds, memory_order_relaxed) == 0)
    return;
  /* T178b EDEN_UP_HEAP_HOLD_REFILL_MIB: the missing pieces as soon as the pool has a block of <n> MiB, in a
   * hold too, one try every 250 ms at most; and its line. Off: as T177 below. */
  const unsigned long long refill_bytes = atomic_load_explicit(&eden_heap_hold_refill_bytes, memory_order_relaxed);
  if (refill_bytes != 0) {
    const unsigned long long now = (unsigned long long)sceKernelGetProcessTime();
    const unsigned before = atomic_load_explicit(&eden_heap_hold_spare_now, memory_order_relaxed);
    if (before < atomic_load_explicit(&eden_heap_hold_spare_pieces_want, memory_order_relaxed) &&
        largest_free >= refill_bytes &&
        now - atomic_load_explicit(&eden_heap_hold_refill_last_us, memory_order_relaxed) >= 250000ull) {
      atomic_store_explicit(&eden_heap_hold_refill_last_us, now, memory_order_relaxed);
      atomic_fetch_add_explicit(&eden_heap_hold_refill_tries, 1, memory_order_relaxed);
      eden_heap_hold_fill();
      if (atomic_load_explicit(&eden_heap_hold_spare_now, memory_order_relaxed) > before) {
        atomic_fetch_add_explicit(&eden_heap_hold_refills, 1, memory_order_relaxed);
        if (eden_heap_hold_active(now))
          atomic_fetch_add_explicit(&eden_heap_hold_refills_in_hold, 1, memory_order_relaxed);
      }
    }
    eden_heap_hold_refill_report(now);
    return;
  }
  if (atomic_load_explicit(&eden_heap_hold_spare_now, memory_order_relaxed) != 0 ||
      largest_free < ((unsigned long long)256 << 20))
    return;
  const unsigned long long now = (unsigned long long)sceKernelGetProcessTime();
  if (eden_heap_hold_active(now) ||
      now - atomic_load_explicit(&eden_heap_hold_last_fill_us, memory_order_relaxed) < 2000000ull)
    return;
  eden_heap_hold_fill();
}

/* T178b: the refill's line, every 10 s while its counts move (from eden_heap_hold_refill: the memory
 * report's thread, no lock of the heap held). */
static void eden_heap_hold_refill_report(unsigned long long now) {
  const unsigned long long last = atomic_load_explicit(&eden_heap_hold_refill_reported_us, memory_order_relaxed);
  if (last != 0 && now - last < 10000000ull)
    return;
  const unsigned long long tries = atomic_load_explicit(&eden_heap_hold_refill_tries, memory_order_relaxed);
  const unsigned long long refills = atomic_load_explicit(&eden_heap_hold_refills, memory_order_relaxed);
  const unsigned long long in_hold = atomic_load_explicit(&eden_heap_hold_refills_in_hold, memory_order_relaxed);
  const unsigned long long partial = atomic_load_explicit(&eden_heap_hold_served_partial, memory_order_relaxed);
  const unsigned long long served = atomic_load_explicit(&eden_heap_hold_served, memory_order_relaxed);
  const unsigned long long refused = atomic_load_explicit(&eden_heap_hold_spare_refused, memory_order_relaxed);
  const unsigned long long sum = tries + refills + in_hold + partial + served + refused;
  atomic_store_explicit(&eden_heap_hold_refill_reported_us, now, memory_order_relaxed);
  if (last != 0 && sum == atomic_load_explicit(&eden_heap_hold_refill_reported_sum, memory_order_relaxed))
    return;
  atomic_store_explicit(&eden_heap_hold_refill_reported_sum, sum, memory_order_relaxed);
  char line[320];
  const int length = snprintf(
      line, sizeof(line),
      "EDEN_UP_HEAP_HOLD_REFILL at %llu MiB: spare %llu of %llu MiB; refills %llu (%llu in a hold) of %llu tries, "
      "refused by the pool %llu; served %llu growths, %llu of them leaving pieces for the next\n",
      atomic_load_explicit(&eden_heap_hold_refill_bytes, memory_order_relaxed) >> 20,
      (unsigned long long)atomic_load_explicit(&eden_heap_hold_spare_now, memory_order_relaxed)
          << (EDEN_HEAP_PIECE_SHIFT - 20),
      (unsigned long long)atomic_load_explicit(&eden_heap_hold_spare_pieces_want, memory_order_relaxed)
          << (EDEN_HEAP_PIECE_SHIFT - 20),
      refills, in_hold, tries, refused, served, partial);
  if (length > 0)
    fputs(line, stderr);
}

/* For the EDEN_UP_HEAP_HOLD line: 0 when off, else 1 and the counts. */
int eden_heap_hold_stats(unsigned long long *values, unsigned count) {
  const unsigned seconds = atomic_load_explicit(&eden_heap_hold_seconds, memory_order_relaxed);
  if (seconds == 0)
    return 0;
  const unsigned long long now = (unsigned long long)sceKernelGetProcessTime();
  const unsigned long long until = atomic_load_explicit(&eden_heap_hold_until_us, memory_order_relaxed);
  const unsigned long long all[14] = {
      seconds,
      eden_heap_hold_active(now) ? (until - now) / 1000000ull + 1 : 0,
      atomic_load_explicit(&eden_heap_hold_begun, memory_order_relaxed),
      atomic_load_explicit(&eden_heap_hold_trims_kept, memory_order_relaxed),
      (unsigned long long)atomic_load_explicit(&eden_heap_hold_spare_now, memory_order_relaxed)
          << (EDEN_HEAP_PIECE_SHIFT - 20),
      atomic_load_explicit(&eden_heap_hold_spare_taken, memory_order_relaxed),
      atomic_load_explicit(&eden_heap_hold_spare_taken_mib, memory_order_relaxed),
      atomic_load_explicit(&eden_heap_hold_spare_refused, memory_order_relaxed),
      atomic_load_explicit(&eden_heap_hold_served, memory_order_relaxed),
      atomic_load_explicit(&eden_heap_hold_served_bytes, memory_order_relaxed),
      atomic_load_explicit(&eden_heap_hold_served_in_hold, memory_order_relaxed),
      atomic_load_explicit(&eden_heap_hold_begun_area, memory_order_relaxed),
      atomic_load_explicit(&eden_heap_hold_capped, memory_order_relaxed),
      atomic_load_explicit(&eden_heap_hold_unheld_trims, memory_order_relaxed),
  };
  for (unsigned i = 0; i < count && i < 14u; ++i)
    values[i] = all[i];
  return 1;
}

/* Adds a space with room for `bytes`: memory for pieces of the range (pieces given back earlier
 * first, else the next ones) and an mspace over them, in a slot given back earlier or a new one.
 * `seen` is the growth the caller tried after: when another thread grew the heap meanwhile, that
 * space is tried first. Returns 0 when the heap is at its limit or the console has no memory left. */
static int eden_heap_grow(size_t bytes, unsigned seen) {
  eden_heap_grow_acquire();
  int grown = 0;
  atomic_store_explicit(&eden_heap_grow_why, EDEN_HEAP_WHY_NONE, memory_order_relaxed);
  const unsigned count = atomic_load_explicit(&eden_heap_space_count, memory_order_relaxed);
  const unsigned used = atomic_load_explicit(&eden_heap_pieces_used, memory_order_relaxed);
  /* Room for the request and for the mspace's own records. */
  const size_t need = bytes + ((size_t)1 << 20);
  const size_t pieces = (need + EDEN_HEAP_PIECE_SIZE - 1) >> EDEN_HEAP_PIECE_SHIFT;
  /* Pieces given back, `pieces` of them in a row, else the next ones. */
  unsigned first = used;
  if (eden_heap_grows && need > bytes) {
    for (unsigned at = 0, run = 0; at < used; ++at) {
      run = atomic_load_explicit(&eden_heap_piece_state[at], memory_order_relaxed) == 1 ? run + 1 : 0;
      if (run == pieces) {
        first = at + 1 - (unsigned)pieces;
        break;
      }
    }
  }
  /* A slot given back, else a new one. */
  unsigned slot = count;
  for (unsigned at = 1; at < count; ++at)
    if (eden_heap_space_span[at] == 0) {
      slot = at;
      break;
    }
  if (atomic_load_explicit(&eden_heap_space_generation, memory_order_relaxed) != seen) {
    grown = 1;
  } else if (eden_heap_grows && need > bytes && slot < EDEN_HEAP_PIECES &&
             ((first < used || pieces <= EDEN_HEAP_PIECES - used) ||
              eden_heap_hold_spare_fits(pieces))) {
    /* `span`: the pieces of the new space (`pieces`, or the spare's when it serves). */
    size_t span = pieces;
    void *start = (char *)ps5_heap_base + ((size_t)first << EDEN_HEAP_PIECE_SHIFT);
    size_t size = span << EDEN_HEAP_PIECE_SHIFT;
    int ready = (first < used || pieces <= EDEN_HEAP_PIECES - used) && !eden_heap_test_no_grow() &&
                eden_heap_commit(start, size) == 0;
    /* EDEN_UP_HEAP_HOLD_AT_LOAD: the pool said no; the spare (committed earlier) is the space. */
    if (!ready && eden_heap_hold_spare_take(pieces, bytes, &first, &span)) {
      start = (char *)ps5_heap_base + ((size_t)first << EDEN_HEAP_PIECE_SHIFT);
      size = span << EDEN_HEAP_PIECE_SHIFT;
      ready = 1;
    }
    if (ready) {
      void *space = sceLibcMspaceCreate("Eden-heap", start, size, 0);
      atomic_fetch_add_explicit(&eden_heap_committed_bytes, size, memory_order_relaxed);
      if (first == used)
        atomic_store_explicit(&eden_heap_pieces_used, used + (unsigned)span, memory_order_relaxed);
      if (space != NULL) {
        for (size_t piece = first; piece < first + span; ++piece) {
          atomic_store_explicit(&eden_heap_piece_space[piece], space, memory_order_release);
          atomic_store_explicit(&eden_heap_piece_slot[piece], slot, memory_order_release);
          atomic_store_explicit(&eden_heap_piece_state[piece], 0, memory_order_relaxed);
        }
        eden_heap_space_first[slot] = first;
        eden_heap_space_span[slot] = (unsigned)span;
        atomic_store_explicit(&eden_heap_space_blocks[slot], 0, memory_order_relaxed);
        atomic_store_explicit(&eden_heap_space_class[slot],
                              getenv("EDEN_HEAP_SCRATCH_SPACES") != NULL ? eden_heap_request_class(bytes) : EDEN_HEAP_CLASS_ANY,
                              memory_order_relaxed);
        atomic_store_explicit(&eden_heap_spaces[slot], space, memory_order_release);
        if (slot == count)
          atomic_store_explicit(&eden_heap_space_count, count + 1, memory_order_release);
        else
          atomic_store_explicit(&eden_heap_space_retired[slot], 0, memory_order_seq_cst);
        atomic_fetch_add_explicit(&eden_heap_space_generation, 1, memory_order_release);
        eden_heap_note_growth(bytes, (unsigned)span);
        /* EDEN_UP_HEAP_STEAL mode 3: the pool gave again, so growth comes before the idle arenas. */
        if (atomic_load_explicit(&eden_heap_steal_mode, memory_order_relaxed) >= 3)
          atomic_store_explicit(&eden_heap_steal_pool_no_us, 0, memory_order_relaxed);
        grown = 1;
      } else {
        /* The memory stays committed but no space is over it: given back at once. */
        if (eden_heap_decommit(start, size) == 1) {
          atomic_fetch_sub_explicit(&eden_heap_committed_bytes, size, memory_order_relaxed);
          for (size_t piece = first; piece < first + span; ++piece)
            atomic_store_explicit(&eden_heap_piece_state[piece], 1, memory_order_relaxed);
        }
        atomic_store_explicit(&eden_heap_grow_why, EDEN_HEAP_WHY_NO_SPACE, memory_order_relaxed);
        eden_heap_grow_refused("EDEN_HEAP_GROW refused: no allocator for a new piece\n");
      }
    } else {
      atomic_store_explicit(&eden_heap_grow_why, EDEN_HEAP_WHY_NO_MEMORY, memory_order_relaxed);
      if (atomic_load_explicit(&eden_heap_steal_mode, memory_order_relaxed) >= 3)
        atomic_store_explicit(&eden_heap_steal_pool_no_us, (unsigned long long)sceKernelGetProcessTime() | 1u,
                              memory_order_relaxed);
      eden_heap_grow_refused("EDEN_HEAP_GROW refused: the console has no memory for a new piece\n");
    }
  } else {
    atomic_store_explicit(&eden_heap_grow_why, EDEN_HEAP_WHY_AT_LIMIT, memory_order_relaxed);
    eden_heap_grow_refused("EDEN_HEAP_GROW refused: the heap is at its limit\n");
  }
  atomic_flag_clear_explicit(&eden_heap_grow_lock, memory_order_release);
  return grown;
}

/* Gives the memory of every space (but the first) that holds no block back to the system: the
 * space is retired, its memory released, its address space kept for a later growth. Returns the
 * bytes given back. */
static size_t eden_heap_trim_with(int hold_applies) {
  if (atomic_load_explicit(&ps5_heap_state, memory_order_acquire) != 2)
    return 0;
  /* Empty arenas to the spares first (their memory stays in their spaces). */
  eden_heap_return_arenas();
  if (!eden_heap_grows)
    return 0;
  /* EDEN_UP_HEAP_HOLD_AT_LOAD: through an area load nothing leaves the heap, but for the game. */
  if (atomic_load_explicit(&eden_heap_hold_seconds, memory_order_relaxed) != 0 &&
      eden_heap_hold_active((unsigned long long)sceKernelGetProcessTime())) {
    if (hold_applies) {
      atomic_fetch_add_explicit(&eden_heap_hold_trims_kept, 1, memory_order_relaxed);
      return 0;
    }
    atomic_fetch_add_explicit(&eden_heap_hold_unheld_trims, 1, memory_order_relaxed);
  }
  eden_heap_grow_acquire();
  size_t returned = 0;
  const unsigned count = atomic_load_explicit(&eden_heap_space_count, memory_order_relaxed);
  for (unsigned slot = 1; slot < count; ++slot) {
    if (eden_heap_space_span[slot] == 0 ||
        atomic_load_explicit(&eden_heap_space_blocks[slot], memory_order_relaxed) != 0)
      continue;
    atomic_store_explicit(&eden_heap_space_retired[slot], 1, memory_order_seq_cst);
    if (atomic_load_explicit(&eden_heap_space_users[slot], memory_order_seq_cst) != 0 ||
        atomic_load_explicit(&eden_heap_space_blocks[slot], memory_order_acquire) != 0) {
      atomic_store_explicit(&eden_heap_space_retired[slot], 0, memory_order_seq_cst);
      continue;
    }
    const unsigned first = eden_heap_space_first[slot], span = eden_heap_space_span[slot];
    const size_t size = (size_t)span << EDEN_HEAP_PIECE_SHIFT;
    const int done = eden_heap_decommit((char *)ps5_heap_base + ((size_t)first << EDEN_HEAP_PIECE_SHIFT), size);
    if (done == 0) {
      /* Kept as it was (dev-settings heap_return=off, or nothing to give back): usable again. */
      atomic_store_explicit(&eden_heap_space_retired[slot], 0, memory_order_seq_cst);
      continue;
    }
    for (unsigned piece = first; piece < first + span; ++piece) {
      atomic_store_explicit(&eden_heap_piece_space[piece], NULL, memory_order_release);
      atomic_store_explicit(&eden_heap_piece_state[piece], done > 0 ? 1 : 2, memory_order_relaxed);
    }
    eden_heap_space_span[slot] = 0;
    atomic_store_explicit(&eden_heap_spaces[slot], NULL, memory_order_release);
    atomic_fetch_sub_explicit(&eden_heap_committed_bytes, size, memory_order_relaxed);
    if (done > 0)
      returned += size;
  }
  atomic_flag_clear_explicit(&eden_heap_grow_lock, memory_order_release);
  atomic_fetch_add_explicit(&eden_heap_returned_bytes, returned, memory_order_relaxed);
  return returned;
}

size_t eden_heap_trim(void) { return eden_heap_trim_with(1); }

/* The guest's block wait, the JIT code cache's waits and operator new's waits: as eden_heap_trim, and
 * a hold (EDEN_UP_HEAP_HOLD_AT_LOAD) does not stop it. */
size_t eden_heap_trim_unheld(void) { return eden_heap_trim_with(0); }

/* A block from the heap's spaces, then from a new one, and when there is no memory for one, from
 * a space of the other class. A small request starts at the space that served the last one (it is
 * asked for millions of times: compiled-code records); a request of 1 MiB or more starts at the
 * first space, which keeps the later ones free of long-lived blocks and able to take the next large
 * request. */
/* One request to one slot's space, unless the slot is retired. A thread counts itself in the
 * slot first and then looks at the flag; eden_heap_trim sets the flag first and then looks at the
 * count (both sequentially consistent): a space is given back only when no thread is in it, and
 * none can enter it once it is retired. */
static void *eden_heap_space_use(unsigned slot, int kind, size_t a, size_t b) {
  atomic_fetch_add_explicit(&eden_heap_space_users[slot], 1, memory_order_seq_cst);
  void *address = NULL;
  if (atomic_load_explicit(&eden_heap_space_retired[slot], memory_order_seq_cst) == 0) {
    address = eden_heap_space_try(atomic_load_explicit(&eden_heap_spaces[slot], memory_order_acquire), kind, a, b);
    if (address != NULL)
      atomic_fetch_add_explicit(&eden_heap_space_blocks[slot], 1, memory_order_relaxed);
  }
  atomic_fetch_sub_explicit(&eden_heap_space_users[slot], 1, memory_order_release);
  return address;
}

/* EDEN_UP_HEAP_STEAL (after eden_heap_arena_rescue): a block from an arena no thread is on, for a
 * request the heap could not grow for (`before_grow` 0), or, in mode 3, one it is about to grow for
 * while the pool keeps refusing (1). NULL: off, or nothing found. Cold and not inlined, so
 * eden_heap_main keeps its code layout. */
static void *eden_heap_steal(int kind, size_t a, size_t b, size_t bytes, int before_grow)
    __attribute__((cold, noinline));

/* Mode 3: the pool refused a growth less than EDEN_HEAP_STEAL_BEFORE_GROW_US ago. */
static int eden_heap_steal_before_grow_due(void) {
  if (atomic_load_explicit(&eden_heap_steal_mode, memory_order_relaxed) < 3)
    return 0;
  const unsigned long long refused = atomic_load_explicit(&eden_heap_steal_pool_no_us, memory_order_relaxed);
  return refused != 0 && (unsigned long long)sceKernelGetProcessTime() - refused < EDEN_HEAP_STEAL_BEFORE_GROW_US;
}

static void *eden_heap_main(int kind, size_t a, size_t b) {
  const size_t bytes = kind == EDEN_HEAP_MALLOC ? a : kind == EDEN_HEAP_CALLOC ? a * b : a + b;
  unsigned wanted = eden_heap_request_class(bytes);
  int stole_before_grow = 0;
  for (;;) {
    const unsigned generation = atomic_load_explicit(&eden_heap_space_generation, memory_order_acquire);
    const unsigned count = atomic_load_explicit(&eden_heap_space_count, memory_order_acquire);
    const int small = bytes < ((size_t)1 << 20);
    const unsigned hint = small ? atomic_load_explicit(&eden_heap_space_hint, memory_order_relaxed) : 0;
    int passed_over = 0;
    for (unsigned i = 0; i < count; ++i) {
      const unsigned index = (hint + i) % count;
      const unsigned kept = atomic_load_explicit(&eden_heap_space_class[index], memory_order_relaxed);
      if (wanted != EDEN_HEAP_CLASS_ANY && kept != wanted && kept != EDEN_HEAP_CLASS_ANY) {
        passed_over = 1;
        continue;
      }
      void *address = eden_heap_space_use(index, kind, a, b);
      if (address != NULL) {
        if (small && i != 0)
          atomic_store_explicit(&eden_heap_space_hint, index, memory_order_relaxed);
        return address;
      }
    }
    if (bytes == 0)
      return NULL;
    if (!stole_before_grow && eden_heap_steal_before_grow_due()) {
      stole_before_grow = 1;
      void *address = eden_heap_steal(kind, a, b, bytes, 1);
      if (address != NULL)
        return address;
    }
    if (!eden_heap_grow(bytes, generation)) {
      if (!passed_over)
        return atomic_load_explicit(&eden_heap_steal_mode, memory_order_relaxed) != 0
                   ? eden_heap_steal(kind, a, b, bytes, 0)
                   : NULL;
      wanted = EDEN_HEAP_CLASS_ANY;
    }
  }
}

/* Allocate and register one arena block from the heap (NULL on failure, or at the limit: of first
 * arenas for a thread's first, of all arenas for another). Under the creation lock. */
static void *eden_heap_create_arena_locked(int first) {
  void *arena = NULL;
  /* A spare first: empty, with its mspace and its tables as they are, and no memory taken. A
   * thread's first arena may be one past the limit of first arenas, which limits what they cost. */
  const unsigned spares = atomic_load_explicit(&eden_heap_spare_count, memory_order_relaxed);
  if (spares != 0) {
    arena = eden_heap_spare[spares - 1];
    atomic_store_explicit(&eden_heap_spare_count, spares - 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&eden_heap_arenas_taken_again, 1, memory_order_relaxed);
    return arena;
  }
  const unsigned count = atomic_load_explicit(&eden_heap_arena_count, memory_order_relaxed);
  /* An entry given back, else a new one. */
  unsigned at = count;
  if (atomic_load_explicit(&eden_heap_arena_holes, memory_order_relaxed) != 0)
    for (unsigned i = 0; i < count; ++i)
      if (eden_heap_arena_list[i] == NULL) {
        at = i;
        break;
      }
  if (at < EDEN_HEAP_MAX_ARENAS && (!first || eden_heap_first_arenas < EDEN_HEAP_FIRST_ARENAS)) {
    void *block = eden_heap_main(EDEN_HEAP_ALIGNED, EDEN_HEAP_ARENA_SIZE, EDEN_HEAP_ARENA_PAYLOAD);
    if (block != NULL) {
      arena = sceLibcMspaceCreate("Eden-arena", block, EDEN_HEAP_ARENA_PAYLOAD, 0);
      if (arena != NULL) {
        const uintptr_t slot = eden_heap_slot(block);
        eden_heap_arena_list[at] = arena;
        eden_heap_arena_blocks_probe[at] = block;
        atomic_fetch_add_explicit(&eden_heap_arena_backing_probe, sceLibcMspaceMallocUsableSize(block), memory_order_relaxed);
        atomic_store_explicit(&eden_heap_arena_live[slot].blocks, 0, memory_order_relaxed);
        atomic_store_explicit(&eden_heap_slots[slot], arena, memory_order_release);
        if (at == count)
          atomic_store_explicit(&eden_heap_arena_count, count + 1, memory_order_release);
        else
          atomic_fetch_sub_explicit(&eden_heap_arena_holes, 1, memory_order_relaxed);
        if (first)
          eden_heap_first_arenas++;
      } else {
        eden_heap_space_free(block);
      }
    }
  }
  return arena;
}

/* The arenas threads are on now (every record's, an ended thread's record too: the next thread
 * on it starts there). Under the creation lock, which the table is used under. */
static void *eden_heap_named_arenas[EDEN_HEAP_THREADS];
static unsigned eden_heap_arenas_named(void **named) {
  unsigned count = 0;
  for (unsigned i = 0; i < EDEN_HEAP_THREADS; ++i) {
    void *arena = __atomic_load_n(&eden_heap_threads[i].arena, __ATOMIC_RELAXED);
    if (arena != NULL)
      named[count++] = arena;
  }
  return count;
}

/* Moves a record's empty arenas to the spares: all but its first, and but those a thread is
 * on (the list of them is read here, so a thread that takes one and is not on it yet, in a path
 * under the creation lock, is excluded by the lock). `ended`: no thread is on the record (its own
 * is ending, or it is claimed with eden_heap_thread_used 2), so its arena is set to its first
 * before the look, and its empty current one may go. Under the creation lock. */
static void eden_heap_return_record(struct eden_heap_thread *record, int ended) {
  unsigned owned = __atomic_load_n(&record->owned, __ATOMIC_ACQUIRE);
  if (owned < 2)
    return;
  if (ended)
    eden_heap_use_arena(record, __atomic_load_n(&record->own[0], __ATOMIC_RELAXED));
  void **named = eden_heap_named_arenas;
  const unsigned names = eden_heap_arenas_named(named);
  const unsigned count = atomic_load_explicit(&eden_heap_arena_count, memory_order_relaxed);
  for (unsigned i = 1; i < owned;) {
    void *arena = __atomic_load_n(&record->own[i], __ATOMIC_RELAXED);
    unsigned at = 0;
    while (at < names && named[at] != arena)
      ++at;
    unsigned index = 0;
    while (index < count && eden_heap_arena_list[index] != arena)
      ++index;
    void *block = index < count ? atomic_load_explicit(&eden_heap_arena_blocks_probe[index], memory_order_relaxed) : NULL;
    if (at != names || block == NULL ||
        atomic_load_explicit(&eden_heap_arena_live[eden_heap_slot(block)].blocks, memory_order_relaxed) != 0) {
      ++i;
      continue;
    }
    /* No block is in it and no thread is on it: nothing can take from it or free into it now.
     * Out of the record, into the spares; its slot, its list entry and its mspace stay as they are
     * (a block that lands in it later finds its owner as before). The spares cannot overflow: the
     * list holds every arena there is. */
    const unsigned spares = atomic_load_explicit(&eden_heap_spare_count, memory_order_relaxed);
    eden_heap_spare[spares] = arena;
    atomic_store_explicit(&eden_heap_spare_count, spares + 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&eden_heap_arenas_returned, 1, memory_order_relaxed);
    for (unsigned k = i; k + 1 < owned; ++k)
      __atomic_store_n(&record->own[k], __atomic_load_n(&record->own[k + 1], __ATOMIC_RELAXED), __ATOMIC_RELAXED);
    --owned;
    __atomic_store_n(&record->owned, owned, __ATOMIC_RELEASE);
  }
}

/* Every record's empty arenas to the spares (eden_heap_trim). A record no thread is on is
 * claimed meanwhile (eden_heap_thread_used 2: eden_heap_self passes it over), so none starts on an
 * arena being moved. Returns the arenas moved. */
static unsigned eden_heap_return_arenas(void) {
  if (eden_heap_keeps_arenas())
    return 0;
  const unsigned long long before = atomic_load_explicit(&eden_heap_arenas_returned, memory_order_relaxed);
  eden_heap_create_acquire();
  for (unsigned i = 0; i < EDEN_HEAP_THREADS; ++i) {
    unsigned char expected = 0;
    const int claimed = atomic_compare_exchange_strong_explicit(&eden_heap_thread_used[i], &expected, 2,
                                                                memory_order_acquire, memory_order_relaxed);
    if (!claimed && expected != 1)
      continue;
    eden_heap_return_record(&eden_heap_threads[i], claimed);
    if (claimed)
      atomic_store_explicit(&eden_heap_thread_used[i], 0, memory_order_release);
  }
  eden_heap_create_release();
  return (unsigned)(atomic_load_explicit(&eden_heap_arenas_returned, memory_order_relaxed) - before);
}

/* This thread's record, claimed on its first allocation. NULL: none free (process-wide counters). */
static struct eden_heap_thread *eden_heap_self(void) {
  if (!eden_heap_arena_key_valid)
    return NULL;
  struct eden_heap_thread *self = pthread_getspecific(eden_heap_arena_key);
  if (self != NULL)
    return self;
  for (unsigned i = 0; i < EDEN_HEAP_THREADS; ++i) {
    unsigned char expected = 0;
    if (!atomic_compare_exchange_strong_explicit(&eden_heap_thread_used[i], &expected, 1,
                                                 memory_order_acquire, memory_order_relaxed))
      continue;
    self = &eden_heap_threads[i];
    self->live = 0;
    self->blocks = 0;
    self->pending = 0;
    self->retry = 0;  /* its arenas (own) stay: this thread uses them */
    self->steal = 0;  /* EDEN_UP_HEAP_STEAL: the last thread's hint is not this one's */
    eden_heap_ll_depth[i] = 0; /* RADV_PS5_LONGLIVED_ARENA: an ended thread's scopes are not this one's */
    eden_heap_ll_retry[i] = 0;
    __atomic_store_n(&self->name[0], 0, __ATOMIC_RELAXED);
    if (pthread_setspecific(eden_heap_arena_key, self) != 0) {
      atomic_store_explicit(&eden_heap_thread_used[i], 0, memory_order_release);
      return NULL;
    }
    return self;
  }
  return NULL;
}

/* EDEN_UP_BUILD_BURST_LOG: a token is valid only from begin until the matching end, while its
 * thread lives. Nested scopes share the outer build's baseline and high-water mark. Nothing here
 * claims a record or allocates. */
unsigned eden_build_heap_begin(void) {
  struct eden_heap_thread *self = eden_heap_self();
  if (self == NULL)
    return 0;
  struct eden_build_heap_record *r = &eden_build_heap_records[self - eden_heap_threads];
  if (r->depth++ == 0) {
    atomic_store_explicit(&r->base, atomic_load_explicit(&r->net, memory_order_relaxed), memory_order_relaxed);
    atomic_store_explicit(&r->allocated_base, atomic_load_explicit(&r->allocated, memory_order_relaxed), memory_order_relaxed);
    atomic_store_explicit(&r->peak, 0, memory_order_relaxed);
  }
  return (unsigned)(self - eden_heap_threads) + 1;
}
void eden_build_heap_read(unsigned token, long long *net,
                          unsigned long long *allocated, unsigned long long *peak) {
  if (token == 0 || token > EDEN_HEAP_THREADS)
    return;
  const struct eden_build_heap_record *r = &eden_build_heap_records[token - 1];
  if (net) *net = atomic_load_explicit(&r->net, memory_order_relaxed) - atomic_load_explicit(&r->base, memory_order_relaxed);
  if (allocated) *allocated = atomic_load_explicit(&r->allocated, memory_order_relaxed) - atomic_load_explicit(&r->allocated_base, memory_order_relaxed);
  if (peak) *peak = (unsigned long long)atomic_load_explicit(&r->peak, memory_order_relaxed);
}
void eden_build_heap_end(unsigned token) {
  if (token && token <= EDEN_HEAP_THREADS) {
    struct eden_build_heap_record *r = &eden_build_heap_records[token - 1];
    if (r->depth) --r->depth;
  }
}
/* EDEN_UP_BUILD_BURST_LOG, read only: record `rec`'s heap pieces grown for its requests (the
 * existing `grown`, in 32 MiB pieces, bumped in eden_heap_note_growth), its sidecar net bytes
 * (inside build scopes only) and its thread name (16 bytes, 0-terminated). Returns 0 when `rec` is
 * past the table. No claim, no lock, no allocation. */
int eden_heap_record_growth(unsigned rec, unsigned *grown, long long *net, char name[16]) {
  if (rec >= EDEN_HEAP_THREADS)
    return 0;
  const struct eden_heap_thread *thread = &eden_heap_threads[rec];
  if (grown) *grown = __atomic_load_n(&thread->grown, __ATOMIC_RELAXED);
  if (net) *net = atomic_load_explicit(&eden_build_heap_records[rec].net, memory_order_relaxed);
  if (name) {
    for (unsigned i = 0; i < sizeof(thread->name); ++i)
      name[i] = __atomic_load_n(&thread->name[i], __ATOMIC_RELAXED);
    name[sizeof(thread->name) - 1] = 0;
  }
  return 1;
}
/* EDEN_UP_BUILD_BURST_LOG, read only: requests that did not go to an arena and were under 1 MiB
 * (EDEN_HEAP_CLASSES under1M), since start. */
unsigned long long eden_heap_under1m_count(void) {
  return (unsigned long long)atomic_load_explicit(&eden_heap_class_count[0], memory_order_relaxed);
}

static void eden_heap_charge(struct eden_heap_thread *self, long bytes, long blocks) {
  eden_build_heap_charge(self, bytes);
  self->live += bytes;
  self->blocks += blocks;
  if (++self->pending >= 64 || self->live > (1l << 20) || self->live < -(1l << 20))
    eden_heap_flush(self);
}

static void eden_heap_account(long bytes, long blocks) {
  struct eden_heap_thread *self = eden_heap_self();
  if (self == NULL)
    eden_heap_apply(bytes, blocks);
  else
    eden_heap_charge(self, bytes, blocks);
}

static void *eden_heap_record(void *address, size_t bytes) {
  if (address == NULL) {
    if (bytes != 0)
      eden_heap_note_failure(bytes, atomic_load_explicit(&eden_heap_grow_why, memory_order_relaxed));
    return ps5_heap_record_allocation(address, bytes != 0); /* counts the failure */
  }
  const size_t usable = eden_heap_usable(address);
  eden_heap_account((long)usable, 1);
  eden_pages_count(address, usable, 1);
  return address;
}

/* Another thread reads what a record uses and holds (eden_heap_adopt, the reports). */
static void eden_heap_use_arena(struct eden_heap_thread *self, void *arena) {
  __atomic_store_n(&self->arena, arena, __ATOMIC_RELAXED);
}

static void eden_heap_keep_arena(struct eden_heap_thread *self, void *arena) {
  const unsigned owned = self->owned;
  __atomic_store_n(&self->own[owned], arena, __ATOMIC_RELAXED);
  __atomic_store_n(&self->owned, owned + 1, __ATOMIC_RELEASE);
}

/* RADV_PS5_LONGLIVED_ARENA=1: 1 when the list's entry `index` is an arena of the long-lived set (by
 * the slot of its first block, as eden_heap_ll_allocate marks it). Off: 0 after one load. */
static int eden_heap_ll_listed(unsigned index) {
  if (!atomic_load_explicit(&eden_heap_ll_on, memory_order_relaxed))
    return 0;
  const void *block = atomic_load_explicit(&eden_heap_arena_blocks_probe[index], memory_order_relaxed);
  if (block == NULL)
    return 0;
  const uintptr_t slot = eden_heap_slot(block);
  return slot < EDEN_HEAP_SLOTS && atomic_load_explicit(&eden_heap_ll_slot[slot], memory_order_relaxed) != 0;
}

/* This thread's arena, created on its first small allocation. NULL = the heap's own spaces. */
static void *eden_heap_thread_arena(void) {
  struct eden_heap_thread *self = eden_heap_self();
  if (self == NULL)
    return NULL;
  if (self->arena != NULL)
    return self->arena;
  eden_heap_create_acquire();
  void *arena = eden_heap_create_arena_locked(1);
  if (arena == NULL) {
    /* Arena limit reached: share the existing arenas round-robin (past the entries given back, and
     * past the long-lived set's: RADV_PS5_LONGLIVED_ARENA=1, kept objects only). */
    const unsigned count = atomic_load_explicit(&eden_heap_arena_count, memory_order_acquire);
    for (unsigned i = 0; i < count && arena == NULL; ++i) {
      const unsigned index = atomic_fetch_add_explicit(&eden_heap_next_shared, 1, memory_order_relaxed) % count;
      if (!eden_heap_ll_listed(index))
        arena = eden_heap_arena_list[index];
    }
  } else if (self->owned < EDEN_HEAP_THREAD_ARENAS) {
    eden_heap_keep_arena(self, arena);
  }
  if (arena != NULL)
    eden_heap_use_arena(self, arena);
  eden_heap_create_release();
  return arena;
}

static void *eden_heap_arena_try(void *arena, size_t count, size_t size, int zero) {
  void *address = zero ? sceLibcMspaceCalloc(arena, count, size) : sceLibcMspaceMalloc(arena, count * size);
  if (address != NULL)
    eden_heap_arena_took(address);
  return address;
}

/* ---- RADV_PS5_LONGLIVED_ARENA (the switch's text is with its data, above eden_heap_apply) ---- */

void eden_heap_set_long_lived(int mode) {
  atomic_store_explicit(&eden_heap_ll_mode, mode, memory_order_relaxed);
  atomic_store_explicit(&eden_heap_ll_on, mode == 1, memory_order_relaxed);
  atomic_store_explicit(&eden_heap_ll_track, mode == 0 || mode == 1, memory_order_relaxed);
}

/* The driver's scopes (lead-mesh-42, weak there). Nested scopes count; off: one load and return. */
void eden_heap_long_lived_enter(void) {
  if (!atomic_load_explicit(&eden_heap_ll_track, memory_order_relaxed))
    return;
  struct eden_heap_thread *self = eden_heap_self();
  if (self == NULL)
    return;
  if (eden_heap_ll_depth[self - eden_heap_threads]++ == 0)
    atomic_fetch_add_explicit(&eden_heap_ll_scopes, 1, memory_order_relaxed);
}

void eden_heap_long_lived_leave(void) {
  if (!atomic_load_explicit(&eden_heap_ll_track, memory_order_relaxed))
    return;
  struct eden_heap_thread *self = eden_heap_self();
  if (self != NULL && eden_heap_ll_depth[self - eden_heap_threads] != 0)
    --eden_heap_ll_depth[self - eden_heap_threads];
}

/* 1: this thread is inside a long-lived scope and the switch routes. */
static int eden_heap_ll_active(void) {
  if (!atomic_load_explicit(&eden_heap_ll_on, memory_order_relaxed))
    return 0;
  struct eden_heap_thread *self = eden_heap_self();
  return self != NULL && eden_heap_ll_depth[self - eden_heap_threads] != 0;
}

/* A small request's routing (the two places that route; the thread cache's checks use
 * eden_heap_ll_active): inside a scope, counted (modes 0 and 1), and 1 when mode 1 routes it. */
static int eden_heap_ll_route(size_t bytes) {
  if (!atomic_load_explicit(&eden_heap_ll_track, memory_order_relaxed))
    return 0;
  struct eden_heap_thread *self = eden_heap_self();
  if (self == NULL || eden_heap_ll_depth[self - eden_heap_threads] == 0)
    return 0;
  atomic_fetch_add_explicit(&eden_heap_ll_requests, 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&eden_heap_ll_bytes, bytes, memory_order_relaxed);
  return atomic_load_explicit(&eden_heap_ll_on, memory_order_relaxed);
}

/* 1: an arena of the long-lived set (so no thread adopts it). Off: 0 after one load. */
static int eden_heap_is_long_lived(const void *arena) {
  if (!atomic_load_explicit(&eden_heap_ll_on, memory_order_relaxed))
    return 0;
  const unsigned count = atomic_load_explicit(&eden_heap_ll_count, memory_order_acquire);
  for (unsigned i = 0; i < count; ++i)
    if (atomic_load_explicit(&eden_heap_ll_arena[i], memory_order_relaxed) == arena)
      return 1;
  return 0;
}

/* 1: a block (in the heap's range) of a long-lived arena; it skips the thread cache when freed. */
static int eden_heap_ll_block(const void *address) {
  if (!atomic_load_explicit(&eden_heap_ll_on, memory_order_relaxed))
    return 0;
  const uintptr_t slot = eden_heap_slot(address);
  return slot < EDEN_HEAP_SLOTS && atomic_load_explicit(&eden_heap_ll_slot[slot], memory_order_relaxed) != 0;
}

static void *eden_heap_ll_try(void *arena, size_t count, size_t size, int zero, size_t alignment) {
  if (alignment == 0)
    return eden_heap_arena_try(arena, count, size, zero);
  void *block = NULL;
  if (sceLibcMspacePosixMemalign(arena, &block, alignment, size) != 0 || block == NULL)
    return NULL;
  eden_heap_arena_took(block);
  return block;
}

/* A block for a kept object from the long-lived set: the entry asked last, then the others, then a
 * new arena (a spare first: eden_heap_create_arena_locked). NULL: the caller's usual path takes it
 * (the set is full, no memory for one more, or the creation lock is this thread's). Under the
 * creation lock past the first try, as eden_heap_arena_allocate. `alignment` 0: malloc/calloc. */
static void *eden_heap_ll_allocate(size_t count, size_t size, int zero, size_t alignment) {
  unsigned made = atomic_load_explicit(&eden_heap_ll_count, memory_order_acquire);
  unsigned current = atomic_load_explicit(&eden_heap_ll_current, memory_order_relaxed);
  if (current < made) {
    void *address = eden_heap_ll_try(atomic_load_explicit(&eden_heap_ll_arena[current], memory_order_relaxed),
                                     count, size, zero, alignment);
    if (address != NULL)
      return address;
  }
  if (atomic_load_explicit(&eden_heap_create_holder, memory_order_relaxed) == (unsigned long)pthread_self()) {
    atomic_fetch_add_explicit(&eden_heap_ll_fallbacks, 1, memory_order_relaxed);
    return NULL; /* the lock is not recursive */
  }
  /* Back-off (as eden_heap_arena_allocate's retry): after a search that found nothing, this thread's
   * next EDEN_HEAP_ARENA_RETRY requests try the current arena only, without the creation lock. */
  struct eden_heap_thread *self = eden_heap_self();
  unsigned *retry = self != NULL ? &eden_heap_ll_retry[self - eden_heap_threads] : NULL;
  if (retry != NULL && *retry != 0) {
    --*retry;
    atomic_fetch_add_explicit(&eden_heap_ll_fallbacks, 1, memory_order_relaxed);
    return NULL;
  }
  void *address = NULL;
  eden_heap_create_acquire();
  made = atomic_load_explicit(&eden_heap_ll_count, memory_order_relaxed);
  for (unsigned i = 0; i < made && address == NULL; ++i) {
    if (i == current)
      continue;
    address = eden_heap_ll_try(atomic_load_explicit(&eden_heap_ll_arena[i], memory_order_relaxed), count, size, zero,
                               alignment);
    if (address != NULL)
      atomic_store_explicit(&eden_heap_ll_current, i, memory_order_relaxed);
  }
  /* A new arena (2 MiB) only for a request under EDEN_HEAP_ARENA_GROW_REQUEST, as a thread's own. */
  if (address == NULL && made < EDEN_HEAP_LL_ARENAS && count * size < EDEN_HEAP_ARENA_GROW_REQUEST) {
    void *fresh = eden_heap_create_arena_locked(0);
    if (fresh != NULL) {
      /* Its slot, as eden_heap_return_record finds it: by its entry in the list. */
      const unsigned listed = atomic_load_explicit(&eden_heap_arena_count, memory_order_relaxed);
      unsigned index = 0;
      while (index < listed && eden_heap_arena_list[index] != fresh)
        ++index;
      void *block = index < listed ? atomic_load_explicit(&eden_heap_arena_blocks_probe[index], memory_order_relaxed)
                                   : NULL;
      if (block != NULL)
        atomic_store_explicit(&eden_heap_ll_slot[eden_heap_slot(block)], 1, memory_order_relaxed);
      atomic_store_explicit(&eden_heap_ll_arena[made], fresh, memory_order_relaxed);
      atomic_store_explicit(&eden_heap_ll_count, made + 1, memory_order_release);
      atomic_store_explicit(&eden_heap_ll_current, made, memory_order_relaxed);
      address = eden_heap_ll_try(fresh, count, size, zero, alignment);
    }
  }
  eden_heap_create_release();
  if (address == NULL) {
    atomic_fetch_add_explicit(&eden_heap_ll_fallbacks, 1, memory_order_relaxed);
    if (retry != NULL)
      *retry = EDEN_HEAP_ARENA_RETRY;
  }
  return address;
}

/* One line of numbers (MiB): what the heap holds and uses, its slack, the slack in whole free 2 MiB
 * pages (what a page give-back could return) and the rest ("pinned": room in pages a live block
 * keeps), the arenas and spares, and the long-lived set (arenas, bytes live in them, requests, the
 * requests it left to the usual path, scopes entered). No allocation, no lock. */
static void eden_heap_ll_snapshot(char *out, size_t room) {
  size_t held = 0, in_use = 0, returned = 0, free_pages = 0, idle_arenas = 0, live = 0;
  eden_heap_usage(&held, &in_use, &returned);
  eden_heap_page_usage(&free_pages, &idle_arenas, &live);
  const size_t slack = held > in_use ? held - in_use : 0;
  const size_t pinned = slack > free_pages ? slack - free_pages : 0;
  const unsigned made = atomic_load_explicit(&eden_heap_ll_count, memory_order_acquire);
  long ll_live = 0;
  for (uintptr_t slot = 0; slot < EDEN_HEAP_SLOTS; ++slot)
    if (atomic_load_explicit(&eden_heap_ll_slot[slot], memory_order_relaxed) != 0)
      ll_live += atomic_load_explicit(&eden_page_live[slot], memory_order_relaxed);
  snprintf(out, room,
           "held=%zu in_use=%zu slack=%zu free_pages=%zu idle_arena_pages=%zu pinned=%zu arenas=%u spare=%u "
           "ll_arenas=%u ll_live=%ld ll_bytes=%llu ll_requests=%llu ll_fallbacks=%llu ll_scopes=%llu (MiB)",
           held >> 20, in_use >> 20, slack >> 20, free_pages >> 20, idle_arenas >> 20, pinned >> 20,
           eden_heap_arenas_created(), eden_heap_arenas_spare(), made, (ll_live > 0 ? ll_live : 0) >> 20,
           (unsigned long long)(atomic_load_explicit(&eden_heap_ll_bytes, memory_order_relaxed) >> 20),
           (unsigned long long)atomic_load_explicit(&eden_heap_ll_requests, memory_order_relaxed),
           (unsigned long long)atomic_load_explicit(&eden_heap_ll_fallbacks, memory_order_relaxed),
           (unsigned long long)atomic_load_explicit(&eden_heap_ll_scopes, memory_order_relaxed));
}

static atomic_uint eden_heap_adopt_next;
static atomic_ullong eden_heap_adopted, eden_heap_adopt_misses;

/* EDEN_UP_HEAP_ADOPT_WIDE=<n> (0880, our own; 17 to 1024, unset or 0: off; read from main in
 * eden_heap_pool_init): a thread out of room asks up to n arenas no thread is on before it makes a
 * new one, instead of EDEN_HEAP_ADOPT_TRIES (16). The 10:17 session of 13 Nov (T165,
 * its crash log) made 112 new arenas in play (EDEN_HEAP_GROWTH arenas=265 at
 * line 404, 377 at line 9117), every one after a miss of the 16 asks (adopt_misses 79 -> 191), while
 * 377 arenas held 200 MiB of room (EDEN_HEAP_SLACK, line 8447; record 0 alone 81 MiB in 128 arenas,
 * line 8431). The heap's pieces grew 1344 -> 1536 MiB in play and nothing of it ever went back (a
 * trim needs a space with no block; every space held arenas, EDEN_HEAP_SPACES 8443-8446). Same room
 * test (EDEN_HEAP_ADOPT_ROOM), same lock, same slack gate; only the number of arenas asked differs.
 * Its price: up to n mspace calls (a malloc and a free each) under the creation lock at a miss.
 * Counted (asks past the 16th, arenas taken there, time in those asks) in "EDEN_HEAP_ADOPT_WIDE" after
 * the 20 s EDEN_HEAP_GROWTH line. Off: one relaxed load at a miss of a thread's own arenas. */
#define EDEN_HEAP_ADOPT_WIDE_MAX EDEN_HEAP_MAX_ARENAS
static atomic_uint eden_heap_adopt_wide;
static atomic_ullong eden_heap_adopt_wide_asks, eden_heap_adopt_wide_hits, eden_heap_adopt_wide_scans;
static atomic_ullong eden_heap_adopt_wide_us, eden_heap_adopt_wide_max_us;

void eden_heap_set_adopt_wide(unsigned arenas) {
  if (arenas <= EDEN_HEAP_ADOPT_TRIES)
    arenas = 0;
  if (arenas > EDEN_HEAP_ADOPT_WIDE_MAX)
    arenas = EDEN_HEAP_ADOPT_WIDE_MAX;
  atomic_store_explicit(&eden_heap_adopt_wide, arenas, memory_order_relaxed);
}

/* The wide part of one look (asks past the 16th): its asks, whether it took an arena, its time. */
static void eden_heap_adopt_wide_note(unsigned asks, int hit, unsigned long long from_us) {
  if (asks == 0)
    return;
  const unsigned long long took = (unsigned long long)sceKernelGetProcessTime() - from_us;
  atomic_fetch_add_explicit(&eden_heap_adopt_wide_scans, 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&eden_heap_adopt_wide_asks, asks, memory_order_relaxed);
  if (hit)
    atomic_fetch_add_explicit(&eden_heap_adopt_wide_hits, 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&eden_heap_adopt_wide_us, took, memory_order_relaxed);
  unsigned long long most = atomic_load_explicit(&eden_heap_adopt_wide_max_us, memory_order_relaxed);
  while (took > most && !atomic_compare_exchange_weak_explicit(&eden_heap_adopt_wide_max_us, &most, took,
                                                               memory_order_relaxed, memory_order_relaxed)) {
  }
}

/* A block from an arena no thread is using, which this thread then uses; NULL when the heap holds
 * little more than is in use, or none of the arenas asked has room. The owner of an arena taken
 * here may come back to it: the two then share it, as threads past the limit always have. Runs
 * once a thread is out of room, which takes at least the room asked for here. Under the creation
 * lock: the list is read. */
static void *eden_heap_adopt(struct eden_heap_thread *self, size_t count, size_t size, int zero) {
  const size_t held = atomic_load_explicit(&eden_heap_committed_bytes, memory_order_relaxed) +
                      atomic_load_explicit(&eden_heap_large_bytes, memory_order_relaxed);
  const size_t live = atomic_load_explicit(&ps5_heap_live_bytes, memory_order_relaxed);
  if (held < live + EDEN_HEAP_ADOPT_SLACK || getenv("EDEN_HEAP_NO_ADOPT") != NULL)
    return NULL;
  /* The arenas threads are on now. A record whose thread has ended uses none. */
  void *busy[EDEN_HEAP_ADOPT_BUSY];
  unsigned busy_count = 0;
  for (unsigned i = 0; i < EDEN_HEAP_THREADS && busy_count < EDEN_HEAP_ADOPT_BUSY; ++i)
    if (atomic_load_explicit(&eden_heap_thread_used[i], memory_order_relaxed) != 0) {
      void *arena = __atomic_load_n(&eden_heap_threads[i].arena, __ATOMIC_RELAXED);
      if (arena != NULL)
        busy[busy_count++] = arena;
    }
  const unsigned arenas = atomic_load_explicit(&eden_heap_arena_count, memory_order_acquire);
  const unsigned start = atomic_fetch_add_explicit(&eden_heap_adopt_next, EDEN_HEAP_ADOPT_TRIES, memory_order_relaxed);
  /* EDEN_UP_HEAP_ADOPT_WIDE: past the first 16 asks, up to its n (0: off, 16 as before). */
  const unsigned wide = atomic_load_explicit(&eden_heap_adopt_wide, memory_order_relaxed);
  const unsigned tries = wide != 0 ? wide : EDEN_HEAP_ADOPT_TRIES;
  unsigned long long wide_from = 0;
  unsigned asked = 0;
  for (unsigned i = 0; i < arenas && asked < tries; ++i) {
    if (asked == EDEN_HEAP_ADOPT_TRIES && wide_from == 0)
      wide_from = (unsigned long long)sceKernelGetProcessTime() | 1u;
    void *other = eden_heap_arena_list[(start + i) % arenas];
    if (other == NULL || eden_heap_is_spare(other))
      continue; /* a spare goes whole to the next thread that needs an arena */
    if (eden_heap_is_long_lived(other))
      continue; /* RADV_PS5_LONGLIVED_ARENA: kept objects only (off: one load) */
    unsigned at = 0;
    while (at < busy_count && busy[at] != other)
      ++at;
    if (at != busy_count)
      continue;
    ++asked;
    void *room = sceLibcMspaceMalloc(other, EDEN_HEAP_ADOPT_ROOM);
    if (room == NULL)
      continue;
    sceLibcMspaceFree(other, room);
    void *address = eden_heap_arena_try(other, count, size, zero);
    if (address != NULL) {
      eden_heap_use_arena(self, other);
      atomic_fetch_add_explicit(&eden_heap_adopted, 1, memory_order_relaxed);
      if (wide_from != 0)
        eden_heap_adopt_wide_note(asked - EDEN_HEAP_ADOPT_TRIES, 1, wide_from);
      return address;
    }
  }
  if (wide_from != 0)
    eden_heap_adopt_wide_note(asked - EDEN_HEAP_ADOPT_TRIES, 0, wide_from);
  atomic_fetch_add_explicit(&eden_heap_adopt_misses, 1, memory_order_relaxed);
  return NULL;
}

/* A small block from this thread's arenas: the one it uses; when that one is full, another of its
 * own with room, which it then uses; when none has, an arena no thread is using, while the heap
 * holds much more than is in use (eden_heap_adopt); else a new one (for a request under
 * EDEN_HEAP_ARENA_GROW_REQUEST, while the thread and the process are under their limits). NULL:
 * the heap's own spaces take the request. Past the arena the thread is on, under the creation
 * lock: the thread's arenas and the list are read, and an empty arena is given back under it. */
static void *eden_heap_arena_allocate(size_t count, size_t size, int zero) {
  void *arena = eden_heap_thread_arena();
  if (arena == NULL)
    return NULL;
  void *address = eden_heap_arena_try(arena, count, size, zero);
  if (address != NULL)
    return address;
  struct eden_heap_thread *self = eden_heap_self();
  if (self == NULL)
    return NULL;
  if (self->retry != 0) {
    --self->retry;
    return NULL;
  }
  eden_heap_create_acquire();
  for (unsigned i = 0; i < self->owned; ++i) {
    void *other = self->own[i];
    if (other == arena)
      continue;
    address = eden_heap_arena_try(other, count, size, zero);
    if (address != NULL) {
      eden_heap_use_arena(self, other);
      eden_heap_create_release();
      return address;
    }
  }
  if (count * size < EDEN_HEAP_ARENA_GROW_REQUEST) {
    address = eden_heap_adopt(self, count, size, zero);
    if (address == NULL) {
      void *fresh = self->owned < EDEN_HEAP_THREAD_ARENAS ? eden_heap_create_arena_locked(0) : NULL;
      if (fresh == NULL) {
        /* At a limit, or no memory for one: each look through the arenas costs a request to each,
         * so the next few requests go straight to the heap's spaces. */
        self->retry = EDEN_HEAP_ARENA_RETRY;
      } else {
        eden_heap_keep_arena(self, fresh);
        eden_heap_use_arena(self, fresh);
        address = eden_heap_arena_try(fresh, count, size, zero);
      }
    }
  }
  eden_heap_create_release();
  return address;
}

/* A small request no arena took, from the heap's spaces, counted with its cycles. */
static void *eden_heap_spill(size_t count, size_t size, int zero) {
  const unsigned long long start = eden_heap_cycles();
  void *address = zero ? eden_heap_main(EDEN_HEAP_CALLOC, count, size)
                       : eden_heap_main(EDEN_HEAP_MALLOC, count * size, 0);
  const unsigned long long cycles = eden_heap_cycles() - start;
  struct eden_heap_thread *self = eden_heap_self();
  if (self != NULL) {
    self->spills++;
    self->spill_cycles += cycles;
  } else {
    atomic_fetch_add_explicit(&eden_heap_spills, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&eden_heap_spill_cycles, cycles, memory_order_relaxed);
  }
  return address;
}

/* Gives an owned block back to its arena, or to the heap's space, counting the latter (its lock is
 * the one every thread shares). */
static void eden_heap_release(void *address) {
  void *arena = eden_heap_arena_owner(address);
  if (arena != NULL) {
    sceLibcMspaceFree(arena, address);
    eden_heap_arena_gave(address);
    return;
  }
  const unsigned long long start = eden_heap_cycles();
  eden_heap_space_free(address);
  const unsigned long long cycles = eden_heap_cycles() - start;
  struct eden_heap_thread *self = eden_heap_self();
  if (self != NULL) {
    self->shared_frees++;
    self->shared_free_cycles += cycles;
  } else {
    atomic_fetch_add_explicit(&eden_heap_shared_frees, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&eden_heap_shared_free_cycles, cycles, memory_order_relaxed);
  }
}

/* A cached block for bytes (0 < bytes <= 16 * (EDEN_TCACHE_BINS - 1)), charged as allocated. */
static void *eden_tcache_take(size_t bytes) {
  const size_t bin = (bytes + 15) >> 4;
  if (bytes == 0 || bin >= EDEN_TCACHE_BINS)
    return NULL;
  struct eden_heap_thread *self = eden_heap_self();
  if (self == NULL || self->bins[bin] == NULL)
    return NULL;
  void *block = self->bins[bin];
  self->bins[bin] = ((void **)block)[0];
  __atomic_store_n(&self->counts[bin], (unsigned char)(self->counts[bin] - 1), __ATOMIC_RELAXED);
  const size_t usable = ((size_t *)block)[1];
  eden_heap_charge(self, (long)usable, 1);
  eden_pages_count(block, usable, 1);
  return block;
}

/* Keeps a freed owned block of this usable size; 0 when the bin is full or out of range. */
static int eden_tcache_keep(void *address, size_t usable) {
  const size_t bin = usable >> 4;
  if (bin == 0 || bin >= EDEN_TCACHE_BINS)
    return 0;
  struct eden_heap_thread *self = eden_heap_self();
  if (self == NULL || self->counts[bin] >= EDEN_TCACHE_LIMIT)
    return 0;
  ((void **)address)[0] = self->bins[bin];
  ((size_t *)address)[1] = usable;
  self->bins[bin] = address;
  __atomic_store_n(&self->counts[bin], (unsigned char)(self->counts[bin] + 1), __ATOMIC_RELAXED);
  eden_heap_charge(self, -(long)usable, -1);
  return 1;
}

/* ---- Where the live blocks come from (a measurement, reported by eden_heap_sites_report) ----
 * One block in EDEN_SITE_RATE, and every block of 1 MiB or more, is followed from its allocation to
 * its free. At the allocation the stack above the allocator is read for the first return addresses
 * that fall in the app's code; the bytes of the block are added to the
 * group of that chain, and taken off again when it is freed. A group is a place in the emulator that
 * asks for memory, as far as the first calls up say. Sampled blocks count EDEN_SITE_RATE times. */
extern const char __eden_text_start[];
extern const char __eden_text_end[];
#define EDEN_SITE_BLOCKS 131072u
/* Run 61: 4096 places were full within 40 s, the words of the stack that are no return address
 * making many chains a new one, and a block from then on had no place: "followed blocks" stood at
 * 444 MiB for the rest of the run while the heap's use went from 741 to 1211 MiB. So: more places, a
 * place with no block alive is taken over, the requested size is no return address, and a block
 * that finds no place is counted (eden_site_missed). */
#define EDEN_SITE_TABLE 16384u
#define EDEN_SITE_PROBES 32u
#define EDEN_SITE_PCS 8u
/* What makes two blocks one place: the nearest words of the chain. The far ones are shown, from
 * the first block of the place, but a stale word out there would split one owner into a place per
 * block, and the list shows only the largest. */
#define EDEN_SITE_KEY 5u
#define EDEN_SITE_REPORTED 24u
#define EDEN_SITE_RATE 128u
#define EDEN_SITE_BIG ((size_t)1 << 20)

static struct {
  _Atomic uintptr_t address; /* 0 free, 1 given back */
  unsigned site;
  unsigned big;
  size_t size;
} eden_site_blocks[EDEN_SITE_BLOCKS];
static struct {
  _Atomic uintptr_t hash;
  uintptr_t pcs[EDEN_SITE_PCS];
  _Atomic long exact_bytes, sampled_bytes, exact_blocks, sampled_blocks;
} eden_site_table[EDEN_SITE_TABLE];

/* Bytes of the blocks that found no place since the process began (the sampled ones at their rate). */
static atomic_long eden_site_missed;

static int eden_site_sampled(const void *address) {
  return (((uintptr_t)address * 0x9E3779B97F4A7C15ull) >> 57) == 0;
}

static unsigned eden_site_slot(const void *address) {
  return (unsigned)((((uintptr_t)address >> 4) * 0x9E3779B97F4A7C15ull) >> 40) % EDEN_SITE_BLOCKS;
}

static __attribute__((noinline)) unsigned eden_site_collect(uintptr_t pcs[EDEN_SITE_PCS], size_t size) {
  const uintptr_t low = (uintptr_t)__eden_text_start + 8, high = (uintptr_t)__eden_text_end;
  const uintptr_t self = (uintptr_t)__builtin_return_address(0);
  const uintptr_t *stack = (const uintptr_t *)__builtin_frame_address(0);
  /* The result is written after the scan: the array of the caller lies in the stack being read. */
  uintptr_t local[EDEN_SITE_PCS] = {0};
  unsigned found = 0;
  /* Never past the page (16 KiB) that holds the frame: a thread's stack can end a few hundred bytes
   * above, and what lies after it is not mapped. */
  const uintptr_t page_end = (((uintptr_t)stack | 0x3FFFu) + 1);
  const unsigned room = (unsigned)((page_end - (uintptr_t)stack) / sizeof(uintptr_t));
  for (unsigned i = 0; i < 256 && i < room && found < EDEN_SITE_PCS; ++i) {
    const uintptr_t value = stack[i];
    /* The app's code is execute-only on the console: its bytes cannot be read to check that the
     * address follows a call (the system ends the process). A stale word of the stack can be taken
     * for a return address; the chain is read with that in mind. */
    if (value < low || value >= high || (found != 0 && local[found - 1] == value))
      continue;
    /* eden_site_alloc itself, and the request: 16 MiB is an address in the app's image. */
    if (value == self || value == size)
      continue;
    local[found++] = value;
  }
  for (unsigned i = 0; i < EDEN_SITE_PCS; ++i)
    pcs[i] = local[i];
  return found;
}

static __attribute__((noinline)) void eden_site_alloc(const void *address, size_t size) {
  if (address == NULL)
    return;
  const unsigned big = size >= EDEN_SITE_BIG;
  if (!big && !eden_site_sampled(address))
    return;
  uintptr_t pcs[EDEN_SITE_PCS] = {0};
  eden_site_collect(pcs, size);
  uintptr_t hash = 0xCBF29CE484222325ull;
  for (unsigned i = 0; i < EDEN_SITE_KEY; ++i)
    hash = (hash ^ pcs[i]) * 0x100000001B3ull;
  hash |= 1;
  unsigned site = EDEN_SITE_TABLE;
  for (unsigned probe = 0; probe < EDEN_SITE_PROBES; ++probe) {
    const unsigned at = (unsigned)((hash >> 12) + probe) % EDEN_SITE_TABLE;
    uintptr_t expected = 0;
    if (atomic_compare_exchange_strong_explicit(&eden_site_table[at].hash, &expected, hash, memory_order_acq_rel,
                                                memory_order_acquire)) {
      for (unsigned i = 0; i < EDEN_SITE_PCS; ++i)
        eden_site_table[at].pcs[i] = pcs[i];
      site = at;
      break;
    }
    if (expected == hash) {
      site = at;
      break;
    }
  }
  /* No room: a place whose blocks are all gone takes the new chain (the report needs no more of
   * it than what is alive; two threads racing here at worst put a block under the other's chain). */
  for (unsigned probe = 0; site == EDEN_SITE_TABLE && probe < EDEN_SITE_PROBES; ++probe) {
    const unsigned at = (unsigned)((hash >> 12) + probe) % EDEN_SITE_TABLE;
    if (atomic_load_explicit(&eden_site_table[at].exact_blocks, memory_order_relaxed) != 0 ||
        atomic_load_explicit(&eden_site_table[at].sampled_blocks, memory_order_relaxed) != 0)
      continue;
    uintptr_t expected = atomic_load_explicit(&eden_site_table[at].hash, memory_order_relaxed);
    if (atomic_compare_exchange_strong_explicit(&eden_site_table[at].hash, &expected, hash, memory_order_acq_rel,
                                                memory_order_acquire)) {
      for (unsigned i = 0; i < EDEN_SITE_PCS; ++i)
        eden_site_table[at].pcs[i] = pcs[i];
      site = at;
    }
  }
  if (site == EDEN_SITE_TABLE) {
    atomic_fetch_add_explicit(&eden_site_missed, big ? (long)size : (long)(size * EDEN_SITE_RATE), memory_order_relaxed);
    return;
  }
  const unsigned first = eden_site_slot(address);
  for (unsigned probe = 0; probe < 16; ++probe) {
    const unsigned at = (first + probe) % EDEN_SITE_BLOCKS;
    uintptr_t value = atomic_load_explicit(&eden_site_blocks[at].address, memory_order_relaxed);
    if (value > 1)
      continue;
    if (atomic_compare_exchange_strong_explicit(&eden_site_blocks[at].address, &value, (uintptr_t)address,
                                                memory_order_acq_rel, memory_order_relaxed)) {
      eden_site_blocks[at].site = site;
      eden_site_blocks[at].big = big;
      eden_site_blocks[at].size = size;
      if (big) {
        atomic_fetch_add_explicit(&eden_site_table[site].exact_bytes, (long)size, memory_order_relaxed);
        atomic_fetch_add_explicit(&eden_site_table[site].exact_blocks, 1, memory_order_relaxed);
      } else {
        atomic_fetch_add_explicit(&eden_site_table[site].sampled_bytes, (long)size, memory_order_relaxed);
        atomic_fetch_add_explicit(&eden_site_table[site].sampled_blocks, 1, memory_order_relaxed);
      }
      return;
    }
  }
}

static void eden_site_free(const void *address, size_t usable) {
  if (usable < EDEN_SITE_BIG && !eden_site_sampled(address))
    return;
  const unsigned first = eden_site_slot(address);
  for (unsigned probe = 0; probe < 16; ++probe) {
    const unsigned at = (first + probe) % EDEN_SITE_BLOCKS;
    uintptr_t value = atomic_load_explicit(&eden_site_blocks[at].address, memory_order_acquire);
    if (value == 0)
      return;
    if (value != (uintptr_t)address)
      continue;
    const unsigned site = eden_site_blocks[at].site, big = eden_site_blocks[at].big;
    const long size = (long)eden_site_blocks[at].size;
    if (!atomic_compare_exchange_strong_explicit(&eden_site_blocks[at].address, &value, 1, memory_order_acq_rel,
                                                 memory_order_relaxed))
      return;
    if (big) {
      atomic_fetch_sub_explicit(&eden_site_table[site].exact_bytes, size, memory_order_relaxed);
      atomic_fetch_sub_explicit(&eden_site_table[site].exact_blocks, 1, memory_order_relaxed);
    } else {
      atomic_fetch_sub_explicit(&eden_site_table[site].sampled_bytes, size, memory_order_relaxed);
      atomic_fetch_sub_explicit(&eden_site_table[site].sampled_blocks, 1, memory_order_relaxed);
    }
    return;
  }
}

/* The places that hold the most, to stderr.log: MiB alive, blocks alive, and the chain of calls
 * (offsets in the app's image, for tools/symbolize-crash.py or llvm-symbolizer). */
void eden_heap_sites_report(void) {
  struct {
    long bytes, blocks;
    unsigned site;
  } top[EDEN_SITE_REPORTED];
  unsigned count = 0;
  long total = 0;
  for (unsigned at = 0; at < EDEN_SITE_TABLE; ++at) {
    if (atomic_load_explicit(&eden_site_table[at].hash, memory_order_relaxed) == 0)
      continue;
    const long bytes =
        atomic_load_explicit(&eden_site_table[at].exact_bytes, memory_order_relaxed) +
        (long)EDEN_SITE_RATE * atomic_load_explicit(&eden_site_table[at].sampled_bytes, memory_order_relaxed);
    const long blocks =
        atomic_load_explicit(&eden_site_table[at].exact_blocks, memory_order_relaxed) +
        (long)EDEN_SITE_RATE * atomic_load_explicit(&eden_site_table[at].sampled_blocks, memory_order_relaxed);
    if (bytes <= 0)
      continue;
    total += bytes;
    unsigned place = count;
    if (count < EDEN_SITE_REPORTED)
      ++count;
    else if (bytes <= top[EDEN_SITE_REPORTED - 1].bytes)
      continue;
    else
      place = EDEN_SITE_REPORTED - 1;
    while (place > 0 && top[place - 1].bytes < bytes) {
      top[place] = top[place - 1];
      --place;
    }
    top[place].bytes = bytes;
    top[place].blocks = blocks;
    top[place].site = at;
  }
  fprintf(stderr, "EDEN_HEAP_SITES followed blocks hold %ld MiB, %ld MiB of blocks found no place so far; the largest places:\n",
          total >> 20, atomic_load_explicit(&eden_site_missed, memory_order_relaxed) >> 20);
  for (unsigned i = 0; i < count; ++i) {
    const unsigned at = top[i].site;
    fprintf(stderr, "EDEN_HEAP_SITE %ld MiB in %ld blocks:", top[i].bytes >> 20, top[i].blocks);
    for (unsigned k = 0; k < EDEN_SITE_PCS; ++k)
      fprintf(stderr, " %lx", (unsigned long)(eden_site_table[at].pcs[k] ? eden_site_table[at].pcs[k] - 0x400000ul : 0));
    fputc('\n', stderr);
  }
}

/* ---- EDEN_UP_BIG_STACKS=1 (0593 up big stacks, our own; log only, off unless exactly 1) ----
 * An exact table of every heap request of 256 KiB or more, by call chain + thread name + "inside a
 * run-time pipeline build" (T136's build scope, EDEN_UP_BUILD_BURST_LOG=1): calls, bytes asked,
 * frees, live now, peak live, heap pieces the requests made the heap take, refusals. A reporter
 * thread (BigStacks, detached, own 256 KiB stack, started in headless/heap_pages.cpp) prints it to
 * stderr.log when a wave starts, every 2 s while the pool is short and every 20 s; the crash report
 * prints its top rows (eden_big_crash_rows); operator new's EDEN_NEW_WAIT start / EDEN_HEAP_FAILURE
 * lines get the refused request's row and the threads that grew the heap (eden_big_failure_text).
 *
 * Off (unset or anything but "1"): nothing is mapped, no thread runs; a request or free of
 * 256 KiB or more pays one relaxed load of eden_big_on and a branch, smaller ones the size compare.
 *
 * The chain (eden_big_collect; no _Unwind_Backtrace: it can lock and allocate). CMake asks for no
 * frame pointers, but the SIE target keeps them by default: in T136's llvm-pie.elf 83 % of Eden's C++
 * functions and 73 % of Mesa's (RADV, NIR, ACO) begin with push rbp; mov rsp, rbp, operator new and
 * ralloc too (leaf functions and some libraries do not). So two lists per row: the frame-pointer
 * chain from the wrapper's caller up (exact; a function without a frame pointer is skipped or ends
 * it), its first EDEN_BIG_KEY links the key; and, as the site tracker above, the stack scan's first
 * code words (every return address, stale words too; tools/symbolize-big-stacks.py drops those that
 * do not follow a call). Both read only from the frame up to the end of its page (16 KiB on the
 * console), or, on a thread that named itself (its stack bounds taken then, outside any allocation,
 * by pthread_attr_get_np), up to its stack's top: the scan never more than EDEN_BIG_SCAN_WORDS
 * words, the chain never more than EDEN_BIG_FP_STEPS links.
 *
 * The hooks run after the allocator returned (or before a free releases the block), so none of
 * the heap's own locks (creation, growth, mspace) is held; the caller's own locks may be, and the
 * hooks take none: atomics on fixed tables, at most 64 + 16 probes, one bounded stack scan and one
 * bounded chain walk, no allocation, no stdio, no sleep; besides the heap's own pthread_getspecific
 * (eden_heap_self) only pthread_self, which takes no lock. A thread inside a hook that comes back in
 * (a signal handler that allocates) is not followed (reentry, counted). */
#include <time.h>
#if defined(PS5_NATIVE)
#include <pthread_np.h>
#endif
#ifndef EDEN_BIG_ROWS /* the host checks make them small to fill them */
#define EDEN_BIG_ROWS 4096u
#endif
#ifndef EDEN_BIG_BLOCKS
#define EDEN_BIG_BLOCKS 32768u
#endif
#define EDEN_BIG_CHAIN 16u       /* the frame-pointer chain: */
#define EDEN_BIG_CHAIN_FIRST 12u /* its first 12 links and its outermost 4 */
#define EDEN_BIG_SCAN 16u        /* the stack scan's first code words */
#define EDEN_BIG_FP_STEPS 256u
#define EDEN_BIG_PCS (EDEN_BIG_CHAIN + EDEN_BIG_SCAN)
#define EDEN_BIG_KEY 6u
#define EDEN_BIG_SCAN_WORDS 4096u /* 32 KiB of stack at most */
/* The page that holds a frame is mapped whole: 16 KiB on the console; 4 KiB assumed elsewhere. */
#if defined(PS5_NATIVE)
#define EDEN_BIG_PAGE_MASK ((uintptr_t)0x3FFF)
#else
#define EDEN_BIG_PAGE_MASK ((uintptr_t)0xFFF)
#endif
#define EDEN_BIG_ROW_PROBES 64u
#define EDEN_BIG_BLOCK_PROBES 16u
#define EDEN_BIG_RISE 12u   /* rows by growth: live at the window's peak minus live at the last table */
#define EDEN_BIG_TOP 6u     /* rows by live at the window's peak (old boot holders too) */
#define EDEN_BIG_RECENT 6u  /* rows by bytes asked since the last table */
#define EDEN_BIG_REFUSED 8u /* rows with new refusals */
#define EDEN_BIG_SAID_NAMES 64u /* distinct thread names (with ok) the bounds lines name */
/* depth: chain links kept | scan words << 8 | deep << 16 | the walk's links in the app's code << 17
 * (more than EDEN_BIG_CHAIN: links between the first 12 and the outermost 4 were left out). */
#define EDEN_BIG_DEPTH_TOTAL(depth) (((depth) >> 17) & 0x1FFu)

struct eden_big_row {
  _Atomic uint64_t hash; /* 0: free */
  atomic_uint ready;     /* key written (pcs, thread, build) */
  atomic_uint upgrading; /* a deeper chain is being written */
  atomic_uint depth;     /* chain count | scan count << 8 | deep << 16 */
  unsigned build;
  unsigned key_at;       /* 0: keyed on the chain; EDEN_BIG_CHAIN: on the scan */
  char thread[16];
  _Atomic uintptr_t pcs[EDEN_BIG_PCS];
  atomic_ullong calls, bytes, frees, freed, refused, refused_last, grew;
  atomic_llong live, blocks, peak;
  /* The reporter's alone. */
  unsigned long long last_calls, last_bytes, last_frees, last_grew, last_refused;
  long long at_peak, last_live;
};
struct eden_big_block {
  _Atomic uintptr_t address; /* 0 free, 1 given back */
  atomic_uint row;
  atomic_uint born_s;
  atomic_ullong size;
};
/* Per heap record (indexed like eden_heap_threads). */
struct eden_big_thread {
  _Atomic uintptr_t owner; /* pthread_self() of the thread whose stack bounds these are; 0: none */
  _Atomic uintptr_t low, high;
  atomic_uint busy;        /* inside a hook */
  atomic_uint refused_row; /* 1 + the row of this thread's last refused request, 0: it got none */
  atomic_ullong refused_bytes; /* the size of this thread's last refused request of 256 KiB or more */
  atomic_uint grown_mark[2];
  /* eden_big_note_stack's result, for the reporter's bounds line: `noted` counts the namings. */
  atomic_uint noted, bounds_ok, bounds_kib, bounds_why;
};
struct eden_big_tables {
  struct eden_big_row rows[EDEN_BIG_ROWS];
  struct eden_big_block blocks[EDEN_BIG_BLOCKS];
  struct eden_big_thread threads[EDEN_HEAP_THREADS];
  /* The reporter's alone: blocks older than 10 s per row, and its picking keys. */
  unsigned scratch[EDEN_BIG_ROWS];
  long long key_rise[EDEN_BIG_ROWS], key_peak[EDEN_BIG_ROWS], key_recent[EDEN_BIG_ROWS], key_refused[EDEN_BIG_ROWS];
  unsigned char taken[EDEN_BIG_ROWS];
  /* The crash handler's alone: live bytes per row in blocks under 10 s old. */
  long long crash_young[EDEN_BIG_ROWS];
};
static struct eden_big_tables *eden_big_tables;
static size_t eden_big_tables_size;
static atomic_uint eden_big_on;
static atomic_llong eden_big_class_live, eden_big_window_peak;
static atomic_ullong eden_big_window_peak_ms, eden_big_untracked, eden_big_rows_full, eden_big_reentry,
    eden_big_no_record, eden_big_refusals, eden_big_deep, eden_big_shallow, eden_big_rows_used;
static atomic_uint eden_big_now_s, eden_big_mark_at;
static atomic_ullong eden_big_now_ms, eden_big_mark_ms[2]; /* the reporter's clock, every 250 ms */
static unsigned eden_big_build_bit;
unsigned long long eden_free_memory_bytes(void) __attribute__((weak));

static unsigned long long eden_big_steady_ms(void) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return (unsigned long long)now.tv_sec * 1000ull + (unsigned long long)now.tv_nsec / 1000000ull;
}

/* This thread's heap pieces grown so far (0: no record). */
static long long eden_big_grown_now(void) {
  struct eden_heap_thread *self = eden_heap_self();
  return self != NULL ? (long long)__atomic_load_n(&self->grown, __ATOMIC_RELAXED) : 0;
}

/* -1: not followed (off, or a request under 256 KiB); else this thread's heap pieces grown so far. */
static inline long long eden_big_begin(size_t bytes) {
  /* acquire: the tables' pointer, written before eden_big_on, is then seen (a plain load on x86). */
  if (bytes < EDEN_HEAP_SMALL_REQUEST || !atomic_load_explicit(&eden_big_on, memory_order_acquire))
    return -1;
  return eden_big_grown_now();
}

/* The chain of the calling thread (see the comment above), in pcs[0, EDEN_BIG_CHAIN): the
 * frame-pointer chain from the wrapper's caller up, exact where every frame keeps a frame pointer
 * (the first EDEN_BIG_CHAIN_FIRST links and the outermost EDEN_BIG_CHAIN - EDEN_BIG_CHAIN_FIRST);
 * in pcs[EDEN_BIG_CHAIN, EDEN_BIG_PCS): the stack scan's first code words, from the slot that holds
 * `skip` (the hook's return address into the wrapper: below it lie the hooks' own frames, with
 * stale words of the allocation that just ran) up, so a frame without a frame pointer is not
 * missed, stale words included. Both read only [frame, end): the frame's page, or a named thread's
 * stack up to its top; a link that leaves that range, goes down or is misaligned ends the chain.
 * Returns chain count | scan count << 8 | deep << 16 | the walk's links in the app's code << 17. */
static __attribute__((noinline, no_sanitize("address"))) unsigned
eden_big_collect(uintptr_t pcs[EDEN_BIG_PCS], uintptr_t skip, size_t size, unsigned record) {
  /* The code: .text and what follows it in the same executable segment up to the next 16 KiB page
   * (operator new lives in __lcxx_override, right after .text). Only compared, never read. */
  const uintptr_t low = (uintptr_t)__eden_text_start + 8,
                  high = ((uintptr_t)__eden_text_end + 0x3FFFu) & ~(uintptr_t)0x3FFF;
  const uintptr_t self = (uintptr_t)__builtin_return_address(0);
  const uintptr_t *stack = (const uintptr_t *)__builtin_frame_address(0);
  const uintptr_t at = (uintptr_t)stack;
  uintptr_t end = (at | EDEN_BIG_PAGE_MASK) + 1; /* the frame's page */
  unsigned deep = 0;
  if (record < EDEN_HEAP_THREADS) {
    const struct eden_big_thread *t = &eden_big_tables->threads[record];
    const uintptr_t top = atomic_load_explicit(&t->high, memory_order_relaxed);
    if (atomic_load_explicit(&t->owner, memory_order_acquire) == (uintptr_t)pthread_self() &&
        at >= atomic_load_explicit(&t->low, memory_order_relaxed) && at < top) {
      end = top & ~(uintptr_t)7;
      deep = 1;
    }
  }
  /* The stack scan. */
  unsigned words = (unsigned)((end - at) / sizeof(uintptr_t));
  if (words > EDEN_BIG_SCAN_WORDS)
    words = EDEN_BIG_SCAN_WORDS;
  unsigned from = 0;
  for (unsigned i = 0; i < words; ++i)
    if (stack[i] == skip) {
      from = i + 1;
      break;
    }
  uintptr_t scan[EDEN_BIG_SCAN] = {0}, chain[EDEN_BIG_CHAIN] = {0};
  unsigned scan_n = 0;
  uintptr_t last = 0;
  for (unsigned i = from; i < words && scan_n < EDEN_BIG_SCAN; ++i) {
    const uintptr_t value = stack[i];
    if (value < low || value >= high || value == last || value == self || value == skip || value == size)
      continue;
    last = value;
    scan[scan_n++] = value;
  }
  /* The frame-pointer chain: records [saved frame pointer, return address], from this frame up;
   * the links up to the one that returns into the wrapper are the hooks'. */
  const unsigned outer_room = EDEN_BIG_CHAIN - EDEN_BIG_CHAIN_FIRST;
  uintptr_t outer[EDEN_BIG_CHAIN - EDEN_BIG_CHAIN_FIRST] = {0};
  unsigned first_n = 0, outer_n = 0, past_skip = 0;
  uintptr_t frame = at;
  for (unsigned steps = 0; steps < EDEN_BIG_FP_STEPS; ++steps) {
    if ((frame & 7u) != 0 || frame < at || frame > end - 2 * sizeof(uintptr_t))
      break;
    const uintptr_t next = ((const uintptr_t *)frame)[0], ret = ((const uintptr_t *)frame)[1];
    if (!past_skip) {
      past_skip = ret == skip;
    } else if (ret >= low && ret < high) {
      if (first_n < EDEN_BIG_CHAIN_FIRST)
        chain[first_n++] = ret;
      else
        outer[outer_n++ % outer_room] = ret;
    } /* else a return into code outside the app's (a system library): not listed, the walk goes on */
    if (next <= frame)
      break;
    frame = next;
  }
  const unsigned kept = outer_n < outer_room ? outer_n : outer_room;
  for (unsigned i = 0; i < kept; ++i)
    chain[first_n + i] = outer[(outer_n - kept + i) % outer_room];
  for (unsigned i = 0; i < EDEN_BIG_CHAIN; ++i)
    pcs[i] = chain[i];
  for (unsigned i = 0; i < EDEN_BIG_SCAN; ++i)
    pcs[EDEN_BIG_CHAIN + i] = scan[i];
  const unsigned total = first_n + outer_n; /* at most EDEN_BIG_FP_STEPS: 9 bits */
  return (first_n + kept) | scan_n << 8 | deep << 16 | total << 17;
}

/* The row of a chain on this thread (made on first sight), or NULL when 64 probes find none. The
 * key: the first EDEN_BIG_KEY links of the frame-pointer chain (exact), or, when it has none, the
 * scan's first EDEN_BIG_KEY words; + the thread's name + the build bit. */
static struct eden_big_row *eden_big_row_for(const uintptr_t pcs[EDEN_BIG_PCS], unsigned depth,
                                             const char name[16], unsigned build) {
  const unsigned key_at = (depth & 0xFFu) != 0 ? 0 : EDEN_BIG_CHAIN;
  uint64_t hash = 0xCBF29CE484222325ull ^ key_at;
  for (unsigned i = 0; i < EDEN_BIG_KEY; ++i)
    hash = (hash ^ pcs[key_at + i]) * 0x100000001B3ull;
  for (unsigned i = 0; i < 16 && name[i]; ++i)
    hash = (hash ^ (uint8_t)name[i]) * 0x100000001B3ull;
  hash = ((hash ^ build) * 0x100000001B3ull) | 1;
  struct eden_big_row *rows = eden_big_tables->rows;
  for (unsigned probe = 0; probe < EDEN_BIG_ROW_PROBES; ++probe) {
    struct eden_big_row *row = &rows[(unsigned)((hash >> 12) + probe) % EDEN_BIG_ROWS];
    uint64_t expected = 0;
    if (atomic_compare_exchange_strong_explicit(&row->hash, &expected, hash, memory_order_acq_rel,
                                                memory_order_acquire)) {
      for (unsigned i = 0; i < EDEN_BIG_PCS; ++i)
        atomic_store_explicit(&row->pcs[i], pcs[i], memory_order_relaxed);
      memcpy(row->thread, name, 16);
      row->build = build;
      row->key_at = key_at;
      atomic_store_explicit(&row->depth, depth, memory_order_relaxed);
      atomic_store_explicit(&row->ready, 1, memory_order_release);
      atomic_fetch_add_explicit(&eden_big_rows_used, 1, memory_order_relaxed);
      return row;
    }
    if (expected != hash)
      continue;
    /* Same hash: the same key unless a collision (checked once the key is written; a row being
     * made right now is taken as it is). */
    if (!atomic_load_explicit(&row->ready, memory_order_acquire))
      return row;
    int same = row->build == build && row->key_at == key_at && memcmp(row->thread, name, 16) == 0;
    for (unsigned i = 0; same && i < EDEN_BIG_KEY; ++i)
      same = atomic_load_explicit(&row->pcs[key_at + i], memory_order_relaxed) == pcs[key_at + i];
    if (!same)
      continue;
    /* A row keeps the deepest chain one of its calls saw (a call near a page's end sees little). */
    const unsigned had = atomic_load_explicit(&row->depth, memory_order_relaxed);
    if ((depth & 0xFFu) + ((depth >> 8) & 0xFFu) > (had & 0xFFu) + ((had >> 8) & 0xFFu)) {
      unsigned free_flag = 0;
      if (atomic_compare_exchange_strong_explicit(&row->upgrading, &free_flag, 1, memory_order_acquire,
                                                  memory_order_relaxed)) {
        for (unsigned i = 0; i < EDEN_BIG_PCS; ++i)
          if (i < key_at || i >= key_at + EDEN_BIG_KEY)
            atomic_store_explicit(&row->pcs[i], pcs[i], memory_order_relaxed);
        atomic_store_explicit(&row->depth, depth, memory_order_relaxed);
        atomic_store_explicit(&row->upgrading, 0, memory_order_release);
      }
    }
    return row;
  }
  atomic_fetch_add_explicit(&eden_big_rows_full, 1, memory_order_relaxed);
  return NULL;
}

/* A high-water mark, bounded: under heavy contention a peak may come out a little low. */
static void eden_big_max(atomic_llong *peak, long long value) {
  long long seen = atomic_load_explicit(peak, memory_order_relaxed);
  for (unsigned tries = 0; value > seen && tries < 16; ++tries)
    if (atomic_compare_exchange_weak_explicit(peak, &seen, value, memory_order_relaxed, memory_order_relaxed))
      return;
}

/* Shared by the allocation and the refusal hooks: guard, record, chain, row. */
static struct eden_big_row *eden_big_capture(uintptr_t skip, size_t size, struct eden_heap_thread **record) {
  struct eden_heap_thread *self = eden_heap_self();
  *record = self;
  if (self == NULL) {
    atomic_fetch_add_explicit(&eden_big_no_record, 1, memory_order_relaxed);
    return NULL;
  }
  const unsigned index = (unsigned)(self - eden_heap_threads);
  struct eden_big_thread *t = &eden_big_tables->threads[index];
  if (atomic_exchange_explicit(&t->busy, 1, memory_order_acquire) != 0) {
    atomic_fetch_add_explicit(&eden_big_reentry, 1, memory_order_relaxed);
    return NULL; /* *record stays: a refusal still clears the thread's refused row */
  }
  uintptr_t pcs[EDEN_BIG_PCS] = {0}; /* on the stack the scan reads: no stale words here */
  const unsigned depth = eden_big_collect(pcs, skip, size, index);
  atomic_fetch_add_explicit(((depth >> 16) & 1u) ? &eden_big_deep : &eden_big_shallow, 1, memory_order_relaxed);
  char name[16];
  for (unsigned i = 0; i < 16; ++i)
    name[i] = __atomic_load_n(&self->name[i], __ATOMIC_RELAXED);
  name[15] = 0;
  for (unsigned i = 0; i < 16; ++i)
    if (name[i] == 0) {
      memset(name + i, 0, 16 - i);
      break;
    }
  /* T136's build scope (only moves with EDEN_UP_BUILD_BURST_LOG=1): its own thread writes depth. */
  const unsigned build = eden_build_heap_records[index].depth > 0;
  struct eden_big_row *row = eden_big_row_for(pcs, depth, name, build);
  atomic_store_explicit(&t->busy, 0, memory_order_release);
  return row;
}

static __attribute__((noinline)) void eden_big_note_alloc(const void *address, size_t size, long long grown_before) {
  struct eden_heap_thread *self;
  struct eden_big_row *row = eden_big_capture((uintptr_t)__builtin_return_address(0), size, &self);
  if (row == NULL) {
    atomic_fetch_add_explicit(&eden_big_untracked, size, memory_order_relaxed);
    return;
  }
  const unsigned grown_now = __atomic_load_n(&self->grown, __ATOMIC_RELAXED);
  if ((long long)grown_now > grown_before)
    atomic_fetch_add_explicit(&row->grew, (unsigned long long)((long long)grown_now - grown_before), memory_order_relaxed);
  atomic_fetch_add_explicit(&row->calls, 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&row->bytes, size, memory_order_relaxed);
  /* The block's place first: one that finds none is counted as untracked, not as live. */
  const uintptr_t value = (uintptr_t)address;
  const unsigned first = (unsigned)(((value >> 4) * 0x9E3779B97F4A7C15ull) >> 40) % EDEN_BIG_BLOCKS;
  for (unsigned probe = 0; probe < EDEN_BIG_BLOCK_PROBES; ++probe) {
    struct eden_big_block *block = &eden_big_tables->blocks[(first + probe) % EDEN_BIG_BLOCKS];
    uintptr_t seen = atomic_load_explicit(&block->address, memory_order_relaxed);
    if (seen > 1)
      continue;
    if (!atomic_compare_exchange_strong_explicit(&block->address, &seen, value, memory_order_acq_rel,
                                                 memory_order_relaxed))
      continue;
    atomic_store_explicit(&block->row, (unsigned)(row - eden_big_tables->rows), memory_order_relaxed);
    atomic_store_explicit(&block->size, size, memory_order_relaxed);
    atomic_store_explicit(&block->born_s, atomic_load_explicit(&eden_big_now_s, memory_order_relaxed),
                          memory_order_relaxed);
    eden_big_max(&row->peak, atomic_fetch_add_explicit(&row->live, (long long)size, memory_order_relaxed) + (long long)size);
    atomic_fetch_add_explicit(&row->blocks, 1, memory_order_relaxed);
    const long long class_live =
        atomic_fetch_add_explicit(&eden_big_class_live, (long long)size, memory_order_relaxed) + (long long)size;
    const long long window = atomic_load_explicit(&eden_big_window_peak, memory_order_relaxed);
    if (class_live > window) {
      eden_big_max(&eden_big_window_peak, class_live);
      atomic_store_explicit(&eden_big_window_peak_ms, atomic_load_explicit(&eden_big_now_ms, memory_order_relaxed), memory_order_relaxed);
    }
    return;
  }
  atomic_fetch_add_explicit(&eden_big_untracked, size, memory_order_relaxed);
}

/* Counts attempts: operator new with EDEN_UP_NEW_WAIT_MS asks again every 20 ms, and each retry is
 * one more refusal (and capture) on the same row. The thread's refused row and size are set on every
 * refusal (row 0 when this one got none), so the failure lines never name an older request. */
static __attribute__((noinline)) void eden_big_note_refused(size_t size) {
  struct eden_heap_thread *self;
  struct eden_big_row *row = eden_big_capture((uintptr_t)__builtin_return_address(0), size, &self);
  atomic_fetch_add_explicit(&eden_big_refusals, 1, memory_order_relaxed);
  if (self != NULL) {
    struct eden_big_thread *t = &eden_big_tables->threads[self - eden_heap_threads];
    atomic_store_explicit(&t->refused_bytes, size, memory_order_relaxed);
    atomic_store_explicit(&t->refused_row, row != NULL ? (unsigned)(row - eden_big_tables->rows) + 1 : 0u,
                          memory_order_relaxed);
  }
  if (row == NULL)
    return;
  atomic_fetch_add_explicit(&row->refused, 1, memory_order_relaxed);
  atomic_store_explicit(&row->refused_last, size, memory_order_relaxed);
}

/* The wrappers' end of a followed request: `grown_before` from eden_big_begin (-1: not followed). */
static inline void eden_big_end(const void *address, size_t size, long long grown_before) {
  if (grown_before < 0)
    return;
  if (address != NULL)
    eden_big_note_alloc(address, size, grown_before);
  else
    eden_big_note_refused(size);
}

/* A followed block leaves the table before its memory goes back (so its address cannot be handed
 * out again meanwhile). Returns 1 + the row, with *size, or 0 when the block is not followed. */
static unsigned eden_big_detach(const void *address, unsigned long long *size) {
  const uintptr_t value = (uintptr_t)address;
  const unsigned first = (unsigned)(((value >> 4) * 0x9E3779B97F4A7C15ull) >> 40) % EDEN_BIG_BLOCKS;
  for (unsigned probe = 0; probe < EDEN_BIG_BLOCK_PROBES; ++probe) {
    struct eden_big_block *block = &eden_big_tables->blocks[(first + probe) % EDEN_BIG_BLOCKS];
    uintptr_t seen = atomic_load_explicit(&block->address, memory_order_acquire);
    if (seen == 0)
      return 0;
    if (seen != value)
      continue;
    const unsigned row = atomic_load_explicit(&block->row, memory_order_relaxed);
    const unsigned long long bytes = atomic_load_explicit(&block->size, memory_order_relaxed);
    if (!atomic_compare_exchange_strong_explicit(&block->address, &seen, 1, memory_order_acq_rel, memory_order_relaxed))
      return 0;
    struct eden_big_row *r = &eden_big_tables->rows[row % EDEN_BIG_ROWS];
    atomic_fetch_sub_explicit(&r->live, (long long)bytes, memory_order_relaxed);
    atomic_fetch_sub_explicit(&r->blocks, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&r->frees, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&r->freed, bytes, memory_order_relaxed);
    atomic_fetch_sub_explicit(&eden_big_class_live, (long long)bytes, memory_order_relaxed);
    *size = bytes;
    return row + 1;
  }
  return 0;
}

/* realloc's old block, back in its row when the realloc failed and the block stays. */
static void eden_big_reattach(const void *address, unsigned row, unsigned long long size) {
  struct eden_big_row *r = &eden_big_tables->rows[(row - 1) % EDEN_BIG_ROWS];
  const uintptr_t value = (uintptr_t)address;
  const unsigned first = (unsigned)(((value >> 4) * 0x9E3779B97F4A7C15ull) >> 40) % EDEN_BIG_BLOCKS;
  atomic_fetch_sub_explicit(&r->frees, 1, memory_order_relaxed);
  atomic_fetch_sub_explicit(&r->freed, size, memory_order_relaxed);
  for (unsigned probe = 0; probe < EDEN_BIG_BLOCK_PROBES; ++probe) {
    struct eden_big_block *block = &eden_big_tables->blocks[(first + probe) % EDEN_BIG_BLOCKS];
    uintptr_t seen = atomic_load_explicit(&block->address, memory_order_relaxed);
    if (seen > 1 ||
        !atomic_compare_exchange_strong_explicit(&block->address, &seen, value, memory_order_acq_rel, memory_order_relaxed))
      continue;
    atomic_store_explicit(&block->row, row - 1, memory_order_relaxed);
    atomic_store_explicit(&block->size, size, memory_order_relaxed);
    atomic_fetch_add_explicit(&r->live, (long long)size, memory_order_relaxed);
    atomic_fetch_add_explicit(&r->blocks, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&eden_big_class_live, (long long)size, memory_order_relaxed);
    return;
  }
  atomic_fetch_add_explicit(&eden_big_untracked, size, memory_order_relaxed);
}

static inline void eden_big_free(const void *address, size_t usable) {
  if (usable < EDEN_HEAP_SMALL_REQUEST || !atomic_load_explicit(&eden_big_on, memory_order_acquire))
    return;
  unsigned long long size;
  (void)eden_big_detach(address, &size);
}

/* EDEN_UP_BIG_STACKS: a thread that names itself (eden_heap_name_thread, outside any allocation)
 * keeps its stack's bounds, so the scan may read past the frame's page up to the stack's top. A
 * fiber's frame lies outside them and reads only its page.
 *
 * The bounds are trusted only when the kernel says that [the page of this call's frame, top) is
 * mapped and readable (review M1: pthread_attr_get_np could name a wrong top, and a garbage
 * frame-pointer link would then read past the stack, a fault inside malloc). The kept low is the
 * start of the readable range that holds this frame, so every deep read [frame, top) of a frame
 * inside [low, top) lies in memory the kernel called readable. The stack lives as long as its
 * thread; eden_big_thread_ended clears the bounds before the record is given back.
 * Returns 1 and the readable [*from, *to) around [at, top), or 0. At most 64 kernel regions. The host
 * stand-in (Linux, mincore) checks the whole [low, top), else [the frame's page, top): mincore says
 * mapped, not readable, and glibc's pthread_getattr_np leaves the guard page out of [low, top). */
#if defined(PS5_NATIVE)
int32_t sceKernelQueryMemoryProtection(void *address, void **start, void **end, uint32_t *protection);
#define EDEN_BIG_PROT_CPU_READ 0x01u /* as RADV's 0x33 mask (CPU read|write, GPU read|write) */
#elif defined(__linux__)
#include <sys/mman.h>
#endif
static int eden_big_readable(uintptr_t low, uintptr_t at, uintptr_t top, uintptr_t *from, uintptr_t *to) {
  const uintptr_t page = EDEN_BIG_PAGE_MASK + 1;
  const uintptr_t first = at & ~EDEN_BIG_PAGE_MASK;
  if (top <= first)
    return 0;
#if defined(PS5_NATIVE)
  (void)low;
  (void)page;
  uintptr_t cursor = first, start_low = 0;
  for (unsigned regions = 0; regions < 64u; ++regions) {
    void *start = NULL, *end = NULL;
    uint32_t protection = 0;
    if (sceKernelQueryMemoryProtection((void *)cursor, &start, &end, &protection) != 0 ||
        (uintptr_t)start > cursor || (uintptr_t)end <= cursor || !(protection & EDEN_BIG_PROT_CPU_READ))
      return 0;
    if (regions == 0)
      start_low = (uintptr_t)start;
    cursor = (uintptr_t)end;
    if (cursor >= top) {
      *from = start_low;
      *to = top;
      return 1;
    }
  }
  return 0;
#elif defined(__linux__)
  /* The host checks: mincore fails with ENOMEM on any page that is not mapped (stacks are read/write). */
  /* [low, top) whole, else (a main thread's stack, mapped as it grows) [the frame's page, top). */
  unsigned char state[256];
  for (unsigned attempt = 0; attempt < 2; ++attempt) {
    const uintptr_t begin = attempt == 0 ? low : first;
    if ((begin & EDEN_BIG_PAGE_MASK) != 0 || begin > first)
      continue;
    uintptr_t cursor = begin;
    while (cursor < top) {
      const uintptr_t pages = (top - cursor + page - 1) / page;
      const size_t now = pages < 256u ? (size_t)pages : 256u;
      if (mincore((void *)cursor, now * page, state) != 0)
        break;
      cursor += now * page;
    }
    if (cursor >= top) {
      *from = begin;
      *to = top;
      return 1;
    }
  }
  return 0;
#else
  (void)page;
  (void)low;
  (void)from;
  (void)to;
  return 0; /* no way to check here: shallow */
#endif
}

static struct eden_big_tables *eden_big_tables_now(void) {
  return __atomic_load_n(&eden_big_tables, __ATOMIC_ACQUIRE);
}

static void eden_big_note_stack(struct eden_heap_thread *self) {
  struct eden_big_tables *tables = eden_big_tables_now();
  if (tables == NULL || self == NULL)
    return;
  struct eden_big_thread *t = &tables->threads[self - eden_heap_threads];
  uintptr_t low = 0, high = 0;
#if defined(PS5_NATIVE) || defined(__FreeBSD__)
  pthread_attr_t attributes;
  if (pthread_attr_init(&attributes) == 0) {
    void *base = NULL;
    size_t bytes = 0;
    if (pthread_attr_get_np(pthread_self(), &attributes) == 0 && pthread_attr_getstack(&attributes, &base, &bytes) == 0) {
      low = (uintptr_t)base;
      high = low + bytes;
    }
    pthread_attr_destroy(&attributes);
  }
#elif defined(__APPLE__)
  high = (uintptr_t)pthread_get_stackaddr_np(pthread_self());
  low = high - pthread_get_stacksize_np(pthread_self());
#elif defined(__linux__)
  pthread_attr_t attributes;
  if (pthread_getattr_np(pthread_self(), &attributes) == 0) {
    void *base = NULL;
    size_t bytes = 0;
    if (pthread_attr_getstack(&attributes, &base, &bytes) == 0) {
      low = (uintptr_t)base;
      high = low + bytes;
    }
    pthread_attr_destroy(&attributes);
  }
#endif
  /* The frame of this call must lie inside, or the bounds are not this stack's. */
  const uintptr_t here = (uintptr_t)__builtin_frame_address(0);
  atomic_store_explicit(&t->owner, 0, memory_order_release);
  unsigned why = 0; /* 0 ok, 1 no bounds, 2 this frame outside or too large, 3 the kernel check */
  uintptr_t from = 0, to = 0;
  if (low == 0 || high <= low)
    why = 1;
  else if (here < low || here >= high || high - low > ((size_t)256 << 20))
    why = 2;
  else if (!eden_big_readable(low, here, high, &from, &to))
    why = 3;
  if (why == 0) {
    if (from > low)
      low = from;
    if (to < high)
      high = to;
    atomic_store_explicit(&t->low, low, memory_order_relaxed);
    atomic_store_explicit(&t->high, high, memory_order_relaxed);
    atomic_store_explicit(&t->owner, (uintptr_t)pthread_self(), memory_order_release);
  }
  atomic_store_explicit(&t->bounds_ok, why == 0, memory_order_relaxed);
  atomic_store_explicit(&t->bounds_why, why, memory_order_relaxed);
  atomic_store_explicit(&t->bounds_kib, why == 0 ? (unsigned)((high - low) >> 10) : 0u, memory_order_relaxed);
  atomic_fetch_add_explicit(&t->noted, 1, memory_order_release);
}

/* A record's thread ends: its bounds go (eden_heap_thread_exit). */
static void eden_big_thread_ended(struct eden_heap_thread *self) {
  struct eden_big_tables *tables = eden_big_tables_now();
  if (tables != NULL && self != NULL)
    atomic_store_explicit(&tables->threads[self - eden_heap_threads].owner, 0, memory_order_release);
}

/* EDEN_UP_BIG_STACKS, from eden_heap_pool_init: map the tables (hint 0x10_0000_0000, as the fiber
 * stacks: a mapping that names no address lands in RADV's 32-bit window). 0: done; else -1 with
 * `why` (map, place) and errno in *error. Nothing is followed until eden_big_stacks_go. */
int eden_big_stacks_map(const char **why, int *error) {
  if (eden_big_tables != NULL)
    return 0;
  const size_t size = (sizeof(struct eden_big_tables) + 16383u) & ~(size_t)16383u;
  void *at = mmap((void *)0x1000000000ull, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  if (at == MAP_FAILED) {
    *why = "map";
    *error = errno;
    return -1;
  }
#if defined(PS5_NATIVE)
  if ((uintptr_t)at < 0x300000000ull || (uintptr_t)at + size > 0x4000000000ull) {
    munmap(at, size);
    *why = "place";
    *error = 0;
    return -1;
  }
#endif
  eden_big_tables_size = size;
  __atomic_store_n(&eden_big_tables, (struct eden_big_tables *)at, __ATOMIC_RELEASE);
  return 0;
}

/* The reporter thread could not be made: nothing will be followed. The pointer goes, the 2.6 MiB stay
 * mapped (review S1: a thread naming itself right now may still write its bounds into them). */
void eden_big_stacks_unmap(void) {
  if (atomic_load_explicit(&eden_big_on, memory_order_relaxed))
    return;
  __atomic_store_n(&eden_big_tables, (struct eden_big_tables *)NULL, __ATOMIC_RELEASE);
}

/* On, once the reporter thread exists. `build_bit`: EDEN_UP_BUILD_BURST_LOG=1 too (else build=0). */
void eden_big_stacks_go(int build_bit) {
  if (eden_big_tables == NULL)
    return;
  eden_big_build_bit = build_bit != 0;
  /* The hooks' clock (blocks' age, the window peak's time) from the start, before the reporter's first tick. */
  const unsigned long long now_ms = eden_big_steady_ms();
  atomic_store_explicit(&eden_big_now_ms, now_ms, memory_order_relaxed);
  atomic_store_explicit(&eden_big_now_s, (unsigned)(now_ms / 1000u), memory_order_relaxed);
  atomic_store_explicit(&eden_big_window_peak_ms, now_ms, memory_order_relaxed);
  fprintf(stderr,
          "EDEN_BIG_STACKS on rows=%u blocks=%u table_at=%p table_kib=%zu code=0x%lx-0x%lx (pcs below are offsets "
          "from code start, = addresses in llvm-pie.elf) chain=%u scan=%u key=%u scan_kib=%u build_bit=%u "
          "steady_ms=%llu\n",
          EDEN_BIG_ROWS, EDEN_BIG_BLOCKS, (void *)eden_big_tables, eden_big_tables_size >> 10,
          (unsigned long)(uintptr_t)__eden_text_start, (unsigned long)(uintptr_t)__eden_text_end, EDEN_BIG_CHAIN,
          EDEN_BIG_SCAN, EDEN_BIG_KEY, EDEN_BIG_SCAN_WORDS * 8u / 1024u, eden_big_build_bit, now_ms);
  atomic_store_explicit(&eden_big_on, 1, memory_order_release);
}

/* ---- EDEN_UP_BIG_STACKS: the reporter thread (BigStacks), outside every hook ---- */
static void eden_big_mib(char *out, size_t room, long long bytes) {
  if (bytes < 0)
    bytes = 0;
  snprintf(out, room, "%lld.%lld", bytes >> 20, ((bytes & ((1ll << 20) - 1)) * 10) >> 20);
}

static const char *eden_big_name(const char name[16], char out[16]) {
  unsigned i = 0;
  for (; i < 15 && name[i]; ++i)
    out[i] = name[i] == ' ' || name[i] == '=' || name[i] == ',' ? '_' : name[i];
  out[i] = 0;
  return i ? out : "?";
}

/* Chosen rows of a table, most first by `key` (bounded insertion; `taken` keeps a row once). */
static unsigned eden_big_pick(unsigned *picked, unsigned have, unsigned most, const long long *key,
                              unsigned char *taken) {
  unsigned start = have;
  if (most == 0)
    return have;
  for (unsigned r = 0; r < EDEN_BIG_ROWS; ++r) {
    if (taken[r] || key[r] <= 0)
      continue;
    unsigned place = have;
    if (have - start < most)
      ++have;
    else if (key[r] <= key[picked[have - 1]])
      continue;
    else
      place = have - 1;
    while (place > start && key[picked[place - 1]] < key[r]) {
      picked[place] = picked[place - 1];
      --place;
    }
    picked[place] = r;
  }
  for (unsigned i = start; i < have; ++i)
    taken[picked[i]] = 1;
  return have;
}

static void eden_big_table(const char *why, unsigned seq, unsigned long long now_ms, size_t heap, size_t heap_2s,
                           unsigned long long pool, int steady) {
  struct eden_big_tables *tables = eden_big_tables_now();
  const unsigned now_s = atomic_load_explicit(&eden_big_now_s, memory_order_relaxed);
  memset(tables->scratch, 0, sizeof(tables->scratch));
  for (unsigned b = 0; b < EDEN_BIG_BLOCKS; ++b) {
    const struct eden_big_block *block = &tables->blocks[b];
    if (atomic_load_explicit(&block->address, memory_order_relaxed) <= 1)
      continue;
    if (atomic_load_explicit(&block->born_s, memory_order_relaxed) + 10u <= now_s)
      ++tables->scratch[atomic_load_explicit(&block->row, memory_order_relaxed) % EDEN_BIG_ROWS];
  }
  for (unsigned r = 0; r < EDEN_BIG_ROWS; ++r) {
    struct eden_big_row *row = &tables->rows[r];
    const int used = atomic_load_explicit(&row->ready, memory_order_acquire) != 0;
    tables->taken[r] = !used;
    tables->key_rise[r] = used && !steady ? row->at_peak - row->last_live : 0;
    tables->key_peak[r] = used && !steady ? row->at_peak : 0;
    tables->key_recent[r] = used ? (long long)(atomic_load_explicit(&row->bytes, memory_order_relaxed) - row->last_bytes) : 0;
    tables->key_refused[r] = used ? (long long)(atomic_load_explicit(&row->refused, memory_order_relaxed) - row->last_refused) : 0;
  }
  /* Growth first (review B: the boot's long-lived holders must not push a wave's rows out), then
   * the largest holders, the busiest since the last table, new refusals. */
  unsigned picked[EDEN_BIG_RISE + EDEN_BIG_TOP + EDEN_BIG_RECENT + EDEN_BIG_REFUSED];
  unsigned count = eden_big_pick(picked, 0, steady ? 0 : EDEN_BIG_RISE, tables->key_rise, tables->taken);
  count = eden_big_pick(picked, count, steady ? 0 : EDEN_BIG_TOP, tables->key_peak, tables->taken);
  count = eden_big_pick(picked, count, EDEN_BIG_RECENT, tables->key_recent, tables->taken);
  count = eden_big_pick(picked, count, EDEN_BIG_REFUSED, tables->key_refused, tables->taken);
  char live[24], peak[24], at_heap[24], at_2s[24], untracked[24];
  eden_big_mib(live, sizeof(live), atomic_load_explicit(&eden_big_class_live, memory_order_relaxed));
  eden_big_mib(peak, sizeof(peak), atomic_load_explicit(&eden_big_window_peak, memory_order_relaxed));
  eden_big_mib(at_heap, sizeof(at_heap), (long long)heap);
  eden_big_mib(at_2s, sizeof(at_2s), (long long)heap_2s);
  eden_big_mib(untracked, sizeof(untracked), (long long)atomic_load_explicit(&eden_big_untracked, memory_order_relaxed));
  char pool_text[24];
  if (pool != 0)
    snprintf(pool_text, sizeof(pool_text), "%llu", pool >> 20);
  else
    snprintf(pool_text, sizeof(pool_text), "unknown");
  fprintf(stderr,
          "EDEN_BIG_TABLE n=%u why=%s steady_ms=%llu class_live_mib=%s win_peak_mib=%s win_peak_ms=%llu heap_mib=%s "
          "heap_2s_mib=+%s pool_mib=%s rows=%llu rows_full=%llu untracked_mib=%s reentry=%llu no_record=%llu "
          "refusals=%llu deep=%llu shallow=%llu shown=%u\n",
          seq, why, now_ms, live, peak, (unsigned long long)atomic_load_explicit(&eden_big_window_peak_ms, memory_order_relaxed),
          at_heap, at_2s, pool_text, (unsigned long long)atomic_load_explicit(&eden_big_rows_used, memory_order_relaxed),
          (unsigned long long)atomic_load_explicit(&eden_big_rows_full, memory_order_relaxed), untracked,
          (unsigned long long)atomic_load_explicit(&eden_big_reentry, memory_order_relaxed),
          (unsigned long long)atomic_load_explicit(&eden_big_no_record, memory_order_relaxed),
          (unsigned long long)atomic_load_explicit(&eden_big_refusals, memory_order_relaxed),
          (unsigned long long)atomic_load_explicit(&eden_big_deep, memory_order_relaxed),
          (unsigned long long)atomic_load_explicit(&eden_big_shallow, memory_order_relaxed), count);
  const uintptr_t code = (uintptr_t)__eden_text_start;
  for (unsigned i = 0; i < count; ++i) {
    struct eden_big_row *row = &tables->rows[picked[i]];
    char line[1024], name[16], at_peak[24], rise[24], now[24], high[24], asked_new[24], asked[24];
    const unsigned long long calls = atomic_load_explicit(&row->calls, memory_order_relaxed),
                             bytes = atomic_load_explicit(&row->bytes, memory_order_relaxed),
                             frees = atomic_load_explicit(&row->frees, memory_order_relaxed),
                             grew = atomic_load_explicit(&row->grew, memory_order_relaxed),
                             refused = atomic_load_explicit(&row->refused, memory_order_relaxed);
    const unsigned depth = atomic_load_explicit(&row->depth, memory_order_relaxed);
    eden_big_mib(at_peak, sizeof(at_peak), row->at_peak);
    const long long rise_bytes = row->at_peak - row->last_live;
    eden_big_mib(rise, sizeof(rise), rise_bytes < 0 ? -rise_bytes : rise_bytes);
    eden_big_mib(now, sizeof(now), atomic_load_explicit(&row->live, memory_order_relaxed));
    eden_big_mib(high, sizeof(high), atomic_load_explicit(&row->peak, memory_order_relaxed));
    eden_big_mib(asked_new, sizeof(asked_new), (long long)(bytes - row->last_bytes));
    eden_big_mib(asked, sizeof(asked), (long long)bytes);
    int at = snprintf(line, sizeof(line),
                      "EDEN_BIG_STACK n=%u row=%u thread=%s build=%u at_peak_mib=%s rise_mib=%s%s live_mib=%s blocks=%lld "
                      "peak_mib=%s calls=+%llu/%llu req_mib=+%s/%s frees=+%llu/%llu grew_mib=+%llu/%llu refused=+%llu/%llu "
                      "last_refused=%llu old10s=%u depth=%u+%u chain_total=%u deep=%u key=%s pcs=",
                      seq, picked[i], eden_big_name(row->thread, name), row->build, at_peak, rise_bytes < 0 ? "-" : "+",
                      rise, now,
                      (long long)atomic_load_explicit(&row->blocks, memory_order_relaxed), high, calls - row->last_calls,
                      calls, asked_new, asked, frees - row->last_frees, frees,
                      (grew - row->last_grew) << (EDEN_HEAP_PIECE_SHIFT - 20), grew << (EDEN_HEAP_PIECE_SHIFT - 20),
                      refused - row->last_refused, refused,
                      (unsigned long long)atomic_load_explicit(&row->refused_last, memory_order_relaxed),
                      tables->scratch[picked[i]], depth & 0xFFu, (depth >> 8) & 0xFFu, EDEN_BIG_DEPTH_TOTAL(depth),
                      (depth >> 16) & 1u, row->key_at == 0 ? "chain" : "scan");
    /* ".." after the 12th chain link: links between it and the outermost 4 were left out. */
    const int gap = EDEN_BIG_DEPTH_TOTAL(depth) > EDEN_BIG_CHAIN;
    for (unsigned k = 0; k < EDEN_BIG_PCS && at > 0 && at < (int)sizeof(line) - 24; ++k) {
      const uintptr_t pc = atomic_load_explicit(&row->pcs[k], memory_order_relaxed);
      if (k == EDEN_BIG_CHAIN)
        line[at++] = '|';
      if (gap && k == EDEN_BIG_CHAIN_FIRST) {
        line[at++] = ',';
        line[at++] = '.';
        line[at++] = '.';
      }
      if (pc == 0)
        continue;
      at += snprintf(line + at, sizeof(line) - (size_t)at, "%s%lx",
                     k == 0 || k == EDEN_BIG_CHAIN ? "" : ",", (unsigned long)(pc - code));
    }
    if (at > 0 && at < (int)sizeof(line) - 1) {
      line[at++] = '\n';
      line[at] = 0;
      fputs(line, stderr);
    }
  }
  /* The next table counts from here. */
  for (unsigned r = 0; r < EDEN_BIG_ROWS; ++r) {
    struct eden_big_row *row = &tables->rows[r];
    if (!atomic_load_explicit(&row->ready, memory_order_acquire))
      continue;
    row->last_calls = atomic_load_explicit(&row->calls, memory_order_relaxed);
    row->last_bytes = atomic_load_explicit(&row->bytes, memory_order_relaxed);
    row->last_frees = atomic_load_explicit(&row->frees, memory_order_relaxed);
    row->last_grew = atomic_load_explicit(&row->grew, memory_order_relaxed);
    row->last_refused = atomic_load_explicit(&row->refused, memory_order_relaxed);
    row->at_peak = row->last_live = atomic_load_explicit(&row->live, memory_order_relaxed); /* the next window starts here */
  }
  const long long class_live = atomic_load_explicit(&eden_big_class_live, memory_order_relaxed);
  atomic_store_explicit(&eden_big_window_peak, class_live, memory_order_relaxed);
  atomic_store_explicit(&eden_big_window_peak_ms, now_ms, memory_order_relaxed);
}

/* The reporter's loop (headless/heap_pages.cpp starts it on its own thread). Every 250 ms: the
 * clock for the hooks, the heap's pieces, the class's live total (at a new high since the last
 * table every row's live is copied: the class's make-up at its peak, within 250 ms), every 2 s the
 * threads' grown pieces (for eden_big_failure_text). A table: at once when a wave starts (class
 * live +128 MiB or heap pieces +256 MiB within 2 s), 2 s after the wave's last table (`after`), every 2 s while the pool's largest free block is under 768 MiB, else every
 * 20 s (six rows). Returns only after eden_big_stacks_stop (host checks). */
static atomic_uint eden_big_stop;
/* One line per thread name and result (review M1): whether that thread's deep reads are allowed.
 * The first EDEN_BIG_SAID_NAMES name+result pairs; the rest only counted on the last line. */
static void eden_big_say_bounds(struct eden_big_tables *tables) {
  static unsigned seen[EDEN_HEAP_THREADS];
  static char said[EDEN_BIG_SAID_NAMES][16];
  static unsigned char said_ok[EDEN_BIG_SAID_NAMES];
  static unsigned said_n, unsaid;
  static const char *const whys[4] = {"ok", "no_bounds", "frame_outside", "kernel_check"};
  for (unsigned r = 0; r < EDEN_HEAP_THREADS; ++r) {
    const struct eden_big_thread *t = &tables->threads[r];
    const unsigned noted = atomic_load_explicit(&t->noted, memory_order_acquire);
    if (noted == seen[r])
      continue;
    seen[r] = noted;
    char name[16], clean[16];
    for (unsigned k = 0; k < 16; ++k)
      name[k] = __atomic_load_n(&eden_heap_threads[r].name[k], __ATOMIC_RELAXED);
    name[15] = 0;
    const unsigned ok = atomic_load_explicit(&t->bounds_ok, memory_order_relaxed) != 0;
    unsigned i = 0;
    while (i < said_n && !(said_ok[i] == ok && memcmp(said[i], name, 16) == 0))
      ++i;
    if (i < said_n)
      continue;
    if (said_n == EDEN_BIG_SAID_NAMES) {
      ++unsaid;
      continue;
    }
    memcpy(said[said_n], name, 16);
    said_ok[said_n++] = (unsigned char)ok;
    const unsigned why = atomic_load_explicit(&t->bounds_why, memory_order_relaxed);
    fprintf(stderr, "EDEN_BIG_STACKS bounds thread=%s kib=%u ok=%u why=%s record=%u names=%u unsaid=%u\n",
            eden_big_name(name, clean), atomic_load_explicit(&t->bounds_kib, memory_order_relaxed), ok,
            whys[why & 3u], r, said_n, unsaid);
  }
}
void eden_big_stacks_stop(void) { /* the host checks' end; nothing on the console calls it */
  atomic_store_explicit(&eden_big_stop, 1, memory_order_relaxed);
}
void eden_big_stacks_run(void) {
  int said = 0;
  size_t history[8] = {0};
  long long live_history[8] = {0};
  unsigned ticks = 0, seq = 0;
  unsigned long long last_table_ms = 0, last_wave_ms = 0, wave_after_ms = 0, mark_ms = 0;
  long long live_at_table = 0, best = 0;
  size_t heap_at_table = 0;
  while (!atomic_load_explicit(&eden_big_stop, memory_order_relaxed)) {
    const struct timespec quarter = {0, 250000000L};
    nanosleep(&quarter, NULL);
    const unsigned long long now_ms = eden_big_steady_ms();
    atomic_store_explicit(&eden_big_now_ms, now_ms, memory_order_relaxed);
    atomic_store_explicit(&eden_big_now_s, (unsigned)(now_ms / 1000u), memory_order_relaxed);
    struct eden_big_tables *tables = eden_big_tables_now();
    if (tables == NULL || !atomic_load_explicit(&eden_big_on, memory_order_acquire))
      continue;
    if (!said) {
      /* Its own stack, as the hooks see a named thread's: about 256 KiB when the bounds work here. */
      said = 1;
      last_table_ms = now_ms; /* the first steady table 20 s from here */
      struct eden_heap_thread *self = eden_heap_self();
      const struct eden_big_thread *t = self != NULL ? &tables->threads[self - eden_heap_threads] : NULL;
      const uintptr_t low = t ? atomic_load_explicit(&t->low, memory_order_relaxed) : 0,
                      high = t ? atomic_load_explicit(&t->high, memory_order_relaxed) : 0;
      const int bounds = t != NULL && atomic_load_explicit(&t->owner, memory_order_acquire) == (uintptr_t)pthread_self();
      fprintf(stderr, "EDEN_BIG_STACKS reporter stack=0x%lx-0x%lx kib=%lu bounds=%d\n", (unsigned long)low,
              (unsigned long)high, (unsigned long)((high - low) >> 10), bounds);
    }
    eden_big_say_bounds(tables);
    const size_t heap = atomic_load_explicit(&eden_heap_committed_bytes, memory_order_relaxed);
    const long long class_live = atomic_load_explicit(&eden_big_class_live, memory_order_relaxed);
    history[ticks % 8u] = heap;
    live_history[ticks % 8u] = class_live;
    ++ticks;
    size_t lowest = heap;
    long long live_lowest = class_live;
    for (unsigned i = 0; i < 8u; ++i) {
      if (i < ticks && history[i] < lowest)
        lowest = history[i];
      if (live_history[i] < live_lowest) /* 0 before the first ticks: the class starts empty */
        live_lowest = live_history[i];
    }
    const size_t heap_2s = heap - lowest;
    if (now_ms - mark_ms >= 2000u) {
      mark_ms = now_ms;
      const unsigned next = atomic_load_explicit(&eden_big_mark_at, memory_order_relaxed) ^ 1u;
      for (unsigned r = 0; r < EDEN_HEAP_THREADS; ++r)
        atomic_store_explicit(&tables->threads[r].grown_mark[next], __atomic_load_n(&eden_heap_threads[r].grown, __ATOMIC_RELAXED),
                              memory_order_relaxed);
      atomic_store_explicit(&eden_big_mark_ms[next], now_ms, memory_order_relaxed);
      atomic_store_explicit(&eden_big_mark_at, next, memory_order_release);
    }
    if (class_live > best) {
      best = class_live;
      for (unsigned r = 0; r < EDEN_BIG_ROWS; ++r)
        tables->rows[r].at_peak = atomic_load_explicit(&tables->rows[r].live, memory_order_relaxed);
    }
    const unsigned long long pool = eden_free_memory_bytes ? eden_free_memory_bytes() : 0;
    /* A wave: the class's live total +128 MiB, or the heap's pieces +256 MiB, within 2 s. A wave
     * table at most every 500 ms, and again within 2 s of the last only while it still rises. */
    const int wave = class_live - live_lowest >= (128ll << 20) || heap_2s >= ((size_t)256 << 20);
    const int rising = class_live - live_at_table >= (64ll << 20) || heap >= heap_at_table + ((size_t)64 << 20);
    const char *why = NULL;
    int steady = 0;
    if (wave && now_ms - last_table_ms >= 500u && (now_ms - last_wave_ms >= 2000u || rising)) {
      why = "wave";
      last_wave_ms = now_ms;
      wave_after_ms = now_ms + 2000u;
    } else if (wave_after_ms != 0 && now_ms >= wave_after_ms) {
      why = "after";
      wave_after_ms = 0;
    } else if (pool != 0 && pool < ((unsigned long long)768 << 20) && now_ms - last_table_ms >= 2000u) {
      why = "short";
    } else if (now_ms - last_table_ms >= 20000u) {
      why = "steady";
      steady = 1;
    }
    if (why == NULL)
      continue;
    eden_big_table(why, ++seq, now_ms, heap, heap_2s, pool, steady);
    last_table_ms = now_ms;
    live_at_table = best = atomic_load_explicit(&eden_big_class_live, memory_order_relaxed);
    heap_at_table = atomic_load_explicit(&eden_heap_committed_bytes, memory_order_relaxed);
  }
}

/* ---- EDEN_UP_BIG_STACKS: read by the crash handler (headless/crash_report.cpp) ----
 * Signal-safe: fixed loops over the mapped tables, relaxed loads, no lock, no allocation, no stdio.
 * A row being written meanwhile can come out half old, half new (counts off by one call). */
struct eden_big_crash_head {
  unsigned long long on, class_live, window_peak, window_peak_ms, now_ms, untracked, rows_full, reentry, refusals,
      rows_used;
};
struct eden_big_crash_row {
  unsigned long long row, live, blocks, peak, calls, bytes, grew_mib, refused, refused_last, depth, build, young;
  char thread[16];
  unsigned long long pcs[EDEN_BIG_PCS]; /* offsets from code start, 0: none */
};
/* Up to `most` rows (at most 16): 6 by young live bytes (blocks under 10 s old: the wave that is
 * killing the run, review B), the largest by live now, and up to 4 rows with refusals; a row once.
 * Returns how many; 0 and head->on 0 when off, or when the caller's layout differs (`row_size`). */
static void eden_big_crash_insert(unsigned *rows, long long *keys, unsigned *n, unsigned room, unsigned r, long long key) {
  if (room == 0 || key <= 0)
    return;
  unsigned place = *n;
  if (*n < room)
    ++*n;
  else if (key <= keys[*n - 1])
    return;
  else
    place = *n - 1;
  while (place > 0 && keys[place - 1] < key) {
    keys[place] = keys[place - 1];
    rows[place] = rows[place - 1];
    --place;
  }
  keys[place] = key;
  rows[place] = r;
}
unsigned eden_big_crash_rows(struct eden_big_crash_head *head, struct eden_big_crash_row *out, unsigned most,
                             unsigned long long row_size) {
  struct eden_big_tables *tables = eden_big_tables_now();
  head->on = tables != NULL && atomic_load_explicit(&eden_big_on, memory_order_relaxed);
  if (!head->on || row_size != sizeof(struct eden_big_crash_row) || most == 0)
    return 0;
  const long long class_live = atomic_load_explicit(&eden_big_class_live, memory_order_relaxed);
  head->class_live = class_live > 0 ? (unsigned long long)class_live : 0;
  const long long peak = atomic_load_explicit(&eden_big_window_peak, memory_order_relaxed);
  head->window_peak = peak > 0 ? (unsigned long long)peak : 0;
  head->window_peak_ms = atomic_load_explicit(&eden_big_window_peak_ms, memory_order_relaxed);
  head->now_ms = atomic_load_explicit(&eden_big_now_ms, memory_order_relaxed);
  head->untracked = atomic_load_explicit(&eden_big_untracked, memory_order_relaxed);
  head->rows_full = atomic_load_explicit(&eden_big_rows_full, memory_order_relaxed);
  head->reentry = atomic_load_explicit(&eden_big_reentry, memory_order_relaxed);
  head->refusals = atomic_load_explicit(&eden_big_refusals, memory_order_relaxed);
  head->rows_used = atomic_load_explicit(&eden_big_rows_used, memory_order_relaxed);
  enum { kMost = 16 };
  if (most > kMost)
    most = kMost;
  const unsigned refused_room = most > 4 ? 4 : 0;
  const unsigned young_room = most - refused_room > 6 ? 6 : (most - refused_room) / 2;
  const unsigned live_room = most - refused_room - young_room;
  /* Young bytes per row: a fixed pass over the blocks (relaxed loads, the crash handler's own array). */
  long long *young = tables->crash_young;
  const unsigned now_s = atomic_load_explicit(&eden_big_now_s, memory_order_relaxed);
  for (unsigned r = 0; r < EDEN_BIG_ROWS; ++r)
    young[r] = 0;
  for (unsigned b = 0; b < EDEN_BIG_BLOCKS; ++b) {
    const struct eden_big_block *block = &tables->blocks[b];
    if (atomic_load_explicit(&block->address, memory_order_relaxed) <= 1)
      continue;
    if (atomic_load_explicit(&block->born_s, memory_order_relaxed) + 10u > now_s)
      young[atomic_load_explicit(&block->row, memory_order_relaxed) % EDEN_BIG_ROWS] +=
          (long long)atomic_load_explicit(&block->size, memory_order_relaxed);
  }
  unsigned young_rows[kMost], live_rows[kMost], refused_rows[4], young_n = 0, live_n = 0, refused_n = 0;
  long long young_key[kMost], live_key[kMost], refused_key[4];
  for (unsigned r = 0; r < EDEN_BIG_ROWS; ++r) {
    const struct eden_big_row *row = &tables->rows[r];
    if (!atomic_load_explicit(&row->ready, memory_order_acquire))
      continue;
    const unsigned long long refused = atomic_load_explicit(&row->refused, memory_order_relaxed);
    if (refused != 0 && refused_room != 0) { /* a refused row is shown with the refused ones */
      eden_big_crash_insert(refused_rows, refused_key, &refused_n, refused_room, r, (long long)refused);
      continue;
    }
    eden_big_crash_insert(young_rows, young_key, &young_n, young_room, r, young[r]);
  }
  for (unsigned r = 0; r < EDEN_BIG_ROWS; ++r) {
    const struct eden_big_row *row = &tables->rows[r];
    if (!atomic_load_explicit(&row->ready, memory_order_acquire))
      continue;
    if (atomic_load_explicit(&row->refused, memory_order_relaxed) != 0 && refused_room != 0)
      continue;
    int taken = 0;
    for (unsigned i = 0; i < young_n && !taken; ++i)
      taken = young_rows[i] == r;
    if (!taken)
      eden_big_crash_insert(live_rows, live_key, &live_n, live_room, r,
                            atomic_load_explicit(&row->live, memory_order_relaxed));
  }
  unsigned order[kMost + kMost + 4], order_n = 0;
  for (unsigned i = 0; i < young_n; ++i)
    order[order_n++] = young_rows[i];
  for (unsigned i = 0; i < live_n; ++i)
    order[order_n++] = live_rows[i];
  for (unsigned i = 0; i < refused_n; ++i)
    order[order_n++] = refused_rows[i];
  const uintptr_t code = (uintptr_t)__eden_text_start;
  unsigned n = 0;
  for (unsigned i = 0; i < order_n && n < most; ++i) {
    const unsigned r = order[i];
    const struct eden_big_row *row = &tables->rows[r];
    struct eden_big_crash_row *o = &out[n++];
    const long long live = atomic_load_explicit(&row->live, memory_order_relaxed);
    const long long blocks = atomic_load_explicit(&row->blocks, memory_order_relaxed);
    const long long high = atomic_load_explicit(&row->peak, memory_order_relaxed);
    o->row = r;
    o->live = live > 0 ? (unsigned long long)live : 0;
    o->blocks = blocks > 0 ? (unsigned long long)blocks : 0;
    o->peak = high > 0 ? (unsigned long long)high : 0;
    o->calls = atomic_load_explicit(&row->calls, memory_order_relaxed);
    o->bytes = atomic_load_explicit(&row->bytes, memory_order_relaxed);
    o->grew_mib = atomic_load_explicit(&row->grew, memory_order_relaxed) << (EDEN_HEAP_PIECE_SHIFT - 20);
    o->refused = atomic_load_explicit(&row->refused, memory_order_relaxed);
    o->refused_last = atomic_load_explicit(&row->refused_last, memory_order_relaxed);
    o->depth = atomic_load_explicit(&row->depth, memory_order_relaxed);
    o->build = row->build;
    o->young = young[r] > 0 ? (unsigned long long)young[r] : 0;
    for (unsigned k = 0; k < 16; ++k)
      o->thread[k] = row->thread[k];
    o->thread[15] = 0;
    for (unsigned k = 0; k < EDEN_BIG_PCS; ++k) {
      const uintptr_t pc = atomic_load_explicit(&row->pcs[k], memory_order_relaxed);
      o->pcs[k] = pc >= code ? pc - code : 0;
    }
  }
  return n;
}

/* ---- EDEN_UP_BIG_STACKS: operator new's EDEN_NEW_WAIT start / EDEN_HEAP_FAILURE lines (main.cpp) ----
 * On the refused thread, after its request failed (not inside the allocator): " big_row=<row of this
 * thread's last refused request when it is `requested` (same size, 256 KiB or more), or none> big_grew_ms=<since the older mark> big_grew=<the 4 records
 * that grew the heap most since then, name:MiB>". "" when off. No allocation, no stdio. */
static unsigned eden_big_put(char *out, unsigned size, unsigned at, const char *text) {
  while (*text && at + 1 < size)
    out[at++] = *text++;
  out[at < size ? at : size - 1] = 0;
  return at;
}
static unsigned eden_big_put_dec(char *out, unsigned size, unsigned at, unsigned long long value) {
  char digits[24];
  unsigned n = 0;
  do {
    digits[n++] = (char)('0' + value % 10u);
    value /= 10u;
  } while (value != 0 && n < sizeof(digits));
  char text[24];
  for (unsigned i = 0; i < n; ++i)
    text[i] = digits[n - 1 - i];
  text[n] = 0;
  return eden_big_put(out, size, at, text);
}
unsigned eden_big_failure_text(char *out, unsigned size, unsigned long long requested) {
  if (size == 0)
    return 0;
  out[0] = 0;
  struct eden_big_tables *tables = eden_big_tables_now();
  if (tables == NULL || !atomic_load_explicit(&eden_big_on, memory_order_relaxed) || size < 2)
    return 0;
  struct eden_heap_thread *self = eden_heap_self();
  unsigned at = eden_big_put(out, size, 0, " big_row=");
  /* Only when this thread's last refused request is this one (review S2): same size, 256 KiB or more. */
  const struct eden_big_thread *mine = self != NULL ? &tables->threads[self - eden_heap_threads] : NULL;
  const unsigned row = mine != NULL && requested >= EDEN_HEAP_SMALL_REQUEST &&
                               atomic_load_explicit(&mine->refused_bytes, memory_order_relaxed) == requested
                           ? atomic_load_explicit(&mine->refused_row, memory_order_relaxed)
                           : 0;
  at = row != 0 ? eden_big_put_dec(out, size, at, row - 1) : eden_big_put(out, size, at, "none");
  const unsigned older = atomic_load_explicit(&eden_big_mark_at, memory_order_acquire) ^ 1u;
  const unsigned long long since = atomic_load_explicit(&eden_big_mark_ms[older], memory_order_relaxed);
  const unsigned long long now_ms = atomic_load_explicit(&eden_big_now_ms, memory_order_relaxed);
  at = eden_big_put(out, size, at, " big_grew_ms=");
  at = eden_big_put_dec(out, size, at, since != 0 && now_ms >= since ? now_ms - since : 0);
  at = eden_big_put(out, size, at, " big_grew=");
  unsigned top[4], grew[4], count = 0;
  for (unsigned r = 0; r < EDEN_HEAP_THREADS; ++r) {
    const unsigned now = __atomic_load_n(&eden_heap_threads[r].grown, __ATOMIC_RELAXED);
    const unsigned then = atomic_load_explicit(&tables->threads[r].grown_mark[older], memory_order_relaxed);
    const unsigned delta = since != 0 && now > then ? now - then : 0;
    if (delta == 0)
      continue;
    unsigned place = count;
    if (count < 4)
      ++count;
    else if (delta <= grew[3])
      continue;
    else
      place = 3;
    while (place > 0 && grew[place - 1] < delta) {
      grew[place] = grew[place - 1];
      top[place] = top[place - 1];
      --place;
    }
    grew[place] = delta;
    top[place] = r;
  }
  if (count == 0)
    at = eden_big_put(out, size, at, "none");
  for (unsigned i = 0; i < count; ++i) {
    char name[16], clean[16];
    for (unsigned k = 0; k < 16; ++k)
      name[k] = __atomic_load_n(&eden_heap_threads[top[i]].name[k], __ATOMIC_RELAXED);
    name[15] = 0;
    if (i)
      at = eden_big_put(out, size, at, ",");
    at = eden_big_put(out, size, at, eden_big_name(name, clean));
    at = eden_big_put(out, size, at, ":");
    at = eden_big_put_dec(out, size, at, (unsigned long long)grew[i] << (EDEN_HEAP_PIECE_SHIFT - 20));
  }
  return at;
}

/* ---- Who holds what the heap does not use (reports, in stderr.log) ---- */

/* The name a thread gave itself, kept with its record (the SetCurrentThreadName hook). */
void eden_heap_name_thread(const char *name) {
  if (atomic_load_explicit(&ps5_heap_state, memory_order_acquire) != 2)
    return;
  struct eden_heap_thread *self = eden_heap_self();
  if (self == NULL)
    return;
  unsigned length = 0;
  for (; length + 1 < sizeof(self->name) && name[length] != 0; ++length)
    __atomic_store_n(&self->name[length], name[length], __ATOMIC_RELAXED);
  __atomic_store_n(&self->name[length], 0, __ATOMIC_RELAXED);
  eden_big_note_stack(self); /* EDEN_UP_BIG_STACKS: nothing when off */
}

/* Every 20 s, and cheap: which requests made the heap grow (MiB, and the largest request), what the
 * arenas are, and how often a thread found room in another's. */
void eden_heap_growth_report(void) {
  static const char *const kinds[EDEN_HEAP_GREW_KINDS] = {"arena", "under256K", "under1M", "under8M", "under32M", "large"};
  char line[320];
  int length = snprintf(line, sizeof(line), "EDEN_HEAP_GROWTH arenas=%u spare=%u reused=%llu given_back=%llu adopted=%llu adopt_misses=%llu spills=%llu grew:",
                        eden_heap_arenas_created(), eden_heap_arenas_spare(),
                        (unsigned long long)atomic_load_explicit(&eden_heap_arenas_taken_again, memory_order_relaxed),
                        (unsigned long long)atomic_load_explicit(&eden_heap_arenas_returned, memory_order_relaxed),
                        (unsigned long long)atomic_load_explicit(&eden_heap_adopted, memory_order_relaxed),
                        (unsigned long long)atomic_load_explicit(&eden_heap_adopt_misses, memory_order_relaxed),
                        (unsigned long long)atomic_load_explicit(&eden_heap_spills, memory_order_relaxed));
  for (unsigned kind = 0; kind < EDEN_HEAP_GREW_KINDS && length > 0 && length < (int)sizeof(line); ++kind)
    length += snprintf(line + length, sizeof(line) - (size_t)length, " %s=%zu/%zu", kinds[kind],
                       (size_t)atomic_load_explicit(&eden_heap_grew_pieces[kind], memory_order_relaxed)
                           << (EDEN_HEAP_PIECE_SHIFT - 20),
                       atomic_load_explicit(&eden_heap_grew_largest[kind], memory_order_relaxed) >> 10);
  fprintf(stderr, "%s (MiB/largest KiB)\n", line);
  /* EDEN_UP_HEAP_ADOPT_WIDE (0880): the looks that went past the first 16 arenas, their asks, the
   * arenas taken there (each a new arena not made: 2 MiB of the heap), and their time. On only. */
  const unsigned wide = atomic_load_explicit(&eden_heap_adopt_wide, memory_order_relaxed);
  if (wide != 0) {
    const unsigned long long wide_us = atomic_load_explicit(&eden_heap_adopt_wide_us, memory_order_relaxed);
    fprintf(stderr, "EDEN_HEAP_ADOPT_WIDE n=%u looks=%llu asks=%llu taken=%llu (%llu MiB of arenas not made) ms=%llu.%03llu max_us=%llu\n",
            wide, (unsigned long long)atomic_load_explicit(&eden_heap_adopt_wide_scans, memory_order_relaxed),
            (unsigned long long)atomic_load_explicit(&eden_heap_adopt_wide_asks, memory_order_relaxed),
            (unsigned long long)atomic_load_explicit(&eden_heap_adopt_wide_hits, memory_order_relaxed),
            (unsigned long long)atomic_load_explicit(&eden_heap_adopt_wide_hits, memory_order_relaxed)
                << (EDEN_HEAP_ARENA_SHIFT - 20),
            wide_us / 1000u, wide_us % 1000u,
            (unsigned long long)atomic_load_explicit(&eden_heap_adopt_wide_max_us, memory_order_relaxed));
  }
  /* Blocks under 32 MiB with memory of their own (EDEN_HEAP_LARGE_MIB). Cycles are the TSC's. */
  fprintf(stderr, "EDEN_HEAP_MID from=%zuKiB taken=%llu taken_mib=%llu refused=%llu alive=%u alive_mib=%zu peak_mib=%zu "
                  "cycles_m=%llu large_entries=%u of %u table_full=%llu\n",
          atomic_load_explicit(&eden_heap_large_from, memory_order_relaxed) >> 10,
          (unsigned long long)atomic_load_explicit(&eden_heap_mid_taken, memory_order_relaxed),
          (unsigned long long)atomic_load_explicit(&eden_heap_mid_taken_bytes, memory_order_relaxed) >> 20,
          (unsigned long long)atomic_load_explicit(&eden_heap_mid_refused, memory_order_relaxed),
          atomic_load_explicit(&eden_heap_mid_count, memory_order_relaxed),
          atomic_load_explicit(&eden_heap_mid_bytes, memory_order_relaxed) >> 20,
          atomic_load_explicit(&eden_heap_mid_peak, memory_order_relaxed) >> 20,
          (unsigned long long)atomic_load_explicit(&eden_heap_mid_cycles, memory_order_relaxed) / 1000000u,
          atomic_load_explicit(&eden_heap_large_count, memory_order_relaxed), EDEN_HEAP_LARGE_BLOCKS,
          (unsigned long long)atomic_load_explicit(&eden_heap_large_full, memory_order_relaxed));
  fprintf(stderr, "EDEN_HEAP_CLASSES requests (count/MiB): under1M=%llu/%llu under2M=%llu/%llu under8M=%llu/%llu under32M=%llu/%llu\n",
          (unsigned long long)atomic_load_explicit(&eden_heap_class_count[0], memory_order_relaxed),
          (unsigned long long)atomic_load_explicit(&eden_heap_class_bytes[0], memory_order_relaxed) >> 20,
          (unsigned long long)atomic_load_explicit(&eden_heap_class_count[1], memory_order_relaxed),
          (unsigned long long)atomic_load_explicit(&eden_heap_class_bytes[1], memory_order_relaxed) >> 20,
          (unsigned long long)atomic_load_explicit(&eden_heap_class_count[2], memory_order_relaxed),
          (unsigned long long)atomic_load_explicit(&eden_heap_class_bytes[2], memory_order_relaxed) >> 20,
          (unsigned long long)atomic_load_explicit(&eden_heap_class_count[3], memory_order_relaxed),
          (unsigned long long)atomic_load_explicit(&eden_heap_class_bytes[3], memory_order_relaxed) >> 20);
  /* RADV_PS5_LONGLIVED_ARENA (0 or 1): the heap's slack and the long-lived set, every 20 s. */
  if (atomic_load_explicit(&eden_heap_ll_mode, memory_order_relaxed) >= 0) {
    char now[320];
    eden_heap_ll_snapshot(now, sizeof(now));
    fprintf(stderr, "EDEN_HEAP_LONG_LIVED now: %s\n", now);
  }
}

/* RADV_PS5_LONGLIVED_ARENA (0 or 1; unset: nothing): the heap before and after a trim, for the boot's
 * end (main.cpp, after the boot pipelines are built) and any other moment a caller names. Two lines;
 * the trim is the one every refusal and session end already makes (eden_heap_trim). */
void eden_heap_long_lived_report(const char *why) {
  if (atomic_load_explicit(&eden_heap_ll_mode, memory_order_relaxed) < 0)
    return;
  char before[320], after[320];
  eden_heap_ll_snapshot(before, sizeof(before));
  const size_t gave = eden_heap_trim();
  eden_heap_ll_snapshot(after, sizeof(after));
  fprintf(stderr, "EDEN_HEAP_LONG_LIVED %s mode=%d before trim: %s\n", why ? why : "-",
          atomic_load_explicit(&eden_heap_ll_mode, memory_order_relaxed), before);
  fprintf(stderr, "EDEN_HEAP_LONG_LIVED %s mode=%d after trim (gave back %zu MiB): %s\n", why ? why : "-",
          atomic_load_explicit(&eden_heap_ll_mode, memory_order_relaxed), gave >> 20, after);
}

/* The room left in an mspace, found by asking it: blocks of `top` while it gives them, then of half
 * that, down to `floor`, all freed again. It takes nothing of the mspace's own records, so it works
 * on any of them; a thread that asks in these microseconds may find it fuller than it is. */
#define EDEN_HEAP_ROOM_BLOCKS 96u
static size_t eden_heap_room(void *space, size_t top, size_t floor, size_t *largest) {
  void *blocks[EDEN_HEAP_ROOM_BLOCKS];
  unsigned count = 0;
  size_t room = 0;
  *largest = 0;
  for (size_t size = top; size >= floor && count < EDEN_HEAP_ROOM_BLOCKS;) {
    void *block = sceLibcMspaceMalloc(space, size);
    if (block == NULL) {
      size >>= 1;
      continue;
    }
    blocks[count++] = block;
    room += size;
    if (*largest == 0)
      *largest = size;
  }
  while (count != 0)
    sceLibcMspaceFree(space, blocks[--count]);
  return room;
}

/* At a refusal, with the time to look: for each thread record with four arenas or 4 MiB of room,
 * what they hold and one digit an arena for its room (0 full, f empty); for each space, its blocks,
 * the arenas among them, and its room. A space whose blocks are all arenas is pinned by them alone.
 * The rest of what the heap holds and does not use is room in pieces under the sizes asked for
 * here, and the mspaces' headers. */
void eden_heap_arena_report(void) {
  if (atomic_load_explicit(&ps5_heap_state, memory_order_acquire) != 2)
    return;
  const size_t arena_floor = (size_t)16 << 10, space_floor = (size_t)256 << 10;
  size_t arena_room = 0, cached = 0, largest = 0;
  unsigned empty = 0;
  for (unsigned record = 0; record < EDEN_HEAP_THREADS; ++record) {
    struct eden_heap_thread *thread = &eden_heap_threads[record];
    unsigned owned = __atomic_load_n(&thread->owned, __ATOMIC_ACQUIRE);
    size_t kept = 0;
    for (unsigned bin = 1; bin < EDEN_TCACHE_BINS; ++bin)
      kept += (size_t)__atomic_load_n(&thread->counts[bin], __ATOMIC_RELAXED) * bin * 16u;
    cached += kept;
    if (owned == 0)
      continue;
    char map[EDEN_HEAP_THREAD_ARENAS + 1];
    size_t room = 0;
    unsigned record_empty = 0;
    /* The record's arenas are read, and asked, under the creation lock: a trim on another thread
     * gives empty ones back under it. No stdio here: it allocates, and may want the lock. */
    eden_heap_create_acquire();
    owned = __atomic_load_n(&thread->owned, __ATOMIC_ACQUIRE);
    for (unsigned i = 0; i < owned; ++i) {
      void *arena = __atomic_load_n(&thread->own[i], __ATOMIC_RELAXED);
      const size_t left = eden_heap_room(arena, (size_t)1 << 20, arena_floor, &largest);
      room += left;
      record_empty += left + ((size_t)64 << 10) >= EDEN_HEAP_ARENA_PAYLOAD;
      map[i] = "0123456789abcdef"[left >= EDEN_HEAP_ARENA_PAYLOAD ? 15 : left * 16 / EDEN_HEAP_ARENA_PAYLOAD];
    }
    eden_heap_create_release();
    map[owned] = 0;
    arena_room += room;
    empty += record_empty;
    if (owned < 4 && room < ((size_t)4 << 20))
      continue;
    char name[sizeof(thread->name)];
    for (unsigned i = 0; i < sizeof(name); ++i)
      name[i] = __atomic_load_n(&thread->name[i], __ATOMIC_RELAXED);
    name[sizeof(name) - 1] = 0;
    fprintf(stderr, "EDEN_HEAP_THREAD %u %s%s arenas=%u room=%zuMiB empty=%u tcache=%zuKiB grew=%uMiB map=%s\n", record,
            name[0] ? name : "?",
            atomic_load_explicit(&eden_heap_thread_used[record], memory_order_relaxed) ? "" : " (ended)", owned,
            room >> 20, record_empty, kept >> 10,
            __atomic_load_n(&thread->grown, __ATOMIC_RELAXED) << (EDEN_HEAP_PIECE_SHIFT - 20), map);
  }
  /* The spares: empty arenas no record holds, 2 MiB each, for the next threads. */
  {
    const unsigned spares = eden_heap_arenas_spare();
    arena_room += (size_t)spares * EDEN_HEAP_ARENA_PAYLOAD;
    empty += spares;
    if (spares != 0)
      fprintf(stderr, "EDEN_HEAP_SPARE arenas=%u (%zuMiB), %llu taken so far\n", spares,
              ((size_t)spares * EDEN_HEAP_ARENA_PAYLOAD) >> 20, eden_heap_arenas_reused());
  }
  /* The arenas in each space. */
  unsigned short arenas_in[EDEN_HEAP_PIECES];
  memset(arenas_in, 0, sizeof(arenas_in));
  const unsigned arenas = atomic_load_explicit(&eden_heap_arena_count, memory_order_acquire);
  for (unsigned i = 0; i < arenas; ++i) {
    const uintptr_t block = (uintptr_t)atomic_load_explicit(&eden_heap_arena_blocks_probe[i], memory_order_relaxed);
    if (block == 0)
      continue; /* given back */
    const unsigned slot = atomic_load_explicit(
        &eden_heap_piece_slot[(block - (uintptr_t)ps5_heap_base) >> EDEN_HEAP_PIECE_SHIFT], memory_order_relaxed);
    if (slot < EDEN_HEAP_PIECES)
      arenas_in[slot]++;
  }
  size_t space_room = 0;
  unsigned only_arenas = 0, on_line = 0;
  char line[512];
  int length = 0;
  const unsigned spaces = atomic_load_explicit(&eden_heap_space_count, memory_order_acquire);
  for (unsigned slot = 0; slot < spaces; ++slot) {
    /* As eden_heap_space_use: a space with a thread in it is not given back. */
    atomic_fetch_add_explicit(&eden_heap_space_users[slot], 1, memory_order_seq_cst);
    void *space = atomic_load_explicit(&eden_heap_space_retired[slot], memory_order_seq_cst) == 0
                      ? atomic_load_explicit(&eden_heap_spaces[slot], memory_order_acquire)
                      : NULL;
    if (space != NULL) {
      const long blocks = atomic_load_explicit(&eden_heap_space_blocks[slot], memory_order_relaxed);
      const size_t room = eden_heap_room(space, EDEN_HEAP_PIECE_SIZE, space_floor, &largest);
      space_room += room;
      only_arenas += slot != 0 && blocks == arenas_in[slot];
      if (length == 0)
        length = snprintf(line, sizeof(line), "EDEN_HEAP_SPACES slot:blocks:arenas:largestKiB:roomMiB");
      length += snprintf(line + length, sizeof(line) - (size_t)length, " %u:%ld:%u:%zu:%zu", slot, blocks,
                         (unsigned)arenas_in[slot], largest >> 10, room >> 20);
      if (++on_line == 12) {
        fprintf(stderr, "%s\n", line);
        on_line = 0;
        length = 0;
      }
    }
    atomic_fetch_sub_explicit(&eden_heap_space_users[slot], 1, memory_order_release);
  }
  if (length != 0)
    fprintf(stderr, "%s\n", line);
  const size_t pieces = atomic_load_explicit(&eden_heap_committed_bytes, memory_order_relaxed);
  const size_t large = atomic_load_explicit(&eden_heap_large_bytes, memory_order_relaxed);
  const size_t live = atomic_load_explicit(&ps5_heap_live_bytes, memory_order_relaxed);
  const size_t slack = pieces + large > live ? pieces + large - live : 0;
  const size_t named = arena_room + space_room + cached;
  fprintf(stderr, "EDEN_HEAP_SLACK %zu MiB not in use: %zu MiB room in %u arenas (%u empty), %zu MiB room in spaces "
                  "(%u held by arenas alone), %zu MiB in thread caches, %zu MiB in smaller holes and headers\n",
          slack >> 20, arena_room >> 20, eden_heap_arenas_created(), empty, space_room >> 20, only_arenas, cached >> 20,
          (slack > named ? slack - named : 0) >> 20);
}

/* EDEN_UP_HEAP_RESCUE=1 (off unless set; read from main in eden_heap_pool_init): a request the
 * heap is about to refuse (no room in the thread's arenas or the heap's spaces, and the pool has
 * no memory to grow with) is asked of every arena, whoever's it is, before it fails. Arenas take
 * requests under 256 KiB only and a thread uses its own, so at a refusal the heap can hold
 * hundreds of MiB it cannot give: at a crash in free play on 10 Nov 2026 it held 1760 MiB with 866 MiB in
 * use, 372 MiB of that in whole free 2 MiB pages, while 968 requests of 280 to 750 KiB failed and
 * the driver's shader compiler, which does not check its allocations, ended the session. An arena
 * serves any thread (mspaces take their own locks, and a block finds its arena by its address), so
 * the block is freed like any other. Up to 4 MiB: written when an arena was 8 MiB; arenas are 2 MiB
 * now (EDEN_HEAP_ARENA_SHIFT), so a request over about 2 MiB is asked of every arena and none can
 * take it. Under the creation lock, as every walk of the arena list. It skips spares, and asks
 * every arena with an mspace call each: EDEN_UP_HEAP_STEAL (below) asks the empty ones first, the
 * spares among them, with a bounded number of calls. */
#define EDEN_HEAP_RESCUE_MAX ((size_t)4 << 20)
static atomic_int eden_heap_rescue_on;
static atomic_uint eden_heap_rescue_next;
static atomic_ullong eden_heap_rescued, eden_heap_rescue_missed;

void eden_heap_set_rescue(int on) {
  atomic_store_explicit(&eden_heap_rescue_on, on, memory_order_relaxed);
}

void eden_heap_rescue_stats(unsigned long long *rescued, unsigned long long *missed) {
  *rescued = atomic_load_explicit(&eden_heap_rescued, memory_order_relaxed);
  *missed = atomic_load_explicit(&eden_heap_rescue_missed, memory_order_relaxed);
}

static void *eden_heap_arena_rescue(size_t count, size_t size, int zero) {
  if (!atomic_load_explicit(&eden_heap_rescue_on, memory_order_relaxed) || count * size > EDEN_HEAP_RESCUE_MAX)
    return NULL;
  /* Not for a request made under the creation lock (a report that prints while it walks the
   * arenas): the lock is not recursive. */
  if (atomic_load_explicit(&eden_heap_create_holder, memory_order_relaxed) == (unsigned long)pthread_self())
    return NULL;
  void *address = NULL;
  eden_heap_create_acquire();
  const unsigned arenas = atomic_load_explicit(&eden_heap_arena_count, memory_order_acquire);
  const unsigned start = atomic_fetch_add_explicit(&eden_heap_rescue_next, 1, memory_order_relaxed);
  for (unsigned i = 0; i < arenas && address == NULL; ++i) {
    void *arena = eden_heap_arena_list[(start + i) % arenas];
    if (arena == NULL || eden_heap_is_spare(arena))
      continue; /* a spare goes whole to the next thread that needs an arena */
    address = eden_heap_arena_try(arena, count, size, zero);
  }
  eden_heap_create_release();
  atomic_fetch_add_explicit(address != NULL ? &eden_heap_rescued : &eden_heap_rescue_missed, 1, memory_order_relaxed);
  return address;
}

/* EDEN_UP_HEAP_STEAL=<mode> (off unless set; read from main in eden_heap_pool_init,
 * headless/heap_pages.cpp). Two sessions of 11 Nov 2026 ended with the pool empty while the heap
 * held 300 to 460 MiB of whole free 2 MiB pages: ours (T121, a 452 KiB request on am:1, std::bad_alloc)
 * and a tester's (T120, 240 refused 160 KiB requests on the GPU thread). What was left of that room
 * at the end was in arenas no thread was on, many of them empty (the tester's: 116 empty, 99 of them
 * the pipeline builders'), and no request of those sizes ever asks such an arena: one of 64 KiB or
 * more asks no arena but its own thread's (eden_heap_arena_allocate), one of 256 KiB or more none at
 * all (eden_heap_allocate). An arena's room cannot go back to the pool (an arena is never freed: the
 * top of this file), but any thread can take a block from it: mspaces take their own locks, and a
 * block finds its arena by its address, so it is freed like any other arena block.
 * Where eden_heap_main is about to return NULL because the heap could not grow:
 *   1  count only: how many arenas no thread is on are empty and how many partly used, and how many
 *      spaces hold nothing but empty arenas no thread is on (what a later step that gives whole
 *      spaces back could return); the request still fails. One count every 100 ms at most, and only
 *      when the creation lock is free at the first try (never a wait the off arm does not have).
 *   2  serve it: the arena this thread was last served from (unless a thread is on it now), then up
 *      to EDEN_HEAP_STEAL_EMPTY_TRIES empty arenas no thread is on (spares too), then up to
 *      EDEN_HEAP_STEAL_PARTLY partly used ones. A spare that served moves to this thread's record
 *      (kept, not used), or stays a spare. After a search for n bytes found nothing, the next
 *      EDEN_HEAP_STEAL_MISS_US do not search for n bytes or more (a burst of refusals); smaller
 *      requests still search, and the last arena is still asked.
 *   3  2, and: a request under 256 KiB that an empty arena served moves that arena to this thread's
 *      record and makes it the one it uses, so the next requests take the lock-free path (never a
 *      record's first arena, never one a thread is on); and for 1 s after the pool refused a growth,
 *      eden_heap_main asks here before it asks the pool again.
 * Nothing is freed, merged, decommitted or given to the pool here.
 * Locks: only the creation lock, taken with a try bounded in time (eden_heap_create_try: a busy lock
 * is a miss, never a wait without end; mspace locks are waited on as on every other path), never by
 * the thread that holds it, never for an arena's own block
 * (eden_heap_create_arena_locked asks eden_heap_main under it), and never under the growth lock
 * (eden_heap_main is not called under it). Under it: atomics, the records and the list, and at most
 * one mspace call per arena tried, which takes that mspace's own lock and calls nothing of ours. No
 * stdio and no wrapped malloc under it. The 10 s line: eden_heap_steal_stats, headless/performance.cpp. */
#define EDEN_HEAP_STEAL_MAX ((size_t)EDEN_HEAP_ARENA_SIZE - ((size_t)64 << 10))
#define EDEN_HEAP_STEAL_EMPTY_TRIES 8u
#define EDEN_HEAP_STEAL_PARTLY 32u
#define EDEN_HEAP_STEAL_MISS_US 5000ull
#define EDEN_HEAP_STEAL_COUNT_US 100000ull
/* The bounded tries of the creation lock, in process time (not in sleeps: how long a 50 us sleep
 * really takes on the console is not measured): at a refusal (the request fails without it), and
 * before a growth (the pool is still there to ask). The wait can pass the budget by one sleep. After
 * a try that waited its budget and gave up, the next EDEN_HEAP_STEAL_BUSY_US try the lock once and
 * never wait (count mode always does), so a burst of refusals while the lock is held long pays the
 * wait at most once in that time. */
#define EDEN_HEAP_STEAL_WAIT_REFUSAL_US 5000ull
#define EDEN_HEAP_STEAL_WAIT_BEFORE_GROW_US 500ull
#define EDEN_HEAP_STEAL_BUSY_US 20000ull
enum {
  EDEN_STEAL_TRIGGER_REFUSAL, EDEN_STEAL_TRIGGER_BEFORE_GROW, EDEN_STEAL_SCANS,
  EDEN_STEAL_SERVED_STICKY, EDEN_STEAL_SERVED_EMPTY, EDEN_STEAL_SERVED_PARTLY, EDEN_STEAL_SERVED_BYTES,
  EDEN_STEAL_FROM_SPARES, EDEN_STEAL_REHOMED, EDEN_STEAL_KEPT_FIRST, EDEN_STEAL_NONE,
  EDEN_STEAL_TOO_BIG, EDEN_STEAL_LOCK_HELD, EDEN_STEAL_LOCK_BUSY, EDEN_STEAL_CACHED, EDEN_STEAL_RATE,
  EDEN_STEAL_WOULD, EDEN_STEAL_TIME_US, EDEN_STEAL_LONGEST_US,
  EDEN_STEAL_LAST_EMPTY, EDEN_STEAL_MIN_EMPTY_PLUS1, EDEN_STEAL_LAST_PARTLY, EDEN_STEAL_LAST_SPACES,
  EDEN_STEAL_LAST_SPACE_MIB, EDEN_STEAL_TEST_REFUSED,
  /* v2: lock busy on a try that did not wait (count mode, or within EDEN_HEAP_STEAL_BUSY_US of a
   * busy give-up); served spares kept on the asking thread's record, not current; sticky arenas
   * dropped because a thread is on them; at the last scan, the arenas threads are on whose mspace is
   * not in its own slot (the slow look-up, NOTES 7.9). */
  EDEN_STEAL_BUSY_NO_WAIT, EDEN_STEAL_SPARE_KEPT, EDEN_STEAL_STICKY_DROPPED, EDEN_STEAL_LAST_ODD,
  EDEN_STEAL_STATS
};
static atomic_ullong eden_heap_steal_stat[EDEN_STEAL_STATS];
static atomic_ullong eden_heap_steal_last_scan_us, eden_heap_steal_last_miss_us, eden_heap_steal_last_miss_bytes,
    eden_heap_steal_busy_us;
static atomic_uint eden_heap_steal_cursor;
/* Scratch, used under the creation lock only: the 2 MiB slots of the arenas threads are on and of
 * every record's first arena, the arenas threads are on whose mspace is not in their own slot, and
 * per space the arenas in it and the empty ones no thread is on. */
#define EDEN_HEAP_STEAL_WORDS ((EDEN_HEAP_SLOTS + 63u) / 64u)
static unsigned long long eden_heap_steal_named[EDEN_HEAP_STEAL_WORDS], eden_heap_steal_first[EDEN_HEAP_STEAL_WORDS];
static void *eden_heap_steal_odd[EDEN_HEAP_THREADS];
static unsigned short eden_heap_steal_arenas_in[EDEN_HEAP_PIECES], eden_heap_steal_idle_in[EDEN_HEAP_PIECES];

void eden_heap_set_steal(int mode) {
  atomic_store_explicit(&eden_heap_steal_mode, mode, memory_order_relaxed);
}

int eden_heap_steal_mode_now(void) {
  return atomic_load_explicit(&eden_heap_steal_mode, memory_order_relaxed);
}

/* The counters, in the order of the enum above (EDEN_STEAL_TEST_REFUSED: EDEN_TEST_HEAP_NO_GROW). */
unsigned eden_heap_steal_stats(unsigned long long *values, unsigned count) {
  for (unsigned i = 0; i < count && i < EDEN_STEAL_STATS; ++i)
    values[i] = i == EDEN_STEAL_TEST_REFUSED ? atomic_load_explicit(&eden_heap_no_grow_refused, memory_order_relaxed)
                                             : atomic_load_explicit(&eden_heap_steal_stat[i], memory_order_relaxed);
  return EDEN_STEAL_STATS;
}

static void eden_heap_steal_bump(unsigned which, unsigned long long by) {
  atomic_fetch_add_explicit(&eden_heap_steal_stat[which], by, memory_order_relaxed);
}

/* The creation lock, or 0. `budget_us` 0: one try, no spin, no sleep. Else 64 spins (as
 * eden_heap_create_acquire), then sleeps of about 50 us while less than `budget_us` of process time
 * has passed since the first miss: the wait is bounded by time, plus at most one sleep. Released
 * with eden_heap_create_release. */
static int eden_heap_create_try(unsigned long long budget_us) {
  if (atomic_flag_test_and_set_explicit(&eden_heap_arena_create_lock, memory_order_acquire)) {
    if (budget_us == 0)
      return 0;
    const unsigned long long from = (unsigned long long)sceKernelGetProcessTime();
    atomic_fetch_add_explicit(&eden_heap_create_waiters, 1, memory_order_relaxed);
    for (unsigned spins = 0; atomic_flag_test_and_set_explicit(&eden_heap_arena_create_lock, memory_order_acquire);
         ++spins) {
      if (spins < 64) {
        __builtin_ia32_pause();
      } else if ((unsigned long long)sceKernelGetProcessTime() - from < budget_us) {
        sceKernelUsleep(50);
      } else {
        atomic_fetch_sub_explicit(&eden_heap_create_waiters, 1, memory_order_relaxed);
        return 0;
      }
    }
    atomic_fetch_sub_explicit(&eden_heap_create_waiters, 1, memory_order_relaxed);
  }
  atomic_store_explicit(&eden_heap_create_holder, (unsigned long)pthread_self(), memory_order_relaxed);
  return 1;
}

/* Within `window_us` of `stamp` (0: never). A stamp later than `now` (another thread wrote it after
 * this one read the clock) counts as within. */
static int eden_heap_steal_within(unsigned long long stamp, unsigned long long now, unsigned long long window_us) {
  return stamp != 0 && (stamp > now || now - stamp < window_us);
}

/* Modes 2 and 3: a search for at least `bytes` found nothing less than EDEN_HEAP_STEAL_MISS_US ago,
 * so a search for `bytes` would find nothing either. Keyed on size: a smaller request may still fit.
 * The size is stored before the time (release) and read after it (acquire); two misses at once can
 * still pair one's time with the other's size, which only skips or allows one search too many. */
static int eden_heap_steal_cached(unsigned long long now, size_t bytes) {
  const unsigned long long missed = atomic_load_explicit(&eden_heap_steal_last_miss_us, memory_order_acquire);
  if (!eden_heap_steal_within(missed, now, EDEN_HEAP_STEAL_MISS_US))
    return 0;
  return (unsigned long long)bytes >= atomic_load_explicit(&eden_heap_steal_last_miss_bytes, memory_order_relaxed);
}

/* Whether a record names `arena` as the one its thread is on now (an ended thread's record too: the
 * next thread on it starts there). Under the creation lock: 512 relaxed loads. */
static int eden_heap_steal_named_now(const void *arena) {
  for (unsigned r = 0; r < EDEN_HEAP_THREADS; ++r)
    if (__atomic_load_n(&eden_heap_threads[r].arena, __ATOMIC_RELAXED) == arena)
      return 1;
  return 0;
}

/* Marks the slot of `arena` (an arena's mspace) in `bits`; 0 when its mspace does not lie in the slot
 * of its own block (then it is looked up by pointer). Under the creation lock. */
static int eden_heap_steal_mark(unsigned long long *bits, void *arena) {
  const uintptr_t slot = eden_heap_slot(arena);
  if (slot >= EDEN_HEAP_SLOTS || atomic_load_explicit(&eden_heap_slots[slot], memory_order_relaxed) != arena)
    return 0;
  bits[slot >> 6] |= 1ull << (slot & 63u);
  return 1;
}

static int eden_heap_steal_marked(const unsigned long long *bits, uintptr_t slot) {
  return slot < EDEN_HEAP_SLOTS && (bits[slot >> 6] >> (slot & 63u)) & 1u;
}

/* Takes `arena` off the spares, if it is one (under the creation lock), counted as a spare taken
 * again, as eden_heap_create_arena_locked counts one. 1: it was. The caller puts it in a record. */
static int eden_heap_steal_unspare(void *arena) {
  const unsigned spares = atomic_load_explicit(&eden_heap_spare_count, memory_order_relaxed);
  for (unsigned i = 0; i < spares; ++i)
    if (eden_heap_spare[i] == arena) {
      eden_heap_spare[i] = eden_heap_spare[spares - 1];
      atomic_store_explicit(&eden_heap_spare_count, spares - 1, memory_order_relaxed);
      atomic_fetch_add_explicit(&eden_heap_arenas_taken_again, 1, memory_order_relaxed);
      eden_heap_steal_bump(EDEN_STEAL_FROM_SPARES, 1);
      return 1;
    }
  return 0;
}

/* Mode 3: the empty arena that just served `self` becomes one of its own and the one it uses. It
 * leaves the record that had it (never as that record's first arena: the next thread on a record
 * starts there), as eden_heap_return_record takes one off; a spare leaves the spares. No thread is on
 * it (the caller checked), and every reader of a record's own arenas reads them under the creation
 * lock, held here. `index`: its entry in eden_heap_arena_list. */
static void eden_heap_steal_rehome(struct eden_heap_thread *self, void *arena, unsigned index) {
  (void)eden_heap_steal_unspare(arena);
  int mine = 0;
  for (unsigned r = 0; r < EDEN_HEAP_THREADS; ++r) {
    struct eden_heap_thread *record = &eden_heap_threads[r];
    unsigned owned = __atomic_load_n(&record->owned, __ATOMIC_ACQUIRE);
    unsigned k = 0;
    while (k < owned && __atomic_load_n(&record->own[k], __ATOMIC_RELAXED) != arena)
      ++k;
    if (k == owned)
      continue;
    if (record == self) {
      mine = 1;
      break;
    }
    if (k == 0) {
      eden_heap_steal_bump(EDEN_STEAL_KEPT_FIRST, 1);
      self->steal = index + 1u;
      return;
    }
    for (; k + 1 < owned; ++k)
      __atomic_store_n(&record->own[k], __atomic_load_n(&record->own[k + 1], __ATOMIC_RELAXED), __ATOMIC_RELAXED);
    __atomic_store_n(&record->owned, owned - 1, __ATOMIC_RELEASE);
    break;
  }
  if (!mine)
    eden_heap_keep_arena(self, arena);
  eden_heap_use_arena(self, arena);
  self->retry = 0;
  self->steal = 0;
  eden_heap_steal_bump(EDEN_STEAL_REHOMED, 1);
}

/* The time a try took, into the totals and the longest (the lock wait included). */
static void eden_heap_steal_timed(unsigned long long start, unsigned long long end) {
  const unsigned long long took = end > start ? end - start : 0;
  eden_heap_steal_bump(EDEN_STEAL_TIME_US, took);
  unsigned long long longest = atomic_load_explicit(&eden_heap_steal_stat[EDEN_STEAL_LONGEST_US], memory_order_relaxed);
  while (took > longest &&
         !atomic_compare_exchange_weak_explicit(&eden_heap_steal_stat[EDEN_STEAL_LONGEST_US], &longest, took,
                                                memory_order_relaxed, memory_order_relaxed)) {
  }
}

static void *eden_heap_steal(int kind, size_t a, size_t b, size_t bytes, int before_grow) {
  const int mode = atomic_load_explicit(&eden_heap_steal_mode, memory_order_relaxed);
  if (mode == 0 || bytes == 0)
    return NULL;
  /* An arena's own block: asked by eden_heap_create_arena_locked under the creation lock. */
  if (kind == EDEN_HEAP_ALIGNED && a == EDEN_HEAP_ARENA_SIZE && b == EDEN_HEAP_ARENA_PAYLOAD)
    return NULL;
  /* eden_heap_main's `bytes` of an aligned request is alignment + size: room for either. */
  const size_t alignment = kind == EDEN_HEAP_ALIGNED ? a : 0;
  const size_t size = kind == EDEN_HEAP_ALIGNED ? b : bytes;
  if (alignment > ((size_t)64 << 10) || bytes > EDEN_HEAP_STEAL_MAX) {
    eden_heap_steal_bump(EDEN_STEAL_TOO_BIG, 1);
    return NULL;
  }
  /* Triggers: requests this could serve (too big ones are counted apart, above). */
  eden_heap_steal_bump(before_grow ? EDEN_STEAL_TRIGGER_BEFORE_GROW : EDEN_STEAL_TRIGGER_REFUSAL, 1);
  /* The lock is not recursive (a report that prints while it walks the arenas, an arena's block). */
  if (atomic_load_explicit(&eden_heap_create_holder, memory_order_relaxed) == (unsigned long)pthread_self()) {
    eden_heap_steal_bump(EDEN_STEAL_LOCK_HELD, 1);
    return NULL;
  }
  const unsigned long long start = (unsigned long long)sceKernelGetProcessTime();
  /* The record before the lock: claiming one (a thread's first request) may call pthread_setspecific. */
  struct eden_heap_thread *self = mode >= 2 ? eden_heap_self() : NULL;
  if (mode == 1) {
    const unsigned long long last = atomic_load_explicit(&eden_heap_steal_last_scan_us, memory_order_relaxed);
    if (eden_heap_steal_within(last, start, EDEN_HEAP_STEAL_COUNT_US)) {
      eden_heap_steal_bump(EDEN_STEAL_RATE, 1);
      return NULL;
    }
  } else if ((self == NULL || self->steal == 0) && eden_heap_steal_cached(start, bytes)) {
    /* No sticky arena to ask, and a search for this much or less just found nothing: no lock. */
    eden_heap_steal_bump(EDEN_STEAL_CACHED, 1);
    return NULL;
  }
  /* Count mode never waits for the lock (the off arm does not); the others wait their budget, or
   * try once within EDEN_HEAP_STEAL_BUSY_US of a give-up. */
  unsigned long long budget = before_grow ? EDEN_HEAP_STEAL_WAIT_BEFORE_GROW_US : EDEN_HEAP_STEAL_WAIT_REFUSAL_US;
  if (mode == 1 ||
      eden_heap_steal_within(atomic_load_explicit(&eden_heap_steal_busy_us, memory_order_relaxed), start,
                             EDEN_HEAP_STEAL_BUSY_US))
    budget = 0;
  if (!eden_heap_create_try(budget)) {
    const unsigned long long end = (unsigned long long)sceKernelGetProcessTime();
    if (budget != 0) {
      atomic_store_explicit(&eden_heap_steal_busy_us, end | 1u, memory_order_relaxed);
      eden_heap_steal_bump(EDEN_STEAL_LOCK_BUSY, 1);
    } else {
      eden_heap_steal_bump(EDEN_STEAL_BUSY_NO_WAIT, 1);
    }
    eden_heap_steal_timed(start, end);
    return NULL;
  }
  void *address = NULL, *from = NULL;
  unsigned from_index = 0;
  int from_empty = 0, scanned = 0;
  /* The arena this thread was served from last: a stream of 256 KiB+ requests (they never use an
   * arena of the thread's own) takes a scan only when it is full. Asked before the 5 ms cache: the
   * cache says the scan found nothing, not that this arena is full. Dropped when it is a spare (whole,
   * for the next thread) or when any record names it now: it became a thread's arena since (its
   * owner went back to it, an adopt, or a spare handed out as a new thread's first). */
  if (self != NULL && self->steal != 0) {
    const unsigned index = self->steal - 1u;
    void *last = index < atomic_load_explicit(&eden_heap_arena_count, memory_order_acquire)
                     ? eden_heap_arena_list[index]
                     : NULL;
    self->steal = 0;
    if (last != NULL && (eden_heap_is_spare(last) || eden_heap_steal_named_now(last))) {
      eden_heap_steal_bump(EDEN_STEAL_STICKY_DROPPED, 1);
    } else if (last != NULL) {
      address = eden_heap_space_try(last, kind, a, b);
      if (address != NULL) {
        eden_heap_arena_took(address);
        eden_heap_steal_bump(EDEN_STEAL_SERVED_STICKY, 1);
        self->steal = index + 1u;
      }
    }
  }
  /* Mode 2/3, nothing from the sticky arena: the cache again, now under the lock (a thread that
   * missed while this one waited counts). */
  if (address == NULL && mode >= 2 && eden_heap_steal_cached(start, bytes)) {
    eden_heap_steal_bump(EDEN_STEAL_CACHED, 1);
  } else if (address == NULL) {
    scanned = 1;
    eden_heap_steal_bump(EDEN_STEAL_SCANS, 1);
    /* Which arenas threads are on (by the slot of their mspace) and which are a record's first. */
    memset(eden_heap_steal_named, 0, sizeof(eden_heap_steal_named));
    memset(eden_heap_steal_first, 0, sizeof(eden_heap_steal_first));
    memset(eden_heap_steal_arenas_in, 0, sizeof(eden_heap_steal_arenas_in));
    memset(eden_heap_steal_idle_in, 0, sizeof(eden_heap_steal_idle_in));
    unsigned odd = 0;
    for (unsigned r = 0; r < EDEN_HEAP_THREADS; ++r) {
      void *on = __atomic_load_n(&eden_heap_threads[r].arena, __ATOMIC_RELAXED);
      if (on != NULL && !eden_heap_steal_mark(eden_heap_steal_named, on))
        eden_heap_steal_odd[odd++] = on;
      if (__atomic_load_n(&eden_heap_threads[r].owned, __ATOMIC_ACQUIRE) != 0) {
        void *first = __atomic_load_n(&eden_heap_threads[r].own[0], __ATOMIC_RELAXED);
        if (first != NULL)
          (void)eden_heap_steal_mark(eden_heap_steal_first, first);
      }
    }
    const uintptr_t base = (uintptr_t)ps5_heap_base;
    const unsigned arenas = atomic_load_explicit(&eden_heap_arena_count, memory_order_acquire);
    const unsigned at = arenas != 0 ? atomic_fetch_add_explicit(&eden_heap_steal_cursor, 1, memory_order_relaxed) % arenas : 0;
    unsigned empty = 0, partly = 0, empty_tries = 0;
    for (unsigned i = 0; i < arenas; ++i) {
      const unsigned index = (at + i) % arenas;
      void *arena = eden_heap_arena_list[index];
      const uintptr_t block = (uintptr_t)atomic_load_explicit(&eden_heap_arena_blocks_probe[index], memory_order_relaxed);
      if (arena == NULL || block == 0)
        continue;
      const uintptr_t slot = eden_heap_slot((const void *)block);
      const unsigned space = atomic_load_explicit(&eden_heap_piece_slot[(block - base) >> EDEN_HEAP_PIECE_SHIFT],
                                                  memory_order_relaxed);
      if (space < EDEN_HEAP_PIECES)
        eden_heap_steal_arenas_in[space]++;
      int named = eden_heap_steal_marked(eden_heap_steal_named, slot);
      for (unsigned k = 0; !named && k < odd; ++k)
        named = eden_heap_steal_odd[k] == arena;
      if (named)
        continue;
      /* Read under the creation lock: every allocation into an arena no thread is on is made
       * under it, and a free lowers the count after the block is back, so 0 here is exact. */
      if (atomic_load_explicit(&eden_heap_arena_live[slot].blocks, memory_order_relaxed) != 0) {
        ++partly;
        continue;
      }
      const int idle = space < EDEN_HEAP_PIECES && !eden_heap_steal_marked(eden_heap_steal_first, slot);
      if (mode >= 2 && address == NULL && empty_tries < EDEN_HEAP_STEAL_EMPTY_TRIES) {
        ++empty_tries;
        address = eden_heap_space_try(arena, kind, a, b);
        if (address != NULL) {
          eden_heap_arena_took(address);
          from = arena;
          from_index = index;
          from_empty = 1;
          eden_heap_steal_bump(EDEN_STEAL_SERVED_EMPTY, 1);
          ++partly; /* not empty any more, and its space not idle */
          continue;
        }
      }
      ++empty;
      if (idle)
        eden_heap_steal_idle_in[space]++;
    }
    /* Spaces that hold nothing but empty arenas no thread is on, none a record's first: what a
     * step that retires such arenas and gives their spaces back could return. */
    unsigned spaces_idle = 0;
    unsigned long long spaces_idle_mib = 0;
    const unsigned spaces = atomic_load_explicit(&eden_heap_space_count, memory_order_acquire);
    for (unsigned slot = 1; slot < spaces && slot < EDEN_HEAP_PIECES; ++slot) {
      const unsigned in = eden_heap_steal_arenas_in[slot];
      if (in == 0 || eden_heap_space_span[slot] == 0 || eden_heap_steal_idle_in[slot] != in ||
          atomic_load_explicit(&eden_heap_space_blocks[slot], memory_order_relaxed) != (long)in)
        continue;
      ++spaces_idle;
      spaces_idle_mib += (unsigned long long)eden_heap_space_span[slot] << (EDEN_HEAP_PIECE_SHIFT - 20);
    }
    /* Then partly used ones, a bounded number of calls (another thread may be freeing into them:
     * their mspace's lock orders that). */
    for (unsigned i = 0, tries = 0; mode >= 2 && address == NULL && i < arenas && tries < EDEN_HEAP_STEAL_PARTLY; ++i) {
      const unsigned index = (at + i) % arenas;
      void *arena = eden_heap_arena_list[index];
      const uintptr_t block = (uintptr_t)atomic_load_explicit(&eden_heap_arena_blocks_probe[index], memory_order_relaxed);
      if (arena == NULL || block == 0)
        continue;
      const uintptr_t slot = eden_heap_slot((const void *)block);
      int named = eden_heap_steal_marked(eden_heap_steal_named, slot);
      for (unsigned k = 0; !named && k < odd; ++k)
        named = eden_heap_steal_odd[k] == arena;
      if (named || atomic_load_explicit(&eden_heap_arena_live[slot].blocks, memory_order_relaxed) == 0)
        continue;
      ++tries;
      address = eden_heap_space_try(arena, kind, a, b);
      if (address != NULL) {
        eden_heap_arena_took(address);
        from = arena;
        from_index = index;
        eden_heap_steal_bump(EDEN_STEAL_SERVED_PARTLY, 1);
      }
    }
    atomic_store_explicit(&eden_heap_steal_stat[EDEN_STEAL_LAST_ODD], odd, memory_order_relaxed);
    atomic_store_explicit(&eden_heap_steal_stat[EDEN_STEAL_LAST_EMPTY], empty, memory_order_relaxed);
    atomic_store_explicit(&eden_heap_steal_stat[EDEN_STEAL_LAST_PARTLY], partly, memory_order_relaxed);
    atomic_store_explicit(&eden_heap_steal_stat[EDEN_STEAL_LAST_SPACES], spaces_idle, memory_order_relaxed);
    atomic_store_explicit(&eden_heap_steal_stat[EDEN_STEAL_LAST_SPACE_MIB], spaces_idle_mib, memory_order_relaxed);
    const unsigned long long low = atomic_load_explicit(&eden_heap_steal_stat[EDEN_STEAL_MIN_EMPTY_PLUS1], memory_order_relaxed);
    if (low == 0 || (unsigned long long)empty + 1u < low)
      atomic_store_explicit(&eden_heap_steal_stat[EDEN_STEAL_MIN_EMPTY_PLUS1], (unsigned long long)empty + 1u,
                            memory_order_relaxed);
    if (mode == 1 && empty != 0)
      eden_heap_steal_bump(EDEN_STEAL_WOULD, 1);
  }
  if (from != NULL && self != NULL) {
    if (mode >= 3 && from_empty && size < EDEN_HEAP_SMALL_REQUEST && alignment <= 4096 &&
        self->owned < EDEN_HEAP_THREAD_ARENAS) {
      eden_heap_steal_rehome(self, from, from_index);
    } else {
      /* A spare that served leaves the spares only into this thread's record, kept but not used
       * (never as its first: the next thread on the record starts there), so it is not lost to
       * them: return_record gives it back to the spares once empty, and the arena table counts it.
       * When the record has no first arena yet or is full, it stays a spare that holds a block: a
       * new thread that takes it shares it, as after an adopt. */
      if (self->owned != 0 && self->owned < EDEN_HEAP_THREAD_ARENAS && eden_heap_steal_unspare(from)) {
        eden_heap_keep_arena(self, from);
        eden_heap_steal_bump(EDEN_STEAL_SPARE_KEPT, 1);
      }
      self->steal = from_index + 1u;
    }
  }
  eden_heap_create_release();
  const unsigned long long end = (unsigned long long)sceKernelGetProcessTime();
  eden_heap_steal_timed(start, end);
  if (mode == 1) {
    atomic_store_explicit(&eden_heap_steal_last_scan_us, start | 1u, memory_order_relaxed);
  } else if (address != NULL) {
    eden_heap_steal_bump(EDEN_STEAL_SERVED_BYTES, size);
  } else if (scanned) {
    /* Size first, then time: a reader that sees the new time sees this size or a later one. */
    atomic_store_explicit(&eden_heap_steal_last_miss_bytes, (unsigned long long)bytes, memory_order_relaxed);
    atomic_store_explicit(&eden_heap_steal_last_miss_us, end | 1u, memory_order_release);
    eden_heap_steal_bump(EDEN_STEAL_NONE, 1);
  }
  return address;
}

/* EDEN_TEST_HEAP_STARVE=<every>[,<attempts>] (a test, off unless set; read from main in
 * eden_heap_pool_init, headless/heap_pages.cpp): on a pipeline builder thread ("VkPipeline..."),
 * every <every>-th request of 256 KiB or more is refused as if the heap had nothing to give, and so
 * are that thread's next <attempts> - 1 requests of that size (default 150: 300 ms of
 * EDEN_UP_HEAP_WAIT_MS's 2 ms steps). No memory is taken away, so nothing else runs short: the
 * pool hog (EDEN_TEST_POOL_HOG_MIB) ended runs in the guest's memory or the driver's before a
 * builder was ever refused. Without EDEN_UP_HEAP_WAIT_MS the first refusal reaches the driver's
 * shader compiler as NULL, as on 10 Nov 2026; with it the builder has to come through. */
static atomic_uint eden_heap_starve_every, eden_heap_starve_attempts;
/* EDEN_TEST_HEAP_STARVE_GPU=1: the test refuses the requests of threads inside a run-time pipeline
 * build (eden_heap_wait_scope: the GPU thread, mostly) instead of the builder threads'. */
static atomic_uint eden_heap_starve_scope;
void eden_heap_set_starve_scope(int on) {
  atomic_store_explicit(&eden_heap_starve_scope, on ? 1u : 0u, memory_order_relaxed);
}
static atomic_ullong eden_heap_starve_seen, eden_heap_starve_begun, eden_heap_starve_refused;

void eden_heap_set_starve(unsigned every, unsigned attempts) {
  atomic_store_explicit(&eden_heap_starve_attempts, attempts != 0 ? attempts : 150u, memory_order_relaxed);
  atomic_store_explicit(&eden_heap_starve_every, every, memory_order_relaxed);
}

void eden_heap_starve_stats(unsigned long long *begun, unsigned long long *refused) {
  *begun = atomic_load_explicit(&eden_heap_starve_begun, memory_order_relaxed);
  *refused = atomic_load_explicit(&eden_heap_starve_refused, memory_order_relaxed);
}

static int eden_heap_starve(size_t bytes) {
  const unsigned every = atomic_load_explicit(&eden_heap_starve_every, memory_order_relaxed);
  if (every == 0 || bytes < EDEN_HEAP_SMALL_REQUEST)
    return 0;
  struct eden_heap_thread *self = eden_heap_self();
  if (self == NULL)
    return 0;
  if (atomic_load_explicit(&eden_heap_starve_scope, memory_order_relaxed) ? self->wait_scope == 0
                                                                           : strncmp(self->name, "VkPipeline", 10) != 0)
    return 0;
  if (self->starve_left == 0) {
    if ((atomic_fetch_add_explicit(&eden_heap_starve_seen, 1, memory_order_relaxed) + 1) % every != 0)
      return 0;
    self->starve_left = atomic_load_explicit(&eden_heap_starve_attempts, memory_order_relaxed);
    atomic_fetch_add_explicit(&eden_heap_starve_begun, 1, memory_order_relaxed);
  }
  --self->starve_left;
  atomic_fetch_add_explicit(&eden_heap_starve_refused, 1, memory_order_relaxed);
  return 1;
}

/* T130: a test refusal while EDEN_UP_HEAP_STEAL is on goes where a real refusal goes when the heap
 * could not grow (eden_heap_main): to eden_heap_steal, with the same arguments; mode 3 also stamps
 * "the pool said no" as eden_heap_grow does, so the next second asks the idle arenas first. Mode 0
 * (unset): NULL at once, as before. Counted for the 20 s heap summary (eden_heap_test_aid_stats). */
static atomic_ullong eden_heap_starve_to_steal, eden_heap_starve_stolen;

static void *eden_heap_starve_refuse(size_t count, size_t size, int zero, size_t bytes) {
  const int mode = atomic_load_explicit(&eden_heap_steal_mode, memory_order_relaxed);
  if (mode == 0)
    return NULL;
  if (mode >= 3)
    atomic_store_explicit(&eden_heap_steal_pool_no_us, (unsigned long long)sceKernelGetProcessTime() | 1u,
                          memory_order_relaxed);
  atomic_fetch_add_explicit(&eden_heap_starve_to_steal, 1, memory_order_relaxed);
  void *address = eden_heap_steal(zero ? EDEN_HEAP_CALLOC : EDEN_HEAP_MALLOC, zero ? count : bytes, zero ? size : 0,
                                  bytes, 0);
  if (address != NULL)
    atomic_fetch_add_explicit(&eden_heap_starve_stolen, 1, memory_order_relaxed);
  return address;
}

/* The two test aids' own counts, whatever the steal mode (the 20 s heap summary,
 * headless/performance.cpp): EDEN_TEST_HEAP_STARVE every, requests seen, shortages begun, requests
 * refused, refusals sent to the steal, served by it; EDEN_TEST_HEAP_NO_GROW every, growths seen,
 * refused. Returns how many values it knows (9). */
unsigned eden_heap_test_aid_stats(unsigned long long *values, unsigned count) {
  const unsigned long long all[9] = {
      atomic_load_explicit(&eden_heap_starve_every, memory_order_relaxed),
      atomic_load_explicit(&eden_heap_starve_seen, memory_order_relaxed),
      atomic_load_explicit(&eden_heap_starve_begun, memory_order_relaxed),
      atomic_load_explicit(&eden_heap_starve_refused, memory_order_relaxed),
      atomic_load_explicit(&eden_heap_starve_to_steal, memory_order_relaxed),
      atomic_load_explicit(&eden_heap_starve_stolen, memory_order_relaxed),
      atomic_load_explicit(&eden_heap_no_grow_every, memory_order_relaxed),
      atomic_load_explicit(&eden_heap_no_grow_seen, memory_order_relaxed),
      atomic_load_explicit(&eden_heap_no_grow_refused, memory_order_relaxed),
  };
  for (unsigned i = 0; i < count && i < 9u; ++i)
    values[i] = all[i];
  return 9u;
}

static void *eden_heap_allocate(size_t count, size_t size, int zero) {
  const size_t bytes = count * size;
  if (eden_heap_starve(bytes))
    return eden_heap_starve_refuse(count, size, zero, bytes);
  if (bytes >= EDEN_HEAP_SMALL_REQUEST && bytes < EDEN_HEAP_LARGE_REQUEST)
    eden_heap_class_note(bytes);
  if (bytes >= atomic_load_explicit(&eden_heap_large_from, memory_order_relaxed) && eden_heap_grows) {
    /* Memory of its own (it arrives zeroed). With every entry taken, or no memory left to take,
     * the heap's spaces: a piece committed earlier may still have the room. */
    void *block = eden_heap_large_allocate(bytes);
    if (block != NULL)
      return block;
    /* EDEN_BIG_IMAGE_NO_SPLIT: a big image's block the reserve could not serve is refused here
     * too, not grown into the heap's spaces (pieces that stay for good). */
    if (eden_heap_big_image_refuses(bytes)) {
      eden_heap_note_failure(bytes, EDEN_HEAP_WHY_NO_MEMORY); /* eden_heap_record counts it */
      return NULL;
    }
  }
  void *address;
  if (bytes < EDEN_HEAP_SMALL_REQUEST) {
    /* RADV_PS5_LONGLIVED_ARENA=1: a kept object goes to the long-lived set (off: one load). */
    address = eden_heap_ll_route(bytes) ? eden_heap_ll_allocate(count, size, zero, 0) : NULL;
    if (address == NULL)
      address = eden_heap_arena_allocate(count, size, zero);
    if (address == NULL)
      address = eden_heap_spill(count, size, zero);
  } else {
    address = zero ? eden_heap_main(EDEN_HEAP_CALLOC, count, size) : eden_heap_main(EDEN_HEAP_MALLOC, bytes, 0);
  }
  if (address == NULL && bytes != 0)
    address = eden_heap_arena_rescue(count, size, zero);
  return address;
}

/* EDEN_UP_HEAP_WAIT_MS=<n> (0 or unset: off; read from main in eden_heap_pool_init, headless/heap_pages.cpp).
 * A pipeline builder thread ("VkPipeline..." by the name on its record) whose request the heap
 * cannot serve asks again every 2 ms, for up to n ms, instead of getting NULL at once. The
 * driver's shader compiler does not check its allocations (RADV's NIR passes over ralloc): a NULL
 * there ended the session with a SIGSEGV in nir_opt_copy_prop_vars (10 Nov 2026, in free
 * play, largest free block 3 MiB; the same on a PS5 Pro in issue 21). While a request waits,
 * eden_heap_wanted() is not 0 and the caches see memory short with none free
 * (headless/performance.cpp, GraphicsMemoryShort, as for a guest write that waits for a block), so
 * the texture collector evicts at its hardest until the request is served. The builders are
 * background threads: a wait holds one pipeline up, not the frame. Still nothing after n ms: NULL
 * as before. eden_heap_wait_stats counts the waits for the 20 s heap line. */
static atomic_uint eden_heap_wait_ms;
/* EDEN_UP_HEAP_WAIT_GPU_MS=<n> (0 or unset: off): the same wait, for up to n ms, for a thread that is
 * building a pipeline itself at run time (the pipeline cache marks the build with
 * eden_heap_wait_scope: the GPU thread at a pipeline's first sight). A crash in free play on 10 Nov 22:07:
 * 64 pipelines were built on the GPU thread in the 4 s before it, and that thread got NULL at once.
 * A wait there holds the frame, so it has its own, shorter limit, and a build whose request waited
 * its whole time and got nothing does not wait again (wait_spent): one build costs at most n ms.
 * The texture collector runs on that same thread and cannot help while it waits; the heap trim and
 * the guest memory packer can. */
static atomic_uint eden_heap_wait_gpu_ms;
static atomic_ullong eden_heap_wait_gpu_count, eden_heap_wait_gpu_total_ms, eden_heap_wait_gpu_lost;
static atomic_uint eden_heap_waiting;
static atomic_ullong eden_heap_wait_count, eden_heap_wait_total_ms, eden_heap_wait_lost;
static atomic_uint eden_heap_wait_longest_ms;

void eden_heap_set_wait_ms(unsigned ms) {
  atomic_store_explicit(&eden_heap_wait_ms, ms, memory_order_relaxed);
}

void eden_heap_set_wait_gpu_ms(unsigned ms) {
  atomic_store_explicit(&eden_heap_wait_gpu_ms, ms, memory_order_relaxed);
}

/* The pipeline cache, around a pipeline it builds on the calling thread at run time (1: in, 0: out). */
void eden_heap_wait_scope(int enter) {
  struct eden_heap_thread *self = eden_heap_self();
  if (self == NULL)
    return;
  if (enter) {
    if (self->wait_scope++ == 0)
      self->wait_spent = 0;
  } else if (self->wait_scope != 0)
    --self->wait_scope;
}

void eden_heap_wait_gpu_stats(unsigned long long *count, unsigned long long *total_ms, unsigned long long *lost) {
  *count = atomic_load_explicit(&eden_heap_wait_gpu_count, memory_order_relaxed);
  *total_ms = atomic_load_explicit(&eden_heap_wait_gpu_total_ms, memory_order_relaxed);
  *lost = atomic_load_explicit(&eden_heap_wait_gpu_lost, memory_order_relaxed);
}

unsigned eden_heap_wanted(void) {
  return atomic_load_explicit(&eden_heap_waiting, memory_order_relaxed);
}

/* EDEN_UP_HLE_OOM_RETRY (core/hle/service/hle_reply_buffers.h): a service thread whose request
 * was refused memory before its service code ran sleeps a moment and tries once more; meanwhile it
 * counts as a waiting request here (1: in, 0: out), so the caches see memory short with none free
 * and the texture collector evicts at its hardest, as for a pipeline builder's wait above. Called
 * only by that switch: unset, eden_heap_waiting moves as before. */
void eden_heap_want(int enter) {
  if (enter)
    atomic_fetch_add_explicit(&eden_heap_waiting, 1, memory_order_relaxed);
  else
    atomic_fetch_sub_explicit(&eden_heap_waiting, 1, memory_order_relaxed);
}

/* EDEN_UP_DOOR_WAIT (0597, our own): run-time pipeline build JOBS in flight on a builder thread.
 * The core counts each whole job (video_core/renderer_vulkan/vk_door_job.h, a scope beside 0592's
 * BuildBurst::Job in the graphics and compute build lambdas; only with EDEN_UP_DOOR_WAIT=1 or
 * EDEN_UP_BUILD_GIVE_BACK=1, read once there), so operator new's handler (headless/main.cpp) can
 * tell "a build is in flight or just ended" without calling into the video core. Counters only:
 * no lock, no wait, no allocation, no line. Nothing calls these with both switches off. */
static atomic_uint eden_door_jobs_now;
static atomic_ullong eden_door_jobs_ended;  /* the "ended" generation: one per finished job */
static atomic_ullong eden_door_last_end_us; /* sceKernelGetProcessTime() of the last end; 0: none */

/* EDEN_UP_BUILD_GIVE_BACK=1 (0598, our own; set by main once the env files are in): when the last
 * builder job in flight ends, the heap gives its empty spaces back to the pool at once
 * (eden_heap_trim: the existing call of operator new's wait, the JIT's wait, TrimHeap and the session
 * end; whole spaces with no block in them, and the empty arenas to the spares), on the builder
 * thread that ended it: after its driver call and its build_mutex block (the draw waiting for it was
 * told first), outside the driver and every lock of the builders, never in a signal handler. Today
 * the heap trims only when someone is short or waits; run1323 died with 520 MiB of whole free
 * pages in the heap and the pool at 0. One give-back at a time (a second one meanwhile is counted
 * busy and skipped). A line only when memory went back:
 * "EDEN_BUILD_GIVE_BACK gave_mib= ms= calls= empty= busy=" (counts since the start). Off: one
 * relaxed load at a job end, and job ends are counted only while one of the two switches is on. */
static atomic_uint eden_door_give_back_on;
static atomic_flag eden_door_giving_back = ATOMIC_FLAG_INIT;
static atomic_ullong eden_door_give_back_calls, eden_door_give_back_empty, eden_door_give_back_busy;

void eden_door_set_give_back(int on) {
  atomic_store_explicit(&eden_door_give_back_on, on ? 1u : 0u, memory_order_relaxed);
}

static void eden_door_give_back(void) {
  if (atomic_flag_test_and_set_explicit(&eden_door_giving_back, memory_order_acquire)) {
    atomic_fetch_add_explicit(&eden_door_give_back_busy, 1, memory_order_relaxed);
    return;
  }
  const unsigned long long from = (unsigned long long)sceKernelGetProcessTime();
  const size_t gave = eden_heap_trim();
  const unsigned long long took = (unsigned long long)sceKernelGetProcessTime() - from;
  const unsigned long long calls = atomic_fetch_add_explicit(&eden_door_give_back_calls, 1, memory_order_relaxed) + 1;
  if (gave == 0)
    atomic_fetch_add_explicit(&eden_door_give_back_empty, 1, memory_order_relaxed);
  atomic_flag_clear_explicit(&eden_door_giving_back, memory_order_release);
  if (gave != 0)
    fprintf(stderr, "EDEN_BUILD_GIVE_BACK gave_mib=%llu ms=%llu.%03llu calls=%llu empty=%llu busy=%llu\n",
            (unsigned long long)(gave >> 20), took / 1000u, took % 1000u, calls,
            (unsigned long long)atomic_load_explicit(&eden_door_give_back_empty, memory_order_relaxed),
            (unsigned long long)atomic_load_explicit(&eden_door_give_back_busy, memory_order_relaxed));
}

void eden_door_job(int enter) {
  if (enter) {
    atomic_fetch_add_explicit(&eden_door_jobs_now, 1, memory_order_acq_rel);
    return;
  }
  unsigned now = atomic_load_explicit(&eden_door_jobs_now, memory_order_relaxed);
  do {
    if (now == 0)
      return; /* an end with no begin (cannot happen with the scope): never wraps */
  } while (!atomic_compare_exchange_weak_explicit(&eden_door_jobs_now, &now, now - 1, memory_order_acq_rel,
                                                  memory_order_relaxed));
  atomic_store_explicit(&eden_door_last_end_us, (unsigned long long)sceKernelGetProcessTime() | 1u,
                        memory_order_relaxed);
  atomic_fetch_add_explicit(&eden_door_jobs_ended, 1, memory_order_release);
  if (now == 1 && atomic_load_explicit(&eden_door_give_back_on, memory_order_relaxed))
    eden_door_give_back(); /* EDEN_UP_BUILD_GIVE_BACK: the last job in flight ended */
}

/* Jobs in flight now; `ended`: jobs finished since start; `since_end_ms`: ms since the last one
 * ended (~0 when none has). Loads and one clock read. */
unsigned eden_door_jobs(unsigned long long *ended, unsigned long long *since_end_ms) {
  if (ended)
    *ended = atomic_load_explicit(&eden_door_jobs_ended, memory_order_acquire);
  if (since_end_ms) {
    const unsigned long long last = atomic_load_explicit(&eden_door_last_end_us, memory_order_relaxed);
    const unsigned long long now = (unsigned long long)sceKernelGetProcessTime();
    *since_end_ms = last == 0 ? ~0ull : now > last ? (now - last) / 1000u : 0;
  }
  return atomic_load_explicit(&eden_door_jobs_now, memory_order_acquire);
}

/* 1 when the calling thread may wait for builder jobs in operator new's handler: by the name on its
 * heap record (RegisterWorker, headless/performance.cpp), the file-service thread "FS" (6 of 12 door
 * deaths) and the guest CPU cores "CPUCore_<n>", and only outside a run-time build scope
 * (eden_heap_wait_scope). Every other thread gets 0: the builders ("VkPipelineBuild"), the GPU
 * threads ("GPUBack", "GPU", "GPUFencingThrea"), "VulkanWorker", the reporters ("MemCensus",
 * "MemoryReport", "BigStacks", the burst reporter), the log flusher (its record has no name: it names
 * itself for the crash report only), any unnamed thread and a thread with no record. Reads this
 * thread's own record without claiming one. `out` (16 bytes, may be NULL): the name, "" when none. */
int eden_door_may_wait(char out[16]) {
  char name[16] = {0};
  int may = 0;
  if (eden_heap_arena_key_valid) {
    const struct eden_heap_thread *self = pthread_getspecific(eden_heap_arena_key);
    if (self != NULL) {
      for (unsigned i = 0; i < sizeof(name); ++i)
        name[i] = __atomic_load_n(&self->name[i], __ATOMIC_RELAXED);
      name[sizeof(name) - 1] = 0;
      may = self->wait_scope == 0 && (strcmp(name, "FS") == 0 || strncmp(name, "CPUCore_", 8) == 0);
    }
  }
  if (out)
    memcpy(out, name, sizeof(name));
  return may;
}

void eden_heap_wait_stats(unsigned long long *count, unsigned long long *total_ms, unsigned *longest_ms,
                          unsigned long long *lost) {
  *count = atomic_load_explicit(&eden_heap_wait_count, memory_order_relaxed);
  *total_ms = atomic_load_explicit(&eden_heap_wait_total_ms, memory_order_relaxed);
  *longest_ms = atomic_load_explicit(&eden_heap_wait_longest_ms, memory_order_relaxed);
  *lost = atomic_load_explicit(&eden_heap_wait_lost, memory_order_relaxed);
}

/* How long this thread may wait for a request, in ms: 0 for most threads. */
static unsigned eden_heap_may_wait(void) {
  const unsigned builders = atomic_load_explicit(&eden_heap_wait_ms, memory_order_relaxed);
  const unsigned scoped = atomic_load_explicit(&eden_heap_wait_gpu_ms, memory_order_relaxed);
  if (builders == 0 && scoped == 0)
    return 0;
  struct eden_heap_thread *self = eden_heap_self();
  if (self == NULL)
    return 0;
  if (strncmp(self->name, "VkPipeline", 10) == 0)
    return builders;
  return self->wait_scope != 0 && !self->wait_spent ? scoped : 0;
}

/* A request that just failed, on a thread that may wait `limit` ms: `again` makes it once more
 * (NULL: not yet). Returns its block, or NULL when the time is up. */
static void *eden_heap_wait_for(void *(*again)(void *), void *request, unsigned limit) {
  struct eden_heap_thread *self = eden_heap_self();
  const int scoped = self != NULL && strncmp(self->name, "VkPipeline", 10) != 0;
  void *block = NULL;
  unsigned waited = 0;
  atomic_fetch_add_explicit(&eden_heap_waiting, 1, memory_order_relaxed);
  while (block == NULL && waited < limit) {
    sceKernelUsleep(2000);
    waited += 2;
    block = again(request);
  }
  atomic_fetch_sub_explicit(&eden_heap_waiting, 1, memory_order_relaxed);
  if (scoped) {
    atomic_fetch_add_explicit(&eden_heap_wait_gpu_count, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&eden_heap_wait_gpu_total_ms, waited, memory_order_relaxed);
    if (block == NULL) {
      atomic_fetch_add_explicit(&eden_heap_wait_gpu_lost, 1, memory_order_relaxed);
      self->wait_spent = 1;
    }
    return block;
  }
  atomic_fetch_add_explicit(&eden_heap_wait_count, 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&eden_heap_wait_total_ms, waited, memory_order_relaxed);
  if (block == NULL)
    atomic_fetch_add_explicit(&eden_heap_wait_lost, 1, memory_order_relaxed);
  unsigned longest = atomic_load_explicit(&eden_heap_wait_longest_ms, memory_order_relaxed);
  while (waited > longest &&
         !atomic_compare_exchange_weak_explicit(&eden_heap_wait_longest_ms, &longest, waited, memory_order_relaxed,
                                                memory_order_relaxed)) {
  }
  return block;
}

struct eden_heap_wait_request {
  size_t count, size, alignment;
  int zero;
};

static void *eden_heap_wait_allocate_again(void *request) {
  const struct eden_heap_wait_request *r = request;
  return eden_heap_allocate(r->count, r->size, r->zero);
}

/* eden_heap_allocate, and on a thread that may wait, again until it is served or the time is up. */
static void *eden_heap_allocate_or_wait(size_t count, size_t size, int zero) {
  void *block = eden_heap_allocate(count, size, zero);
  unsigned limit;
  if (block == NULL && count * size != 0 && (limit = eden_heap_may_wait()) != 0) {
    struct eden_heap_wait_request request = {count, size, 0, zero};
    block = eden_heap_wait_for(eden_heap_wait_allocate_again, &request, limit);
  }
  return block;
}

void *__wrap_malloc(size_t size) {
  if (!ps5_heap_ready())
    return __real_malloc(size);
  const long long big = eden_big_begin(size); /* EDEN_UP_BIG_STACKS: -1 when off */
  /* RADV_PS5_LONGLIVED_ARENA=1: a kept object takes no cached block (those are the thread's arenas'). */
  void *cached = eden_heap_ll_active() ? NULL : eden_tcache_take(size);
  void *block = cached != NULL ? cached : eden_heap_record(eden_heap_allocate_or_wait(1, size, 0), size);
  eden_site_alloc(block, size);
  eden_big_end(block, size, big);
  return block;
}

void *__wrap_calloc(size_t count, size_t size) {
  if (!ps5_heap_ready())
    return __real_calloc(count, size);
  if (size != 0 && count > (size_t)-1 / size) {
    eden_heap_failed((size_t)-1, EDEN_HEAP_WHY_OVERFLOW);
    return NULL;
  }
  const long long big = eden_big_begin(count * size); /* EDEN_UP_BIG_STACKS: -1 when off */
  void *cached = eden_heap_ll_active() ? NULL : eden_tcache_take(count * size); /* as __wrap_malloc */
  if (cached != NULL) {
    eden_site_alloc(cached, count * size);
    eden_big_end(cached, count * size, big);
    return memset(cached, 0, count * size);
  }
  void *block = eden_heap_record(eden_heap_allocate_or_wait(count, size, 1), count * size);
  eden_site_alloc(block, count * size);
  eden_big_end(block, count * size, big);
  return block;
}

static void *eden_realloc_inner(void *address, size_t size) {
  if (address == NULL)
    return __wrap_malloc(size);
  const int in_range = eden_heap_in_range(address);
  const int large = in_range ? -1 : eden_heap_large_find(address);
  if (!in_range && large < 0)
    return __real_realloc(address, size);
  const size_t before = in_range ? sceLibcMspaceMallocUsableSize(address) : eden_heap_large[large].size;
  if (large >= 0) {
    if (size == 0) {
      eden_heap_account(-(long)before, -1);
      eden_heap_large_free(large);
      return NULL;
    }
    /* A large block keeps its memory while the new size still needs most of it. */
    if (size <= before && size >= before / 2)
      return address;
  } else if (size < atomic_load_explicit(&eden_heap_large_from, memory_order_relaxed) || !eden_heap_grows) {
    void *owner = eden_heap_owner(address);
    void *result = sceLibcMspaceRealloc(owner, address, size);
    if (result) {
      const size_t after = sceLibcMspaceMallocUsableSize(result);
      eden_heap_account((long)after - (long)before, 0);
      eden_pages_count(address, before, -1);
      eden_pages_count(result, after, 1);
      return result;
    }
    if (size == 0) {
      /* Preserve platform realloc(p,0) behavior; NULL does not tell us whether
       * p was freed. Mark the counters inconclusive instead of guessing. */
      atomic_fetch_add_explicit(&ps5_heap_ambiguous_zero_reallocs, 1, memory_order_relaxed);
      return NULL;
    }
  }
  /* The block cannot stay where it is (its arena or its piece of the heap is full, or it
   * crosses the size at which blocks get memory of their own): move it. */
  void *moved = eden_heap_allocate_or_wait(1, size, 0);
  if (moved == NULL) {
    eden_heap_failed(size, atomic_load_explicit(&eden_heap_grow_why, memory_order_relaxed));
    return NULL;
  }
  memcpy(moved, address, before < size ? before : size);
  const size_t after = eden_heap_usable(moved);
  eden_heap_account((long)after - (long)before, 0);
  eden_pages_count(moved, after, 1);
  if (large >= 0) {
    eden_heap_large_free(large);
  } else {
    eden_pages_count(address, before, -1);
    eden_heap_free_owned(address);
  }
  return moved;
}

void *__wrap_realloc(void *address, size_t size) {
  if (address == NULL)
    return __wrap_malloc(size);
  size_t usable = 0;
  const int owned = ps5_heap_owns(address);
  if (owned) {
    usable = eden_heap_usable(address);
    eden_site_free(address, usable);
  }
  /* EDEN_UP_BIG_STACKS: the old block leaves the table first (its address may come back as the
   * result), and goes back to its row if the realloc fails and the block stays. One load when off. */
  const int big_on = owned && (usable >= EDEN_HEAP_SMALL_REQUEST || size >= EDEN_HEAP_SMALL_REQUEST) &&
                     atomic_load_explicit(&eden_big_on, memory_order_relaxed);
  unsigned long long big_old = 0;
  const unsigned big_row = big_on && usable >= EDEN_HEAP_SMALL_REQUEST ? eden_big_detach(address, &big_old) : 0;
  const long long big = big_on && size >= EDEN_HEAP_SMALL_REQUEST ? eden_big_grown_now() : -1;
  void *result = eden_realloc_inner(address, size);
  eden_site_alloc(result, size);
  if (big_row != 0 && result == NULL && size != 0)
    eden_big_reattach(address, big_row, big_old);
  eden_big_end(result, size, big);
  return result;
}

void __wrap_free(void *address) {
  if (eden_heap_in_range(address)) {
    const size_t usable = sceLibcMspaceMallocUsableSize(address);
    eden_site_free(address, usable);
    eden_big_free(address, usable); /* EDEN_UP_BIG_STACKS */
    eden_pages_count(address, usable, -1);
    /* RADV_PS5_LONGLIVED_ARENA=1: a kept object's block goes straight back to its arena (off: one load). */
    if (!eden_heap_ll_block(address) && eden_tcache_keep(address, usable))
      return;
    eden_heap_account(-(long)usable, -1);
    eden_heap_release(address);
    return;
  }
  const int large = eden_heap_large_find(address);
  if (large >= 0) {
    eden_site_free(address, eden_heap_large[large].size);
    eden_big_free(address, eden_heap_large[large].size); /* EDEN_UP_BIG_STACKS */
    eden_heap_account(-(long)eden_heap_large[large].size, -1);
    eden_heap_large_free(large);
  } else
    __real_free(address);
}

/* One try of posix_memalign's request: the block, or NULL. */
static void *eden_heap_aligned_try(size_t alignment, size_t size) {
  void *block = NULL;
  if (size < EDEN_HEAP_SMALL_REQUEST && alignment <= 4096) {
    /* RADV_PS5_LONGLIVED_ARENA=1: a kept object goes to the long-lived set (off: one load). */
    if (eden_heap_ll_route(size) && (block = eden_heap_ll_allocate(1, size, 0, alignment)) != NULL)
      return block;
    void *arena = eden_heap_thread_arena();
    if (arena != NULL) {
      if (sceLibcMspacePosixMemalign(arena, &block, alignment, size) == 0) {
        eden_heap_arena_took(block);
        return block;
      }
      block = NULL;
    }
  }
  /* A large block's own memory starts on a page; the heap's spaces align anything. */
  if (alignment <= 4096 && size >= atomic_load_explicit(&eden_heap_large_from, memory_order_relaxed) && eden_heap_grows)
    block = eden_heap_large_allocate(size);
  if (block == NULL)
    block = eden_heap_main(EDEN_HEAP_ALIGNED, alignment, size);
  return block;
}

static void *eden_heap_wait_aligned_again(void *request) {
  const struct eden_heap_wait_request *r = request;
  return eden_heap_aligned_try(r->alignment, r->size);
}

int __wrap_posix_memalign(void **address, size_t alignment, size_t size) {
  if (!ps5_heap_ready())
    return __real_posix_memalign(address, alignment, size);
  /* An alignment no space accepts must not make the heap grow looking for one that does. */
  if (alignment < sizeof(void *) || (alignment & (alignment - 1)) != 0)
    return EINVAL;
  int result = ENOMEM;
  const long long big = eden_big_begin(size); /* EDEN_UP_BIG_STACKS: -1 when off */
  {
    void *block = eden_heap_aligned_try(alignment, size);
    /* EDEN_UP_HEAP_WAIT_MS: a pipeline builder asks again until it is served or the time is up. */
    unsigned limit;
    if (block == NULL && size != 0 && (limit = eden_heap_may_wait()) != 0) {
      struct eden_heap_wait_request request = {1, size, alignment, 0};
      block = eden_heap_wait_for(eden_heap_wait_aligned_again, &request, limit);
    }
    if (block != NULL) {
      *address = block;
      result = 0;
    }
  }
  if (result == 0) {
    eden_heap_record(*address, size);
    eden_site_alloc(*address, size);
    eden_big_end(*address, size, big);
  } else {
    eden_heap_failed(size, atomic_load_explicit(&eden_heap_grow_why, memory_order_relaxed));
    eden_big_end(NULL, size, big);
  }
  return result;
}

size_t __wrap_malloc_usable_size(const void *address) {
  return ps5_heap_owns(address) ? eden_heap_usable(address) : __real_malloc_usable_size(address);
}

