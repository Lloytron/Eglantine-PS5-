// SPDX-License-Identifier: GPL-3.0-or-later
// ---- The guest's memory as a service: shared until written (included by memory_pages.cpp) ----
//
// After Gregory Price's "A Compressed RAM Service" (Linux Plumbers 2026, mm/cram.c): memory that
// is read where it lies and promoted when it is written. There the device compresses; here the
// emulator is the device, and the two forms a processor can read without a fault are a slot that
// shares the block of zeroes and a slot that shares a block with another slot of the same bytes.
//
//   Zero    the slot shows the shared block of zeroes, read-only
//   Own     the slot has a block of its own, read-write
//   Shared  the slot shows a block it shares with other slots of the same bytes, read-only
//   Fine    the slot is 128 pages of 16 KiB, read-only: its pages of zeroes show the zero block, the
//           others a compact block that holds only them (a written slot with many pages of zeroes);
//           a write makes it whole again, and it waits longer each time before it may be Fine again
//   Probe   an Own slot that keeps its block but can be neither read nor written: the first touch of
//           either kind faults and makes it Own again, so one that stays Probe has not been touched
//   Out     a Probe slot untouched for long: its bytes are in a file, its block is gone, and its
//           address space is reserved; any touch faults and reads it back (EDEN_GUEST_SWAP, below)
//
// A write to a Zero or Shared slot faults; HandleGuestWriteFault gives the slot a block of its
// own (zeroed, or a copy) and the write repeats. So memory is no longer taken when the guest's
// kernel hands pages out, only when they are written; a slot the guest's kernel clears or frees
// whole goes back to the zero block; and a scanner gives back slots that hold only zeroes and
// lets slots of the same bytes share one block. A slot is locked (bit 7 of its state) while it
// changes; a writer that faults on a locked slot waits for it.
//
// 64-bit games only: a 32-bit game's fastmem window aliases the backing, which takes it whole.

namespace {
enum : std::uint8_t { SlotZero = 0, SlotOwn = 1, SlotShared = 2, SlotFine = 3, SlotProbe = 4, SlotOut = 5, SlotLock = 0x80, SlotMask = 0x7f };
constexpr std::size_t FinePage = 16384, FinePages = SparseSlot / FinePage;
// A written slot becomes Fine with at least this many pages of zeroes (32: 512 KiB given back).
// EDEN_CRAM_FINE_MIN_PAGES=<1 to 127> changes it (off unless asked for). EDEN_CRAM_ZEROS says what a
// lower number would give back. The price: a write to a Fine slot wants a whole 2 MiB block at once,
// so more Fine slots means more blocks wanted just when the pool is short.
std::size_t fine_min_zero_pages = 32;
// A slot made whole again by a write waits 1, 2, 4 ... 1024 scan passes before it may be Fine again.
constexpr unsigned FineBackoffMax = 10;

// Duplicate sharing is off unless asked for (dev-settings cram_dedup=on): on the console it found a
// dozen 2 MiB pairs that were written apart within a minute, and with it on a large game held 20 FPS
// instead of 30 (the GPU worker waited for every fence, once a frame) where it held 30 without.
std::atomic<bool> service_requested{true}, scan_requested{true}, dedup_requested{false}, fine_requested{true};
// EDEN_CRAM_FREE_RETURN=1: a whole slot the guest's kernel frees goes back then, not when its pages are
// next handed out (off unless asked for: no build has ever run with it, see GuestMemoryFreed).
std::atomic<bool> free_return_requested{false};
std::atomic<SparseRange*> service_range{nullptr}; // the guest range the service runs for
std::atomic<SparseRange*> probe_range{nullptr};   // the self-test's range
std::atomic<int> service_state{0};                // 0 not tested, 1 works here, 2 does not
const char* service_refused = "";

struct Counters {
    std::atomic<std::uint64_t> faults{0}, promotions{0}, copies{0}, waits{0};
    std::atomic<std::uint64_t> returns_clear{0}, returns_free{0}, returns_scan{0}, dedup{0};
    std::atomic<std::uint64_t> reserve_used{0}, host_reserve_used{0}, failures{0}, passes{0}, scan_ns{0};
    std::atomic<std::uint64_t> fined{0}, unfined{0}, fine_failures{0};
};
Counters counters;
// Fine slots made in place (EDEN_CRAM_FINE_IN_PLACE=1, below at MakeFine), and tails the system kept.
std::atomic<std::uint64_t> fined_in_place{0}, fine_tail_refused{0};
void ResetCounters() noexcept {
    fined_in_place.store(0, std::memory_order_relaxed);
    fine_tail_refused.store(0, std::memory_order_relaxed);
    for (auto* counter : {&counters.faults, &counters.promotions, &counters.copies, &counters.waits,
                          &counters.returns_clear, &counters.returns_free, &counters.returns_scan, &counters.dedup,
                          &counters.reserve_used, &counters.host_reserve_used, &counters.failures, &counters.passes, &counters.scan_ns,
                          &counters.fined, &counters.unfined, &counters.fine_failures})
        counter->store(0, std::memory_order_relaxed);
}

// ---- EDEN_UP_L961_ZERO_TRIM=1: zortos293/Lichii#961's schedule for the scanner (off unless set) ----
//
// Upstream (Ghostwolf13 #431, Zortos #961) takes resident all-zero guest pages out of the Windows
// working set, in every title, a pass every 5 s while a pass finds 64 MiB or more, backing off to
// once a minute. The scanner below already does the PS5's form of it for every title (a slot of
// zeroes goes back whole, a written slot with many 16 KiB pages of zeroes becomes Fine), and does it
// safely: the slot is fenced before the check, so a write meanwhile faults and waits. What it lacks
// is the schedule: one pass every 20 s, however much each pass finds. With the switch:
//   - a pass that gave back 64 MiB or more brings the next one in 5 s; otherwise the wait doubles,
//     up to EDEN_UP_L961_ZERO_TRIM_MAX_S (default 20, T83's fixed wait; 60 is upstream's);
//   - a pass that comes sooner than that after the last full one is a quick pass: it looks at Own
//     slots only. Armed slots (EDEN_GUEST_SWAP) cannot have changed while armed, and lending each one
//     costs two mprotects; full passes lend them, at T83's pace;
//   - the Fine backoff counts full passes, and only a full pass ends one, so a slot made whole waits
//     as long as it does in T83 (in full passes, at most MAX_S apart);
//   - EDEN_CRAM_ZEROS is kept from full passes only (a quick pass does not see armed slots).
// Readouts: EDEN_UP_L961 lines (each pass that gave back 64 MiB or more, and beside each EDEN_CRAM).
std::atomic<bool> l961_zero_trim{false};
std::atomic<unsigned> l961_max_s{20};
constexpr std::uint64_t L961Busy = std::uint64_t{64} << 20;
constexpr unsigned L961FastS = 5;
std::atomic<std::uint64_t> l961_whole{0}, l961_fine{0}, l961_zero_filled{0}; // bytes, this session
std::atomic<std::uint64_t> l961_passes{0}, l961_full{0}, l961_busy{0}, l961_interval_s{0};
std::atomic<std::uint64_t> full_passes{0}; // passes that lent armed slots (all of them without the switch)
// The pass count the Fine backoff runs on: every pass in T83, full passes with the switch.
std::uint64_t BackoffPasses() noexcept {
    return l961_zero_trim.load(std::memory_order_relaxed) ? full_passes.load(std::memory_order_relaxed)
                                                          : counters.passes.load(std::memory_order_relaxed);
}
void L961Add(std::atomic<std::uint64_t>& counter, std::uint64_t bytes) noexcept {
    if (l961_zero_trim.load(std::memory_order_relaxed)) counter.fetch_add(bytes, std::memory_order_relaxed);
}

bool LockSlot(SparseRange& range, std::size_t slot, bool count_wait = false) noexcept {
    auto& state = range.state[slot];
    bool waited = false;
    std::uint64_t wait_since = 0; // EDEN_UP_STALL_PROBE: when the spin began
    for (;;) {
        std::uint8_t value = state.load(std::memory_order_relaxed);
        if (!(value & SlotLock) &&
            state.compare_exchange_weak(value, value | SlotLock, std::memory_order_acquire, std::memory_order_relaxed))
            break;
        if (!waited && StallProbe::On()) wait_since = StallProbe::NowNs();
        waited = true;
        sched_yield();
    }
    if (waited && count_wait) counters.waits.fetch_add(1, std::memory_order_relaxed);
    if (waited) StallProbe::SlotWaited(wait_since);
    StallProbe::SlotTaken();
    return waited;
}
std::size_t FineKept(SparseRange& range, std::size_t slot) noexcept {
    return range.fine[slot] & 0xff;
}
std::uint8_t SlotState(SparseRange& range, std::size_t slot) noexcept {
    return range.state[slot].load(std::memory_order_acquire) & SlotMask;
}
// Sets the state and lets the slot go.
void UnlockSlot(SparseRange& range, std::size_t slot, std::uint8_t to) noexcept {
    StallProbe::SlotReleased();
    range.owned[slot].store(to != SlotZero, std::memory_order_release);
    range.state[slot].store(to, std::memory_order_release);
}
std::uintptr_t SlotAddress(SparseRange& range, std::size_t slot) noexcept {
    return range.begin.load(std::memory_order_relaxed) + slot * SparseSlot;
}

// ---- Blocks: the pool, a reserve for when the pool says no, and shared blocks' references ----
//
// The reserve is the guest's headroom in the pool, held as blocks of its own: 1 GiB that a game's
// writes take when the pool has no block left. It used to be 32 MiB here, plus 1 GiB that the
// graphics caches only counted as taken (headless/performance.cpp): that 1 GiB stayed in the pool,
// where Eden's heap, the driver and the caches themselves could take it, and where 2 MiB blocks
// could no longer be cut from what was left. Tears of the Kingdom's writes found no block four
// times with the pool's largest free block at 1 or 2 MiB. Held here, nothing else can take it or
// cut it up; the caches see the pool as it is, less what is not held yet, and evict as soon as a
// burst of writes has used half of it (performance.h ViewOfPool). It is topped up every 100 ms
// (ReserveKeeper) and given back with the guest's memory when the session ends.
constexpr std::size_t ReserveBlocks = 256;      // 512 MiB (diagnostic62 shared-pool test)
constexpr std::size_t ReserveAtStart = 16;      // taken as the service starts; the keeper takes the rest
// What the reserve is to hold, in blocks: ReserveBlocks, or EDEN_NX2_CRAM_RESERVE_MIB (read as the
// service starts; Aflakon, ocarina-cold-memory f0515ee16). Ocarina of Time in Kokiri Forest: the
// graphics allocator found no memory while the reserve still held 840 MiB that nothing used
// (crash-20261106-095505). The floor (EDEN_GUEST_RESERVE_FLOOR) should stay at or under it.
std::atomic<std::size_t> reserve_target{ReserveBlocks};
// The pool's largest block is under 1 GiB (headless/performance.cpp, every 100 ms). The reserve was
// only for when the pool says no, and that moment never came: the graphics allocations ran out first,
// with 512 MiB idle here (run 38, run 39: reserve=512, reserve_used=0) while the game's writes went
// on cutting 2 MiB blocks out of what was left for them. So while the pool is short the writes live
// off the reserve down to a floor, a refused buffer can have what is left above it (lend, below), and
// the keeper fills it no further than the floor. The floor is one for both, so the saving is one: at
// most the reserve less the floor. 128 blocks (256 MiB) until guest growth has been measured with it:
// a write that finds no block waits 2 s and then ends the session, and a host write that does (a
// fence's download holds the buffer cache's lock) stops the thread the collector runs on.
// EDEN_GUEST_RESERVE_FLOOR=<blocks>; 256 keeps the reserve as it was.
std::atomic<bool> pool_is_short{false};
std::atomic<std::size_t> reserve_floor{128};
std::int64_t reserve[ReserveBlocks];
std::atomic<std::size_t> reserve_count{0};
std::atomic_flag reserve_lock = ATOMIC_FLAG_INIT;
// EDEN_UP_RADV_OOM_LEND (eden_guest_reserve_lend_quiet, for the driver's out-of-memory callback): blocks
// lent under the floor stay lent for 2 s, the keeper filling the reserve no further than the floor that
// lend went down to until then, so it does not take them back before the driver has asked again. The
// lender writes them (any thread inside the driver, under the callback's mutex), the keeper reads them;
// 0 until the first such lend, and then the keeper only compares one time with the clock.
#ifdef PS5_NATIVE
std::atomic<std::uint64_t> keeper_hold_until_ns{0};
std::atomic<std::size_t> keeper_hold_floor{0};
#endif
constexpr std::size_t SharedBlocks = 4096;
std::int64_t shared_block[SharedBlocks];
int shared_refs[SharedBlocks];
std::atomic_flag shared_lock = ATOMIC_FLAG_INIT;

void Spin(std::atomic_flag& flag) noexcept {
    while (flag.test_and_set(std::memory_order_acquire)) sched_yield();
}

// ---- 0900 (T175c): the GPU threads, and EDEN_UP_FAULT_RESERVE (off unless set) ----
//
// The GPU threads name themselves (core patch 0900, eden_guest_gpu_thread_mark): GPUBack (1) when the
// GPU work is split, GPU (2). Kept as thread ids, so a fault handler asks with two compares.
template <typename Id>
std::uintptr_t ThreadKey(Id id) noexcept {
    if constexpr (std::is_pointer_v<Id>) return reinterpret_cast<std::uintptr_t>(id);
    else return static_cast<std::uintptr_t>(id);
}
std::atomic<std::uintptr_t> gpu_thread_ids[3]{};
// EDEN_UP_GPU_READ_THROUGH is on (set as the service starts; the fault handler asks it too, 0901).
std::atomic<bool> read_through_on{false};
int GpuThreadKind() noexcept {
    const std::uintptr_t self = ThreadKey(pthread_self());
    for (int kind = 1; kind <= 2; ++kind)
        if (gpu_thread_ids[kind].load(std::memory_order_relaxed) == self) return kind;
    return 0;
}
// A slot leaving Out (brought back by a fault, or given back) counts one here, before its lock goes:
// EDEN_UP_GPU_READ_THROUGH's unpacked copy of a slot is good while the slot's count is the same.
constexpr std::size_t OutGenSlots = 16384; // 32 GiB of slots; others are never kept unpacked
std::atomic<std::uint32_t> out_generation[OutGenSlots]{};
void NoteLeftOut(std::size_t slot) noexcept {
    if (slot < OutGenSlots) out_generation[slot].fetch_add(1, std::memory_order_acq_rel);
}

#ifdef PS5_NATIVE
// EDEN_UP_FAULT_RESERVE=<blocks> (1: 8 blocks; 2 to 64: that many): a second reserve of zeroed 2 MiB
// blocks that only the fault handler takes, for a slot it brings back (Out) or promotes, and only after
// the pool and the reserve above said no (ReserveBlock's last step). Nothing else asks it: the graphics
// lends, the driver's lend, the JIT's lend, the table reserve and the swap's packing see only the
// reserve above. The keeper fills it before the reserve above, every 100 ms, so it is the first to get
// memory when memory comes back. T166 runs 1399 and 1402: the reserve was 0 at the area load, and the
// GPU thread's fault on a packed slot found no block for 10 s. EDEN_UP_FAULT_GPU=1 (with it): the last
// FaultReserveGpuKeep blocks are for the GPU threads' faults only.
constexpr std::size_t FaultReserveMost = 64, FaultReserveDefault = 8, FaultReserveGpuKeep = 2;
std::atomic<std::size_t> fault_reserve_target{0}; // 0: off
std::int64_t fault_reserve[FaultReserveMost];
std::atomic<std::size_t> fault_reserve_count{0};
std::atomic_flag fault_reserve_lock = ATOMIC_FLAG_INIT;
std::atomic<bool> fault_gpu_first{false}; // EDEN_UP_FAULT_GPU
// Set by HandleGuestWriteFault around its promotions: only they may take from it.
thread_local bool fault_reserve_open = false;
struct FaultReserveCounters {
    std::atomic<std::uint64_t> taken{0}, taken_gpu{0}, empty{0}, kept_for_gpu{0}, filled{0};
    std::atomic<std::uint64_t> gpu_keep_waived{0}; // T178b: CPU takes of the last FaultReserveGpuKeep with read-through on
    std::atomic<std::size_t> lowest{FaultReserveMost};
};
FaultReserveCounters fault_reserve_counters;
bool FaultReserveBlock(std::int64_t* physical) noexcept {
    if (!fault_reserve_open || fault_reserve_target.load(std::memory_order_relaxed) == 0) return false;
    const bool gpu = GpuThreadKind() != 0;
    // T178b: with EDEN_UP_GPU_READ_THROUGH (=1 and up) the GPU threads read cold slots without a block, so
    // EDEN_UP_FAULT_GPU keeps nothing back from the game's CPU threads any more. run1418 (T177 F1): a CPU
    // thread's write fault waited 5,975 ms and died with kept_for_gpu=3203 while taken_gpu stayed 0.
    const bool keep_for_gpu = !gpu && fault_gpu_first.load(std::memory_order_relaxed);
    const bool waive = keep_for_gpu && read_through_on.load(std::memory_order_relaxed);
    const std::size_t keep = keep_for_gpu && !waive ? FaultReserveGpuKeep : 0;
    Spin(fault_reserve_lock);
    std::size_t count = fault_reserve_count.load(std::memory_order_relaxed);
    const bool ok = count > keep;
    if (ok) {
        *physical = fault_reserve[--count];
        fault_reserve_count.store(count, std::memory_order_relaxed);
    }
    fault_reserve_lock.clear(std::memory_order_release);
    auto& c = fault_reserve_counters;
    if (!ok) {
        (count != 0 ? c.kept_for_gpu : c.empty).fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (waive && count < FaultReserveGpuKeep) c.gpu_keep_waived.fetch_add(1, std::memory_order_relaxed);
    (gpu ? c.taken_gpu : c.taken).fetch_add(1, std::memory_order_relaxed);
    for (std::size_t low = c.lowest.load(std::memory_order_relaxed);
         count < low && !c.lowest.compare_exchange_weak(low, count, std::memory_order_relaxed);)
        ;
    return true;
}
#endif

// ---- T178b EDEN_UP_TABLE_RESERVE=<blocks> (off unless set) and the table commit's wait ----
//
// run1417 (T177, M1, 13 Nov): 18 s into the load into the area past the field, a page table slot of the
// device memory manager (SparseLargeVector<DeviceMemoryManager::TrackedEntry>::CommitPage <-
// DeviceMemoryManager::Map <- nvhost_as_gpu Nx2Map, on the service thread am:2) found no block: the pool's
// largest free piece 1 MiB, the guest reserve 0, and CommitSparsePage ended the app ("EDEN_SPARSE_TABLES
// commit failed: out of memory"). With EDEN_UP_TABLE_RESERVE=<1 to 64> (1: 8) the tables get a reserve of
// zeroed 2 MiB blocks of their own: filled as the service starts and by the keeper every 100 ms before every
// other reserve, and taken only by a table slot the pool refused (TableReserveBlock, then the guest reserve as
// before). Nothing else asks it: no lend, no fault, no bring-back, no packing. Emptied when the service stops.
// The wait (TableCommitWait, below at the fault's wait; always there, see memory_pages.cpp CommitSparsePage)
// follows the counters here too.
constexpr std::size_t TableReserveMost = 64, TableReserveDefault = 8;
struct TableCommitCounters {
    std::atomic<std::uint64_t> reserve_taken{0}, reserve_empty{0}, reserve_filled{0}, guest_reserve_taken{0};
    std::atomic<std::uint64_t> waits{0}, won{0}, lost{0}, wait_ms{0}, most_ms{0}, trims{0}, trim_bytes{0};
    std::atomic<std::uint64_t> give_backs{0}, given_bytes{0}, tries{0};
    std::atomic<std::size_t> lowest{SIZE_MAX}; // checker: SIZE_MAX = never taken (was 64: wrong at a target of 64)
};
TableCommitCounters table_commit_counters;
#ifdef PS5_NATIVE
std::atomic<std::size_t> table_reserve_target{0}; // 0: off
std::int64_t table_reserve[TableReserveMost];
std::atomic<std::size_t> table_reserve_count{0};
std::atomic_flag table_reserve_lock = ATOMIC_FLAG_INIT;
// A block of the tables' own reserve. Off: one relaxed load.
bool TableOwnReserveBlock(std::int64_t* physical) noexcept {
    if (table_reserve_target.load(std::memory_order_relaxed) == 0) return false;
    Spin(table_reserve_lock);
    std::size_t count = table_reserve_count.load(std::memory_order_relaxed);
    const bool ok = count > 0;
    if (ok) {
        *physical = table_reserve[--count];
        table_reserve_count.store(count, std::memory_order_relaxed);
    }
    table_reserve_lock.clear(std::memory_order_release);
    auto& c = table_commit_counters;
    if (!ok) {
        c.reserve_empty.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    c.reserve_taken.fetch_add(1, std::memory_order_relaxed);
    for (std::size_t low = c.lowest.load(std::memory_order_relaxed);
         count < low && !c.lowest.compare_exchange_weak(low, count, std::memory_order_relaxed);)
        ;
    return true;
}
#endif

#ifdef PS5_NATIVE
// A zeroed block from the reserve (the pool refused one). `last`: nothing is asked after this, so a
// fault's promotion may take the fault reserve (0900) when this one is empty.
bool ReserveBlock(std::int64_t* physical, bool last = true) noexcept {
    Spin(reserve_lock);
    const std::size_t count = reserve_count.load(std::memory_order_relaxed);
    const bool ok = count > 0;
    if (ok) {
        *physical = reserve[count - 1];
        reserve_count.store(count - 1, std::memory_order_relaxed);
    }
    reserve_lock.clear(std::memory_order_release);
    if (ok) counters.reserve_used.fetch_add(1, std::memory_order_relaxed);
    return ok || (last && FaultReserveBlock(physical));
}
// The same for a page table slot, which has no other way out.
bool TableReserveBlock(std::int64_t* physical) noexcept {
    if (TableOwnReserveBlock(physical)) return true; // T178b EDEN_UP_TABLE_RESERVE first (off: one relaxed load)
    const bool ok = std::getenv("EDEN_TABLE_NO_RESERVE") == nullptr && ReserveBlock(physical, false); // never the fault reserve (0900)
    if (ok) table_commit_counters.guest_reserve_taken.fetch_add(1, std::memory_order_relaxed);
    return ok;
}
// A zeroed block from the reserve first, while the pool is short.
bool SparedBlock(std::int64_t* physical) noexcept {
    return pool_is_short.load(std::memory_order_relaxed) &&
           reserve_count.load(std::memory_order_relaxed) > reserve_floor.load(std::memory_order_relaxed) &&
           ReserveBlock(physical, false);
}
// Zeroed blocks into the reserve until it holds `blocks` (at most ReserveBlocks) or the pool says no.
void FillReserve(std::size_t blocks = ReserveBlocks) noexcept {
    blocks = std::min({blocks, ReserveBlocks, reserve_target.load(std::memory_order_relaxed)});
    for (;;) {
        Spin(reserve_lock);
        const bool full = reserve_count.load(std::memory_order_relaxed) >= blocks;
        reserve_lock.clear(std::memory_order_release);
        if (full) return;
        std::int64_t physical = -1;
        if (!ZeroedBlock(&physical, true)) return;
        Spin(reserve_lock);
        const std::size_t count = reserve_count.load(std::memory_order_relaxed);
        if (count < blocks) {
            reserve[count] = physical;
            reserve_count.store(count + 1, std::memory_order_relaxed);
            physical = -1;
        }
        reserve_lock.clear(std::memory_order_release);
        if (physical >= 0 && sceKernelReleaseDirectMemory(physical, SparseSlot) != 0) std::abort();
    }
}
void EmptyReserve() noexcept {
    Spin(reserve_lock);
    for (std::size_t i = 0; i < reserve_count.load(std::memory_order_relaxed); ++i)
        if (sceKernelReleaseDirectMemory(reserve[i], SparseSlot) != 0) std::abort();
    reserve_count.store(0, std::memory_order_relaxed);
    reserve_lock.clear(std::memory_order_release);
    // 0900 EDEN_UP_FAULT_RESERVE's blocks go with it (0 held when the switch is off).
    Spin(fault_reserve_lock);
    for (std::size_t i = 0; i < fault_reserve_count.load(std::memory_order_relaxed); ++i)
        if (sceKernelReleaseDirectMemory(fault_reserve[i], SparseSlot) != 0) std::abort();
    fault_reserve_count.store(0, std::memory_order_relaxed);
    fault_reserve_lock.clear(std::memory_order_release);
}
// 0900 EDEN_UP_FAULT_RESERVE: zeroed blocks from the pool until it holds its target or the pool says
// no (the keeper, before the reserve above; and as the service starts). Off: one relaxed load.
void FillFaultReserve() noexcept {
    const std::size_t target = fault_reserve_target.load(std::memory_order_relaxed);
    if (target == 0) return;
    for (;;) {
        Spin(fault_reserve_lock);
        const bool full = fault_reserve_count.load(std::memory_order_relaxed) >= target;
        fault_reserve_lock.clear(std::memory_order_release);
        if (full) return;
        std::int64_t physical = -1;
        if (!ZeroedBlock(&physical, true)) return;
        Spin(fault_reserve_lock);
        const std::size_t count = fault_reserve_count.load(std::memory_order_relaxed);
        if (count < target) {
            fault_reserve[count] = physical;
            fault_reserve_count.store(count + 1, std::memory_order_relaxed);
            physical = -1;
        }
        fault_reserve_lock.clear(std::memory_order_release);
        if (physical < 0) fault_reserve_counters.filled.fetch_add(1, std::memory_order_relaxed);
        else if (sceKernelReleaseDirectMemory(physical, SparseSlot) != 0) std::abort();
    }
}
// T178b EDEN_UP_TABLE_RESERVE: zeroed blocks from the pool (where a table slot's own block comes from:
// ZeroedBlock, not the guest's band) until it holds its target or the pool says no. As the service starts,
// and from the keeper before every other reserve. Off: one relaxed load.
void FillTableReserve() noexcept {
    const std::size_t target = table_reserve_target.load(std::memory_order_relaxed);
    if (target == 0) return;
    for (;;) {
        Spin(table_reserve_lock);
        const bool full = table_reserve_count.load(std::memory_order_relaxed) >= target;
        table_reserve_lock.clear(std::memory_order_release);
        if (full) return;
        std::int64_t physical = -1;
        if (!ZeroedBlock(&physical, false)) return;
        Spin(table_reserve_lock);
        const std::size_t count = table_reserve_count.load(std::memory_order_relaxed);
        if (count < target) {
            table_reserve[count] = physical;
            table_reserve_count.store(count + 1, std::memory_order_relaxed);
            physical = -1;
        }
        table_reserve_lock.clear(std::memory_order_release);
        if (physical < 0) table_commit_counters.reserve_filled.fetch_add(1, std::memory_order_relaxed);
        else if (sceKernelReleaseDirectMemory(physical, SparseSlot) != 0) std::abort();
    }
}
void EmptyTableReserve() noexcept {
    Spin(table_reserve_lock);
    for (std::size_t i = 0; i < table_reserve_count.load(std::memory_order_relaxed); ++i)
        if (sceKernelReleaseDirectMemory(table_reserve[i], SparseSlot) != 0) std::abort();
    table_reserve_count.store(0, std::memory_order_relaxed);
    table_reserve_lock.clear(std::memory_order_release);
}
#endif

// One more slot shows `physical`; the first one makes it a shared block. False when the table is full.
bool ShareBlock(std::int64_t physical) noexcept {
    Spin(shared_lock);
    std::size_t free_entry = SharedBlocks;
    bool done = false;
    for (std::size_t i = 0; i < SharedBlocks && !done; ++i) {
        if (shared_refs[i] > 0 && shared_block[i] == physical) {
            ++shared_refs[i];
            done = true;
        } else if (shared_refs[i] == 0 && free_entry == SharedBlocks) {
            free_entry = i;
        }
    }
    if (!done && free_entry < SharedBlocks) {
        shared_block[free_entry] = physical;
        shared_refs[free_entry] = 1;
        done = true;
    }
    shared_lock.clear(std::memory_order_release);
    return done;
}
// One slot less shows `physical`: the references left (0: the block is no longer shown).
int UnshareBlock(std::int64_t physical) noexcept {
    Spin(shared_lock);
    int left = 0;
    for (std::size_t i = 0; i < SharedBlocks; ++i)
        if (shared_refs[i] > 0 && shared_block[i] == physical) {
            left = --shared_refs[i];
            break;
        }
    shared_lock.clear(std::memory_order_release);
    return left;
}

// ---- Cold slots (EDEN_GUEST_SWAP, off unless asked for) ----
//
// A processor's accessed bits are out of reach here, so a slot shows it is cold by what does not
// happen: it is armed (Probe) a few at a time, and the first touch comes back through the fault.
//   EDEN_GUEST_SWAP=1        measures: slots are armed and counted, nothing leaves memory
//   EDEN_GUEST_SWAP=2        also, while the pool is short, writes cold slots to a file
//   EDEN_GUEST_SWAP=3        also, while the pool is short, packs cold slots in memory (PackOut, below)
//   EDEN_GUEST_SWAP_FILE     the file (default /data/eglantineps5/guest-swap.bin)
//   EDEN_GUEST_SWAP_MIB      the most that may be in the file (default 2048)
//   EDEN_GUEST_SWAP_COLD_S   how long a slot stays untouched to count as cold (default 30)
//   EDEN_UP_SWAP_OUT_WHILE_BACK=1  a tick in which more than 8 slots came back used to send nothing
//                            out (no thrash). At an area change that is every tick for half a minute:
//                            on 10 Nov 2026 22:07 (leaving Kokiri Forest in free play) 499 slots came back
//                            in 18 s while 354-404 MiB of cold slots stayed in, and the pool ran dry.
//                            With the switch such a tick still sends out the slots that have been
//                            untouched for twice EDEN_GUEST_SWAP_COLD_S: they sat through the burst.
//   EDEN_GUEST_SWAP_ALL_POOLS=1  arms every Own slot, as run 59 did, not only the game's pool
//   EDEN_GUEST_SWAP_AHEAD_MIB=<n>  modes 2 and 3 also send cold slots out while the pool's largest
//                            free block is under n MiB, not only once the pool is short (under 1 GiB,
//                            headless/performance.cpp): the pool fell from 2.4 GiB to 0.2 GiB within
//                            20 s at Ocarina's first 3D frame (run 149), faster than cold slots can
//                            leave, and at an area change the slots that come back make the thrash
//                            guard hold everything. Packed in calm play, the room is there before.
//   EDEN_GUEST_SWAP_AHEAD_MIB=auto  picks n by firmware (kern.sdk_version): 12288 below 13, 2048 from 13
//                            on, and 2048 when the firmware cannot be read (below 12 is untested; the
//                            line says so). Any case, spaces before it allowed. 13.x consoles showed about
//                            1 GiB less free at the same point, and with 12288 they ran out of memory
//                            at the first door; 2048 is the value testers run there. A number still wins,
//                            and unset is still 0. One EDEN_SWAP_AHEAD line says what was chosen.
// Only the game's application pool is armed. The guest's kernel keeps real C++ objects in this
// backing too (KThread and the other slab heaps, with mutexes and condition variables in them), and
// the system's and the applets' pools hold what the emulator itself reads all the time. The core
// says where the game's pool lies (eden_guest_swap_pool); until it has, nothing is armed.
// The real GPU never sees this memory: no mapping of it has GPU flags and no Vulkan import takes it.
// The emulated GPU, the JIT and the emulator read it with the processor, and those faults come here.
// What cannot fault is a system call: given a pointer into a Probe or Out slot it returns EFAULT, so
// the core copies IPC buffers while this runs (tools/eglantine-patches/0125, eden_guest_swap_active).
constexpr unsigned SwapTickMs = 2000;
constexpr std::uint32_t SwapRearmTicks = 8;      // a touched slot stays Own this long (16 s)
constexpr std::size_t SwapArmShare = 8;          // an eighth of the slots is looked at each tick
constexpr std::size_t SwapOutPerTick = 64;       // at most 64 MiB/s to the file (128 MiB a tick)
constexpr std::uint64_t SwapBackInPerTick = 8;   // more reads back than this in a tick: nothing goes out
constexpr std::uint32_t SwapMostPlaces = 4096;   // 8 GiB of file
constexpr std::size_t SwapAges = 8;
std::atomic<int> swap_mode{0};                   // 0 off, 1 measure, 2 file, 3 packed in memory
// Slots that ever went out (core dma_pusher.cpp, EDEN_NX2_JUNK_TRACE, asks through
// eden_guest_swap_address_state): one bit a slot, 16384 slots cover a 32 GiB range.
constexpr std::size_t SwapEverOutSlots = 16384;
std::atomic<std::uint64_t> swap_ever_out[SwapEverOutSlots / 64]{};
std::atomic<std::uint32_t> swap_tick{SwapRearmTicks + 1};
std::uint32_t swap_cold_ticks = 15;
bool swap_out_while_back = false;                // EDEN_UP_SWAP_OUT_WHILE_BACK
// SwapOut from two threads (the swapper's tick, and a refused graphics allocation asking through
// eden_guest_memory_reclaim) one at a time: PackOut packs through one static buffer.
std::mutex swap_out_mutex;
bool swap_all_pools = false;
std::size_t swap_ahead_bytes = 0;                // EDEN_GUEST_SWAP_AHEAD_MIB, 0 = only while short
std::atomic<std::size_t> swap_pool_largest{0};   // the pool's largest free block at the last tick
std::atomic<std::uint64_t> swap_pool_first{0}, swap_pool_end{0}; // the slots of the game's pool
std::uint32_t swap_places = 1024;                // places in the file
const char* swap_path = "/data/eglantineps5/guest-swap.bin";
const char* swap_refused = "";
int swap_fd = -1;
std::uint32_t swap_free[SwapMostPlaces];
std::uint32_t swap_free_count = 0, swap_next_place = 0;
std::atomic_flag swap_lock = ATOMIC_FLAG_INIT;
struct SwapCounters {
    std::atomic<std::uint64_t> armed{0}, touched{0}, out{0}, in{0}, out_ns{0}, in_ns{0}, in_most_ns{0}, failures{0};
    std::atomic<std::uint64_t> age[SwapAges]{}; // touches by how long the slot was armed: under 2^n ticks
    std::atomic<std::uint64_t> sampled{0}, packed{0}; // bytes given to LZ4, and what it made of them
    std::atomic<std::uint64_t> returned_armed{0};     // slots given back while they were Probe
};
SwapCounters swap_counters;
// Packed slots (mode 3): an Out slot with a block (range.physical) holds its bytes LZ4-packed in it,
// range.swap being the packed size; one without (physical -1) is in the file. A slot is packed only
// when a quarter MiB comes back.
constexpr std::size_t PackMinGainPages = 16;
std::atomic<std::size_t> packed_pages_held{0};
std::atomic<std::uint64_t> pack_not_worth{0};
std::size_t PackedPages(std::uint32_t bytes) noexcept {
    return (bytes + FinePage - 1) / FinePage;
}
std::uint64_t NowNs() noexcept {
    timespec now{};
    clock_gettime(CLOCK_MONOTONIC, &now);
    return static_cast<std::uint64_t>(now.tv_sec) * 1000000000ull + static_cast<std::uint64_t>(now.tv_nsec);
}
// A place in the file, the lowest never used first: a file on exFAT only ever grows at its end.
bool TakeSwapPlace(std::uint32_t* place) noexcept {
    Spin(swap_lock);
    bool ok = true;
    if (swap_free_count != 0) *place = swap_free[--swap_free_count];
    else if (swap_next_place < swap_places) *place = swap_next_place++;
    else ok = false;
    swap_lock.clear(std::memory_order_release);
    return ok;
}
void GiveSwapPlace(std::uint32_t place) noexcept {
    Spin(swap_lock);
    swap_free[swap_free_count++] = place;
    swap_lock.clear(std::memory_order_release);
}

// ---- EDEN_UP_PACK_SHARED, EDEN_UP_PACK_PRESSURE_MS, EDEN_UP_PACK_ON_REFUSAL (off unless set) ----
//
// After PS5_Lichii's compactor by Mihawk (mihawk-99/PS5_Lichii v1.00.00, commit 2cc7711af,
// src/guest_memory_service.inc: the slab store, CompactStore, ScannerLoop's pace, PackCold and
// GuestMemoryPackNow; GPL-3.0-or-later; its frontend is ProsperoEden's). Read as the service starts
// (ReadPackSwitches), after the self-test, which keeps testing the old way.
//
//   EDEN_UP_PACK_SHARED=1 (needs EDEN_GUEST_SWAP=3): a packed slot's bytes go into a store of whole
//     2 MiB blocks (slabs) filled one copy after another, not into a block of 16 KiB pages of its own.
//     When the open slab has no room, the block of the slot being packed becomes the next one (its
//     bytes are in the packing buffer by then), so packing never asks the pool for memory. A slab goes
//     back with its last copy. Run 739 (t=129 s) held 464 MiB of packed bytes in 461 blocks of their
//     own, 496 MiB in 696 blocks under 2 MiB in all (EDEN_POOL_LEDGER): holes a 2 MiB block or a
//     graphics block cannot use. After each swap tick, while the slabs hold over a quarter more than
//     their copies plus 32 MiB, the copies of the emptiest closed slab (under half full) move into the
//     open one and it goes back whole (Lichii's CompactStore, up to 16 slabs a time). It wins over
//     EDEN_GUEST_SWAP_PACK_IN_PLACE. A copy's place is kept in range.physical (-2 and below; -1 is a
//     slot in the file, 0 and up a block), its size in range.swap. The refusal path below packs
//     written slots well only with it: PackOut asks the pool for a block for each slot's bytes.
//   EDEN_UP_PACK_PRESSURE_MS=<n> (50 to 2000; needs EDEN_GUEST_SWAP=2 or 3): every n ms while the
//     pool's largest free block is under EDEN_UP_PACK_PRESSURE_MIB (default 2560), armed slots untouched
//     for EDEN_UP_PACK_PRESSURE_COLD_S (default 4; in 2 s ticks, at least one) go out, the lowest in the
//     pool first, up to 256 MiB of slots a pass; a pass that found less arms every Own slot of the
//     game's pool the hand would arm (untouched for 16 s) for the next ones. Lichii: every 250 ms under
//     2.5 GiB free, 256 MiB a pass, Cold slots one pass old, then a sweep. Ours waits for a slot to be
//     untouched for 30 s (EDEN_GUEST_SWAP_COLD_S) on a 2 s tick: at run 739's door nothing was
//     ("untouched 30s=0MiB") while 2498 MiB lay armed.
//   EDEN_UP_PACK_ON_REFUSAL=1 (needs EDEN_GUEST_SWAP=2 or 3): a refused graphics allocation that asks
//     64 MiB or more of eden_guest_memory_reclaim (a buffer of 32 MiB or more: the port's
//     ReleaseRefused asks twice the buffer, and 32 MiB for an image) gets all of it there and then,
//     cold slots first, then slots armed this tick, then written (Own) slots of the game's pool, the
//     longest untouched first, armed and packed at once. Lichii's GuestMemoryPackNow does this for
//     64 MiB or more; there the caller asks again up to 156 times, so its sweep packs written slots on
//     the next call, while ours asks once. Without EDEN_GUEST_SWAP_RECLAIM, smaller requests get nothing.
// Pressure passes and refusals pass over a slot that would not pack in the last 300 s (Lichii's
// RefusedWait). The fault handler only drops copies (atomics; a slab's last copy releases it, system
// calls only), and a slab is never released while a view of it is mapped: a pack or an unpack holds
// its own copy, the compactor a reference of its own. Packing and compacting run under
// swap_out_mutex; a copy moves only under its slot's lock, the lock an unpack holds.
std::atomic<bool> pack_shared{false}, pack_on_refusal{false};
std::atomic<unsigned> pack_pressure_ms{0};
std::atomic<std::size_t> pack_pool_largest{0};
struct PackCounters {
    std::atomic<std::uint64_t> packed{0}, from_slot{0}, from_pool{0}, released{0}, moved{0}, compacted{0}, leaked{0};
    std::atomic<std::uint64_t> pressure_passes{0}, pressure_out{0}, pressure_freed{0}, pressure_armed{0};
    std::atomic<std::uint64_t> refusals{0}, refusal_out{0}, refusal_freed{0}, refusal_written{0}, refusal_short{0};
    std::atomic<std::uint64_t> refusal_ns{0}, refusal_most_ns{0};
};
PackCounters pack_counters;
#ifdef PS5_NATIVE
std::size_t pack_pressure_bytes = std::size_t{2560} << 20;
std::uint32_t pack_pressure_cold_ticks = 2;
constexpr std::size_t PackRefusalLarge = std::size_t{64} << 20;
constexpr std::size_t PackPressurePassBytes = std::size_t{256} << 20;
constexpr std::uint32_t PackRefusedTicks = 150; // 300 s
std::atomic<std::uint32_t> pack_skip_until[SwapEverOutSlots]{};
// What the last PackShared gave back to the pool (under swap_out_mutex): 2 MiB, or 0 when the slot's
// block became the open slab.
std::size_t pack_shared_given = 0;
bool PackSkipped(std::size_t slot, std::uint32_t now) noexcept {
    return slot < SwapEverOutSlots &&
           static_cast<std::int32_t>(pack_skip_until[slot].load(std::memory_order_relaxed) - now) > 0;
}
void PackSkip(std::size_t slot, std::uint32_t now) noexcept {
    if (slot < SwapEverOutSlots) pack_skip_until[slot].store(now + PackRefusedTicks, std::memory_order_relaxed);
}
void NoteEverOut(std::size_t slot) noexcept {
    if (slot < SwapEverOutSlots)
        swap_ever_out[slot / 64].fetch_or(std::uint64_t{1} << (slot % 64), std::memory_order_relaxed);
}
#endif
constexpr std::int64_t StorePlace(std::uint32_t slab, std::size_t offset) noexcept {
    return -2 - ((static_cast<std::int64_t>(slab) << 21) | static_cast<std::int64_t>(offset));
}
constexpr bool InStore(std::int64_t place) noexcept {
    return place <= -2;
}
constexpr std::uint32_t StoreSlab(std::int64_t place) noexcept {
    return static_cast<std::uint32_t>((-2 - place) >> 21);
}
constexpr std::size_t StoreOffset(std::int64_t place) noexcept {
    return static_cast<std::size_t>((-2 - place) & ((std::int64_t{1} << 21) - 1));
}
static_assert(SparseSlot == (std::size_t{1} << 21), "a copy's offset in its slab is 21 bits");
constexpr std::uint32_t CopyBytes(std::uint32_t size) noexcept {
    return (size + 63) & ~63u;
}
std::atomic<std::size_t> pack_store_slabs{0}, pack_store_bytes{0}, pack_store_copies{0};
#ifdef PS5_NATIVE
constexpr std::uint32_t PackSlabs = 4096; // 8 GiB of slabs
constexpr std::uint32_t SlabOpen = 1u << 31;
constexpr std::uint32_t NoSlab = UINT32_MAX;
struct PackSlab {
    std::atomic<bool> used{false};
    std::atomic<std::uint32_t> live{0};  // copies in it and views held, plus SlabOpen while it is the open one
    std::atomic<std::uint32_t> bytes{0}; // its copies, in 64-byte steps
    std::atomic<std::int64_t> physical{-1};
};
PackSlab pack_slabs[PackSlabs];
std::uint32_t open_slab = NoSlab; // under swap_out_mutex
std::size_t open_used = 0;        // under swap_out_mutex
// Its last copy or view went (also in the fault handler: system calls only).
void ReleaseSlab(std::uint32_t index) noexcept {
    PackSlab& slab = pack_slabs[index];
    const std::int64_t physical = slab.physical.exchange(-1, std::memory_order_acq_rel);
    if (physical >= 0 && sceKernelReleaseDirectMemory(physical, SparseSlot) != 0) std::abort();
    slab.bytes.store(0, std::memory_order_relaxed);
    guest_committed.fetch_sub(SparseSlot, std::memory_order_relaxed);
    pack_store_slabs.fetch_sub(1, std::memory_order_relaxed);
    pack_counters.released.fetch_add(1, std::memory_order_relaxed);
    slab.used.store(false, std::memory_order_release);
}
void LetGoSlab(std::uint32_t index) noexcept {
    if (pack_slabs[index].live.fetch_sub(1, std::memory_order_acq_rel) == 1) ReleaseSlab(index);
}
// One more reference to a slab that still has one (false: it is going or gone).
bool HoldSlab(std::uint32_t index) noexcept {
    auto& live = pack_slabs[index].live;
    std::uint32_t value = live.load(std::memory_order_acquire);
    while (value != 0)
        if (live.compare_exchange_weak(value, value + 1, std::memory_order_acq_rel, std::memory_order_acquire)) return true;
    return false;
}
// A copy goes (an unpack, a slot given back, the session's end): the slot is locked.
void DropCopy(std::uint32_t index, std::uint32_t size) noexcept {
    pack_slabs[index].bytes.fetch_sub(CopyBytes(size), std::memory_order_relaxed);
    pack_store_bytes.fetch_sub(size, std::memory_order_relaxed);
    pack_store_copies.fetch_sub(1, std::memory_order_relaxed);
    LetGoSlab(index);
}
std::uint32_t FreeSlabEntry() noexcept {
    for (std::uint32_t i = 0; i < PackSlabs; ++i)
        if (!pack_slabs[i].used.load(std::memory_order_acquire)) return i;
    return NoSlab;
}
// Under swap_out_mutex: `physical`, a whole block, is the open slab now (entry `index`, from
// FreeSlabEntry); the one before is closed and goes back if nothing is left in it. Not while a view
// of the one before is mapped.
void OpenSlab(std::uint32_t index, std::int64_t physical) noexcept {
    PackSlab& fresh = pack_slabs[index];
    fresh.physical.store(physical, std::memory_order_relaxed);
    fresh.bytes.store(0, std::memory_order_relaxed);
    fresh.live.store(SlabOpen, std::memory_order_relaxed);
    fresh.used.store(true, std::memory_order_release);
    guest_committed.fetch_add(SparseSlot, std::memory_order_relaxed);
    pack_store_slabs.fetch_add(1, std::memory_order_relaxed);
    const std::uint32_t closed = open_slab;
    open_slab = index;
    open_used = 0;
    if (closed != NoSlab && pack_slabs[closed].live.fetch_sub(SlabOpen, std::memory_order_acq_rel) == SlabOpen)
        ReleaseSlab(closed);
}
// The session's end (every copy dropped, no thread left to pack): the open slab is closed, and a slab
// still held is counted and given back.
void ClosePackStore() noexcept {
    const std::uint32_t closed = open_slab;
    open_slab = NoSlab;
    open_used = 0;
    if (closed != NoSlab && pack_slabs[closed].live.fetch_sub(SlabOpen, std::memory_order_acq_rel) == SlabOpen)
        ReleaseSlab(closed);
    std::size_t left = 0;
    for (std::uint32_t i = 0; i < PackSlabs; ++i)
        if (pack_slabs[i].used.load(std::memory_order_acquire)) {
            ++left;
            pack_slabs[i].live.store(0, std::memory_order_relaxed);
            ReleaseSlab(i);
        }
    pack_counters.leaked.fetch_add(left, std::memory_order_relaxed);
    char line[128];
    const int length = std::snprintf(line, sizeof(line), "EDEN_UP_PACK store closed: %zu slabs were still held, %llu released in all\n",
                                     left, (unsigned long long)pack_counters.released.load(std::memory_order_relaxed));
    if (length > 0) Report(line);
}
#endif

// ---- The changes of a slot (the slot is locked) ----

// Zero -> Own: a zeroed block of its own, read-write.
bool PromoteZero(SparseRange& range, std::size_t slot) noexcept {
    const std::uintptr_t at = SlotAddress(range, slot);
#ifdef PS5_NATIVE
    std::int64_t physical = -1;
    if (!SparedBlock(&physical) && !ZeroedBlock(&physical, true) && !ReserveBlock(&physical)) return false;
    if (!MapSlot(at, PROT_READ | PROT_WRITE, physical)) {
        (void)sceKernelReleaseDirectMemory(physical, SparseSlot);
        return false;
    }
    range.physical[slot] = physical;
#else
    if (mprotect(reinterpret_cast<void*>(at), SparseSlot, PROT_READ | PROT_WRITE) != 0) return false;
    range.physical[slot] = 0;
#endif
    guest_committed.fetch_add(SparseSlot, std::memory_order_relaxed);
    return true;
}

// Shared -> Own: the last slot to show a block takes it back read-write; the others copy it.
bool PromoteShared(SparseRange& range, std::size_t slot) noexcept {
    const std::uintptr_t at = SlotAddress(range, slot);
#ifdef PS5_NATIVE
    const std::int64_t shared = range.physical[slot];
    Spin(shared_lock);
    bool last = false;
    for (std::size_t i = 0; i < SharedBlocks; ++i)
        if (shared_refs[i] > 0 && shared_block[i] == shared) {
            last = shared_refs[i] == 1;
            if (last) shared_refs[i] = 0;
            break;
        }
    shared_lock.clear(std::memory_order_release);
    if (last) return mprotect(reinterpret_cast<void*>(at), SparseSlot, PROT_READ | PROT_WRITE) == 0;
    std::int64_t physical = -1;
    if (eden_direct_guest(SparseSlot, SparseSlot, 12, &physical) != 0 && !ReserveBlock(&physical)) return false;
    void* view = reinterpret_cast<void*>(slot_view_hint.load(std::memory_order_relaxed));
    if (sceKernelMapDirectMemory(&view, SparseSlot, PROT_READ | PROT_WRITE, 0, physical, SparseSlot) != 0) {
        (void)sceKernelReleaseDirectMemory(physical, SparseSlot);
        return false;
    }
    std::memcpy(view, reinterpret_cast<const void*>(at), SparseSlot);
    if (munmap(view, SparseSlot) != 0) std::abort();
    if (!MapSlot(at, PROT_READ | PROT_WRITE, physical)) {
        (void)sceKernelReleaseDirectMemory(physical, SparseSlot);
        return false;
    }
    range.physical[slot] = physical;
    if (UnshareBlock(shared) == 0 && sceKernelReleaseDirectMemory(shared, SparseSlot) != 0) std::abort();
    guest_committed.fetch_add(SparseSlot, std::memory_order_relaxed);
    counters.copies.fetch_add(1, std::memory_order_relaxed);
    return true;
#else
    (void)range;
    (void)slot;
    (void)at;
    return false; // the host never shares
#endif
}

// Own, Shared or Fine -> Zero: the slot shows the zero block again and its memory goes back.
void ReturnSlot(SparseRange& range, std::size_t slot) noexcept {
    const std::uintptr_t at = SlotAddress(range, slot);
    const std::uint8_t state = SlotState(range, slot);
    if (state == SlotOut) { // no block: its place in the file goes back; a packed block goes back itself
        NoteLeftOut(slot); // 0900: an unpacked copy of it is no longer good
#ifdef PS5_NATIVE
        if (!MapSlot(at, PROT_READ, zero_block)) std::abort();
        if (InStore(range.physical[slot])) { // EDEN_UP_PACK_SHARED: its copy in the store goes
            DropCopy(StoreSlab(range.physical[slot]), range.swap[slot]);
            range.physical[slot] = -1;
            return;
        }
        if (range.physical[slot] >= 0) {
            const std::size_t kept = PackedPages(range.swap[slot]);
            if (sceKernelReleaseDirectMemory(range.physical[slot], kept * FinePage) != 0) std::abort();
            guest_committed.fetch_sub(kept * FinePage, std::memory_order_relaxed);
            packed_pages_held.fetch_sub(kept, std::memory_order_relaxed);
            range.physical[slot] = -1;
            return;
        }
#endif
        GiveSwapPlace(range.swap[slot]);
        range.physical[slot] = -1;
        return;
    }
    if (state == SlotProbe) {
        // Given back as the Own slot it is: MapSlot over a mapping without access is a path nothing
        // has tested on the console, and this is the one every game takes when it frees in bulk.
        if (mprotect(reinterpret_cast<void*>(at), SparseSlot, PROT_READ | PROT_WRITE) != 0) std::abort();
        swap_counters.returned_armed.fetch_add(1, std::memory_order_relaxed);
    }
    if (state == SlotFine) {
        const std::size_t kept = FineKept(range, slot);
#ifdef PS5_NATIVE
        if (!MapSlot(at, PROT_READ, zero_block)) std::abort();
        if (kept != 0 && sceKernelReleaseDirectMemory(range.physical[slot], kept * FinePage) != 0) std::abort();
#else
        if (madvise(reinterpret_cast<void*>(at), SparseSlot, MADV_DONTNEED) != 0) std::abort();
#endif
        guest_committed.fetch_sub(kept * FinePage, std::memory_order_relaxed);
        range.fine[slot] &= 0xf00;
        range.physical[slot] = -1;
        return;
    }
#ifdef PS5_NATIVE
    if (!MapSlot(at, PROT_READ, zero_block)) std::abort();
    if (state == SlotShared) {
        if (UnshareBlock(range.physical[slot]) == 0) {
            if (sceKernelReleaseDirectMemory(range.physical[slot], SparseSlot) != 0) std::abort();
            guest_committed.fetch_sub(SparseSlot, std::memory_order_relaxed);
        }
    } else {
        if (sceKernelReleaseDirectMemory(range.physical[slot], SparseSlot) != 0) std::abort();
        guest_committed.fetch_sub(SparseSlot, std::memory_order_relaxed);
    }
#else
    (void)state;
    if (madvise(reinterpret_cast<void*>(at), SparseSlot, MADV_DONTNEED) != 0 ||
        mprotect(reinterpret_cast<void*>(at), SparseSlot, PROT_READ) != 0)
        std::abort();
    guest_committed.fetch_sub(SparseSlot, std::memory_order_relaxed);
#endif
    range.physical[slot] = -1;
}

bool AllZero(std::uintptr_t at) noexcept {
    const auto* word = reinterpret_cast<const volatile std::uint64_t*>(at);
    for (std::size_t i = 0; i < SparseSlot / 8; i += 8) {
        if (word[i] | word[i + 1] | word[i + 2] | word[i + 3] | word[i + 4] | word[i + 5] | word[i + 6] | word[i + 7])
            return false;
    }
    return true;
}

bool PageZero(std::uintptr_t at) noexcept {
    const auto* word = reinterpret_cast<const volatile std::uint64_t*>(at);
    for (std::size_t i = 0; i < FinePage / 8; i += 8)
        if (word[i] | word[i + 1] | word[i + 2] | word[i + 3] | word[i + 4] | word[i + 5] | word[i + 6] | word[i + 7])
            return false;
    return true;
}

// Fine -> Own: a written Fine slot becomes one block of 2 MiB again (a copy of what it shows), and
// stays whole for the rest of the session.
bool PromoteFine(SparseRange& range, std::size_t slot) noexcept {
    const std::uintptr_t at = SlotAddress(range, slot);
    const std::size_t kept = FineKept(range, slot);
#ifdef PS5_NATIVE
    std::int64_t physical = -1;
    if (!SparedBlock(&physical) && eden_direct_guest(SparseSlot, SparseSlot, 12, &physical) != 0 &&
        !ReserveBlock(&physical))
        return false;
    void* view = reinterpret_cast<void*>(slot_view_hint.load(std::memory_order_relaxed));
    if (sceKernelMapDirectMemory(&view, SparseSlot, PROT_READ | PROT_WRITE, 0, physical, SparseSlot) != 0) {
        (void)sceKernelReleaseDirectMemory(physical, SparseSlot);
        return false;
    }
    std::memcpy(view, reinterpret_cast<const void*>(at), SparseSlot);
    if (munmap(view, SparseSlot) != 0) std::abort();
    if (!MapSlot(at, PROT_READ | PROT_WRITE, physical)) {
        (void)sceKernelReleaseDirectMemory(physical, SparseSlot);
        return false;
    }
    if (kept != 0 && sceKernelReleaseDirectMemory(range.physical[slot], kept * FinePage) != 0) std::abort();
    range.physical[slot] = physical;
#else
    if (mprotect(reinterpret_cast<void*>(at), SparseSlot, PROT_READ | PROT_WRITE) != 0) return false;
#endif
    guest_committed.fetch_add(SparseSlot - kept * FinePage, std::memory_order_relaxed);
    const unsigned times = std::min<unsigned>(((range.fine[slot] >> 8) & 0xf) + 1, FineBackoffMax);
    range.fine[slot] = static_cast<std::uint16_t>(times << 8);
    range.fine_after[slot] = static_cast<std::uint32_t>(BackoffPasses()) + (1u << times);
    counters.unfined.fetch_add(1, std::memory_order_relaxed);
    return true;
}

#ifdef PS5_NATIVE
bool MapRun(std::uintptr_t at, std::size_t bytes, std::int64_t physical) noexcept {
    void* address = reinterpret_cast<void*>(at);
    return sceKernelMapDirectMemory(&address, bytes, PROT_READ, MAP_FIXED, physical, FinePage) == 0 &&
           address == reinterpret_cast<void*>(at);
}

// EDEN_CRAM_FINE_IN_PLACE=1 (off unless asked for): a Fine slot keeps its own block. Its kept pages move
// to the front of the block and the tail goes back to the pool, so no block is asked of the pool at the
// moment the pool has none: run 101 (Mario Kart) counted 257 Fine failures, all after the largest free
// block reached 0, about 130 slots a pass with some 160 MiB of zero pages between them that stayed held.
// Asked once, at the first Fine slot: a block's second half is released on its own, then its first. A
// system that refuses the tail leaves the switch off for the session (fine_in_place=0 in the report).
std::atomic<int> fine_in_place{0}; // 0 not asked yet, 1 on, 2 off
bool FineInPlace() noexcept {
    const int state = fine_in_place.load(std::memory_order_relaxed);
    if (state != 0) return state == 1;
    const char* wanted = std::getenv("EDEN_CRAM_FINE_IN_PLACE");
    if (wanted == nullptr || wanted[0] != '1') {
        fine_in_place.store(2, std::memory_order_relaxed);
        return false;
    }
    std::int64_t block = -1;
    if (eden_direct_guest(SparseSlot, SparseSlot, 12, &block) != 0) return false; // no block now: asked again later
    const int tail = sceKernelReleaseDirectMemory(block + SparseSlot / 2, SparseSlot / 2);
    if (sceKernelReleaseDirectMemory(block, tail == 0 ? SparseSlot / 2 : SparseSlot) != 0) std::abort();
    fine_in_place.store(tail == 0 ? 1 : 2, std::memory_order_relaxed);
    char line[128];
    const int length = std::snprintf(line, sizeof(line), "EDEN_CRAM_FINE_IN_PLACE %s: a block's tail %s (result %d)\n",
                                     tail == 0 ? "on" : "off", tail == 0 ? "went back on its own" : "was refused", tail);
    if (length > 0) Report(line);
    return tail == 0;
}

// The kept pages to the front of the slot's own block, or (`back`) where they were, with the zero pages
// zeroed again. Through a view: the slot is unmapped meanwhile, so no reader sees a page half moved.
// Forward, a page only ever moves down, over a zero page or a page already moved; back, in the other
// order, so nothing is overwritten before it is read.
bool MovePagesInPlace(std::int64_t whole, const bool (&zero)[FinePages], std::size_t kept, bool back) noexcept {
    void* view = reinterpret_cast<void*>(slot_view_hint.load(std::memory_order_relaxed));
    if (sceKernelMapDirectMemory(&view, SparseSlot, PROT_READ | PROT_WRITE, 0, whole, SparseSlot) != 0) return false;
    auto* bytes = static_cast<std::uint8_t*>(view);
    if (!back) {
        std::size_t next = 0;
        for (std::size_t page = 0; page < FinePages; ++page)
            if (!zero[page]) {
                if (next != page) std::memcpy(bytes + next * FinePage, bytes + page * FinePage, FinePage);
                ++next;
            }
    } else {
        std::size_t next = kept;
        for (std::size_t page = FinePages; page-- > 0;)
            if (!zero[page]) {
                --next;
                if (next != page) std::memcpy(bytes + page * FinePage, bytes + next * FinePage, FinePage);
            }
        for (std::size_t page = 0; page < FinePages; ++page)
            if (zero[page]) std::memset(bytes + page * FinePage, 0, FinePage);
    }
    if (munmap(view, SparseSlot) != 0) std::abort();
    return true;
}
#endif

// Own -> Fine (the slot is locked and fenced, and has `zero` marked in a mask of its pages): its other
// pages go into a compact block, and the slot is mapped again page run by page run, read-only.
// The 2 MiB mapping is taken away whole first (a large mapping is never split); a thread that
// touches the slot meanwhile faults, waits for the lock and repeats its access.
// In place (FineInPlace), the compact block is the slot's own: its tail goes back at the end.
bool MakeFine(SparseRange& range, std::size_t slot, const bool (&zero)[FinePages], std::size_t kept) noexcept {
    const std::uintptr_t at = SlotAddress(range, slot);
#ifdef PS5_NATIVE
    const std::int64_t whole = range.physical[slot];
    const bool in_place = FineInPlace();
    std::int64_t compact = whole;
    if (in_place) {
        if (munmap(reinterpret_cast<void*>(at), SparseSlot) != 0) std::abort();
        if (!MovePagesInPlace(whole, zero, kept, false)) {
            if (!MapSlot(at, PROT_READ, whole)) std::abort();
            return false;
        }
    } else {
        if (eden_direct_guest(kept * FinePage, FinePage, 12, &compact) != 0) return false;
        void* view = reinterpret_cast<void*>(slot_view_hint.load(std::memory_order_relaxed));
        if (sceKernelMapDirectMemory(&view, kept * FinePage, PROT_READ | PROT_WRITE, 0, compact, FinePage) != 0) {
            (void)sceKernelReleaseDirectMemory(compact, kept * FinePage);
            return false;
        }
        std::size_t next = 0;
        for (std::size_t page = 0; page < FinePages; ++page)
            if (!zero[page]) {
                std::memcpy(static_cast<std::uint8_t*>(view) + next * FinePage,
                            reinterpret_cast<const void*>(at + page * FinePage), FinePage);
                ++next;
            }
        if (munmap(view, kept * FinePage) != 0) std::abort();
        if (munmap(reinterpret_cast<void*>(at), SparseSlot) != 0) std::abort();
    }
    bool ok = true;
    std::size_t next = 0;
    for (std::size_t page = 0; page < FinePages && ok;) {
        std::size_t end = page + 1;
        while (end < FinePages && zero[end] == zero[page]) ++end;
        const std::size_t bytes = (end - page) * FinePage;
        ok = MapRun(at + page * FinePage, bytes, zero[page] ? zero_block : compact + std::int64_t(next * FinePage));
        if (!zero[page]) next += end - page;
        page = end;
    }
    if (!ok) {
        // The slot shows its block again, as it was (fenced: read-only until the caller unfences).
        if (in_place && !MovePagesInPlace(whole, zero, kept, true)) std::abort();
        if (!MapSlot(at, PROT_READ, whole)) std::abort();
        if (!in_place) (void)sceKernelReleaseDirectMemory(compact, kept * FinePage);
        return false;
    }
    if (in_place) {
        // The tail is nobody's now. A refusal (never seen after the check above) keeps it held and counted.
        if (sceKernelReleaseDirectMemory(whole + std::int64_t(kept * FinePage), SparseSlot - kept * FinePage) != 0) {
            fine_tail_refused.fetch_add(1, std::memory_order_relaxed);
            guest_committed.fetch_add(SparseSlot - kept * FinePage, std::memory_order_relaxed);
        }
        fined_in_place.fetch_add(1, std::memory_order_relaxed);
    } else {
        if (sceKernelReleaseDirectMemory(whole, SparseSlot) != 0) std::abort();
    }
    range.physical[slot] = compact;
#else
    for (std::size_t page = 0; page < FinePages; ++page)
        if (zero[page] && madvise(reinterpret_cast<void*>(at + page * FinePage), FinePage, MADV_DONTNEED) != 0)
            std::abort();
#endif
    guest_committed.fetch_sub(SparseSlot - kept * FinePage, std::memory_order_relaxed);
    range.fine[slot] = static_cast<std::uint16_t>((range.fine[slot] & 0xf00) | kept);
    counters.fined.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// Probe -> Own: it was touched. Counted by how long it lay untouched, the measure of what is cold.
bool TouchProbe(SparseRange& range, std::size_t slot) noexcept {
    const int saved = errno;
    const bool ok = mprotect(reinterpret_cast<void*>(SlotAddress(range, slot)), SparseSlot, PROT_READ | PROT_WRITE) == 0;
    errno = saved;
    if (!ok) return false;
    const std::uint32_t now = swap_tick.load(std::memory_order_relaxed), ticks = now - range.swap[slot];
    std::size_t bucket = 0;
    while (bucket + 1 < SwapAges && (std::uint64_t{1} << (bucket + 1)) <= ticks) ++bucket;
    swap_counters.age[bucket].fetch_add(1, std::memory_order_relaxed);
    swap_counters.touched.fetch_add(1, std::memory_order_relaxed);
    range.swap[slot] = now;
    return true;
}

#ifdef PS5_NATIVE
bool SwapRead(void* to, std::uint32_t place) noexcept {
    auto* at = static_cast<std::uint8_t*>(to);
    std::size_t done = 0;
    while (done < SparseSlot) {
        const auto got = pread(swap_fd, at + done, SparseSlot - done, static_cast<off_t>(place) * SparseSlot + done);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return false;
        done += static_cast<std::size_t>(got);
    }
    return true;
}
bool SwapWrite(const void* from, std::uint32_t place) noexcept {
    const auto* at = static_cast<const std::uint8_t*>(from);
    std::size_t done = 0;
    while (done < SparseSlot) {
        const auto put = pwrite(swap_fd, at + done, SparseSlot - done, static_cast<off_t>(place) * SparseSlot + done);
        if (put < 0 && errno == EINTR) continue;
        if (put <= 0) return false;
        done += static_cast<std::size_t>(put);
    }
    return true;
}
#endif

#if defined(PS5_NATIVE) && defined(EDEN_SWAP_LZ4)
// ---- Packed slots (EDEN_GUEST_SWAP=3): a cold slot's bytes, LZ4-packed, in a smaller block ----
//
// Mode 2 without the drive: Probe -> Out keeps the packed bytes in a block of 16 KiB pages of its own
// and the 2 MiB block goes back; a touch unpacks them into a fresh block in the fault. Run 59's samples
// (Zelda) packed to 61%: a slot gives back about 0.78 MiB, 2.6 times less than the file, for a fault
// of about a millisecond and no drive. A slot that packs worse than a quarter MiB, or finds no block
// for its bytes, is armed anew and looked at again when it is cold again. The swapper is one thread,
// so the packing buffer is one, static (2 MiB of flexible memory).
//
// EDEN_GUEST_SWAP_PACK_IN_PLACE=1 (off unless asked for): the packed bytes go to the front of the
// slot's own block and the block's tail goes back, so no block is asked of the pool. Run 161 (Ocarina,
// hand play) reached the first 3D frame with the pool at 0 MiB for ten seconds: three ticks found no
// block for the packed bytes and 900 MiB of cold slots waited. Asked once, at the first packed slot,
// as EDEN_CRAM_FINE_IN_PLACE asks: a block's second half is released on its own, then its first (run
// 126 saw the console take a tail back, 1,162 slots). A refusal leaves the switch off for the session.
//
// 0870 EDEN_GUEST_SWAP_PACK_IN_PLACE=2 (waveydude's W-20; src/pack_place.h): a block of its own first, in place only when the
// pool has none. A tail is a hole under 2 MiB that only another packed copy can take: W-19 stopped
// with 1,138 slots packed in place, 1,342 MiB of the pool free and its largest free block 3 MiB, and
// every crash from W-13 on had 430 to 1,245 MiB free the same way. A copy in a block of its own lands
// in the lowest hole that fits it (an older tail), and its slot's block goes back whole.
std::atomic<int> pack_in_place{0}; // 0 not asked yet, 1 on, 2 off
std::atomic<bool> pack_pool_first{false}; // =2, read with the first packed slot (PackInPlace)
std::atomic<std::uint64_t> packed_in_place{0}, pack_tail_refused{0}, packed_pool_first{0};
bool PackInPlace() noexcept {
    const int state = pack_in_place.load(std::memory_order_relaxed);
    if (state != 0) return state == 1;
    const Eden::PackPlace::Mode wanted = Eden::PackPlace::ModeOf(std::getenv("EDEN_GUEST_SWAP_PACK_IN_PLACE"));
    if (wanted == Eden::PackPlace::Mode::Own) {
        pack_in_place.store(2, std::memory_order_relaxed);
        return false;
    }
    pack_pool_first.store(wanted == Eden::PackPlace::Mode::PoolFirst, std::memory_order_relaxed);
    std::int64_t block = -1;
    if (eden_direct_guest(SparseSlot, SparseSlot, 12, &block) != 0) return false; // no block now: asked again later
    const int tail = sceKernelReleaseDirectMemory(block + SparseSlot / 2, SparseSlot / 2);
    if (sceKernelReleaseDirectMemory(block, tail == 0 ? SparseSlot / 2 : SparseSlot) != 0) std::abort();
    pack_in_place.store(tail == 0 ? 1 : 2, std::memory_order_relaxed);
    char line[128];
    const int length = std::snprintf(line, sizeof(line), "EDEN_SWAP_PACK_IN_PLACE %s: a block's tail %s (result %d)\n",
                                     tail != 0 ? "off" : wanted == Eden::PackPlace::Mode::PoolFirst ? "on when the pool has no block" : "on",
                                     tail == 0 ? "went back on its own" : "was refused", tail);
    if (length > 0) Report(line);
    return tail == 0;
}
// The packing buffer (2 MiB of flexible memory), one for PackOut and EDEN_UP_PACK_SHARED's PackShared:
// both run under swap_out_mutex (the self-test packs before the swapper starts).
char pack_buffer[LZ4_COMPRESSBOUND(SparseSlot)];
bool PackOut(SparseRange& range, std::size_t slot) noexcept {
    auto& packed = pack_buffer;
    LockSlot(range, slot);
    if (SlotState(range, slot) != SlotProbe) {
        UnlockSlot(range, slot, SlotState(range, slot));
        return false;
    }
    const std::uintptr_t at = SlotAddress(range, slot);
    const std::uint64_t start = NowNs();
    if (mprotect(reinterpret_cast<void*>(at), SparseSlot, PROT_READ) != 0) std::abort();
    const int size = LZ4_compress_default(reinterpret_cast<const char*>(at), packed, static_cast<int>(SparseSlot),
                                          static_cast<int>(sizeof(packed)));
    swap_counters.sampled.fetch_add(SparseSlot, std::memory_order_relaxed);
    swap_counters.packed.fetch_add(size > 0 ? static_cast<std::uint64_t>(size) : SparseSlot, std::memory_order_relaxed);
    const std::size_t kept = size > 0 ? PackedPages(static_cast<std::uint32_t>(size)) : FinePages;
    const bool worth = kept + PackMinGainPages <= FinePages;
    std::int64_t block = -1;
    void* view = reinterpret_cast<void*>(slot_view_hint.load(std::memory_order_relaxed));
    using Eden::PackPlace::Mode;
    using Eden::PackPlace::Place;
    const Mode mode = !PackInPlace() ? Mode::Own
                      : pack_pool_first.load(std::memory_order_relaxed) ? Mode::PoolFirst : Mode::InPlace;
    // A block for the packed bytes, with a view to write them through. Nothing is held on a refusal.
    bool have_block = false;
    if (Eden::PackPlace::AsksPoolFirst(worth, mode)) {
        if (eden_direct_guest(kept * FinePage, FinePage, 12, &block) == 0) {
            have_block = sceKernelMapDirectMemory(&view, kept * FinePage, PROT_READ | PROT_WRITE, 0, block, FinePage) == 0;
            if (!have_block) {
                (void)sceKernelReleaseDirectMemory(block, kept * FinePage);
                view = reinterpret_cast<void*>(slot_view_hint.load(std::memory_order_relaxed));
            }
        }
        if (!have_block) block = -1;
    }
    const Place place = Eden::PackPlace::Choose(worth, mode, have_block);
    if (place == Place::InPlace) {
        // The slot's own block keeps the packed bytes in its first `kept` pages. The tail goes back
        // first, while the packed copy holds every byte: a refusal leaves the block whole and the slot
        // armed, nothing lost. Then the front is written through a view, as MakeFine writes a block.
        const std::int64_t whole = range.physical[slot];
        if (munmap(reinterpret_cast<void*>(at), SparseSlot) != 0) std::abort();
        void* again = reinterpret_cast<void*>(at);
        if (sceKernelReserveVirtualRange(&again, SparseSlot, MAP_FIXED, SparseSlot) != 0 || again != reinterpret_cast<void*>(at))
            std::abort();
        if (sceKernelReleaseDirectMemory(whole + std::int64_t(kept * FinePage), SparseSlot - kept * FinePage) != 0) {
            if (!MapSlot(at, PROT_READ, whole)) std::abort();
            if (mprotect(reinterpret_cast<void*>(at), SparseSlot, PROT_NONE) != 0) std::abort();
            range.swap[slot] = swap_tick.load(std::memory_order_relaxed);
            pack_tail_refused.fetch_add(1, std::memory_order_relaxed);
            swap_counters.failures.fetch_add(1, std::memory_order_relaxed);
            UnlockSlot(range, slot, SlotProbe);
            return false;
        }
        if (sceKernelMapDirectMemory(&view, kept * FinePage, PROT_READ | PROT_WRITE, 0, whole, FinePage) != 0) std::abort();
        std::memcpy(view, packed, static_cast<std::size_t>(size));
        if (munmap(view, kept * FinePage) != 0) std::abort();
        range.swap[slot] = static_cast<std::uint32_t>(size);
        guest_committed.fetch_sub(SparseSlot - kept * FinePage, std::memory_order_relaxed);
        packed_pages_held.fetch_add(kept, std::memory_order_relaxed);
        packed_in_place.fetch_add(1, std::memory_order_relaxed);
        swap_counters.out.fetch_add(1, std::memory_order_relaxed);
        swap_counters.out_ns.fetch_add(NowNs() - start, std::memory_order_relaxed);
        UnlockSlot(range, slot, SlotOut);
        return true;
    }
    if (place != Place::OwnBlock) {
        if (mprotect(reinterpret_cast<void*>(at), SparseSlot, PROT_NONE) != 0) std::abort();
        range.swap[slot] = swap_tick.load(std::memory_order_relaxed);
        (worth ? swap_counters.failures : pack_not_worth).fetch_add(1, std::memory_order_relaxed);
        UnlockSlot(range, slot, SlotProbe);
        return false;
    }
    std::memcpy(view, packed, static_cast<std::size_t>(size));
    if (munmap(view, kept * FinePage) != 0) std::abort();
    const std::int64_t whole = range.physical[slot];
    if (munmap(reinterpret_cast<void*>(at), SparseSlot) != 0) std::abort();
    void* again = reinterpret_cast<void*>(at);
    if (sceKernelReserveVirtualRange(&again, SparseSlot, MAP_FIXED, SparseSlot) != 0 || again != reinterpret_cast<void*>(at))
        std::abort();
    if (sceKernelReleaseDirectMemory(whole, SparseSlot) != 0) std::abort();
    range.physical[slot] = block;
    range.swap[slot] = static_cast<std::uint32_t>(size);
    guest_committed.fetch_sub(SparseSlot - kept * FinePage, std::memory_order_relaxed);
    packed_pages_held.fetch_add(kept, std::memory_order_relaxed);
    if (mode == Mode::PoolFirst) packed_pool_first.fetch_add(1, std::memory_order_relaxed);
    swap_counters.out.fetch_add(1, std::memory_order_relaxed);
    swap_counters.out_ns.fetch_add(NowNs() - start, std::memory_order_relaxed);
    UnlockSlot(range, slot, SlotOut);
    return true;
}

// Out (packed) -> Own, in a fault handler: a block, the bytes unpacked into it through two views, and
// the slot shows them in one step. The packed block goes back. Nothing here allocates.
bool UnpackIn(SparseRange& range, std::size_t slot, std::int64_t given = -1) noexcept {
    const int saved = errno;
    const std::uint64_t start = NowNs();
    const std::int64_t packed_block = range.physical[slot];
    const std::uint32_t size = range.swap[slot];
    const std::size_t kept = PackedPages(size);
    std::int64_t physical = given; // T177a: a block the caller took already (-1: none, asked here as before)
    if (physical < 0 && !SparedBlock(&physical) && eden_direct_guest(SparseSlot, SparseSlot, 12, &physical) != 0 &&
        !ReserveBlock(&physical)) {
        errno = saved;
        return false;
    }
    void* to = reinterpret_cast<void*>(slot_view_hint.load(std::memory_order_relaxed));
    void* from = to;
    bool ok = sceKernelMapDirectMemory(&to, SparseSlot, PROT_READ | PROT_WRITE, 0, physical, SparseSlot) == 0;
    if (ok) {
        ok = sceKernelMapDirectMemory(&from, kept * FinePage, PROT_READ, 0, packed_block, FinePage) == 0;
        if (ok) {
            ok = LZ4_decompress_safe(static_cast<const char*>(from), static_cast<char*>(to), static_cast<int>(size),
                                     static_cast<int>(SparseSlot)) == static_cast<int>(SparseSlot);
            if (munmap(from, kept * FinePage) != 0) std::abort();
        }
        if (munmap(to, SparseSlot) != 0) std::abort();
    }
    if (!ok || !MapSlot(SlotAddress(range, slot), PROT_READ | PROT_WRITE, physical)) {
        (void)sceKernelReleaseDirectMemory(physical, SparseSlot);
        swap_counters.failures.fetch_add(1, std::memory_order_relaxed);
        Note(ok ? "EDEN_SWAP a packed slot could not be mapped again\n" : "EDEN_SWAP a packed slot could not be unpacked\n");
        errno = saved;
        return false;
    }
    range.physical[slot] = physical;
    range.swap[slot] = swap_tick.load(std::memory_order_relaxed);
    if (sceKernelReleaseDirectMemory(packed_block, kept * FinePage) != 0) std::abort();
    guest_committed.fetch_add(SparseSlot - kept * FinePage, std::memory_order_relaxed);
    packed_pages_held.fetch_sub(kept, std::memory_order_relaxed);
    const std::uint64_t took = NowNs() - start;
    swap_counters.in.fetch_add(1, std::memory_order_relaxed);
    swap_counters.in_ns.fetch_add(took, std::memory_order_relaxed);
    if (took > swap_counters.in_most_ns.load(std::memory_order_relaxed))
        swap_counters.in_most_ns.store(took, std::memory_order_relaxed);
    errno = saved;
    return true;
}

// EDEN_UP_PACK_SHARED: Probe -> Out into the store, under swap_out_mutex (pack_shared_given says what
// went back). Fenced while it is packed, as PackOut packs. The copy goes through a view into the open
// slab or, when that has no room, into the slot's own block, which becomes the open slab; nothing is
// written there before the slot stops showing it. A view that cannot be made leaves the slot armed,
// as a block PackOut cannot get does. One that packs worse than a quarter MiB back is armed anew, as
// in PackOut, and passed over by pressure passes and refusals for 300 s.
bool PackShared(SparseRange& range, std::size_t slot) noexcept {
    auto& packed = pack_buffer; // PackOut's (under swap_out_mutex): no second 2 MiB
    pack_shared_given = 0;
    LockSlot(range, slot);
    if (SlotState(range, slot) != SlotProbe) {
        UnlockSlot(range, slot, SlotState(range, slot));
        return false;
    }
    const std::uintptr_t at = SlotAddress(range, slot);
    const std::uint64_t start = NowNs();
    const std::uint32_t now = swap_tick.load(std::memory_order_relaxed);
    if (mprotect(reinterpret_cast<void*>(at), SparseSlot, PROT_READ) != 0) std::abort();
    const int size = LZ4_compress_default(reinterpret_cast<const char*>(at), packed, static_cast<int>(SparseSlot),
                                          static_cast<int>(sizeof(packed)));
    swap_counters.sampled.fetch_add(SparseSlot, std::memory_order_relaxed);
    swap_counters.packed.fetch_add(size > 0 ? static_cast<std::uint64_t>(size) : SparseSlot, std::memory_order_relaxed);
    const std::size_t need = size > 0 ? CopyBytes(static_cast<std::uint32_t>(size)) : SparseSlot;
    const bool worth = size > 0 && need + PackMinGainPages * FinePage <= SparseSlot;
    const bool room = open_slab != NoSlab && open_used + need <= SparseSlot;
    const std::uint32_t entry = worth && !room ? FreeSlabEntry() : NoSlab;
    const std::int64_t whole = range.physical[slot];
    void* view = reinterpret_cast<void*>(slot_view_hint.load(std::memory_order_relaxed));
    if (!worth || (!room && entry == NoSlab) ||
        sceKernelMapDirectMemory(&view, SparseSlot, PROT_READ | PROT_WRITE, 0,
                                 room ? pack_slabs[open_slab].physical.load(std::memory_order_relaxed) : whole,
                                 SparseSlot) != 0) {
        if (mprotect(reinterpret_cast<void*>(at), SparseSlot, PROT_NONE) != 0) std::abort();
        range.swap[slot] = now;
        if (!worth) {
            pack_not_worth.fetch_add(1, std::memory_order_relaxed);
            PackSkip(slot, now);
        } else {
            swap_counters.failures.fetch_add(1, std::memory_order_relaxed);
        }
        UnlockSlot(range, slot, SlotProbe);
        return false;
    }
    // The slot stops showing its block: a touch from here on faults and waits for the lock.
    if (munmap(reinterpret_cast<void*>(at), SparseSlot) != 0) std::abort();
    void* again = reinterpret_cast<void*>(at);
    if (sceKernelReserveVirtualRange(&again, SparseSlot, MAP_FIXED, SparseSlot) != 0 || again != reinterpret_cast<void*>(at))
        std::abort();
    if (!room) {
        OpenSlab(entry, whole);
        pack_counters.from_slot.fetch_add(1, std::memory_order_relaxed);
    }
    const std::uint32_t slab = open_slab;
    const std::size_t offset = open_used;
    open_used += need;
    pack_slabs[slab].live.fetch_add(1, std::memory_order_relaxed);
    pack_slabs[slab].bytes.fetch_add(static_cast<std::uint32_t>(need), std::memory_order_relaxed);
    std::memcpy(static_cast<char*>(view) + offset, packed, static_cast<std::size_t>(size));
    if (munmap(view, SparseSlot) != 0) std::abort();
    if (room) {
        if (sceKernelReleaseDirectMemory(whole, SparseSlot) != 0) std::abort();
        pack_shared_given = SparseSlot;
    }
    guest_committed.fetch_sub(SparseSlot, std::memory_order_relaxed); // the slot's block; a slab made of it counts in OpenSlab
    range.physical[slot] = StorePlace(slab, offset);
    range.swap[slot] = static_cast<std::uint32_t>(size);
    pack_store_bytes.fetch_add(static_cast<std::size_t>(size), std::memory_order_relaxed);
    pack_store_copies.fetch_add(1, std::memory_order_relaxed);
    pack_counters.packed.fetch_add(1, std::memory_order_relaxed);
    swap_counters.out.fetch_add(1, std::memory_order_relaxed);
    swap_counters.out_ns.fetch_add(NowNs() - start, std::memory_order_relaxed);
    UnlockSlot(range, slot, SlotOut);
    return true;
}

// Out (in the store) -> Own, in a fault handler (the slot is locked, so its copy cannot move): as
// UnpackIn, from a view of the copy's slab. The copy goes once the slot shows its bytes again, and its
// slab with its last copy. Nothing here allocates.
bool UnpackShared(SparseRange& range, std::size_t slot, std::int64_t given = -1) noexcept {
    const int saved = errno;
    const std::uint64_t start = NowNs();
    const std::int64_t place = range.physical[slot];
    const std::uint32_t slab = StoreSlab(place);
    const std::uint32_t size = range.swap[slot];
    std::int64_t physical = given; // T177a: a block the caller took already (-1: none, asked here as before)
    if (physical < 0 && !SparedBlock(&physical) && eden_direct_guest(SparseSlot, SparseSlot, 12, &physical) != 0 &&
        !ReserveBlock(&physical)) {
        errno = saved;
        return false;
    }
    void* to = reinterpret_cast<void*>(slot_view_hint.load(std::memory_order_relaxed));
    void* from = to;
    bool ok = sceKernelMapDirectMemory(&to, SparseSlot, PROT_READ | PROT_WRITE, 0, physical, SparseSlot) == 0;
    if (ok) {
        ok = sceKernelMapDirectMemory(&from, SparseSlot, PROT_READ, 0, pack_slabs[slab].physical.load(std::memory_order_acquire),
                                      SparseSlot) == 0;
        if (ok) {
            ok = LZ4_decompress_safe(static_cast<const char*>(from) + StoreOffset(place), static_cast<char*>(to),
                                     static_cast<int>(size), static_cast<int>(SparseSlot)) == static_cast<int>(SparseSlot);
            if (munmap(from, SparseSlot) != 0) std::abort();
        }
        if (munmap(to, SparseSlot) != 0) std::abort();
    }
    if (!ok || !MapSlot(SlotAddress(range, slot), PROT_READ | PROT_WRITE, physical)) {
        (void)sceKernelReleaseDirectMemory(physical, SparseSlot);
        swap_counters.failures.fetch_add(1, std::memory_order_relaxed);
        Note(ok ? "EDEN_UP_PACK a slot in the store could not be mapped again\n"
                : "EDEN_UP_PACK a slot in the store could not be unpacked\n");
        errno = saved;
        return false;
    }
    range.physical[slot] = physical;
    range.swap[slot] = swap_tick.load(std::memory_order_relaxed);
    DropCopy(slab, size);
    guest_committed.fetch_add(SparseSlot, std::memory_order_relaxed);
    const std::uint64_t took = NowNs() - start;
    swap_counters.in.fetch_add(1, std::memory_order_relaxed);
    swap_counters.in_ns.fetch_add(took, std::memory_order_relaxed);
    if (took > swap_counters.in_most_ns.load(std::memory_order_relaxed))
        swap_counters.in_most_ns.store(took, std::memory_order_relaxed);
    errno = saved;
    return true;
}
#endif

// Out -> Own, in a fault handler: a block, its bytes from the file through a view of their own, and
// the slot shows them in one step. Nothing here allocates. The place in the file goes back at once:
// the slot can be written now, and a later trip writes it again. T177a: `given` (0 and up) is a block
// the caller took already (EDEN_UP_GPU_READ_THROUGH=2's bring-back); it is used, or released on failure.
bool SwapIn(SparseRange& range, std::size_t slot, std::int64_t given = -1) noexcept {
#ifdef PS5_NATIVE
#ifdef EDEN_SWAP_LZ4
    if (InStore(range.physical[slot])) return UnpackShared(range, slot, given); // EDEN_UP_PACK_SHARED's store
    if (range.physical[slot] >= 0) return UnpackIn(range, slot, given); // packed in memory, not in the file
#endif
    const int saved = errno;
    const std::uint64_t start = NowNs();
    const std::uint32_t place = range.swap[slot];
    std::int64_t physical = given; // T177a: a block the caller took already (-1: none, asked here as before)
    if (physical < 0 && !SparedBlock(&physical) && eden_direct_guest(SparseSlot, SparseSlot, 12, &physical) != 0 &&
        !ReserveBlock(&physical)) {
        errno = saved;
        return false;
    }
    void* view = reinterpret_cast<void*>(slot_view_hint.load(std::memory_order_relaxed));
    if (sceKernelMapDirectMemory(&view, SparseSlot, PROT_READ | PROT_WRITE, 0, physical, SparseSlot) != 0) {
        (void)sceKernelReleaseDirectMemory(physical, SparseSlot);
        errno = saved;
        return false;
    }
    const bool read = SwapRead(view, place);
    if (munmap(view, SparseSlot) != 0) std::abort();
    if (!read || !MapSlot(SlotAddress(range, slot), PROT_READ | PROT_WRITE, physical)) {
        (void)sceKernelReleaseDirectMemory(physical, SparseSlot);
        swap_counters.failures.fetch_add(1, std::memory_order_relaxed);
        Note(read ? "EDEN_SWAP a slot could not be mapped again\n" : "EDEN_SWAP a slot could not be read back from the file\n");
        errno = saved;
        return false;
    }
    range.physical[slot] = physical;
    range.swap[slot] = swap_tick.load(std::memory_order_relaxed);
    GiveSwapPlace(place);
    guest_committed.fetch_add(SparseSlot, std::memory_order_relaxed);
    const std::uint64_t took = NowNs() - start;
    swap_counters.in.fetch_add(1, std::memory_order_relaxed);
    swap_counters.in_ns.fetch_add(took, std::memory_order_relaxed);
    if (took > swap_counters.in_most_ns.load(std::memory_order_relaxed))
        swap_counters.in_most_ns.store(took, std::memory_order_relaxed);
    errno = saved;
    return true;
#else
    (void)range;
    (void)slot;
    (void)given;
    return false; // the host never swaps
#endif
}

// CommitSparsePage for a range with slot states.
bool ServiceCommit(SparseRange& range, std::size_t slot) noexcept {
    LockSlot(range, slot);
    const std::uint8_t state = SlotState(range, slot);
    const bool ok = state == SlotZero ? PromoteZero(range, slot) : state == SlotFine ? PromoteFine(range, slot) : true;
    UnlockSlot(range, slot, ok && (state == SlotZero || state == SlotFine) ? SlotOwn : state);
    return ok;
}

// ---- The fault ----
SparseRange* WriteRangeOf(std::uintptr_t address) noexcept {
    for (SparseRange* range : {service_range.load(std::memory_order_acquire), probe_range.load(std::memory_order_acquire)}) {
        if (range == nullptr || range->state == nullptr) continue;
        const auto begin = range->begin.load(std::memory_order_acquire);
        if (begin != 0 && address >= begin && address < range->end) return range;
    }
    return nullptr;
}
std::atomic<std::uintptr_t> probe_expected{0}; // the self-test's write, whatever si_addr says
std::atomic<bool> probe_mismatch{false};
// The last write the service refused, for the crash report: the address, then why (1: it faulted
// again and again with nothing to change, 2: no block was left for its slot) and the slot's state.
std::atomic<std::uintptr_t> refused_address{0};
std::atomic<unsigned> refused_detail{0}; // why | state << 8
void Refused(std::uintptr_t address, unsigned why, std::uint8_t state) noexcept {
    refused_detail.store(why | unsigned{state} << 8, std::memory_order_relaxed);
    refused_address.store(address, std::memory_order_release);
}
} // namespace

// The last write the service refused (headless/crash_report.cpp): its address, 0 for none.
extern "C" std::uintptr_t eden_guest_write_refused(unsigned* why, unsigned* state) noexcept {
    const std::uintptr_t address = refused_address.load(std::memory_order_acquire);
    const unsigned detail = refused_detail.load(std::memory_order_relaxed);
    *why = detail & 0xff;
    *state = detail >> 8;
    return address;
}

#ifdef PS5_NATIVE
extern "C" const char __eden_text_start[] __attribute__((weak));
extern "C" const char __eden_text_end[] __attribute__((weak));
#endif

namespace {
// Who writes the slots that get promoted. A fault handler (ServiceSignal) reads the interrupted
// instruction's address and the first return address of the app's code above the stack pointer:
// a write by the emulated CPU has the JIT's entry point there, a write by the emulator itself the
// function that called the copy. The promotions are counted by that place (ServiceReport).
constexpr std::size_t WriterPcs = 4;
struct WriterSite {
    std::atomic<std::uintptr_t> key{0};
    std::uintptr_t pcs[WriterPcs]{};
    std::atomic<std::uint64_t> count{0};
};
constexpr std::size_t WriterSites = 512;
WriterSite writer_sites[WriterSites];
std::atomic<std::uint64_t> promotions_in_app_code{0}, promotions_elsewhere{0};
thread_local std::uintptr_t fault_pcs[WriterPcs] = {};
// What wrote, by where the instruction lies. "Other code" is two things: code the JIT made, the game's
// own stores, and the system's memcpy and memset (the app imports them), the emulator's copies. A JIT
// frame's stack is dynarmic's 64 spill slots (1 KiB) that hold what earlier calls left there, so its
// words name nothing: those writes are one row, of zeroes, and do not fill the table.
enum : std::uint8_t { WriterUnknown, WriterJit, WriterApp, WriterLibrary, WriterKinds };
std::atomic<std::uint64_t> promotions_by_kind[WriterKinds];
thread_local std::uint8_t fault_kind = WriterUnknown;

// The kernel's calls that gave Own slots back by clearing them (ClearGuestMemory), by where they came
// from: HostMemory::ClearBackingRegion jumps there, so the return address is the kernel's own.
struct ClearSite {
    std::atomic<std::uintptr_t> pc{0};
    std::atomic<std::uint64_t> calls{0}, slots{0};
};
constexpr std::size_t ClearSites = 16;
ClearSite clear_sites[ClearSites];
void NoteClearSite(std::uintptr_t pc, std::size_t slots) noexcept {
    for (ClearSite& site : clear_sites) {
        std::uintptr_t expected = 0;
        if (site.pc.compare_exchange_strong(expected, pc, std::memory_order_acq_rel) || expected == pc) {
            site.calls.fetch_add(1, std::memory_order_relaxed);
            site.slots.fetch_add(slots, std::memory_order_relaxed);
            return;
        }
    }
}

// The latest promotions, one word each so that a fault handler can write one and a report read it
// without a lock: the slot, the writer's place in writer_sites (WriterSites: no place left), whether
// the app's code wrote, and when (eden_guest_memory_growth). The counts above are for the whole
// session; when the pool runs out the question is what the last few hundred MiB went to.
constexpr std::size_t RecentCount = 1024; // 2 GiB of slots
constexpr unsigned RecentSlotBits = 16, RecentSiteBits = 10, RecentTimeShift = RecentSlotBits + RecentSiteBits + 1;
static_assert(WriterSites < (1u << RecentSiteBits));
std::atomic<std::uint64_t> recent_promotions[RecentCount];
std::atomic<std::uint64_t> recent_total{0};
std::uint64_t MonotonicMs() noexcept {
    timespec now{};
    clock_gettime(CLOCK_MONOTONIC, &now);
    return static_cast<std::uint64_t>(now.tv_sec) * 1000 + static_cast<std::uint64_t>(now.tv_nsec) / 1000000;
}
void NoteRecent(std::size_t slot, std::size_t site) noexcept {
    const std::uint64_t word = MonotonicMs() << RecentTimeShift |
                               std::uint64_t{fault_kind == WriterApp} << (RecentSlotBits + RecentSiteBits) |
                               std::uint64_t{site} << RecentSlotBits | (slot & ((1u << RecentSlotBits) - 1));
    recent_promotions[recent_total.fetch_add(1, std::memory_order_relaxed) % RecentCount].store(word, std::memory_order_relaxed);
}

void NoteWriter(std::size_t slot) noexcept {
    (fault_kind == WriterApp ? promotions_in_app_code : promotions_elsewhere).fetch_add(1, std::memory_order_relaxed);
    promotions_by_kind[fault_kind].fetch_add(1, std::memory_order_relaxed);
    std::uintptr_t key = 0xCBF29CE484222325ull;
    for (const std::uintptr_t pc : fault_pcs) key = (key ^ pc) * 0x100000001B3ull;
    key |= 1;
    std::size_t at = static_cast<std::size_t>((key >> 12) % WriterSites);
    for (std::size_t probe = 0; probe < 16; ++probe, at = (at + 1) % WriterSites) {
        std::uintptr_t expected = 0;
        if (writer_sites[at].key.compare_exchange_strong(expected, key, std::memory_order_acq_rel)) {
            for (std::size_t i = 0; i < WriterPcs; ++i) writer_sites[at].pcs[i] = fault_pcs[i];
            writer_sites[at].count.fetch_add(1, std::memory_order_relaxed);
            NoteRecent(slot, at);
            return;
        }
        if (expected == key) {
            writer_sites[at].count.fetch_add(1, std::memory_order_relaxed);
            NoteRecent(slot, at);
            return;
        }
    }
    NoteRecent(slot, WriterSites);
}
} // namespace

namespace {
// Writers waiting for a block (WaitForBlock), and how many waits ended with one.
std::atomic<unsigned> memory_waiters{0};
std::atomic<std::uint64_t> memory_waits_won{0}, memory_waits_lost{0};

// No block was left for a written slot. Most of what the pool gives out is held by the graphics
// caches, which free what is not drawn once they see memory short (headless/performance.cpp reads
// eden_guest_memory_wanted, and while a writer waits the caches see no free memory at all). The
// game's own memory comes first: a large open-world game wrote 1.8 GB of new memory within 10 s
// after 23 minutes of play, faster than the caches free at the end of each frame, and the write
// that found no block ended the session. So the writer waits here, in its fault handler, and tries
// again every 20 ms while the caches give memory back: for up to EDEN_UP_GUEST_WRITE_WAIT_MS (his
// name EDEN_GUEST_WRITE_WAIT_MS is read too), 2 s unset (headless/memory_wait.h: a native port's game
// pages gave up at 3 s and survived at 10 s at an area change). Only then is it refused. The limit is
// read at the first wait, which is long after the game's env file is in, and kept. Unset: the old
// 100 steps and the old kernel-log line, word for word.
unsigned GuestWriteWaitMs() noexcept {
    static std::atomic<unsigned> kept{0};
    unsigned ms = kept.load(std::memory_order_relaxed);
    if (ms == 0) {
        ms = ::Eden::MemoryWait::ReadMs2("EDEN_UP_GUEST_WRITE_WAIT_MS", "EDEN_GUEST_WRITE_WAIT_MS", 2000, 20, 60000);
        kept.store(ms, std::memory_order_relaxed);
    }
    return ms;
}

// ---- 0900 EDEN_UP_FAULT_LOG=1 (off unless set): the fault handler's waits for a block on stderr ----
//
// The notes below go to the kernel's log only (sceKernelDebugOutText), which a console run does not
// keep. With the switch, each wait writes two lines to stderr with write(2) (no stdio, no heap, no
// lock: a fault handler may run them): when the first try found no block, and how the wait ended.
//   EDEN_UP_FAULT_WAIT start t_ms=<since the service started> thread=<GPUBack|GPU|cpu-jit|cpu-app|cpu-lib|other>
//     slot=<n> state=<zero|own|shared|fine|probe|out> pool_reserve=<blocks> fault_reserve=<blocks> waiters=<n>
//   EDEN_UP_FAULT_WAIT end ... waited_ms=<n> outcome=<got-block|trim|lost>
// A GPU thread's wait is always written (up to FaultLogMost lines a session); the others at most one
// every FaultLogGapMs. Counted in any case (EDEN_UP_GPU_READ line).
std::atomic<int> fault_log_fd{-1};           // stderr's descriptor, taken as the service starts; -1 off
std::atomic<bool> fault_gpu_trim{false};     // EDEN_UP_FAULT_GPU: a GPU thread's wait trims the heap at once
std::uint64_t fault_log_start_ms = 0;
constexpr unsigned FaultLogMost = 400, FaultLogGapMs = 200;
std::atomic<unsigned> fault_log_lines{0};
std::atomic<std::uint64_t> fault_log_last_ms{0};
struct FaultWaitCounters {
    std::atomic<std::uint64_t> waits{0}, gpu_waits{0}, gpu_won{0}, gpu_lost{0}, gpu_trim_won{0}, gpu_out{0}, gpu_wait_ms{0};
    std::atomic<std::uint64_t> most_ms{0};
};
FaultWaitCounters fault_wait_counters;
// A line built in place: text and numbers only, for a fault handler.
struct FaultLine {
    char text[320];
    std::size_t used = 0;
    FaultLine& Put(const char* part) noexcept {
        while (*part != '\0' && used + 1 < sizeof(text)) text[used++] = *part++;
        return *this;
    }
    FaultLine& Num(std::uint64_t value) noexcept {
        char digits[24];
        std::size_t count = 0;
        do {
            digits[count++] = static_cast<char>('0' + value % 10);
            value /= 10;
        } while (value != 0 && count < sizeof(digits));
        while (count != 0 && used + 1 < sizeof(text)) text[used++] = digits[--count];
        return *this;
    }
    void Write() noexcept {
        const int fd = fault_log_fd.load(std::memory_order_relaxed);
        if (fd < 0) return;
        if (used + 1 < sizeof(text)) text[used++] = '\n';
        const int saved = errno;
        (void)!write(fd, text, used);
        errno = saved;
    }
};
const char* SlotStateName(std::uint8_t state) noexcept {
    static constexpr const char* names[] = {"zero", "own", "shared", "fine", "probe", "out"};
    return state < 6 ? names[state] : "?";
}
const char* FaultThreadName(int gpu_kind) noexcept {
    if (gpu_kind == 1) return "GPUBack";
    if (gpu_kind == 2) return "GPU";
    // Any other thread, by where the faulting instruction lay (ServiceSignal sets fault_kind).
    return fault_kind == WriterJit ? "cpu-jit" : fault_kind == WriterApp ? "app" : fault_kind == WriterLibrary ? "lib" : "other";
}
// Whether this wait gets its two lines (decided at its start).
bool FaultLogWanted(bool gpu, std::uint64_t now_ms) noexcept {
    if (fault_log_fd.load(std::memory_order_relaxed) < 0) return false;
    if (fault_log_lines.load(std::memory_order_relaxed) >= FaultLogMost) return false;
    if (!gpu) {
        std::uint64_t last = fault_log_last_ms.load(std::memory_order_relaxed);
        if (now_ms - last < FaultLogGapMs ||
            !fault_log_last_ms.compare_exchange_strong(last, now_ms, std::memory_order_relaxed))
            return false;
    }
    fault_log_lines.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// ---- T178b: a game CPU thread's last try, and the emergency give-back ----
//
// run1418 (T177, F1, 13 Nov): a game CPU thread's write fault (CPUCore_0) waited its whole limit and the
// fault was passed on (SIGSEGV). With the switch on, a non-GPU thread whose WaitForBlock found nothing makes
// one more attempt after EmergencyGiveBack (below, after the swap's code): the heap's empty spaces back to the
// pool (the unheld trim, as the wait's own trims), then up to EmergencySlots cold slots packed or written out
// now, on this thread, if the swap is not busy (swap_out_mutex by try_lock only). On only where the fault used
// to be passed on, which ends the app:
//   EDEN_UP_FAULT_LAST_TRY=1  on; =0 off; unset: on while EDEN_UP_GPU_READ_THROUGH is on (=1 and up), else off
// Read as the service starts (ReadGpuReadSwitches). Counted: EDEN_UP_FAULT_LAST_TRY line beside EDEN_UP_GPU_READ;
// with EDEN_UP_FAULT_LOG=1 one stderr line per try (write(2), as the wait's lines).
std::atomic<bool> fault_last_try{false};
struct FaultLastTryCounters {
    std::atomic<std::uint64_t> tries{0}, won{0}, lost{0}, given_bytes{0}, swap_busy{0}, slots_out{0};
};
FaultLastTryCounters fault_last_try_counters;
// Bytes given back (the trim's, plus the slots' estimate); `slots` and `busy` say what the swap did.
std::size_t EmergencyGiveBack(std::size_t* slots, bool* busy) noexcept;

template <typename Promote>
bool WaitForBlock(Promote&& promote, std::size_t slot = 0, std::uint8_t state = 0) noexcept {
    const std::uint64_t block_since = StallProbe::On() ? StallProbe::NowNs() : 0; // EDEN_UP_STALL_PROBE
    memory_waiters.fetch_add(1, std::memory_order_acq_rel);
    bool ok = false;
    const unsigned wait_ms = GuestWriteWaitMs();
    const unsigned limit = ::Eden::MemoryWait::Steps(wait_ms);
    // 0900: who waits (EDEN_UP_FAULT_GPU, EDEN_UP_FAULT_LOG). Off: one compare of a -1 and two loads.
    const int gpu_kind = GpuThreadKind();
    const std::uint64_t wait_start_ms = MonotonicMs();
    fault_wait_counters.waits.fetch_add(1, std::memory_order_relaxed);
    if (gpu_kind != 0) {
        fault_wait_counters.gpu_waits.fetch_add(1, std::memory_order_relaxed);
        if (state == SlotOut) fault_wait_counters.gpu_out.fetch_add(1, std::memory_order_relaxed);
    }
    const bool logged = FaultLogWanted(gpu_kind != 0, wait_start_ms);
    const auto log_line = [&](const char* what) noexcept {
        FaultLine line;
        line.Put("EDEN_UP_FAULT_WAIT ").Put(what).Put(" t_ms=").Num(wait_start_ms - fault_log_start_ms)
            .Put(" thread=").Put(FaultThreadName(gpu_kind)).Put(" slot=").Num(slot).Put(" state=").Put(SlotStateName(state))
#ifdef PS5_NATIVE
            .Put(" pool_reserve=").Num(reserve_count.load(std::memory_order_relaxed))
            .Put(" fault_reserve=").Num(fault_reserve_count.load(std::memory_order_relaxed))
#endif
            .Put(" waiters=").Num(memory_waiters.load(std::memory_order_relaxed));
        return line;
    };
    if (logged) log_line("start").Write();
    bool trim_won = false;
    // 0900 EDEN_UP_FAULT_GPU=1: a GPU thread holds the texture and buffer caches' locks here, and only
    // it frees graphics memory, so waiting for the caches cannot help it. Its promotion has had the
    // fault reserve already (ReserveBlock's last step); the one give-back known to be safe in a fault
    // handler is the heap's trim (0761 calls it here too), done at once, then the wait as before.
    if (gpu_kind != 0 && fault_gpu_trim.load(std::memory_order_relaxed) && ::eden_heap_trim != nullptr) {
        (void)::eden_heap_trim();
        ok = promote();
        trim_won = ok;
    }
    if (trim_won) fault_wait_counters.gpu_trim_won.fetch_add(1, std::memory_order_relaxed);
    // 0761 EDEN_UP_BLOCK_WAIT_TRIM=1 (waveydude d63c890, Wave Ocarina 07): the heap's empty spaces
    // go back to the pool at the wait's first try and every TrimEveryMs (500 ms) after, and the
    // block is asked for again right then. When the waiting thread is the GPU thread, the graphics
    // caches free nothing while it waits; whole empty heap pages only come back through a trim.
    // His W-05 later-area run: GPUBack faulted with the pool at 1 MiB and 26 MiB of empty heap
    // pages. Off: one cached static read per wait, then the loop as before.
    static const bool block_wait_trim = Common::PeSwitch("EDEN_UP_BLOCK_WAIT_TRIM");
    constexpr unsigned trim_every_tries = static_cast<unsigned>(::Eden::MemoryWait::TrimEveryMs / 20);
    for (unsigned tries = 0; tries < limit && !ok; ++tries) {
        if (block_wait_trim && tries % trim_every_tries == 0 && ::eden_heap_trim != nullptr) {
            // EDEN_UP_HEAP_HOLD_AT_LOAD's hold does not stop this one: the game's memory first.
            (void)(::eden_heap_trim_unheld != nullptr ? ::eden_heap_trim_unheld() : ::eden_heap_trim());
            ok = promote();
            if (ok) break;
        }
        const timespec pause{0, 20'000'000};
        nanosleep(&pause, nullptr);
        ok = promote();
    }
    memory_waiters.fetch_sub(1, std::memory_order_acq_rel);
    if (block_since != 0) StallProbe::BlockWaited(block_since);
    (ok ? memory_waits_won : memory_waits_lost).fetch_add(1, std::memory_order_relaxed);
    Note(ok ? "EDEN_CRAM a written slot waited for the graphics caches to give memory back, and got a block\n"
         : wait_ms == 2000
             ? "EDEN_CRAM a written slot waited 2 s for memory and found no block\n"
             : "EDEN_CRAM a written slot waited its whole limit (EDEN_UP_GUEST_WRITE_WAIT_MS) for memory and found no block\n");
    // 0900: how it ended (EDEN_UP_FAULT_LOG), and the GPU threads' totals (EDEN_UP_GPU_READ line).
    const std::uint64_t waited_ms = MonotonicMs() - wait_start_ms;
    if (gpu_kind != 0) {
        (ok ? fault_wait_counters.gpu_won : fault_wait_counters.gpu_lost).fetch_add(1, std::memory_order_relaxed);
        fault_wait_counters.gpu_wait_ms.fetch_add(waited_ms, std::memory_order_relaxed);
    }
    if (waited_ms > fault_wait_counters.most_ms.load(std::memory_order_relaxed))
        fault_wait_counters.most_ms.store(waited_ms, std::memory_order_relaxed);
    if (logged)
        log_line("end").Put(" waited_ms=").Num(waited_ms).Put(" outcome=").Put(trim_won ? "trim" : ok ? "got-block" : "lost").Write();
    return ok;
}
} // namespace

// How many writes of the guest's memory are waiting for a block now (headless/performance.cpp):
// while any is, the graphics caches are told no memory is free.
extern "C" unsigned eden_guest_memory_wanted(void) noexcept {
    return memory_waiters.load(std::memory_order_acquire);
}

// Last resort for a host heap piece, after the direct pool refused even 2 MiB.
// Transfer ownership under the reserve lock: never release then race the keeper
// to reacquire the same memory. The host's committed-block table owns/frees it.
// Leave at least half the configured reserve for guest write faults, and give
// waiting guest writers priority. This does not change the guest's own fallback.
bool TakeHostReserveBlock(std::int64_t* physical) noexcept {
#ifdef PS5_NATIVE
    const char* enabled = std::getenv("EDEN_HOST_RESERVE_TRANSFER");
    if (!physical || !enabled || std::strcmp(enabled, "1") != 0) return false;
    Spin(reserve_lock);
    const auto count = reserve_count.load(std::memory_order_relaxed);
    const bool allowed = service_range.load(std::memory_order_acquire) != nullptr &&
        memory_waiters.load(std::memory_order_acquire) == 0 && count > ReserveBlocks / 2;
    if (allowed) {
        *physical = reserve[count - 1];
        reserve_count.store(count - 1, std::memory_order_relaxed);
    }
    reserve_lock.clear(std::memory_order_release);
    if (allowed) counters.host_reserve_used.fetch_add(1, std::memory_order_relaxed);
    return allowed;
#else
    (void)physical;
    return false;
#endif
}

// A graphics buffer was refused with nothing else left to give, which ends the session: blocks of the
// reserve go back to the pool, the newest first, down to the floor. Buffers only: an image has its
// fallback, and it can ask for more than this could ever give. Not while a guest write waits for a
// block. The keeper leaves them alone while the pool is short. What went back, in bytes.
extern "C" std::size_t eden_guest_reserve_lend(std::size_t bytes) noexcept {
#ifdef PS5_NATIVE
    if (service_range.load(std::memory_order_acquire) == nullptr ||
        memory_waiters.load(std::memory_order_acquire) != 0) return 0;
    const std::size_t floor = reserve_floor.load(std::memory_order_relaxed);
    std::size_t lent = 0, adjacent = 0;
    Spin(reserve_lock);
    std::size_t count = reserve_count.load(std::memory_order_relaxed);
    while (lent < bytes && count > floor) {
        const std::int64_t block = reserve[count - 1];
        if (sceKernelReleaseDirectMemory(block, SparseSlot) != 0) break;
        if (count >= 2 && reserve[count - 2] + static_cast<std::int64_t>(SparseSlot) == block) ++adjacent;
        --count;
        lent += SparseSlot;
    }
    reserve_count.store(count, std::memory_order_relaxed);
    reserve_lock.clear(std::memory_order_release);
    // Single 2 MiB blocks serve a buffer through the driver's pieces (Mesa f5610fe27) or, next to each
    // other, in one range: how many were, for the log.
    std::fprintf(stderr, "EDEN_RESERVE_LENT %zu MiB, %zu of %zu blocks beside the next; %zu blocks left\n", lent >> 20,
                 adjacent, lent / SparseSlot, count);
    return lent;
#else
    (void)bytes;
    return 0;
#endif
}

#ifdef PS5_NATIVE
namespace {
// EDEN_UP_RADV_OOM_LEND's counters (eden_guest_reserve_lend_stats): relaxed, any thread.
std::atomic<std::uint64_t> oom_lend_calls{0}, oom_lend_bytes{0}, oom_lend_adjacent{0}, oom_lend_refused_waiting{0},
    oom_lend_none_left{0};
// The blocks a lend to the driver always leaves (or the floor, if that is lower), whatever the switch
// says: guest writes the pool refuses and page table slots take the reserve down to 0, and a write
// that then finds none ends the session 2 s later.
constexpr std::size_t OomLendKeep = 8;
} // namespace
#endif

// EDEN_UP_RADV_OOM_LEND (headless/performance.cpp, the driver's out-of-memory callback): what
// eden_guest_reserve_lend does, down to `below_floor` blocks under the floor, and without its line: the
// caller is a thread inside the driver with the pool empty, where stdio may ask the heap for memory (the
// callback's own line says what went). Not while a guest write waits for a block, as there; the keeper
// fills no further than the lend's floor for the next 2 s. What went back, in bytes.
extern "C" std::size_t eden_guest_reserve_lend_quiet(std::size_t bytes, std::size_t below_floor) noexcept {
#ifdef PS5_NATIVE
    if (service_range.load(std::memory_order_acquire) == nullptr) return 0;
    oom_lend_calls.fetch_add(1, std::memory_order_relaxed);
    if (memory_waiters.load(std::memory_order_acquire) != 0) {
        oom_lend_refused_waiting.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    const std::size_t floor_now = reserve_floor.load(std::memory_order_relaxed);
    const std::size_t keep = std::min(floor_now, OomLendKeep);
    const std::size_t floor = floor_now > below_floor + keep ? floor_now - below_floor : keep;
    std::size_t lent = 0, adjacent = 0;
    Spin(reserve_lock);
    std::size_t count = reserve_count.load(std::memory_order_relaxed);
    while (lent < bytes && count > floor) {
        const std::int64_t block = reserve[count - 1];
        if (sceKernelReleaseDirectMemory(block, SparseSlot) != 0) break;
        if (count >= 2 && reserve[count - 2] + static_cast<std::int64_t>(SparseSlot) == block) ++adjacent;
        --count;
        lent += SparseSlot;
    }
    reserve_count.store(count, std::memory_order_relaxed);
    reserve_lock.clear(std::memory_order_release);
    if (lent != 0) {
        keeper_hold_floor.store(floor, std::memory_order_relaxed);
        keeper_hold_until_ns.store(NowNs() + 2'000'000'000ull, std::memory_order_release);
    } else {
        oom_lend_none_left.fetch_add(1, std::memory_order_relaxed);
    }
    oom_lend_bytes.fetch_add(lent, std::memory_order_relaxed);
    oom_lend_adjacent.fetch_add(adjacent, std::memory_order_relaxed);
    return lent;
#else
    (void)bytes;
    (void)below_floor;
    return 0;
#endif
}

// ---- EDEN_UP_JIT_LEND_UNDER_FLOOR: a code cache piece from the reserve, under its floor ----
//
// The JIT's code cache asks for its next 16 MiB step one 2 MiB piece at a time (memory_pages.cpp
// CommitCodeStep). A piece the pool refuses could come from the reserve only above its floor
// (EDEN_JIT_RESERVE_LEND); with the reserve under it the JIT waited 2 s and ended the session (owner's
// T130 play, 11 Nov: 28 blocks held, floor 32, 0 lent, abort). With this switch such a piece takes one
// block of the reserve straight over, under the reserve's lock (never given to the pool and asked
// back, where the driver or the heap could take it first), down to JitLendKeep blocks and never
// further. One block per refused piece, so never more than the step still needs. Only CommitCodeStep
// calls it. Not while a game write waits for a block (that write has emptied the reserve anyway).
// The keeper fills the reserve again from the pool as it does after any game write took from it.
//   EDEN_UP_JIT_LEND_UNDER_FLOOR=1  count only: what would have been lent; the piece is refused as before
//   EDEN_UP_JIT_LEND_UNDER_FLOOR=2  lend
// Read once as the service starts. Unset or 0: one relaxed load on the refused path, and the old path.
//
// The minimum: guest writes that the pool refuses take the reserve one block each and may empty it,
// and a write that then finds none waits 2 s and ends the session. The owner's T130 session took the
// reserve 11 blocks under its floor of 32 at the passage stall (21 left, the deepest of the session)
// and 12 blocks within one ledger window there (reserve_used 103 -> 115); everywhere else 1-7. 16 blocks
// (32 MiB) cover that burst with 4 to spare, twice what the driver's lend keeps (OomLendKeep).
#ifdef PS5_NATIVE
namespace {
constexpr std::size_t JitLendKeep = 16;
std::atomic<int> jit_lend_mode{0};
// Counters for the line every 10 s (ReserveKeeper): relaxed, any thread.
std::atomic<std::uint64_t> jit_lend_asked{0}, jit_lend_lent{0}, jit_lend_would{0}, jit_lend_at_keep{0},
    jit_lend_writer_waits{0};
std::atomic<std::size_t> jit_lend_lowest{ReserveBlocks}; // the fewest blocks a lend left
} // namespace
#endif

// 1: a block of the reserve is now the caller's (`physical`); 0: none (switch off, a write waits, or the
// reserve is at the minimum; in count mode also when one could have gone). `left`: blocks after it.
// `why`: 0 off, 1 count only (one could have gone), 2 at the minimum, 3 a game write waits.
// Called by CommitCodeStep with code_mutex held; takes the reserve's spin lock for a few loads only
// (no system call inside it, unlike the lends that give blocks to the pool).
int JitLendUnderFloor(std::int64_t* physical, std::size_t* left, int* why) noexcept {
#ifdef PS5_NATIVE
    const int mode = jit_lend_mode.load(std::memory_order_relaxed);
    *why = 0;
    if (mode == 0 || service_range.load(std::memory_order_acquire) == nullptr) return 0;
    jit_lend_asked.fetch_add(1, std::memory_order_relaxed);
    if (memory_waiters.load(std::memory_order_acquire) != 0) {
        jit_lend_writer_waits.fetch_add(1, std::memory_order_relaxed);
        *left = reserve_count.load(std::memory_order_relaxed);
        *why = 3;
        return 0;
    }
    Spin(reserve_lock);
    std::size_t count = reserve_count.load(std::memory_order_relaxed);
    const bool can = count > JitLendKeep;
    if (can && mode == 2) {
        *physical = reserve[count - 1];
        reserve_count.store(--count, std::memory_order_relaxed);
    }
    reserve_lock.clear(std::memory_order_release);
    *left = count;
    if (!can) {
        jit_lend_at_keep.fetch_add(1, std::memory_order_relaxed);
        *why = 2;
        return 0;
    }
    if (mode != 2) {
        jit_lend_would.fetch_add(1, std::memory_order_relaxed);
        *why = 1;
        return 0;
    }
    jit_lend_lent.fetch_add(1, std::memory_order_relaxed);
    for (std::size_t low = jit_lend_lowest.load(std::memory_order_relaxed);
         count < low && !jit_lend_lowest.compare_exchange_weak(low, count, std::memory_order_relaxed);)
        ;
    return 1;
#else
    (void)physical;
    (void)left;
    *why = 0;
    return 0;
#endif
}

#ifdef PS5_NATIVE
namespace {
// From the keeper, every 100 ms while the switch is on: the 10 s line. (Its test is T132's shared
// EDEN_TEST_JIT_COMMIT_REFUSE, memory_pages.cpp: a refusal at the need comes here after the old lend.)
void JitLendTick(std::uint64_t now) noexcept {
    static std::uint64_t reported_ns = 0; // the keeper's own
    const int mode = jit_lend_mode.load(std::memory_order_relaxed);
    if (mode != 0 && now - reported_ns >= 10'000'000'000ull) {
        reported_ns = now;
        const std::size_t lowest = jit_lend_lowest.load(std::memory_order_relaxed);
        char line[400];
        const int length = std::snprintf(
            line, sizeof(line),
            "EDEN_UP_JIT_LEND_UNDER_FLOOR mode %d: %llu refused pieces asked, %llu lent (%llu MiB), %llu would "
            "have been lent (count only), %llu found the reserve at its minimum of %zu, %llu refused while a game "
            "write waited; reserve %zu blocks (floor %zu, target %zu), fewest after a lend %zu (0: no lend yet)\n",
            mode, (unsigned long long)jit_lend_asked.load(std::memory_order_relaxed),
            (unsigned long long)jit_lend_lent.load(std::memory_order_relaxed),
            (unsigned long long)jit_lend_lent.load(std::memory_order_relaxed) * (SparseSlot >> 20),
            (unsigned long long)jit_lend_would.load(std::memory_order_relaxed),
            (unsigned long long)jit_lend_at_keep.load(std::memory_order_relaxed), JitLendKeep,
            (unsigned long long)jit_lend_writer_waits.load(std::memory_order_relaxed),
            reserve_count.load(std::memory_order_relaxed), reserve_floor.load(std::memory_order_relaxed),
            reserve_target.load(std::memory_order_relaxed), lowest == ReserveBlocks ? std::size_t{0} : lowest);
        if (length > 0) Report(line);
    }
}

// As the service starts: the switch.
void ReadJitLendSwitches() noexcept {
    const char* const mode = std::getenv("EDEN_UP_JIT_LEND_UNDER_FLOOR");
    const int asked = mode != nullptr && (mode[0] == '1' || mode[0] == '2') && mode[1] == '\0' ? mode[0] - '0' : 0;
    jit_lend_mode.store(asked, std::memory_order_relaxed);
    for (auto* counter : {&jit_lend_asked, &jit_lend_lent, &jit_lend_would, &jit_lend_at_keep, &jit_lend_writer_waits})
        counter->store(0, std::memory_order_relaxed);
    jit_lend_lowest.store(ReserveBlocks, std::memory_order_relaxed);
    char line[240];
    if (asked != 0) {
        const int length = std::snprintf(
            line, sizeof(line),
            "EDEN_UP_JIT_LEND_UNDER_FLOOR mode %d: a code cache piece the pool and the reserve above its floor "
            "refuse %s a reserve block, down to %zu blocks (floor %zu)\n",
            asked, asked == 2 ? "takes" : "counts what it would take of", JitLendKeep,
            reserve_floor.load(std::memory_order_relaxed));
        if (length > 0) Report(line);
    }
}
} // namespace
#endif

// The guest's headroom (the reserve): what it holds and what it is to hold, in bytes; both 0 while
// the service does not run or keeps no reserve (the host). headless/performance.cpp counts what is
// not held yet as taken, as all of it was before.
extern "C" void eden_guest_memory_reserve(std::size_t* held, std::size_t* target) noexcept {
#ifdef PS5_NATIVE
    const bool running = service_range.load(std::memory_order_acquire) != nullptr;
    *held = running ? reserve_count.load(std::memory_order_relaxed) * SparseSlot : 0;
    // Spent on purpose while the pool is short, so not counted as taken from what the caches see
    // (performance.h ViewOfPool): they would read a free pool as full for most of a session.
    const std::size_t blocks =
        pool_is_short.load(std::memory_order_relaxed) ? reserve_floor.load(std::memory_order_relaxed)
                                                      : reserve_target.load(std::memory_order_relaxed);
    *target = running ? blocks * SparseSlot : 0;
#else
    *held = 0;
    *target = 0;
#endif
}

// Whether the pool's largest block is short (headless/performance.cpp, RefreshFreeMemory).
extern "C" void eden_guest_memory_pool_short(int is_short) noexcept {
    pool_is_short.store(is_short != 0, std::memory_order_relaxed);
}

// How the waits ended so far, and the writes refused for want of a block (headless/performance.cpp
// ReportMemory prints them beside what holds the pool).
extern "C" void eden_guest_memory_waits(unsigned long long* won, unsigned long long* lost,
                                        unsigned long long* refused) noexcept {
    *won = memory_waits_won.load(std::memory_order_relaxed);
    *lost = memory_waits_lost.load(std::memory_order_relaxed);
    *refused = counters.failures.load(std::memory_order_relaxed);
}

// A write fault at `address`: true when it was a write to a slot of the guest's memory that is
// now writable (the faulting instruction repeats), false for anything else.
bool HandleGuestWriteFault(void* fault) noexcept {
    auto address = reinterpret_cast<std::uintptr_t>(fault);
    if (const auto expected = probe_expected.load(std::memory_order_acquire); expected != 0 && address != expected) {
        probe_mismatch.store(true, std::memory_order_relaxed);
        address = expected;
    }
    SparseRange* const range = WriteRangeOf(address);
    if (range == nullptr) return false;
    const std::size_t slot = (address - range->begin.load(std::memory_order_relaxed)) / SparseSlot;
    // The same address faulting again and again with nothing to change is not a slot to promote
    // but a real fault. A promotion starts the count again (below).
    static thread_local std::uintptr_t last = 0;
    static thread_local unsigned repeats = 0;
    repeats = address == last ? repeats + 1 : 1;
    last = address;
    if (repeats > 16) {
        repeats = 0;
        Refused(address, 1, SlotState(*range, slot));
        Note("EDEN_CRAM refused a write: 17 faults in a row at one address, its slot unchanged\n");
        return false;
    }
    counters.faults.fetch_add(1, std::memory_order_relaxed);
    const bool waited = LockSlot(*range, slot, true);
    const std::uint8_t state = SlotState(*range, slot);
    // 0901 EDEN_UP_GPU_READ_THROUGH: a GPU read takes an Out or Probe slot's lock for each piece it
    // copies (thousands for one upload), and a thread faulting there kept finding it held: each time it
    // returned to fault again, never counted as progress, and the 17th time in a row ended as "refused a
    // write". A slot still Out or Probe after the wait cannot be accessed as it is, so with the switch
    // the handler brings it in itself now, as it does when it finds the slot free.
    const bool still_cold = state == SlotOut || state == SlotProbe;
    if (waited && !(still_cold && read_through_on.load(std::memory_order_relaxed))) {
        // Someone changed the slot meanwhile (or is mapping it again): the access may now succeed,
        // and a read must never make a Fine slot whole. It faults again if it was a write.
        UnlockSlot(*range, slot, state);
        return true;
    }
    const auto promote = [&]() noexcept {
        bool done = true;
        if (state == SlotFine) {
            done = PromoteFine(*range, slot);
            if (done) NoteWriter(slot);
        } else if (state == SlotZero) {
            done = PromoteZero(*range, slot);
            if (done) {
                counters.promotions.fetch_add(1, std::memory_order_relaxed);
                NoteWriter(slot);
            }
        } else if (state == SlotShared) {
            done = PromoteShared(*range, slot);
        } else if (state == SlotProbe) {
            done = TouchProbe(*range, slot);
        } else if (state == SlotOut) {
            done = SwapIn(*range, slot);
        }
        return done;
    };
#ifdef PS5_NATIVE
    fault_reserve_open = true; // 0900 EDEN_UP_FAULT_RESERVE: these promotions may take it (ReserveBlock's last step)
#endif
    bool ok = promote();
    // 0901 EDEN_UP_GPU_READ_THROUGH: an Out slot is not held while its fault sleeps for a block. Held,
    // it stopped GPUBack's read-through of that slot for the whole wait (up to
    // EDEN_UP_GUEST_WRITE_WAIT_MS), and GPUBack is the thread that frees graphics memory: the T166
    // case where a game thread faults on a packed slot first. Between tries the slot is let go as it
    // was (Out, its packed copy untouched: a failed SwapIn gives back what it took); each try takes the
    // lock again and gives up waiting when someone else changed the slot (brought it in, gave it back),
    // and the access is then repeated. Off: the wait as before, with the slot held.
    bool held = true, changed = false;
    std::uint8_t now_state = state;
    const auto loose_promote = [&]() noexcept {
        if (!held) {
            LockSlot(*range, slot);
            held = true;
            now_state = SlotState(*range, slot);
            if (now_state != SlotOut) {
                changed = true;
                return true; // nothing more to wait for: the access repeats
            }
        }
        if (promote()) return true;
        UnlockSlot(*range, slot, SlotOut);
        held = false;
        return false;
    };
    if (!ok) {
        if (state == SlotOut && read_through_on.load(std::memory_order_relaxed)) {
            UnlockSlot(*range, slot, SlotOut);
            held = false;
            ok = WaitForBlock(loose_promote, slot, state);
            if (!held) {
                LockSlot(*range, slot);
                held = true;
                now_state = SlotState(*range, slot);
                if (now_state != SlotOut) ok = changed = true;
            }
        } else {
            ok = WaitForBlock(promote, slot, state);
        }
        // T178b EDEN_UP_FAULT_LAST_TRY (at fault_last_try): the wait ran out on a non-GPU thread, and the fault
        // would now be passed on. The slot is held here again (both paths above), so the give-back passes it over.
        if (!ok && !changed && fault_last_try.load(std::memory_order_relaxed) && GpuThreadKind() == 0) {
            auto& t = fault_last_try_counters;
            t.tries.fetch_add(1, std::memory_order_relaxed);
            std::size_t slots_out = 0;
            bool busy = false;
            const std::size_t given = EmergencyGiveBack(&slots_out, &busy);
            t.given_bytes.fetch_add(given, std::memory_order_relaxed);
            t.slots_out.fetch_add(slots_out, std::memory_order_relaxed);
            if (busy) t.swap_busy.fetch_add(1, std::memory_order_relaxed);
            ok = promote();
            (ok ? t.won : t.lost).fetch_add(1, std::memory_order_relaxed);
            if (fault_log_fd.load(std::memory_order_relaxed) >= 0) {
                FaultLine line;
                line.Put("EDEN_UP_FAULT_LAST_TRY t_ms=").Num(MonotonicMs() - fault_log_start_ms)
                    .Put(" thread=").Put(FaultThreadName(0)).Put(" slot=").Num(slot).Put(" state=").Put(SlotStateName(state))
                    .Put(" given_kib=").Num(given >> 10).Put(" slots_out=").Num(slots_out).Put(" swap_busy=").Num(busy ? 1 : 0)
                    .Put(" outcome=").Put(ok ? "got-block" : "lost");
                line.Write();
            }
        }
    }
#ifdef PS5_NATIVE
    fault_reserve_open = false;
#endif
    if (changed) {
        // 0901: brought in or given back by another thread while this one slept: as the `waited` case.
        UnlockSlot(*range, slot, now_state);
        return true;
    }
    if (ok && state == SlotOut) NoteLeftOut(slot); // 0900: before the lock goes
    UnlockSlot(*range, slot, ok ? SlotOwn : state);
    if (!ok) {
        counters.failures.fetch_add(1, std::memory_order_relaxed);
        Refused(address, 2, state);
        Note("EDEN_CRAM out of memory: a written slot of the guest's memory found no block\n");
    } else if (state != SlotOwn) {
        // The slot was made writable. A game that writes one place each time its slot was given
        // back (cleared, freed or scanned) faults there again and again over a session, each time
        // with something to do: counted as repeats, the 17th such write ended the game after
        // 15 to 60 minutes (Swordpdf/ProsperoEden#246).
        repeats = 0;
    }
    return ok;
}

// 0900: core's GPUBack (1) and GPU (2) threads call this as they start (common/guest_read_through.h).
extern "C" void eden_guest_gpu_thread_mark(int kind) {
    if (kind < 1 || kind > 2) return;
    gpu_thread_ids[kind].store(ThreadKey(pthread_self()), std::memory_order_relaxed);
#ifdef PS5_NATIVE
    fault_reserve_open = false; // its thread-local made here, not first in a fault handler
#endif
}

namespace {
struct sigaction service_previous_segv{}, service_previous_bus{};

void ServiceSignal(int signal, siginfo_t* info, void* context) {
#ifdef PS5_NATIVE
    fault_kind = WriterUnknown;
    for (std::uintptr_t& pc : fault_pcs) pc = 0;
    if (context != nullptr) {
        const auto low = reinterpret_cast<std::uintptr_t>(__eden_text_start) + 8;
        const auto high = reinterpret_cast<std::uintptr_t>(__eden_text_end);
        const std::uintptr_t rip = Eden::Fastmem::ContextRip(context);
        fault_kind = rip >= low && rip < high ? WriterApp : InCodeCache(rip) ? WriterJit : WriterLibrary;
        const auto* stack = reinterpret_cast<const std::uintptr_t*>(Eden::Fastmem::ContextRsp(context));
        std::size_t found = 0;
        // Never past the page (16 KiB) that holds the stack pointer: a thread's stack can end a few
        // hundred bytes above it, and what lies after it is not mapped.
        const auto page_end = ((reinterpret_cast<std::uintptr_t>(stack) | 0x3FFFu) + 1);
        const auto room = static_cast<unsigned>((page_end - reinterpret_cast<std::uintptr_t>(stack)) / sizeof(std::uintptr_t));
        for (unsigned i = 0; fault_kind != WriterJit && i < 160 && i < room && found < WriterPcs; ++i) {
            if (stack[i] >= low && stack[i] < high && (found == 0 || fault_pcs[found - 1] != stack[i])) {
                fault_pcs[found++] = stack[i];
            }
        }
    }
#endif
    if (HandleGuestWriteFault(info ? info->si_addr : nullptr)) return;
    const struct sigaction& previous = signal == SIGSEGV ? service_previous_segv : service_previous_bus;
    if (previous.sa_flags & SA_SIGINFO) {
        previous.sa_sigaction(signal, info, context);
    } else if (previous.sa_handler == SIG_DFL) {
        std::signal(signal, SIG_DFL); // the faulting instruction repeats and ends the process
    } else if (previous.sa_handler != SIG_IGN) {
        previous.sa_handler(signal);
    }
}

bool InstallServiceSignal() noexcept {
    static std::once_flag once;
    static bool installed = false;
    std::call_once(once, [] {
        struct sigaction action{};
        action.sa_sigaction = ServiceSignal;
        action.sa_flags = SA_SIGINFO | SA_RESTART;
        sigemptyset(&action.sa_mask);
        installed = sigaction(SIGSEGV, &action, &service_previous_segv) == 0 &&
                    sigaction(SIGBUS, &action, &service_previous_bus) == 0;
    });
    return installed;
}

// ---- 0900 EDEN_UP_GPU_READ_THROUGH=1 (off unless set): the GPU's reads of a cold slot, no block ----
//
// T166 drain A/B (runs 1399, 1402): at an area load the pool was full and the GPU back thread's
// memcpy from a packed slot faulted (texture upload: ReadBlockImpl <- GuestMemory::Read <-
// UploadImageContents; buffer sync: WalkBlock <- ReadBlockUnsafe <- AheadGuard::Read). Bringing the
// slot back wants a 2 MiB block; there was none, and the fault handler waited 10 s on the one thread
// that holds the texture and buffer caches' locks and so the only one that could free graphics memory.
// With the switch, core patch 0900 sends the GPU side's reads of the guest's memory here
// (DeviceMemoryManager ReadBlock/ReadBlockUnsafe and their page walk, MemoryManager ReadBlockImpl and
// ReadBlockUnsafe, and a read-only GuestMemory, which takes a copy rather than a span over such a slot).
// For each 2 MiB slot the read covers:
//   - Out (packed or in the file): under the reader's scratch mutex, then the slot's lock, its bytes are copied
//     from where they are: the file read straight into the destination, a packed copy unpacked once into
//     a 2 MiB scratch (kept while the slot stays Out, so the 4 KiB pieces of one upload unpack it once)
//     and the wanted range copied from there. No block is taken and the slot stays Out. The exact bytes:
//     while the slot is locked nothing can bring it back, write it, move its copy or give it back (each
//     of those takes the lock), and a slot that left Out since the scratch was filled has a new
//     out_generation (counted before its lock went), so an old unpacked copy is never used.
//   - Probe (armed, cold, still with its block): made Own under its lock as the fault would make it
//     (TouchProbe: an mprotect, no block), then the plain copy.
//   - anything else: the plain copy, as before (Zero shows the zero block, Shared and Fine are readable).
// Nothing under the slot's lock waits for the pool: a view of memory the slot already holds, LZ4, a
// pread, a memcpy into the destination (never the guest's memory: such a read takes the old copy). The
// lock order is a scratch mutex, then the slot; a fault handler takes the slot's lock only, and the
// swapper, the scanner and the compactor never take a scratch mutex. If anything fails, the piece is
// copied the old way (a fault, as in T173). Core 0901 sends the direct-pointer reads here too (the
// frame-ahead guard's capture, descriptor tables, Read<T>, bank hashes, uniform spans). Not covered:
// spans handed out before the slot went cold, the CPU's JIT, GPU writes, the guard's verify mode.
//
// 0901 (checker review): no lock is shared by all readers any more. Each reader kind has its own
// scratch and mutex: GPUBack (1) and GPU (2) have one thread each, so neither ever waits for another
// reader (one global mutex let a reader stuck behind a slot's lock hold up GPUBack's other cold reads,
// Probe ones too, and two readers on two slots unpacked again for every piece). Every other thread
// (shader pre-pass workers and the like) shares kind 0's. And with this switch a fault handler no
// longer holds an Out slot while it sleeps for a block (HandleGuestWriteFault), so a reader waits for
// a slot's lock only while someone works on it.
struct ReadScratch {
    std::mutex mutex;     // this scratch and its kept fields
    char* data = nullptr; // 2 MiB, made once as the service starts with the switch, then kept
    SparseRange* kept_range = nullptr;
    std::size_t kept_slot = SIZE_MAX;
    std::int64_t kept_place = 0;
    std::uint32_t kept_size = 0, kept_generation = 0;
};
ReadScratch read_scratch[3];
struct ReadThroughCounters {
    std::atomic<std::uint64_t> out_pieces{0}, out_bytes{0}, unpacks{0}, kept_hits{0}, unpack_ns{0}, file_reads{0};
    std::atomic<std::uint64_t> probes{0}, lock_waits{0}, failures{0}, spans_copied{0};
};
ReadThroughCounters read_through_counters;

// ---- T177a EDEN_UP_GPU_READ_THROUGH=2 (off unless set): bring back first, read through only without a block ----
//
// T175's first console run with =1 (run1411-ab175-F1-fix) survived the crash load, but a GPU read of an Out
// slot copies from the packed bytes and leaves it Out, so the slots the GPU reads every frame were unpacked
// again and again (186,085 unpacks, 106 s of LZ4, about 600 a second after the load) and the new area ran at
// 11.6 fps where the older build's survivor ran 15.6. Before the read-through such a read faulted once,
// the fault brought the slot back, and later reads were plain copies. With =2, a read of an Out slot (packed
// or in the file), by any reader kind (GPUBack, GPU, and kind 0: the pre-pass workers and the frame-ahead
// guard's other callers: what they read every frame is just as hot), goes:
//   1. A block, taken at once, before the slot's lock and without waiting: the pool, then the reserve while
//      it holds more than its floor (EDEN_GUEST_RESERVE_FLOOR: the writes' share while the pool is short),
//      then EDEN_UP_FAULT_RESERVE while it holds more than half its target and never its last
//      FaultReserveGpuKeep (2) blocks: those are the faults' (game writes, which cannot read through, and
//      EDEN_UP_FAULT_GPU's GPU faults), so a bring-back, which has a way out, never empties it. Nothing while
//      a game write waits for a block (memory_waiters): that write comes first.
//   2. With a block: the slot's lock, and if it is still Out the fault handler's own swap-in (SwapIn with
//      that block: the same unpack or file read, swap counters, packed-copy and slab release, place in the
//      file given back, the slot stamped as touched now), NoteLeftOut before the lock goes (the kept unpacked
//      scratch is stale from then on), the copy from the slot under its lock, and the slot let go Own.
//      Changed meanwhile: the block goes back where it came from and the read goes on as =1 does.
//   3. Without a block: read through as =1 does (same code, same counters), and the slot is marked wanted:
//      the keeper, every 100 ms, after it refills EDEN_UP_FAULT_RESERVE and before it refills the reserve,
//      brings wanted slots back with the same step 1 sources (at most WantedPerTick a tick; a slot not
//      served within WantedKeepMs is dropped). After a failure, reads go through without asking the pool for
//      BringBackGapMs (20 ms), so a dry pool costs no system call per read.
//   4. Repeat guard: a slot read through in RepeatReads (4) separate reads within RepeatWindowMs (1 s) is
//      tried at once, past the gap, with the same floors as step 1 ("while blocks are free"). A separate
//      read is one that comes BringBackGapMs or more after the slot's last read-through: the 4 KiB pieces
//      of one upload count once (checker: counted per piece, the guard fired inside every upload and its
//      deeper floors, reserve to 16 and fault reserve to 2, became the floors whenever the pool was dry).
// Probe and Zero slots, Shared and Fine, and dest inside the guest: as =1. Lock order: the slot's lock only
// (no scratch mutex is held for a bring-back; the read-through after it takes scratch, then slot, as =1);
// under the slot's lock only what the fault handler's swap-in does there (views, LZ4 or pread, MapSlot,
// spin locks of the reserve and the file's places), never a wait for the pool: the block is taken first.
// The fault handler is unchanged: with =2 read_through_on is set as with =1, so its 0901 paths are the same.
//
// T178b changes to =2 (run1417, T177 M1, 13 Nov: at the load without the stand =2 had just brought back 341
// slots, 304 with pool blocks and 37 from EDEN_UP_FAULT_RESERVE, when a page table slot found no block and the
// app ended; the pool's largest free piece was 1 MiB):
//   - headroom: a bring-back takes a block (the pool, or the reserve above its floor) only while the pool's
//     largest free piece is above EDEN_UP_GPU_READ_HEADROOM_MIB (digits; unset: 256 with =2 and =3; 0: no
//     headroom, as T177). Asked of the system at most every HeadroomRecheckMs (10 ms), the answer kept between;
//     not known counts as under. Under it the read goes through as =1 and the slot is marked wanted; the
//     keeper's wanted list asks the same (TakeBringBackBlock is the one place both ask).
//   - EDEN_UP_FAULT_RESERVE is no source any more with =2: it stays the faults'. =3 is =2 with it as the last
//     source (T177's =2 floors: above half its target, never its last 2), the headroom rule as in =2.
//   - EDEN_UP_GPU_READ line: read_through=3 for =3; headroom_mib, headroom_refused, headroom_largest_mib after
//     the =2 fields.
std::atomic<int> read_through_mode{0}; // 0 off, 1 read through (T175), 2 bring back first (T177a, T178b), 3 = 2 + fault reserve
std::atomic<std::size_t> bring_back_headroom{0}; // T178b: bytes; 0 none
struct BringBackCounters {
    std::atomic<std::uint64_t> pool{0}, reserve{0}, fault{0}, deep{0}, failed{0}, skipped{0}, swap_in_failed{0};
    std::atomic<std::uint64_t> after_fail{0}, wanted{0}, served{0}, expired{0}, raced{0}, ns{0};
    std::atomic<std::uint64_t> headroom_refused{0}; // T178b
};
BringBackCounters bring_back_counters;
#ifdef PS5_NATIVE
constexpr std::uint32_t BringBackGapMs = 20, RepeatReads = 4, RepeatWindowMs = 1000, WantedKeepMs = 2000;
constexpr std::size_t WantedPerTick = 32, BringBackDeepKeep = 16;
enum BlockFrom : int { FromNone = 0, FromPool, FromReserve, FromFault };
std::uint64_t bring_back_start_ms = 0;
std::atomic<std::uint32_t> bring_back_failed_ms{0}; // when the last bring-back found no block (0: never)
std::atomic<std::uint32_t> wanted_ms[OutGenSlots]{}; // when the slot was last read through for want of a block
std::atomic<std::uint32_t> repeat_since_ms[OutGenSlots]{}, repeat_reads[OutGenSlots]{};
std::atomic<std::size_t> wanted_pending{0};
// Milliseconds since the service started, plus 1 (0 means never).
std::uint32_t BringBackNowMs() noexcept {
    return static_cast<std::uint32_t>(MonotonicMs() - bring_back_start_ms) + 1;
}
// T178b headroom: the pool's largest free piece, asked at most every HeadroomRecheckMs (0: not known).
constexpr std::uint32_t HeadroomRecheckMs = 10;
std::atomic<std::uint32_t> headroom_checked_ms{0};
std::atomic<std::size_t> headroom_largest{0};
bool BringBackHeadroom() noexcept {
    const std::size_t floor = bring_back_headroom.load(std::memory_order_relaxed);
    if (floor == 0) return true;
    const std::uint32_t now = BringBackNowMs();
    const std::uint32_t at = headroom_checked_ms.load(std::memory_order_relaxed);
    std::size_t largest = headroom_largest.load(std::memory_order_relaxed);
    if (at == 0 || static_cast<std::int32_t>(now - at) >= static_cast<std::int32_t>(HeadroomRecheckMs)) {
        std::int64_t start = 0;
        largest = 0;
        const std::int64_t total = sceKernelGetDirectMemorySize();
        if (total <= 0 || sceKernelAvailableDirectMemorySize(0, total, FinePage, &start, &largest) != 0) largest = 0;
        headroom_largest.store(largest, std::memory_order_relaxed);
        headroom_checked_ms.store(now, std::memory_order_relaxed);
    }
    return largest > floor;
}
// T178b checker: set by TakeBringBackBlock when its FromNone was the headroom's, so a reader counts it
// once (headroom_refused), not also as bring_back_failed.
thread_local bool bring_back_headroom_said_no = false;
// Step 1 above: a block at once, or FromNone. The same floors for every caller (the repeat guard too).
int TakeBringBackBlock(std::int64_t* physical) noexcept {
    bring_back_headroom_said_no = false;
    if (memory_waiters.load(std::memory_order_acquire) != 0) return FromNone;
    if (!BringBackHeadroom()) { // T178b: the pool is too full to give a block to a read that has a way out
        bring_back_counters.headroom_refused.fetch_add(1, std::memory_order_relaxed);
        bring_back_headroom_said_no = true;
        return FromNone;
    }
    if (eden_direct_guest(SparseSlot, SparseSlot, 12, physical) == 0) return FromPool;
    const std::size_t floor = std::max(reserve_floor.load(std::memory_order_relaxed), BringBackDeepKeep);
    Spin(reserve_lock);
    std::size_t count = reserve_count.load(std::memory_order_relaxed);
    bool ok = count > floor;
    if (ok) {
        *physical = reserve[count - 1];
        reserve_count.store(count - 1, std::memory_order_relaxed);
    }
    reserve_lock.clear(std::memory_order_release);
    if (ok) {
        counters.reserve_used.fetch_add(1, std::memory_order_relaxed); // as ReserveBlock counts it
        return FromReserve;
    }
    // T178b: EDEN_UP_FAULT_RESERVE is the faults' with =2; only =3 takes from it (T177's =2 floors).
    if (read_through_mode.load(std::memory_order_relaxed) != 3) return FromNone;
    const std::size_t target = fault_reserve_target.load(std::memory_order_relaxed);
    if (target == 0) return FromNone;
    const std::size_t keep = std::max(FaultReserveGpuKeep, target / 2);
    Spin(fault_reserve_lock);
    count = fault_reserve_count.load(std::memory_order_relaxed);
    ok = count > keep;
    if (ok) {
        *physical = fault_reserve[--count];
        fault_reserve_count.store(count, std::memory_order_relaxed);
    }
    fault_reserve_lock.clear(std::memory_order_release);
    if (!ok) return FromNone;
    auto& c = fault_reserve_counters;
    for (std::size_t low = c.lowest.load(std::memory_order_relaxed);
         count < low && !c.lowest.compare_exchange_weak(low, count, std::memory_order_relaxed);)
        ;
    return FromFault;
}
// A block TakeBringBackBlock gave that was not used: back where it came from (or the pool when that is full).
void GiveBringBackBlock(std::int64_t physical, int from) noexcept {
    bool kept = false;
    if (from == FromReserve) {
        Spin(reserve_lock);
        const std::size_t count = reserve_count.load(std::memory_order_relaxed);
        if (count < ReserveBlocks) {
            reserve[count] = physical;
            reserve_count.store(count + 1, std::memory_order_relaxed);
            kept = true;
        }
        reserve_lock.clear(std::memory_order_release);
    } else if (from == FromFault) {
        Spin(fault_reserve_lock);
        const std::size_t count = fault_reserve_count.load(std::memory_order_relaxed);
        if (count < FaultReserveMost) {
            fault_reserve[count] = physical;
            fault_reserve_count.store(count + 1, std::memory_order_relaxed);
            kept = true;
        }
        fault_reserve_lock.clear(std::memory_order_release);
    }
    if (!kept) (void)sceKernelReleaseDirectMemory(physical, SparseSlot);
}
void CountBroughtBack(int from, std::uint64_t since_ns) noexcept {
    auto& b = bring_back_counters;
    (from == FromPool ? b.pool : from == FromReserve ? b.reserve : b.fault).fetch_add(1, std::memory_order_relaxed);
    b.ns.fetch_add(NowNs() - since_ns, std::memory_order_relaxed);
}
// The slot is locked and Out, `physical` from TakeBringBackBlock: the fault handler's swap-in with it (the
// block is used, or released on failure). True: the slot is Own now (the caller lets it go Own).
bool BringBackLocked(SparseRange& range, std::size_t slot, std::int64_t physical) noexcept {
    if (!SwapIn(range, slot, physical)) {
        bring_back_counters.swap_in_failed.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    NoteLeftOut(slot); // before the lock goes, as the fault handler does
    return true;
}
// Step 3: a read went through for want of a block: the repeat guard's count, and the keeper's list.
void NoteReadThroughWanted(std::size_t slot, std::uint32_t now) noexcept {
    bring_back_counters.after_fail.fetch_add(1, std::memory_order_relaxed);
    if (slot >= OutGenSlots) return;
    const std::uint32_t last = wanted_ms[slot].exchange(now, std::memory_order_relaxed);
    if (last == 0) {
        wanted_pending.fetch_add(1, std::memory_order_relaxed);
        bring_back_counters.wanted.fetch_add(1, std::memory_order_relaxed);
    }
    // The repeat guard counts separate reads: not the next piece of the same upload (checker fix).
    // Signed differences: another thread may have stored a later time than this thread's `now`.
    if (last != 0 && static_cast<std::int32_t>(now - last) < static_cast<std::int32_t>(BringBackGapMs)) return;
    const std::uint32_t since = repeat_since_ms[slot].load(std::memory_order_relaxed);
    if (since == 0 || static_cast<std::int32_t>(now - since) > static_cast<std::int32_t>(RepeatWindowMs)) {
        repeat_since_ms[slot].store(now, std::memory_order_relaxed);
        repeat_reads[slot].store(1, std::memory_order_relaxed);
    } else {
        repeat_reads[slot].fetch_add(1, std::memory_order_relaxed);
    }
}
#endif
// Steps 1, 2 and 4 for one slot's part of a read; the slot looked Out. True: copied (the slot is back in);
// false: the caller reads it as =1 does.
bool BringBackRead(SparseRange& range, std::size_t slot, const void* source, std::uint8_t* dest, std::size_t size) noexcept {
#ifdef PS5_NATIVE
    auto& b = bring_back_counters;
    const std::uint32_t now = BringBackNowMs();
    bool deep = false;
    if (slot < OutGenSlots) {
        const std::uint32_t since = repeat_since_ms[slot].load(std::memory_order_relaxed);
        deep = since != 0 && static_cast<std::int32_t>(now - since) <= static_cast<std::int32_t>(RepeatWindowMs) &&
               repeat_reads[slot].load(std::memory_order_relaxed) + 1 >= RepeatReads;
    }
    const std::uint32_t failed = bring_back_failed_ms.load(std::memory_order_relaxed);
    if (!deep && failed != 0 && now - failed < BringBackGapMs) {
        b.skipped.fetch_add(1, std::memory_order_relaxed);
        NoteReadThroughWanted(slot, now);
        return false;
    }
    if (deep && slot < OutGenSlots) { // the guard fires once per RepeatReads separate reads, whatever comes of it
        repeat_since_ms[slot].store(now, std::memory_order_relaxed);
        repeat_reads[slot].store(0, std::memory_order_relaxed);
    }
    const std::uint64_t start = NowNs();
    std::int64_t physical = -1;
    const int from = TakeBringBackBlock(&physical);
    if (from == FromNone) {
        bring_back_failed_ms.store(now, std::memory_order_relaxed); // the same 20 ms gap after a headroom refusal
        if (!bring_back_headroom_said_no) b.failed.fetch_add(1, std::memory_order_relaxed);
        NoteReadThroughWanted(slot, now);
        return false;
    }
    LockSlot(range, slot);
    const std::uint8_t state = SlotState(range, slot);
    if (state != SlotOut) { // brought back, given back or written meanwhile
        UnlockSlot(range, slot, state);
        GiveBringBackBlock(physical, from);
        b.raced.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (!BringBackLocked(range, slot, physical)) {
        UnlockSlot(range, slot, SlotOut);
        NoteReadThroughWanted(slot, now);
        return false;
    }
    std::memcpy(dest, source, size); // Own and mapped read-write; dest lies outside the guest (GuestReadThrough)
    UnlockSlot(range, slot, SlotOwn);
    CountBroughtBack(from, start);
    if (deep) b.deep.fetch_add(1, std::memory_order_relaxed);
    return true;
#else
    (void)range;
    (void)slot;
    (void)source;
    (void)dest;
    (void)size;
    return false;
#endif
}
#ifdef PS5_NATIVE
// The keeper, every 100 ms, after EDEN_UP_FAULT_RESERVE is refilled and before the reserve is (=2 only; otherwise one relaxed load).
void ServeWantedBack() noexcept {
    if (wanted_pending.load(std::memory_order_relaxed) == 0 || read_through_mode.load(std::memory_order_relaxed) < 2) return; // =2 and =3
    SparseRange* const range = service_range.load(std::memory_order_acquire);
    if (range == nullptr || range->state == nullptr) return;
    auto& b = bring_back_counters;
    const std::size_t slots = std::min(OutGenSlots, (range->end - range->begin.load(std::memory_order_relaxed)) / SparseSlot);
    const std::uint32_t now = BringBackNowMs();
    const auto drop = [&](std::size_t slot) noexcept {
        if (wanted_ms[slot].exchange(0, std::memory_order_relaxed) != 0) wanted_pending.fetch_sub(1, std::memory_order_relaxed);
    };
    std::size_t served = 0;
    bool dry = false;
    for (std::size_t slot = 0; slot < slots; ++slot) {
        const std::uint32_t when = wanted_ms[slot].load(std::memory_order_relaxed);
        if (when == 0) continue;
        if (SlotState(*range, slot) != SlotOut) { // back in already (a read, a fault) or given back
            drop(slot);
            continue;
        }
        if (static_cast<std::int32_t>(now - when) > static_cast<std::int32_t>(WantedKeepMs)) { // signed: a reader may store a later time
            drop(slot);
            b.expired.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        if (dry || served >= WantedPerTick) continue;
        const std::uint64_t start = NowNs();
        std::int64_t physical = -1;
        const int from = TakeBringBackBlock(&physical);
        if (from == FromNone) {
            dry = true; // kept for the next tick
            continue;
        }
        LockSlot(*range, slot);
        const std::uint8_t state = SlotState(*range, slot);
        if (state != SlotOut) {
            UnlockSlot(*range, slot, state);
            GiveBringBackBlock(physical, from);
            drop(slot);
            continue;
        }
        const bool ok = BringBackLocked(*range, slot, physical);
        UnlockSlot(*range, slot, ok ? SlotOwn : SlotOut);
        drop(slot);
        if (ok) {
            ++served;
            CountBroughtBack(from, start);
            b.served.fetch_add(1, std::memory_order_relaxed);
        }
    }
}
#endif

// The slot is locked and Out, sc.mutex is held: [offset, offset + size) of its bytes to `dest`.
bool ReadOutLocked(ReadScratch& sc, SparseRange& range, std::size_t slot, std::size_t offset, void* dest,
                   std::size_t size) noexcept {
#ifdef PS5_NATIVE
    auto& c = read_through_counters;
    const std::int64_t place = range.physical[slot];
    const std::uint32_t stored = range.swap[slot]; // the packed size, or the place in the file
    if (place == -1) { // EDEN_GUEST_SWAP=2: in the file
        if (swap_fd < 0) return false;
        auto* at = static_cast<std::uint8_t*>(dest);
        std::size_t done = 0;
        while (done < size) {
            const auto got = pread(swap_fd, at + done, size - done,
                                   static_cast<off_t>(stored) * static_cast<off_t>(SparseSlot) + static_cast<off_t>(offset + done));
            if (got < 0 && errno == EINTR) continue;
            if (got <= 0) return false;
            done += static_cast<std::size_t>(got);
        }
        c.file_reads.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
#ifdef EDEN_SWAP_LZ4
    if (sc.data == nullptr) return false;
    const bool keepable = slot < OutGenSlots;
    const std::uint32_t generation = keepable ? out_generation[slot].load(std::memory_order_acquire) : 0;
    if (keepable && sc.kept_range == &range && sc.kept_slot == slot && sc.kept_place == place &&
        sc.kept_size == stored && sc.kept_generation == generation) {
        c.kept_hits.fetch_add(1, std::memory_order_relaxed);
    } else {
        const std::uint64_t start = NowNs();
        sc.kept_slot = SIZE_MAX; // nothing good in the scratch until this unpack is done
        const bool store = InStore(place); // EDEN_UP_PACK_SHARED: a copy in a slab
        const std::int64_t block = store ? pack_slabs[StoreSlab(place)].physical.load(std::memory_order_acquire) : place;
        const std::size_t mapped = store ? SparseSlot : PackedPages(stored) * FinePage;
        if (block < 0 || stored == 0 || stored > LZ4_COMPRESSBOUND(SparseSlot)) return false;
        void* view = reinterpret_cast<void*>(slot_view_hint.load(std::memory_order_relaxed));
        if (sceKernelMapDirectMemory(&view, mapped, PROT_READ, 0, block, store ? SparseSlot : FinePage) != 0) return false;
        const char* from = static_cast<const char*>(view) + (store ? StoreOffset(place) : 0);
        const bool ok = LZ4_decompress_safe(from, sc.data, static_cast<int>(stored),
                                            static_cast<int>(SparseSlot)) == static_cast<int>(SparseSlot);
        if (munmap(view, mapped) != 0) std::abort();
        if (!ok) return false;
        if (keepable) {
            sc.kept_range = &range;
            sc.kept_slot = slot;
            sc.kept_place = place;
            sc.kept_size = stored;
            sc.kept_generation = generation;
        }
        c.unpacks.fetch_add(1, std::memory_order_relaxed);
        c.unpack_ns.fetch_add(NowNs() - start, std::memory_order_relaxed);
    }
    std::memcpy(dest, sc.data + offset, size);
    return true;
#else
    (void)sc;
    (void)offset;
    (void)dest;
    (void)size;
    return false;
#endif
#else
    (void)sc;
    (void)range;
    (void)slot;
    (void)offset;
    (void)dest;
    (void)size;
    return false;
#endif
}

// One slot's part of a read.
void ReadPiece(SparseRange& range, std::size_t slot, std::size_t offset, std::uint8_t* dest, std::size_t size) noexcept {
    const auto* source = reinterpret_cast<const void*>(SlotAddress(range, slot) + offset);
    const std::uint8_t state = range.state[slot].load(std::memory_order_acquire) & SlotMask;
    if (state != SlotOut && state != SlotProbe) { // mapped and readable (a locked one too: as before)
        std::memcpy(dest, source, size);
        return;
    }
    // T177a EDEN_UP_GPU_READ_THROUGH=2: an Out slot is brought back first when a block is there at once.
    if (state == SlotOut && read_through_mode.load(std::memory_order_relaxed) >= 2 && // =2 and =3 (T178b)
        BringBackRead(range, slot, source, dest, size))
        return;
    auto& c = read_through_counters;
    bool copied = false;
    {
        ReadScratch& sc = read_scratch[GpuThreadKind()]; // 0901: this reader kind's own
        const std::lock_guard hold{sc.mutex};
        if (LockSlot(range, slot)) c.lock_waits.fetch_add(1, std::memory_order_relaxed);
        const std::uint8_t now = SlotState(range, slot);
        if (now == SlotOut) {
            copied = ReadOutLocked(sc, range, slot, offset, dest, size);
            if (copied) {
                c.out_pieces.fetch_add(1, std::memory_order_relaxed);
                c.out_bytes.fetch_add(size, std::memory_order_relaxed);
            } else {
                c.failures.fetch_add(1, std::memory_order_relaxed);
            }
            UnlockSlot(range, slot, SlotOut);
        } else if (now == SlotProbe) {
            const bool touched = TouchProbe(range, slot);
            if (touched) c.probes.fetch_add(1, std::memory_order_relaxed);
            UnlockSlot(range, slot, touched ? SlotOwn : SlotProbe);
        } else {
            UnlockSlot(range, slot, now);
        }
    }
    if (!copied) std::memcpy(dest, source, size); // the old way: a fault brings the slot back if it must
}

// core 0900's Common::GuestReadThrough::copy_fn. False: not a read of the guest's memory (or into it).
bool GuestReadThrough(void* dest, const void* source, std::size_t size) {
    if (!read_through_on.load(std::memory_order_relaxed)) return false;
    SparseRange* const range = service_range.load(std::memory_order_acquire);
    if (range == nullptr || range->state == nullptr) return false;
    const std::uintptr_t begin = range->begin.load(std::memory_order_relaxed), end = range->end;
    const auto from = reinterpret_cast<std::uintptr_t>(source), to = reinterpret_cast<std::uintptr_t>(dest);
    if (begin == 0 || from < begin || from >= end || size > end - from) return false;
    if (to < end && to + size > begin) return false; // into the guest's memory: the copy as before
    auto* out = static_cast<std::uint8_t*>(dest);
    std::uintptr_t at = from;
    for (std::size_t left = size; left != 0;) {
        const std::size_t slot = (at - begin) / SparseSlot, offset = (at - begin) % SparseSlot;
        const std::size_t piece = std::min(left, SparseSlot - offset);
        ReadPiece(*range, slot, offset, out, piece);
        out += piece;
        at += piece;
        left -= piece;
    }
    return true;
}

// core 0900's Common::GuestReadThrough::readable_fn: false when a span would cover a cold slot.
bool GuestSpanReadable(const void* source, std::size_t size) {
    if (size == 0 || !read_through_on.load(std::memory_order_relaxed)) return true;
    SparseRange* const range = service_range.load(std::memory_order_acquire);
    if (range == nullptr || range->state == nullptr) return true;
    const std::uintptr_t begin = range->begin.load(std::memory_order_relaxed), end = range->end;
    const auto from = reinterpret_cast<std::uintptr_t>(source);
    if (begin == 0 || from < begin || from >= end || size > end - from) return true;
    for (std::size_t slot = (from - begin) / SparseSlot, last = (from + size - 1 - begin) / SparseSlot; slot <= last; ++slot) {
        const std::uint8_t state = range->state[slot].load(std::memory_order_acquire) & SlotMask;
        if (state == SlotOut || state == SlotProbe) {
            read_through_counters.spans_copied.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
    }
    return true;
}
} // namespace

// core 0900 (video_core/memory_manager.cpp). Weak: the host checks link this file without the core.
extern "C" void eden_core_set_guest_read_through(bool (*copy)(void*, const void*, std::size_t),
                                                  bool (*readable)(const void*, std::size_t)) __attribute__((weak));

namespace {
// As the service starts: EDEN_UP_GPU_READ_THROUGH, EDEN_UP_FAULT_RESERVE, EDEN_UP_FAULT_GPU,
// EDEN_UP_FAULT_LOG (0900). Unset: nothing set, nothing filled, no line.
bool GpuReadSwitchesOn() noexcept {
#ifdef PS5_NATIVE
    if (fault_reserve_target.load(std::memory_order_relaxed) != 0 || fault_gpu_first.load(std::memory_order_relaxed))
        return true;
#endif
    return read_through_on.load(std::memory_order_relaxed) || fault_log_fd.load(std::memory_order_relaxed) >= 0 ||
           fault_last_try.load(std::memory_order_relaxed); // T178b EDEN_UP_FAULT_LAST_TRY=1 alone
}
void ReadGpuReadSwitches() noexcept {
    fault_log_start_ms = MonotonicMs();
    fault_log_lines.store(0, std::memory_order_relaxed);
    fault_log_last_ms.store(0, std::memory_order_relaxed);
    fault_log_fd.store(Common::PeSwitch("EDEN_UP_FAULT_LOG") ? fileno(stderr) : -1, std::memory_order_relaxed);
    const bool gpu = Common::PeSwitch("EDEN_UP_FAULT_GPU");
    fault_gpu_trim.store(gpu, std::memory_order_relaxed);
#ifdef PS5_NATIVE
    std::size_t blocks = 0;
    if (const char* value = std::getenv("EDEN_UP_FAULT_RESERVE")) {
        const unsigned long asked = std::strtoul(value, nullptr, 10);
        blocks = asked == 1 ? FaultReserveDefault : std::min<std::size_t>(asked, FaultReserveMost);
    }
    fault_reserve_target.store(blocks, std::memory_order_relaxed);
    fault_gpu_first.store(gpu, std::memory_order_relaxed);
    FillFaultReserve();
#endif
    // T177a: =1 as T175 (Common::PeSwitch's rule, word for word), =2 bring back first; anything else off.
    const char* const through_value = std::getenv("EDEN_UP_GPU_READ_THROUGH");
    // T178b: =3 too (=2 with EDEN_UP_FAULT_RESERVE as a bring-back's last source).
    const int asked_mode = Common::PeSwitch("EDEN_UP_GPU_READ_THROUGH") ? 1
                           : through_value != nullptr && (through_value[0] == '2' || through_value[0] == '3') &&
                                     through_value[1] == '\0'
                               ? through_value[0] - '0'
                               : 0;
    // T178b EDEN_UP_GPU_READ_HEADROOM_MIB (=2 and =3 only): digits, 0 to 8192; unset or anything else: 256.
    {
        std::size_t mib = 256;
        if (const char* value = std::getenv("EDEN_UP_GPU_READ_HEADROOM_MIB"); value != nullptr && *value != '\0') {
            std::size_t asked = 0;
            const char* at = value;
            while (*at >= '0' && *at <= '9' && asked <= 8192) asked = asked * 10 + static_cast<std::size_t>(*at++ - '0');
            if (*at == '\0' && asked <= 8192) mib = asked;
        }
        bring_back_headroom.store(asked_mode >= 2 ? mib << 20 : 0, std::memory_order_relaxed);
        bring_back_counters.headroom_refused.store(0, std::memory_order_relaxed);
#ifdef PS5_NATIVE
        headroom_checked_ms.store(0, std::memory_order_relaxed);
        headroom_largest.store(0, std::memory_order_relaxed);
#endif
    }
    bool through = asked_mode != 0;
    bool scratches = true; // 0901: one per reader kind (3 x 2 MiB)
    for (ReadScratch& sc : read_scratch) {
        if (through && sc.data == nullptr) {
            sc.data = new (std::nothrow) char[SparseSlot];
            if (sc.data != nullptr) std::memset(sc.data, 0, SparseSlot); // its pages in now
        }
        scratches = scratches && sc.data != nullptr;
        const std::lock_guard hold{sc.mutex};
        sc.kept_slot = SIZE_MAX;
    }
    through = through && scratches && eden_core_set_guest_read_through != nullptr;
    // T177a: the mode before read_through_on, and =2's tables empty (nothing reads them before the hook is set).
    for (auto* counter : {&bring_back_counters.pool, &bring_back_counters.reserve, &bring_back_counters.fault,
                          &bring_back_counters.deep, &bring_back_counters.failed, &bring_back_counters.skipped,
                          &bring_back_counters.swap_in_failed, &bring_back_counters.after_fail, &bring_back_counters.wanted,
                          &bring_back_counters.served, &bring_back_counters.expired, &bring_back_counters.raced,
                          &bring_back_counters.ns})
        counter->store(0, std::memory_order_relaxed);
#ifdef PS5_NATIVE
    bring_back_start_ms = MonotonicMs();
    bring_back_failed_ms.store(0, std::memory_order_relaxed);
    for (std::size_t slot = 0; slot < OutGenSlots; ++slot) {
        wanted_ms[slot].store(0, std::memory_order_relaxed);
        repeat_since_ms[slot].store(0, std::memory_order_relaxed);
        repeat_reads[slot].store(0, std::memory_order_relaxed);
    }
    wanted_pending.store(0, std::memory_order_relaxed);
#endif
    read_through_mode.store(through ? asked_mode : 0, std::memory_order_relaxed);
    read_through_on.store(through, std::memory_order_relaxed);
    if (eden_core_set_guest_read_through != nullptr)
        eden_core_set_guest_read_through(through ? &GuestReadThrough : nullptr, through ? &GuestSpanReadable : nullptr);
    // T178b EDEN_UP_FAULT_LAST_TRY (at fault_last_try): =1 on, =0 off, unset: on while read-through is on.
    {
        const char* const last = std::getenv("EDEN_UP_FAULT_LAST_TRY");
        const bool asked_on = last != nullptr && last[0] == '1' && last[1] == '\0';
        const bool asked_off = last != nullptr && last[0] == '0' && last[1] == '\0';
        fault_last_try.store(asked_on || (through && !asked_off), std::memory_order_relaxed);
        for (auto* counter : {&fault_last_try_counters.tries, &fault_last_try_counters.won, &fault_last_try_counters.lost,
                              &fault_last_try_counters.given_bytes, &fault_last_try_counters.swap_busy,
                              &fault_last_try_counters.slots_out})
            counter->store(0, std::memory_order_relaxed);
    }
    if (!GpuReadSwitchesOn()) return;
    if (through || fault_last_try.load(std::memory_order_relaxed)) {
        char more[300];
        const int written = std::snprintf(
            more, sizeof(more),
            "EDEN_UP_GPU_READ T178b: headroom=%zuMiB (bring-backs, =2 and =3) fault_reserve_source=%d fault_last_try=%d "
            "fault_gpu_keeps_from_cpu=%d\n",
            bring_back_headroom.load(std::memory_order_relaxed) >> 20, through && asked_mode == 3 ? 1 : 0,
            fault_last_try.load(std::memory_order_relaxed) ? 1 : 0, gpu && !through ? 1 : 0);
        if (written > 0) Report(more);
    }
    char line[320];
    const int length = std::snprintf(
        line, sizeof(line),
        "EDEN_UP_GPU_READ on: read_through=%d%s fault_reserve=%zu blocks (filled %zu) fault_gpu=%d fault_log=%d "
        "(EDEN_GUEST_SWAP=%d)\n",
        through ? asked_mode : 0,
        asked_mode != 0 && !through ? " (refused: no scratch or no core hook)" : "",
#ifdef PS5_NATIVE
        fault_reserve_target.load(std::memory_order_relaxed), fault_reserve_count.load(std::memory_order_relaxed),
#else
        std::size_t{0}, std::size_t{0},
#endif
        gpu ? 1 : 0, fault_log_fd.load(std::memory_order_relaxed) >= 0 ? 1 : 0, swap_mode.load(std::memory_order_relaxed));
    if (length > 0) Report(line);
}
// The service stops: the core's reads are plain copies again.
void StopGpuRead() noexcept {
    read_through_on.store(false, std::memory_order_relaxed);
    read_through_mode.store(0, std::memory_order_relaxed); // T177a
    if (eden_core_set_guest_read_through != nullptr) eden_core_set_guest_read_through(nullptr, nullptr);
}
// One line with the counts (beside each EDEN_CRAM line, and every 10 s from the keeper when they moved).
void GpuReadReport(const char* when) noexcept {
    if (!GpuReadSwitchesOn()) return;
    const auto& r = read_through_counters;
    const auto& w = fault_wait_counters;
    char line[1500];
    int length = std::snprintf(
        line, sizeof(line),
        "EDEN_UP_GPU_READ %s read_through=%d out_reads=%llu out_MiB=%llu unpacks=%llu unpack_ms=%llu kept_hits=%llu "
        "file_reads=%llu probes_touched=%llu lock_waits=%llu failed=%llu spans_copied=%llu; fault_reserve=%zu/%zu "
        "taken=%llu taken_gpu=%llu empty=%llu kept_for_gpu=%llu lowest=%zu filled=%llu; block_waits=%llu "
        "gpu_waits=%llu gpu_waits_out=%llu gpu_won=%llu gpu_lost=%llu gpu_trim_won=%llu gpu_wait_ms=%llu "
        "longest_wait_ms=%llu\n",
        when, read_through_on.load(std::memory_order_relaxed) ? read_through_mode.load(std::memory_order_relaxed) : 0,
        (unsigned long long)r.out_pieces.load(std::memory_order_relaxed),
        (unsigned long long)(r.out_bytes.load(std::memory_order_relaxed) >> 20),
        (unsigned long long)r.unpacks.load(std::memory_order_relaxed),
        (unsigned long long)(r.unpack_ns.load(std::memory_order_relaxed) / 1000000),
        (unsigned long long)r.kept_hits.load(std::memory_order_relaxed),
        (unsigned long long)r.file_reads.load(std::memory_order_relaxed),
        (unsigned long long)r.probes.load(std::memory_order_relaxed),
        (unsigned long long)r.lock_waits.load(std::memory_order_relaxed),
        (unsigned long long)r.failures.load(std::memory_order_relaxed),
        (unsigned long long)r.spans_copied.load(std::memory_order_relaxed),
#ifdef PS5_NATIVE
        fault_reserve_count.load(std::memory_order_relaxed), fault_reserve_target.load(std::memory_order_relaxed),
        (unsigned long long)fault_reserve_counters.taken.load(std::memory_order_relaxed),
        (unsigned long long)fault_reserve_counters.taken_gpu.load(std::memory_order_relaxed),
        (unsigned long long)fault_reserve_counters.empty.load(std::memory_order_relaxed),
        (unsigned long long)fault_reserve_counters.kept_for_gpu.load(std::memory_order_relaxed),
        fault_reserve_counters.lowest.load(std::memory_order_relaxed) == FaultReserveMost
            ? fault_reserve_target.load(std::memory_order_relaxed)
            : fault_reserve_counters.lowest.load(std::memory_order_relaxed),
        (unsigned long long)fault_reserve_counters.filled.load(std::memory_order_relaxed),
#else
        std::size_t{0}, std::size_t{0}, 0ull, 0ull, 0ull, 0ull, std::size_t{0}, 0ull,
#endif
        (unsigned long long)w.waits.load(std::memory_order_relaxed),
        (unsigned long long)w.gpu_waits.load(std::memory_order_relaxed),
        (unsigned long long)w.gpu_out.load(std::memory_order_relaxed),
        (unsigned long long)w.gpu_won.load(std::memory_order_relaxed),
        (unsigned long long)w.gpu_lost.load(std::memory_order_relaxed),
        (unsigned long long)w.gpu_trim_won.load(std::memory_order_relaxed),
        (unsigned long long)w.gpu_wait_ms.load(std::memory_order_relaxed),
        (unsigned long long)w.most_ms.load(std::memory_order_relaxed));
    // T177a =2: the bring-back's fields after these (=1 and off: the line as T175 wrote it).
    if (length > 0 && static_cast<std::size_t>(length) < sizeof(line) && read_through_mode.load(std::memory_order_relaxed) >= 2) {
        const auto& b = bring_back_counters;
        const auto n = [](const std::atomic<std::uint64_t>& v) noexcept { return (unsigned long long)v.load(std::memory_order_relaxed); };
        --length; // over the line's end
        const int more = std::snprintf(
            line + length, sizeof(line) - static_cast<std::size_t>(length),
            "; brought_back=%llu pool=%llu reserve=%llu fault_reserve=%llu deep=%llu bring_back_ms=%llu "
            "bring_back_failed=%llu bring_back_skipped=%llu swap_in_failed=%llu raced=%llu read_through_after_fail=%llu "
            "wanted=%llu wanted_served=%llu wanted_expired=%llu wanted_now=%zu headroom_mib=%zu headroom_refused=%llu "
            "headroom_largest_mib=%zu\n",
            n(b.pool) + n(b.reserve) + n(b.fault), n(b.pool), n(b.reserve), n(b.fault), n(b.deep), n(b.ns) / 1000000,
            n(b.failed), n(b.skipped), n(b.swap_in_failed), n(b.raced), n(b.after_fail), n(b.wanted), n(b.served),
            n(b.expired),
#ifdef PS5_NATIVE
            wanted_pending.load(std::memory_order_relaxed),
#else
            std::size_t{0},
#endif
            bring_back_headroom.load(std::memory_order_relaxed) >> 20, n(b.headroom_refused),
#ifdef PS5_NATIVE
            headroom_largest.load(std::memory_order_relaxed) >> 20
#else
            std::size_t{0}
#endif
        );
        length = more > 0 && static_cast<std::size_t>(more) < sizeof(line) - static_cast<std::size_t>(length) ? length + more
                                                                                                             : length + 1;
    }
    if (length > 0) Report(line);
    // T178b: the CPU side's two changes (with read-through on, or EDEN_UP_FAULT_LAST_TRY=1): one more line.
    const auto& t = fault_last_try_counters;
    const bool cpu_side = fault_last_try.load(std::memory_order_relaxed) || read_through_on.load(std::memory_order_relaxed);
    if (cpu_side) {
        char more[320];
        const int written = std::snprintf(
            more, sizeof(more),
            "EDEN_UP_FAULT_LAST_TRY %s on=%d tries=%llu won=%llu lost=%llu given_MiB=%llu slots_out=%llu swap_busy=%llu; "
            "cpu_took_gpu_keep=%llu\n",
            when, fault_last_try.load(std::memory_order_relaxed) ? 1 : 0,
            (unsigned long long)t.tries.load(std::memory_order_relaxed), (unsigned long long)t.won.load(std::memory_order_relaxed),
            (unsigned long long)t.lost.load(std::memory_order_relaxed),
            (unsigned long long)(t.given_bytes.load(std::memory_order_relaxed) >> 20),
            (unsigned long long)t.slots_out.load(std::memory_order_relaxed),
            (unsigned long long)t.swap_busy.load(std::memory_order_relaxed),
#ifdef PS5_NATIVE
            (unsigned long long)fault_reserve_counters.gpu_keep_waived.load(std::memory_order_relaxed)
#else
            0ull
#endif
        );
        if (written > 0) Report(more);
    }
}
// From the keeper every 100 ms: the line every 10 s while any count moved.
void GpuReadTick(std::uint64_t now_ns) noexcept {
    static std::uint64_t reported_ns = 0, seen = 0; // the keeper's own
    if (now_ns - reported_ns < 10'000'000'000ull || !GpuReadSwitchesOn()) return;
    const auto& r = read_through_counters;
    const auto& w = fault_wait_counters;
    std::uint64_t moved = r.out_pieces.load(std::memory_order_relaxed) + r.probes.load(std::memory_order_relaxed) +
                          r.failures.load(std::memory_order_relaxed) + r.spans_copied.load(std::memory_order_relaxed) +
                          w.waits.load(std::memory_order_relaxed);
    moved += bring_back_counters.pool.load(std::memory_order_relaxed) + bring_back_counters.reserve.load(std::memory_order_relaxed) +
             bring_back_counters.fault.load(std::memory_order_relaxed) + bring_back_counters.after_fail.load(std::memory_order_relaxed); // T177a (0 unless =2)
    moved += fault_last_try_counters.tries.load(std::memory_order_relaxed) +
             bring_back_counters.headroom_refused.load(std::memory_order_relaxed); // T178b (0 unless on)
#ifdef PS5_NATIVE
    moved += fault_reserve_counters.taken.load(std::memory_order_relaxed) +
             fault_reserve_counters.taken_gpu.load(std::memory_order_relaxed) +
             fault_reserve_counters.empty.load(std::memory_order_relaxed);
#endif
    reported_ns = now_ns;
    if (moved == seen) return;
    seen = moved;
    GpuReadReport("tick");
}

// Gives a range slot states (a guest range, or the self-test's).
bool AddStates(SparseRange& range) noexcept {
    const std::size_t slots = (range.end - range.begin.load(std::memory_order_relaxed)) / SparseSlot;
    auto* state = new (std::nothrow) std::atomic<std::uint8_t>[slots];
    auto* fine = state ? new (std::nothrow) std::uint16_t[slots]() : nullptr;
    auto* fine_after = fine ? new (std::nothrow) std::uint32_t[slots]() : nullptr;
    auto* swap = fine_after ? new (std::nothrow) std::uint32_t[slots]() : nullptr;
    if (swap == nullptr) {
        delete[] state;
        delete[] fine;
        delete[] fine_after;
        return false;
    }
    range.fine = fine;
    range.fine_after = fine_after;
    range.swap = swap;
    for (std::size_t slot = 0; slot < slots; ++slot)
        state[slot].store(range.owned[slot].load(std::memory_order_relaxed) ? SlotOwn : SlotZero, std::memory_order_relaxed);
    range.state = state;
    return true;
}

// ---- The scanner ----
std::thread scanner;
std::mutex scanner_mutex;
std::condition_variable scanner_wake;
bool scanner_stop = false;
// EDEN_UP_STALL_PROBE (stall_probe.h): a thread that notices stalls and says what the service was
// doing in them.
std::thread stall_probe;
std::atomic<bool> stall_probe_stop{false};
void StartStallProbe() {
    if (!StallProbe::ReadSwitch() || stall_probe.joinable()) return;
    StallProbe::Get().start_ns.store(StallProbe::NowNs(), std::memory_order_relaxed);
    stall_probe_stop.store(false, std::memory_order_relaxed);
    stall_probe = std::thread([] {
        if (::eden_crash_name_thread) ::eden_crash_name_thread("StallProbe");  // crash report, thread dump
        StallProbe::ProbeLoop([] { return stall_probe_stop.load(std::memory_order_relaxed); },
                              [](std::uint64_t ns) {
                                  const timespec pause{static_cast<time_t>(ns / 1'000'000'000ull),
                                                       static_cast<long>(ns % 1'000'000'000ull)};
                                  nanosleep(&pause, nullptr);
                              },
                              StallProbe::NowNs, [](const char* line) { Report(line); });
    });
    Report("EDEN_UP_STALL_PROBE on: an EDEN_STALL line for each wake of a 10 ms sleep 100 ms or more late, "
           "with the slot lock waits and holds and the block waits of 50 ms or more since the last one "
           "(count/longest by role), and EDEN_SLOT_LOCK totals beside each EDEN_CRAM line\n");
}
void StopStallProbe() {
    stall_probe_stop.store(true, std::memory_order_relaxed);
    if (stall_probe.joinable()) stall_probe.join();
}
#ifdef PS5_NATIVE
// The reserve kept full: every 100 ms, so a burst of writes that used it finds it whole again as
// soon as the caches have given memory back (the scanner only passes every 20 s).
std::thread keeper;
void TableCommitTick(std::uint64_t now_ns) noexcept; // T178b, below with the table commit's wait

void ReserveKeeper() {
    if (::eden_crash_name_thread) ::eden_crash_name_thread("CramKeeper");  // crash report, thread dump
    StallProbe::role = StallProbe::Keeper;
    std::unique_lock lock{scanner_mutex};
    while (!scanner_stop) {
        scanner_wake.wait_for(lock, std::chrono::milliseconds(100), [] { return scanner_stop; });
        if (scanner_stop) break;
        lock.unlock();
        std::size_t blocks = pool_is_short.load(std::memory_order_relaxed) ? reserve_floor.load(std::memory_order_relaxed)
                                                                           : reserve_target.load(std::memory_order_relaxed);
        // EDEN_UP_RADV_OOM_LEND: for 2 s after a lend to the driver, no further than that lend's floor;
        // then one line, the hold cleared (a lend meanwhile has set a new one and keeps it), and the
        // fill as before.
        if (std::uint64_t hold = keeper_hold_until_ns.load(std::memory_order_acquire); hold != 0) {
            if (NowNs() < hold) {
                blocks = std::min(blocks, keeper_hold_floor.load(std::memory_order_relaxed));
            } else if (keeper_hold_until_ns.compare_exchange_strong(hold, 0, std::memory_order_relaxed)) {
                char line[128];
                const int length = std::snprintf(line, sizeof(line),
                                                 "EDEN_UP_RADV_OOM_LEND hold over: the reserve holds %zu blocks, filled "
                                                 "again up to %zu\n",
                                                 reserve_count.load(std::memory_order_relaxed), blocks);
                if (length > 0) Report(line);
            }
        }
        FillTableReserve(); // T178b EDEN_UP_TABLE_RESERVE before all (off: one relaxed load)
        FillFaultReserve(); // 0900 EDEN_UP_FAULT_RESERVE first (off: one relaxed load)
        // T177a EDEN_UP_GPU_READ_THROUGH=2: wanted slots back, after the faults' reserve and before the
        // reserve (checker: served first, up to 32 bring-backs a tick took the blocks the faults' reserve was
        // waiting for after a dry spell). Otherwise: one relaxed load.
        ServeWantedBack();
        FillReserve(blocks);
        // EDEN_UP_JIT_LEND_UNDER_FLOOR's line (off: one relaxed load).
        if (jit_lend_mode.load(std::memory_order_relaxed) != 0) JitLendTick(NowNs());
        GpuReadTick(NowNs()); // 0900's line every 10 s while its counts move (off: nothing)
        TableCommitTick(NowNs()); // T178b: the tables' line every 10 s while its counts move (none: nothing)
        lock.lock();
    }
}
#endif

#ifdef PS5_NATIVE
// Own -> Probe, unless it was touched a moment ago. A hot slot pays one fault every SwapRearmTicks.
bool ArmSlot(SparseRange& range, std::size_t slot, bool at_once = false) noexcept {
    LockSlot(range, slot);
    const std::uint8_t state = SlotState(range, slot);
    const std::uint32_t now = swap_tick.load(std::memory_order_relaxed);
    if (state != SlotOwn || (!at_once && now - range.swap[slot] < SwapRearmTicks)) {
        UnlockSlot(range, slot, state);
        return false;
    }
    if (mprotect(reinterpret_cast<void*>(SlotAddress(range, slot)), SparseSlot, PROT_NONE) != 0) std::abort();
    range.swap[slot] = now;
    swap_counters.armed.fetch_add(1, std::memory_order_relaxed);
    UnlockSlot(range, slot, SlotProbe);
    return true;
}

// Probe -> Out. The slot is fenced while it is written, so readers go on and a writer waits for the
// lock; a Probe slot that is touched meanwhile goes back to Own only once this is over. The mapping
// goes away whole and the address space is reserved at once, as the heap gives a block back.
bool SwapOut(SparseRange& range, std::size_t slot) noexcept {
#ifdef EDEN_SWAP_LZ4
    if (swap_mode.load(std::memory_order_relaxed) == 3)
        return pack_shared.load(std::memory_order_relaxed) ? PackShared(range, slot) : PackOut(range, slot);
#endif
    LockSlot(range, slot);
    std::uint32_t place = 0;
    if (SlotState(range, slot) != SlotProbe || !TakeSwapPlace(&place)) {
        UnlockSlot(range, slot, SlotState(range, slot));
        return false;
    }
    const std::uintptr_t at = SlotAddress(range, slot);
    const std::uint64_t start = NowNs();
    if (mprotect(reinterpret_cast<void*>(at), SparseSlot, PROT_READ) != 0) std::abort();
    if (!SwapWrite(reinterpret_cast<const void*>(at), place)) {
        if (mprotect(reinterpret_cast<void*>(at), SparseSlot, PROT_NONE) != 0) std::abort();
        GiveSwapPlace(place);
        swap_counters.failures.fetch_add(1, std::memory_order_relaxed);
        UnlockSlot(range, slot, SlotProbe);
        return false;
    }
    const std::int64_t block = range.physical[slot];
    if (munmap(reinterpret_cast<void*>(at), SparseSlot) != 0) std::abort();
    void* again = reinterpret_cast<void*>(at);
    if (sceKernelReserveVirtualRange(&again, SparseSlot, MAP_FIXED, SparseSlot) != 0 || again != reinterpret_cast<void*>(at))
        std::abort();
    if (sceKernelReleaseDirectMemory(block, SparseSlot) != 0) std::abort();
    range.physical[slot] = -1;
    range.swap[slot] = place;
    guest_committed.fetch_sub(SparseSlot, std::memory_order_relaxed);
    swap_counters.out.fetch_add(1, std::memory_order_relaxed);
    swap_counters.out_ns.fetch_add(NowNs() - start, std::memory_order_relaxed);
    UnlockSlot(range, slot, SlotOut);
    return true;
}

// What LZ4 would make of a slot that just became cold: four samples of 64 KiB, read fenced.
void SampleColdSlot(SparseRange& range, std::size_t slot) noexcept {
#ifdef EDEN_SWAP_LZ4
    constexpr int Sample = 65536;
    static char packed[LZ4_COMPRESSBOUND(Sample)];
    LockSlot(range, slot);
    if (SlotState(range, slot) == SlotProbe) {
        const std::uintptr_t at = SlotAddress(range, slot);
        if (mprotect(reinterpret_cast<void*>(at), SparseSlot, PROT_READ) != 0) std::abort();
        for (std::size_t offset = 0; offset < SparseSlot; offset += SparseSlot / 4) {
            const int size = LZ4_compress_default(reinterpret_cast<const char*>(at + offset), packed, Sample, sizeof(packed));
            swap_counters.sampled.fetch_add(Sample, std::memory_order_relaxed);
            swap_counters.packed.fetch_add(size > 0 ? static_cast<std::uint64_t>(size) : Sample, std::memory_order_relaxed);
        }
        if (mprotect(reinterpret_cast<void*>(at), SparseSlot, PROT_NONE) != 0) std::abort();
    }
    UnlockSlot(range, slot, SlotState(range, slot));
#else
    (void)range;
    (void)slot;
#endif
}

// One tick: the next share of the slots is armed; slots that stayed Probe long enough are cold; and
// while the pool is short the cold ones go out, lowest in the pool first, so that blocks that lie
// side by side leave together (a pool of 2 MiB holes serves a buffer, not an image).
void SwapTick(SparseRange& range, std::size_t& hand, std::vector<std::pair<std::int64_t, std::size_t>>& cold) noexcept {
    const std::size_t slots = (range.end - range.begin.load(std::memory_order_relaxed)) / SparseSlot;
    static std::uint64_t in_before = 0;
    const std::uint32_t now = swap_tick.fetch_add(1, std::memory_order_relaxed) + 1;
    const std::uint64_t first = swap_all_pools ? 0 : swap_pool_first.load(std::memory_order_relaxed);
    const std::uint64_t end = swap_all_pools ? slots : swap_pool_end.load(std::memory_order_relaxed);
    for (std::size_t i = 0; i < slots / SwapArmShare + 1; ++i, hand = (hand + 1) % slots)
        if (hand >= first && hand < end && SlotState(range, hand) == SlotOwn) (void)ArmSlot(range, hand);
    cold.clear();
    for (std::size_t slot = 0; slot < slots; ++slot) {
        if (SlotState(range, slot) != SlotProbe) continue;
        const std::uint32_t age = now - range.swap[slot];
        if (age == swap_cold_ticks) SampleColdSlot(range, slot);
        if (age >= swap_cold_ticks) cold.emplace_back(range.physical[slot], slot);
    }
    const std::uint64_t in_now = swap_counters.in.load(std::memory_order_relaxed);
    const std::uint64_t came_back = in_now - in_before;
    const bool thrashing = came_back > SwapBackInPerTick;
    in_before = in_now;
    const int mode = swap_mode.load(std::memory_order_relaxed);
    if (mode != 2 && mode != 3) return;
    if (thrashing) {
        // EDEN_UP_SWAP_OUT_WHILE_BACK: only what stayed untouched for twice the cold time goes.
        if (!swap_out_while_back) return;
        cold.erase(std::remove_if(cold.begin(), cold.end(),
                                  [&](const std::pair<std::int64_t, std::size_t>& entry) {
                                      return now - range.swap[entry.second] < 2 * swap_cold_ticks;
                                  }),
                   cold.end());
        if (cold.empty()) return;
    }
    bool wanted = pool_is_short.load(std::memory_order_relaxed) || memory_waiters.load(std::memory_order_relaxed) != 0;
    if (swap_ahead_bytes != 0) {
        // Ahead of the shortage: what the next graphics allocation would get, asked here on the
        // swapper's own thread (the caches ask the same every 100 ms). Nothing goes while the pool
        // cannot be asked.
        std::int64_t start = 0;
        std::size_t largest = 0;
        const std::int64_t total = sceKernelGetDirectMemorySize();
        if (total > 0 && sceKernelAvailableDirectMemorySize(0, total, FinePage, &start, &largest) == 0) {
            swap_pool_largest.store(largest, std::memory_order_relaxed);
            wanted = wanted || largest < swap_ahead_bytes;
        }
    }
    if (!wanted) return;
    std::sort(cold.begin(), cold.end());
    const std::lock_guard out_lock{swap_out_mutex};
    const std::uint64_t failed = swap_counters.failures.load(std::memory_order_relaxed);
    std::size_t sent = 0;
    const auto say_while_back = [&] {
        static unsigned said = 0;
        if (!thrashing || sent == 0 || said >= 64) return;
        ++said;
        char line[160];
        std::snprintf(line, sizeof line, "EDEN_SWAP %zu long-cold slot(s) went out in a tick with %llu back (%zu were long cold)\n",
                      sent, static_cast<unsigned long long>(came_back), cold.size());
        Report(line);
    };
    for (std::size_t i = 0; i < cold.size() && i < SwapOutPerTick; ++i)
        if (SwapOut(range, cold[i].second)) {
            ++sent;
            if (cold[i].second < SwapEverOutSlots)
                swap_ever_out[cold[i].second / 64].fetch_or(std::uint64_t{1} << (cold[i].second % 64), std::memory_order_relaxed);
        } else if (swap_counters.failures.load(std::memory_order_relaxed) != failed) {
            if (mode == 3) break; // no block for the packed bytes now: the next tick asks again
            // A drive that is full or gone: what is out stays readable, nothing more goes.
            swap_mode.store(1, std::memory_order_relaxed);
            Report("EDEN_SWAP the file refused a slot: measuring only from now on\n");
            break;
        }
    say_while_back();
}

// ---- EDEN_UP_PACK_* (at pack_shared, above): the store's compactor, pressure passes, refusals ----

// The pool's largest free block, as the caches and EDEN_GUEST_SWAP_AHEAD_MIB ask it (0: not known).
std::size_t PoolLargest() noexcept {
    std::int64_t start = 0;
    std::size_t largest = 0;
    const std::int64_t total = sceKernelGetDirectMemorySize();
    if (total <= 0 || sceKernelAvailableDirectMemorySize(0, total, FinePage, &start, &largest) != 0) return 0;
    pack_pool_largest.store(largest, std::memory_order_relaxed);
    return largest;
}

// SwapOut, and what went back to the pool (under swap_out_mutex). A slot that would not pack is
// passed over by pressure passes and refusals for 300 s.
bool PackOutAny(SparseRange& range, std::size_t slot, std::size_t* given) noexcept {
    *given = 0;
    const std::uint64_t not_worth = pack_not_worth.load(std::memory_order_relaxed);
    if (!SwapOut(range, slot)) {
        if (pack_not_worth.load(std::memory_order_relaxed) != not_worth) PackSkip(slot, swap_tick.load(std::memory_order_relaxed));
        return false;
    }
    if (SlotState(range, slot) == SlotOut) { // read after the fact, as eden_guest_memory_reclaim reads it
        const std::int64_t physical = range.physical[slot];
        *given = InStore(physical) ? pack_shared_given
                 : physical < 0    ? SparseSlot
                                   : SparseSlot - std::min(SparseSlot, PackedPages(range.swap[slot]) * FinePage);
    }
    NoteEverOut(slot);
    return true;
}

// EDEN_UP_PACK_SHARED: copies that went (unpacked, given back) leave holes in their slabs. While the
// slabs hold over a quarter more than their copies plus 32 MiB, the copies of the emptiest closed slab
// under half full move into the open slab, each under its slot's lock, and that slab goes back whole;
// up to 16 slabs a call (Lichii's CompactStore). The open slab takes a fresh block from the pool only
// while no guest write waits for one. The victim is held while a view of it is mapped: an unpack of
// one of its other slots may drop what was its last copy meanwhile.
void CompactPackStore(SparseRange& range) noexcept {
    const std::lock_guard out_lock{swap_out_mutex};
    const std::size_t slots = (range.end - range.begin.load(std::memory_order_relaxed)) / SparseSlot;
    for (int round = 0; round < 16; ++round) {
        const std::size_t held = pack_store_slabs.load(std::memory_order_relaxed) * SparseSlot;
        if (held <= pack_store_bytes.load(std::memory_order_relaxed) / 4 * 5 + (std::size_t{32} << 20)) return;
        std::uint32_t victim = NoSlab, least = static_cast<std::uint32_t>(SparseSlot / 2);
        for (std::uint32_t i = 0; i < PackSlabs; ++i) {
            const PackSlab& slab = pack_slabs[i];
            if (!slab.used.load(std::memory_order_acquire) || (slab.live.load(std::memory_order_relaxed) & SlabOpen)) continue;
            const std::uint32_t bytes = slab.bytes.load(std::memory_order_relaxed);
            if (bytes < least) {
                least = bytes;
                victim = i;
            }
        }
        if (victim == NoSlab || !HoldSlab(victim)) return;
        void* from = reinterpret_cast<void*>(slot_view_hint.load(std::memory_order_relaxed));
        if (sceKernelMapDirectMemory(&from, SparseSlot, PROT_READ, 0, pack_slabs[victim].physical.load(std::memory_order_relaxed),
                                     SparseSlot) != 0) {
            LetGoSlab(victim);
            return;
        }
        void* to = nullptr;
        std::uint32_t to_slab = NoSlab;
        bool stop = false;
        for (std::size_t slot = 0; slot < slots && !stop; ++slot) {
            // A read without the lock: only a hint, checked again locked. A slot locked now is passed
            // over: an unpack that waits for a block (WaitForBlock, up to 2 s) holds its lock.
            if (SlotState(range, slot) != SlotOut || !InStore(range.physical[slot]) || StoreSlab(range.physical[slot]) != victim)
                continue;
            std::uint8_t value = range.state[slot].load(std::memory_order_relaxed);
            if ((value & SlotLock) || !range.state[slot].compare_exchange_strong(value, value | SlotLock, std::memory_order_acquire,
                                                                                 std::memory_order_relaxed))
                continue;
            StallProbe::SlotTaken(); // UnlockSlot below lets it go
            const std::int64_t place = range.physical[slot];
            if (SlotState(range, slot) != SlotOut || !InStore(place) || StoreSlab(place) != victim) {
                UnlockSlot(range, slot, SlotState(range, slot));
                continue;
            }
            const std::uint32_t size = range.swap[slot];
            const std::size_t need = CopyBytes(size);
            if (open_slab == NoSlab || open_used + need > SparseSlot) {
                std::int64_t block = -1;
                const std::uint32_t entry = FreeSlabEntry();
                if (memory_waiters.load(std::memory_order_acquire) != 0 || entry == NoSlab ||
                    eden_direct_guest(SparseSlot, SparseSlot, 12, &block) != 0) {
                    UnlockSlot(range, slot, SlotOut);
                    stop = true;
                    break;
                }
                if (to != nullptr && munmap(to, SparseSlot) != 0) std::abort();
                to = nullptr; // the slab it showed may go back now (OpenSlab closes it)
                OpenSlab(entry, block);
                pack_counters.from_pool.fetch_add(1, std::memory_order_relaxed);
            }
            if (to == nullptr || to_slab != open_slab) {
                if (to != nullptr && munmap(to, SparseSlot) != 0) std::abort();
                to = reinterpret_cast<void*>(slot_view_hint.load(std::memory_order_relaxed));
                if (sceKernelMapDirectMemory(&to, SparseSlot, PROT_READ | PROT_WRITE, 0,
                                             pack_slabs[open_slab].physical.load(std::memory_order_relaxed), SparseSlot) != 0) {
                    to = nullptr;
                    UnlockSlot(range, slot, SlotOut);
                    stop = true;
                    break;
                }
                to_slab = open_slab;
            }
            const std::size_t offset = open_used;
            open_used += need;
            pack_slabs[open_slab].live.fetch_add(1, std::memory_order_relaxed);
            pack_slabs[open_slab].bytes.fetch_add(static_cast<std::uint32_t>(need), std::memory_order_relaxed);
            std::memcpy(static_cast<char*>(to) + offset, static_cast<const char*>(from) + StoreOffset(place), size);
            range.physical[slot] = StorePlace(open_slab, offset);
            pack_slabs[victim].bytes.fetch_sub(static_cast<std::uint32_t>(need), std::memory_order_relaxed);
            LetGoSlab(victim); // never its last reference: this call holds one
            pack_counters.moved.fetch_add(1, std::memory_order_relaxed);
            UnlockSlot(range, slot, SlotOut);
        }
        if (to != nullptr && munmap(to, SparseSlot) != 0) std::abort();
        if (munmap(from, SparseSlot) != 0) std::abort();
        if (pack_slabs[victim].bytes.load(std::memory_order_relaxed) == 0)
            pack_counters.compacted.fetch_add(1, std::memory_order_relaxed);
        LetGoSlab(victim); // goes back here when no copy is left in it
        if (stop) return;
    }
}

// EDEN_UP_PACK_PRESSURE_MS: one pass while the pool's largest free block is under
// EDEN_UP_PACK_PRESSURE_MIB. Armed slots untouched for EDEN_UP_PACK_PRESSURE_COLD_S go out, the lowest
// in the pool first, up to 256 MiB of slots, the pool asked again every 16 slots; swap_out_mutex is
// taken a slot at a time, so a refused graphics allocation can come in between. A pass that found
// fewer arms the game pool's Own slots that the hand would arm (untouched for SwapRearmTicks), for the
// next passes: Lichii sweeps every Own slot then.
void PackPressurePass(SparseRange& range, std::vector<std::pair<std::int64_t, std::size_t>>& cold) noexcept {
    const int mode = swap_mode.load(std::memory_order_relaxed);
    if (mode != 2 && mode != 3) return;
    const std::size_t largest = PoolLargest();
    if (largest == 0 || largest >= pack_pressure_bytes) return;
    pack_counters.pressure_passes.fetch_add(1, std::memory_order_relaxed);
    const std::size_t slots = (range.end - range.begin.load(std::memory_order_relaxed)) / SparseSlot;
    const std::uint32_t now = swap_tick.load(std::memory_order_relaxed);
    cold.clear();
    for (std::size_t slot = 0; slot < slots; ++slot)
        if (SlotState(range, slot) == SlotProbe && now - range.swap[slot] >= pack_pressure_cold_ticks && !PackSkipped(slot, now))
            cold.emplace_back(range.physical[slot], slot);
    std::sort(cold.begin(), cold.end());
    std::size_t went = 0, freed = 0, tried = 0;
    bool over = false; // the pool came back over the mark during the pass
    const std::uint64_t failed = swap_counters.failures.load(std::memory_order_relaxed);
    for (const auto& entry : cold) {
        const std::size_t slot = entry.second;
        if (tried * SparseSlot >= PackPressurePassBytes) break;
        if (tried % 16 == 15) {
            const std::size_t again = PoolLargest();
            over = again == 0 || again >= pack_pressure_bytes;
            if (over) break;
        }
        if (range.state[slot].load(std::memory_order_relaxed) & SlotLock) continue; // busy now: the next pass
        ++tried;
        std::size_t given = 0;
        bool out = false;
        {
            const std::lock_guard out_lock{swap_out_mutex};
            out = PackOutAny(range, slot, &given);
        }
        if (out) {
            ++went;
            freed += given;
        } else if (swap_counters.failures.load(std::memory_order_relaxed) != failed) {
            break; // no view or block now, or the file refused: the next pass asks again
        }
    }
    pack_counters.pressure_out.fetch_add(went, std::memory_order_relaxed);
    pack_counters.pressure_freed.fetch_add(freed, std::memory_order_relaxed);
    if (over || tried * SparseSlot >= PackPressurePassBytes) return;
    const std::uint64_t first = swap_all_pools ? 0 : swap_pool_first.load(std::memory_order_relaxed);
    const std::uint64_t end = swap_all_pools ? slots : std::min<std::uint64_t>(swap_pool_end.load(std::memory_order_relaxed), slots);
    std::size_t armed = 0;
    for (std::uint64_t slot = first; slot < end; ++slot)
        if (range.state[slot].load(std::memory_order_relaxed) == SlotOwn && now - range.swap[slot] >= SwapRearmTicks)
            armed += ArmSlot(range, slot) ? 1 : 0; // Own and not locked now
    pack_counters.pressure_armed.fetch_add(armed, std::memory_order_relaxed);
}

// The swapper with EDEN_UP_PACK_PRESSURE_MS: a pass every n ms, the tick when it is due (2 s after the
// last one ended, as in SwapLoop), and the store's compactor after each.
void PackPressureLoop(SparseRange* range) {
    if (::eden_crash_name_thread) ::eden_crash_name_thread("GuestSwapper");  // crash report, thread dump
    const std::size_t slots = (range->end - range->begin.load(std::memory_order_relaxed)) / SparseSlot;
    std::vector<std::pair<std::int64_t, std::size_t>> cold, pressed;
    cold.reserve(slots);
    pressed.reserve(slots);
    std::size_t hand = 0;
    const auto pace = std::chrono::milliseconds(pack_pressure_ms.load(std::memory_order_relaxed));
    auto next_tick = std::chrono::steady_clock::now() + std::chrono::milliseconds(SwapTickMs);
    std::unique_lock lock{scanner_mutex};
    while (!scanner_stop) {
        scanner_wake.wait_for(lock, pace, [] { return scanner_stop; });
        if (scanner_stop) break;
        lock.unlock();
        if (std::chrono::steady_clock::now() >= next_tick) {
            SwapTick(*range, hand, cold);
            next_tick = std::chrono::steady_clock::now() + std::chrono::milliseconds(SwapTickMs);
        }
        PackPressurePass(*range, pressed);
        if (pack_shared.load(std::memory_order_relaxed)) CompactPackStore(*range);
        lock.lock();
    }
}

// EDEN_UP_PACK_ON_REFUSAL: `want` (64 MiB or more) given back now, on the thread that was refused:
// cold slots (armed, untouched for a tick), then slots armed this tick, the lowest in the pool first,
// then the game pool's Own slots, the longest untouched first, each armed and packed at once (Lichii's
// emergency sweep makes every Own slot Cold when too few are). Until `want` went back, three times its
// slots were tried, or 256 slots went; a slot that would not pack in the last 300 s and a slot locked
// now are passed over. A stall, not an end: 2 to 5 ms a slot. One refusal at a time, under
// swap_out_mutex. The list asks nothing of the heap (the pool is short now): a key a slot, its group
// in the top two bits, then its order, then the slot; slots past SwapEverOutSlots are not looked at.
static_assert(SwapEverOutSlots <= (std::size_t{1} << 14), "a slot is 14 bits of a refusal's key");
std::size_t PackOnRefusal(SparseRange& range, std::size_t want) noexcept {
    const std::uint64_t start = NowNs();
    const std::lock_guard out_lock{swap_out_mutex};
    const std::size_t slots = (range.end - range.begin.load(std::memory_order_relaxed)) / SparseSlot;
    const std::uint32_t now = swap_tick.load(std::memory_order_relaxed);
    const std::uint64_t first = swap_all_pools ? 0 : swap_pool_first.load(std::memory_order_relaxed);
    const std::uint64_t end = swap_all_pools ? slots : swap_pool_end.load(std::memory_order_relaxed);
    static std::uint64_t candidates[SwapEverOutSlots];
    std::size_t count = 0;
    for (std::size_t slot = 0; slot < std::min(slots, SwapEverOutSlots); ++slot) {
        const std::uint8_t value = range.state[slot].load(std::memory_order_relaxed);
        if ((value & SlotLock) || PackSkipped(slot, now)) continue;
        std::uint64_t group = 0, order = 0;
        // The swapper's tick can pass `now` meanwhile (SwapTick counts it outside swap_out_mutex): a slot
        // armed or touched since is 0 ticks old, not 2^32 - 1 (cold, and the first written slot taken).
        const std::uint32_t since = now - range.swap[slot];
        const std::uint32_t age = static_cast<std::int32_t>(since) > 0 ? since : 0;
        if ((value & SlotMask) == SlotProbe) { // 0 cold, 1 armed this tick: the lowest block first
            group = age >= 1 ? 0 : 1;
            order = static_cast<std::uint64_t>(range.physical[slot]) / SparseSlot;
        } else if ((value & SlotMask) == SlotOwn && slot >= first && slot < end) { // 2 written: the longest untouched first
            group = 2;
            order = UINT32_MAX - age;
        } else {
            continue;
        }
        candidates[count++] = group << 62 | (order & ((std::uint64_t{1} << 48) - 1)) << 14 | slot;
    }
    std::sort(candidates, candidates + count);
    const std::size_t most_tries = 3 * (want / SparseSlot) + 8;
    std::size_t freed = 0, went = 0, tried = 0, written = 0;
    const std::uint64_t failed = swap_counters.failures.load(std::memory_order_relaxed);
    for (std::size_t i = 0; i < count; ++i) {
        if (freed >= want || went >= 256 || tried >= most_tries) break;
        const std::size_t slot = static_cast<std::size_t>(candidates[i] & 0x3fff);
        const bool own = (candidates[i] >> 62) == 2;
        // Locked now: given back and written meanwhile, its write may wait up to 2 s for a block.
        if (range.state[slot].load(std::memory_order_relaxed) & SlotLock) continue;
        if (own && !ArmSlot(range, slot, true)) continue; // touched or changed meanwhile
        ++tried;
        std::size_t given = 0;
        if (PackOutAny(range, slot, &given)) {
            ++went;
            freed += given;
            written += own ? 1 : 0;
        } else if (swap_counters.failures.load(std::memory_order_relaxed) != failed) {
            break; // no view or block now, or the file refused
        }
    }
    const std::uint64_t took = NowNs() - start;
    pack_counters.refusals.fetch_add(1, std::memory_order_relaxed);
    pack_counters.refusal_out.fetch_add(went, std::memory_order_relaxed);
    pack_counters.refusal_freed.fetch_add(freed, std::memory_order_relaxed);
    pack_counters.refusal_written.fetch_add(written, std::memory_order_relaxed);
    if (freed < want) pack_counters.refusal_short.fetch_add(1, std::memory_order_relaxed);
    pack_counters.refusal_ns.fetch_add(took, std::memory_order_relaxed);
    if (took > pack_counters.refusal_most_ns.load(std::memory_order_relaxed))
        pack_counters.refusal_most_ns.store(took, std::memory_order_relaxed);
    char line[192];
    const int length = std::snprintf(line, sizeof(line),
                                     "EDEN_UP_PACK_ON_REFUSAL asked %zu MiB: %zu MiB given back by %zu slots (%zu written), "
                                     "%zu tried, %llu ms\n",
                                     want >> 20, freed >> 20, went, written, tried, (unsigned long long)(took / 1000000));
    if (length > 0) Report(line);
    return freed;
}

// T178b EmergencyGiveBack's swap part (below): up to EmergencySlots cold slots (armed and untouched for a
// tick or more: Probe), packed (EDEN_GUEST_SWAP=3) or written to the file (=2) now, on the calling thread.
// Called from a fault handler (a game CPU thread whose wait ran out) and from a table commit's wait, so:
//   - swap_out_mutex by try_lock only: held, the swapper or a refusal is giving back already (busy);
//   - a slot locked now is passed over (the caller's own slot is locked by the caller);
//   - no heap, no stdio: a fixed loop over the slot states, PackOutAny as the pressure pass calls it (its
//     kernel calls, LZ4 into the static buffer swap_out_mutex guards, a smaller block for the packed bytes);
//   - at most EmergencySlots slots and 3 x as many tried, so a few tens of ms (2 to 5 ms a slot, PackOnRefusal).
// Bytes given back, as PackOutAny counts them.
constexpr std::size_t EmergencySlots = 16; // up to 32 MiB
std::size_t EmergencyPackColdSlots(std::size_t* slots, bool* busy) noexcept {
    *slots = 0;
    *busy = false;
    const int mode = swap_mode.load(std::memory_order_relaxed);
    SparseRange* const range = service_range.load(std::memory_order_acquire);
    if ((mode != 2 && mode != 3) || range == nullptr || range->state == nullptr) return 0;
    std::unique_lock out_lock{swap_out_mutex, std::try_to_lock};
    if (!out_lock.owns_lock()) {
        *busy = true;
        return 0;
    }
    const std::size_t count = (range->end - range->begin.load(std::memory_order_relaxed)) / SparseSlot;
    const std::uint32_t now = swap_tick.load(std::memory_order_relaxed);
    const std::uint64_t failed = swap_counters.failures.load(std::memory_order_relaxed);
    std::size_t freed = 0, went = 0, tried = 0;
    for (std::size_t slot = 0; slot < count && went < EmergencySlots && tried < 3 * EmergencySlots; ++slot) {
        const std::uint8_t value = range->state[slot].load(std::memory_order_relaxed);
        if ((value & SlotLock) || (value & SlotMask) != SlotProbe) continue;
        if (static_cast<std::int32_t>(now - range->swap[slot]) < 1) continue; // armed this tick: may be hot
        if (PackSkipped(slot, now)) continue;
        ++tried;
        std::size_t given = 0;
        if (PackOutAny(*range, slot, &given)) {
            ++went;
            freed += given;
        } else if (swap_counters.failures.load(std::memory_order_relaxed) != failed) {
            break; // no block for the packed bytes, or the file refused: nothing more now
        }
    }
    *slots = went;
    return freed;
}

std::thread swapper;
void SwapLoop(SparseRange* range) {
    StallProbe::role = StallProbe::Swapper;
    if (pack_pressure_ms.load(std::memory_order_relaxed) != 0) { // EDEN_UP_PACK_PRESSURE_MS
        PackPressureLoop(range);
        return;
    }
    if (::eden_crash_name_thread) ::eden_crash_name_thread("GuestSwapper");  // crash report, thread dump
    std::vector<std::pair<std::int64_t, std::size_t>> cold;
    cold.reserve((range->end - range->begin.load(std::memory_order_relaxed)) / SparseSlot);
    std::size_t hand = 0;
    std::unique_lock lock{scanner_mutex};
    while (!scanner_stop) {
        scanner_wake.wait_for(lock, std::chrono::milliseconds(SwapTickMs), [] { return scanner_stop; });
        if (scanner_stop) break;
        lock.unlock();
        SwapTick(*range, hand, cold);
        if (pack_shared.load(std::memory_order_relaxed)) CompactPackStore(*range); // EDEN_UP_PACK_SHARED
        lock.lock();
    }
}

bool OpenSwapFile() noexcept {
    swap_free_count = 0;
    swap_next_place = 0;
    if (swap_fd < 0) swap_fd = open(swap_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    return swap_fd >= 0;
}
void CloseSwapFile() noexcept {
    if (swap_fd < 0) return;
    (void)ftruncate(swap_fd, 0);
    (void)close(swap_fd);
    swap_fd = -1;
}

// In the self-test, on its slot 1 (Own, `word` in it): the fault of a read and of a write on a Probe
// and an Out slot, every byte back where it was, and what the file does. A refusal switches this off
// and nothing else. The first step that can fail on the console is the first PROT_NONE.
const char* SwapCheck(SparseRange& range, volatile std::uint64_t* word, char* speeds, std::size_t room) noexcept {
    const std::size_t slot = 1;
    auto* whole = reinterpret_cast<volatile std::uint64_t*>(SlotAddress(range, slot));
    constexpr std::size_t Last = SparseSlot / 8 - 1;
    *word = 21;
    whole[Last] = 0x1122334455667788ull;
    if (!ArmSlot(range, slot, true) || SlotState(range, slot) != SlotProbe) return "arm";
    const std::uint64_t start = NowNs();
    const bool read_back = *word == 21;
    const std::uint64_t touch_ns = NowNs() - start;
    if (!read_back || SlotState(range, slot) != SlotOwn) return "probe-read";
    if (!ArmSlot(range, slot, true)) return "arm-again";
    *word = 22;
    if (*word != 22 || SlotState(range, slot) != SlotOwn) return "probe-write";
    // A slot given back while it is armed reads as zeroes and is written again like any other.
    if (!ArmSlot(range, slot, true)) return "arm-return";
    LockSlot(range, slot);
    ReturnSlot(range, slot);
    UnlockSlot(range, slot, SlotZero);
    if (*word != 0 || whole[Last] != 0) return "probe-return";
    *word = 22;
    whole[Last] = 0x1122334455667788ull;
    if (*word != 22 || SlotState(range, slot) != SlotOwn) return "probe-return-write";
    if (!OpenSwapFile()) {
        std::snprintf(speeds, room, "touch_ns=%llu file=none", (unsigned long long)touch_ns);
        return swap_mode.load(std::memory_order_relaxed) == 2 ? "file" : "ok";
    }
    constexpr unsigned Trips = 8;
    std::uint64_t out_ns = 0, in_ns = 0;
    for (unsigned trip = 0; trip < Trips; ++trip) {
        const auto held = guest_committed.load(std::memory_order_relaxed);
        if (!ArmSlot(range, slot, true)) return "arm-out";
        std::uint64_t mark = NowNs();
        if (!SwapOut(range, slot) || SlotState(range, slot) != SlotOut) return "out";
        out_ns += NowNs() - mark;
        const std::size_t freed = held - guest_committed.load(std::memory_order_relaxed);
        if (swap_mode.load(std::memory_order_relaxed) == 3 ? freed < SparseSlot / 2 : freed != SparseSlot) return "out-held";
        mark = NowNs();
        if (trip % 2 == 0) {
            if (*word != 22 + trip) return "in-read"; // the read that brings it back
        } else {
            *word = 23 + trip; // the write that does
        }
        in_ns += NowNs() - mark;
        if (SlotState(range, slot) != SlotOwn || whole[Last] != 0x1122334455667788ull) return "in-state";
        *word = 23 + trip;
    }
    if (!ArmSlot(range, slot, true) || !SwapOut(range, slot)) return "out-last";
    LockSlot(range, slot); // a slot given back while it is out: its place too, and it reads as zeroes
    ReturnSlot(range, slot);
    UnlockSlot(range, slot, SlotZero);
    if (*word != 0 || (swap_mode.load(std::memory_order_relaxed) != 3 && swap_free_count != 1)) return "return";
    std::snprintf(speeds, room, "touch_ns=%llu out_ms=%llu in_ms=%llu write_MBps=%llu read_MBps=%llu",
                  (unsigned long long)touch_ns, (unsigned long long)(out_ns / Trips / 1000000),
                  (unsigned long long)(in_ns / Trips / 1000000),
                  (unsigned long long)(Trips * SparseSlot * 1000ull / (out_ns + 1)),
                  (unsigned long long)(Trips * SparseSlot * 1000ull / (in_ns + 1)));
    return "ok";
}
#endif

// T178b: the emergency give-back (at fault_last_try above): the heap's empty spaces (the unheld trim: a hold
// does not stop it, as for the guest's own waits), then the swap's part (EmergencyPackColdSlots, console only).
std::size_t EmergencyGiveBack(std::size_t* slots, bool* busy) noexcept {
    *slots = 0;
    *busy = false;
    std::size_t given = 0;
    if (::eden_heap_trim_unheld != nullptr) given += ::eden_heap_trim_unheld();
    else if (::eden_heap_trim != nullptr) given += ::eden_heap_trim();
#ifdef PS5_NATIVE
    given += EmergencyPackColdSlots(slots, busy);
#endif
    return given;
}

// ---- T178b: a table slot's commit waits instead of ending the app (memory_pages.cpp CommitSparsePage) ----
//
// Where T177 called abort ("EDEN_SPARSE_TABLES commit failed: out of memory"), the commit now waits and tries
// again, the way a game write's fault does (WaitForBlock): it counts as a waiting write (memory_waiters: the
// graphics caches see no free memory and evict, the driver's, the JIT's and the reserve's lends and
// EDEN_UP_GPU_READ_THROUGH=2's bring-backs stand back), trims the heap at once and every 500 ms (the unheld
// trim), lets the keeper refill EDEN_UP_TABLE_RESERVE (first, every 100 ms), makes one emergency give-back
// (EmergencyGiveBack) after 1 s, and asks for a block every 20 ms: the pool, then EDEN_UP_TABLE_RESERVE, then
// the guest reserve (OwnSlot, as before). Up to EDEN_UP_TABLE_WAIT_MS (0 to 60000 ms, digits only; unset or
// anything else: 10000; 0: no wait, T177's abort at once). Only then the old failure lines and the abort.
// sparse_mutex is NOT held while it sleeps: it is taken for each try only (one global mutex for every sparse
// table's commits, the ranges' creation and their release).
// The thread that waits keeps every lock its caller holds. Read in the core source (core-t33 checkout of 6 Nov,
// not checked against T177's patched core): the device memory manager's table (run1417) is committed in
// DeviceMemoryManager::Map under its mapping_guard, called from nvhost_as_gpu::Nx2Map under the address
// space's mutex (and, for a new device range, Nx2GpuMemoryRegistry's device_mutex), on a service thread (am:2);
// the code there says its callers hold nx2:gpu locks that the GPU thread takes under the texture cache's lock.
// The GPU's own page table (EDEN_GMMU_SPARSE, MemoryManager::PageTableOp) is committed on the GPU thread inside
// gmmu->Map, which runs under the texture cache's lock; the guest's page table on a game thread inside the
// guest kernel's mapping calls. So while one of them waits, the GPU thread may stop on those locks, and then
// the caches free nothing: what can still come is the heap's trim, the swap's give-back, and blocks other
// threads let go. The wait is bounded, so the worst case is the old abort, 10 s later (a hang watcher may
// report the stall first).
// Lines (stderr): EDEN_UP_TABLE_COMMIT wait start ... and EDEN_UP_TABLE_COMMIT wait end ... outcome=<got-block|
// lost>, the first 64 waits of a session in full, later ones counted only; EDEN_UP_TABLE_COMMIT totals every
// 10 s from the keeper while its counts move.
unsigned TableWaitMs() noexcept {
    static std::atomic<int> kept{-1};
    int ms = kept.load(std::memory_order_relaxed);
    if (ms < 0) {
        ms = 10000;
        if (const char* value = std::getenv("EDEN_UP_TABLE_WAIT_MS"); value != nullptr && *value != '\0') {
            unsigned long asked = 0;
            const char* at = value;
            while (*at >= '0' && *at <= '9' && asked <= 60000) asked = asked * 10 + static_cast<unsigned long>(*at++ - '0');
            if (*at == '\0' && asked <= 60000) ms = static_cast<int>(asked);
        }
        kept.store(ms, std::memory_order_relaxed);
    }
    return static_cast<unsigned>(ms);
}
// One try, under sparse_mutex: true when the slot is owned now (by this try or another thread's).
bool TableCommitTry(SparseRange& range, std::size_t slot) noexcept {
    const std::lock_guard lock{sparse_mutex};
    if (range.owned[slot].load(std::memory_order_relaxed)) return true;
    const auto begin = range.begin.load(std::memory_order_relaxed);
    if (!OwnSlot(begin + slot * SparseSlot, &range.physical[slot], range.guest)) return false;
    (range.guest ? guest_committed : sparse_committed).fetch_add(SparseSlot, std::memory_order_relaxed);
    range.owned[slot].store(true, std::memory_order_release);
    return true;
}
std::atomic<unsigned> table_wait_lines{0};
constexpr unsigned TableWaitLinesMost = 64;
bool TableCommitWait(SparseRange& range, std::size_t slot) noexcept {
    const unsigned limit_ms = TableWaitMs();
    if (limit_ms == 0) return false;
    auto& c = table_commit_counters;
    c.waits.fetch_add(1, std::memory_order_relaxed);
    const bool logged = table_wait_lines.fetch_add(1, std::memory_order_relaxed) < TableWaitLinesMost;
    const std::uint64_t start_ms = MonotonicMs();
    const auto say = [&](const char* what, bool ok, std::uint64_t waited, unsigned tries, unsigned trims,
                         std::size_t given) noexcept {
        if (!logged) return;
        char line[400];
        const int length = std::snprintf(
            line, sizeof(line),
            "EDEN_UP_TABLE_COMMIT wait %s range=%p slot=%zu guest=%u limit_ms=%u table_reserve=%zu/%zu pool_reserve=%zu "
            "waiters=%u%s%s waited_ms=%llu tries=%u trims=%u given_MiB=%zu\n",
            what, reinterpret_cast<void*>(range.begin.load(std::memory_order_relaxed)), slot, range.guest ? 1u : 0u,
            limit_ms,
#ifdef PS5_NATIVE
            table_reserve_count.load(std::memory_order_relaxed), table_reserve_target.load(std::memory_order_relaxed),
            reserve_count.load(std::memory_order_relaxed),
#else
            std::size_t{0}, std::size_t{0}, std::size_t{0},
#endif
            memory_waiters.load(std::memory_order_relaxed), *what == 'e' ? " outcome=" : "",
            *what == 'e' ? (ok ? "got-block" : "lost") : "", (unsigned long long)waited, tries, trims, given >> 20);
        if (length > 0) Report(line);
    };
    say("start", false, 0, 0, 0, 0);
    memory_waiters.fetch_add(1, std::memory_order_acq_rel);
    bool ok = false, gave = false;
    unsigned tries = 0, trims = 0;
    std::size_t given = 0;
    constexpr unsigned TrimEveryTries = 25; // 500 ms at 20 ms a try
    for (;;) {
        const std::uint64_t waited = MonotonicMs() - start_ms;
        if (tries % TrimEveryTries == 0 && (::eden_heap_trim_unheld != nullptr || ::eden_heap_trim != nullptr)) {
            const std::size_t trimmed = ::eden_heap_trim_unheld != nullptr ? ::eden_heap_trim_unheld() : ::eden_heap_trim();
            ++trims;
            given += trimmed;
            c.trims.fetch_add(1, std::memory_order_relaxed);
            c.trim_bytes.fetch_add(trimmed, std::memory_order_relaxed);
        }
        if (!gave && waited >= 1000) {
            gave = true;
            std::size_t slots = 0;
            bool busy = false;
            const std::size_t back = EmergencyGiveBack(&slots, &busy);
            given += back;
            c.give_backs.fetch_add(1, std::memory_order_relaxed);
            c.given_bytes.fetch_add(back, std::memory_order_relaxed);
        }
        ++tries;
        c.tries.fetch_add(1, std::memory_order_relaxed);
        if (TableCommitTry(range, slot)) {
            ok = true;
            break;
        }
        if (waited >= limit_ms) break;
        const timespec pause{0, 20'000'000};
        nanosleep(&pause, nullptr);
    }
    memory_waiters.fetch_sub(1, std::memory_order_acq_rel);
    const std::uint64_t waited = MonotonicMs() - start_ms;
    (ok ? c.won : c.lost).fetch_add(1, std::memory_order_relaxed);
    c.wait_ms.fetch_add(waited, std::memory_order_relaxed);
    if (waited > c.most_ms.load(std::memory_order_relaxed)) c.most_ms.store(waited, std::memory_order_relaxed);
    say("end", ok, waited, tries, trims, given);
    return ok;
}
// From the keeper every 100 ms: the totals every 10 s while they move (nothing when no table ever waited or
// took from EDEN_UP_TABLE_RESERVE).
void TableCommitTick(std::uint64_t now_ns) noexcept {
    static std::uint64_t reported_ns = 0, seen = 0; // the keeper's own
    if (now_ns - reported_ns < 10'000'000'000ull) return;
    reported_ns = now_ns;
    const auto& c = table_commit_counters;
    const auto n = [](const std::atomic<std::uint64_t>& v) noexcept { return (unsigned long long)v.load(std::memory_order_relaxed); };
#ifdef PS5_NATIVE
    const bool reserve_on = table_reserve_target.load(std::memory_order_relaxed) != 0;
#else
    const bool reserve_on = false;
#endif
    // Without EDEN_UP_TABLE_RESERVE only a wait makes the line (T177 had none: its tables never waited).
    const std::uint64_t moved =
        (reserve_on ? n(c.reserve_taken) + n(c.reserve_empty) + n(c.reserve_filled) + n(c.guest_reserve_taken) : 0) +
        n(c.waits) + n(c.won) + n(c.lost);
    if (moved == 0) return;
    if (moved == seen) return;
    seen = moved;
    char line[480];
    std::size_t lowest = c.lowest.load(std::memory_order_relaxed);
#ifdef PS5_NATIVE
    if (lowest == SIZE_MAX) lowest = table_reserve_target.load(std::memory_order_relaxed); // never taken: the target
#endif
    if (lowest == SIZE_MAX) lowest = 0;
    const int length = std::snprintf(
        line, sizeof(line),
        "EDEN_UP_TABLE_COMMIT totals: table_reserve=%zu/%zu taken=%llu empty=%llu filled=%llu lowest=%zu "
        "guest_reserve_taken=%llu; waits=%llu won=%llu lost=%llu wait_ms=%llu longest_ms=%llu tries=%llu trims=%llu "
        "trim_MiB=%llu give_backs=%llu given_MiB=%llu\n",
#ifdef PS5_NATIVE
        table_reserve_count.load(std::memory_order_relaxed), table_reserve_target.load(std::memory_order_relaxed),
#else
        std::size_t{0}, std::size_t{0},
#endif
        n(c.reserve_taken), n(c.reserve_empty), n(c.reserve_filled), lowest,
        n(c.guest_reserve_taken), n(c.waits), n(c.won), n(c.lost), n(c.wait_ms), n(c.most_ms), n(c.tries), n(c.trims),
        n(c.trim_bytes) >> 20, n(c.give_backs), n(c.given_bytes) >> 20);
    if (length > 0) Report(line);
}

// The switches, read once a guest range starts.
void ReadSwapSwitches() noexcept {
#ifdef PS5_NATIVE
    const char* mode = std::getenv("EDEN_GUEST_SWAP");
    swap_mode.store(mode && mode[0] == '3' ? 3 : mode && mode[0] == '2' ? 2 : mode && mode[0] == '1' ? 1 : 0,
                    std::memory_order_relaxed);
    if (const char* path = std::getenv("EDEN_GUEST_SWAP_FILE")) swap_path = path;
    if (const char* mib = std::getenv("EDEN_GUEST_SWAP_MIB"))
        swap_places = std::min<std::uint32_t>(static_cast<std::uint32_t>(std::strtoul(mib, nullptr, 10) / (SparseSlot >> 20)), SwapMostPlaces);
    if (const char* seconds = std::getenv("EDEN_GUEST_SWAP_COLD_S"))
        swap_cold_ticks = std::max<std::uint32_t>(static_cast<std::uint32_t>(std::strtoul(seconds, nullptr, 10) * 1000 / SwapTickMs), 2);
    if (const char* back = std::getenv("EDEN_UP_SWAP_OUT_WHILE_BACK")) swap_out_while_back = back[0] == '1' && back[1] == '\0';
    const char* all = std::getenv("EDEN_GUEST_SWAP_ALL_POOLS");
    swap_all_pools = all && all[0] == '1';
    swap_ahead_bytes = 0;
    if (const char* ahead = std::getenv("EDEN_GUEST_SWAP_AHEAD_MIB")) {
        unsigned int sdk = 0;
        std::size_t size = sizeof(sdk);
        const bool known = sysctlbyname("kern.sdk_version", &sdk, &size, nullptr, 0) == 0 && sdk != 0;
        // The top byte is the major version in hex digits: 0x12400009 is 12.40, 0x13... is 13.x.
        const unsigned major = sdk >> 24;
        const bool from_13 = !known || major >= 0x13;
        // Spaces and tabs before the value are skipped, and "auto" is matched in any case
        // (" auto", "Auto", "AUTO\r" are all auto).
        while (*ahead == ' ' || *ahead == '\t') ++ahead;
        const auto lower = [](char c) { return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c; };
        const bool automatic = lower(ahead[0]) == 'a' && lower(ahead[1]) == 'u' && lower(ahead[2]) == 't' &&
                               lower(ahead[3]) == 'o';
        char* end = nullptr;
        const unsigned long number = automatic ? 0ul : std::strtoul(ahead, &end, 10);
        const bool parsed = !automatic && end != ahead;
        const unsigned long mib = automatic ? (from_13 ? 2048ul : 12288ul) : number;
        swap_ahead_bytes = static_cast<std::size_t>(mib) << 20;
        char line[240];
        const int length = std::snprintf(line, sizeof(line),
                                         "EDEN_SWAP_AHEAD %luMiB (%s; firmware sdk_version %s0x%08x; swap mode %d)\n",
                                         mib,
                                         !automatic && !parsed ? "not a number: 0, swap-ahead off"
                                         : !automatic          ? "a number in the settings, used as it is"
                                         : !known              ? "auto, firmware not known: the 13.x value"
                                         : from_13             ? "auto, firmware 13 or newer"
                                         : major == 0x12       ? "auto, firmware 12.x"
                                                               : "auto, firmware below 12, not tested: the 12.x value",
                                         known ? "" : "unknown ", sdk, swap_mode.load(std::memory_order_relaxed));
        if (length > 0) Report(line);
    }
    const char* high = std::getenv("EDEN_SLOT_VIEW_HIGH");
    slot_view_hint.store(high && high[0] == '1' ? slot_view_high : cpu_mapping_hint, std::memory_order_relaxed);
#endif
}

// EDEN_UP_PACK_* (at pack_shared, above), read as the service starts, once the swap's mode is final
// (after the self-test and the file). Unset, every one is off and nothing below runs.
void ReadPackSwitches() noexcept {
#ifdef PS5_NATIVE
    const int mode = swap_mode.load(std::memory_order_relaxed);
    const bool shared = Common::PeSwitch("EDEN_UP_PACK_SHARED");
    const bool refusal = Common::PeSwitch("EDEN_UP_PACK_ON_REFUSAL");
    unsigned long ms = 0;
    if (const char* value = std::getenv("EDEN_UP_PACK_PRESSURE_MS")) ms = std::strtoul(value, nullptr, 10);
    if (ms != 0) ms = std::clamp<unsigned long>(ms, 50, SwapTickMs);
    pack_shared.store(shared && mode == 3, std::memory_order_relaxed);
    pack_on_refusal.store(refusal && (mode == 2 || mode == 3), std::memory_order_relaxed);
    pack_pressure_ms.store(mode == 2 || mode == 3 ? static_cast<unsigned>(ms) : 0u, std::memory_order_relaxed);
    pack_pressure_bytes = std::size_t{2560} << 20;
    pack_pressure_cold_ticks = 2;
    if (ms != 0) {
        if (const char* value = std::getenv("EDEN_UP_PACK_PRESSURE_MIB")) {
            const unsigned long mib = std::strtoul(value, nullptr, 10);
            if (mib >= 64 && mib <= 16384) pack_pressure_bytes = static_cast<std::size_t>(mib) << 20;
        }
        if (const char* value = std::getenv("EDEN_UP_PACK_PRESSURE_COLD_S")) {
            const unsigned long seconds = std::strtoul(value, nullptr, 10);
            if (seconds >= 1 && seconds <= 60)
                pack_pressure_cold_ticks = std::max<std::uint32_t>(static_cast<std::uint32_t>(seconds * 1000 / SwapTickMs), 1);
        }
    }
    if (!shared && !refusal && ms == 0) return;
    for (auto& until : pack_skip_until) until.store(0, std::memory_order_relaxed);
    for (auto* counter : {&pack_counters.packed, &pack_counters.from_slot, &pack_counters.from_pool, &pack_counters.released,
                          &pack_counters.moved, &pack_counters.compacted, &pack_counters.leaked, &pack_counters.pressure_passes,
                          &pack_counters.pressure_out, &pack_counters.pressure_freed, &pack_counters.pressure_armed,
                          &pack_counters.refusals, &pack_counters.refusal_out, &pack_counters.refusal_freed,
                          &pack_counters.refusal_written, &pack_counters.refusal_short, &pack_counters.refusal_ns,
                          &pack_counters.refusal_most_ns})
        counter->store(0, std::memory_order_relaxed);
    pack_pool_largest.store(0, std::memory_order_relaxed);
    char line[256];
    const int length = std::snprintf(line, sizeof(line),
                                     "EDEN_UP_PACK shared=%d pressure_ms=%u pressure_mib=%zu cold_s=%u on_refusal=%d "
                                     "(EDEN_GUEST_SWAP=%d; shared needs 3, the others 2 or 3)\n",
                                     pack_shared.load() ? 1 : 0, pack_pressure_ms.load(), pack_pressure_bytes >> 20,
                                     pack_pressure_cold_ticks * SwapTickMs / 1000, pack_on_refusal.load() ? 1 : 0, mode);
    if (length > 0) Report(line);
#endif
}

// A second line beside EDEN_CRAM: how many slots lie untouched and for how long, and what a touch found.
void SwapReport(SparseRange& range) noexcept {
    if (swap_mode.load(std::memory_order_relaxed) == 0) return;
    const std::size_t slots = (range.end - range.begin.load(std::memory_order_relaxed)) / SparseSlot;
    const std::uint32_t now = swap_tick.load(std::memory_order_relaxed);
    constexpr std::uint32_t Seconds[4] = {30, 60, 120, 300};
    std::size_t probe = 0, out = 0, untouched[4] = {};
    for (std::size_t slot = 0; slot < slots; ++slot) {
        const auto state = SlotState(range, slot);
        out += state == SlotOut;
        if (state != SlotProbe) continue;
        ++probe;
        for (std::size_t i = 0; i < 4; ++i) untouched[i] += (now - range.swap[slot]) * SwapTickMs / 1000 >= Seconds[i];
    }
    const auto& c = swap_counters;
    const std::uint64_t sampled = c.sampled.load(), outs = c.out.load(), ins = c.in.load();
    constexpr std::size_t Mib = SparseSlot >> 20;
    char line[640];
    const int length = std::snprintf(line, sizeof(line),
        "EDEN_SWAP mode=%d armed=%zuMiB out=%zuMiB untouched 30s=%zuMiB 60s=%zuMiB 120s=%zuMiB 300s=%zuMiB; "
        "touches=%llu by ticks armed <2:%llu <4:%llu <8:%llu <16:%llu <32:%llu <64:%llu <128:%llu more:%llu; "
        "went_out=%llu avg_ms=%llu came_in=%llu avg_ms=%llu most_ms=%llu failures=%llu lz4=%llu%% of %lluMiB "
        "returned_armed=%llu all_pools=%d game_pool_slots=%llu..%llu ahead=%zuMiB pool_largest=%zuMiB\n",
        swap_mode.load(std::memory_order_relaxed), probe * Mib, out * Mib, untouched[0] * Mib, untouched[1] * Mib,
        untouched[2] * Mib, untouched[3] * Mib, (unsigned long long)c.touched.load(),
        (unsigned long long)c.age[0].load(), (unsigned long long)c.age[1].load(), (unsigned long long)c.age[2].load(),
        (unsigned long long)c.age[3].load(), (unsigned long long)c.age[4].load(), (unsigned long long)c.age[5].load(),
        (unsigned long long)c.age[6].load(), (unsigned long long)c.age[7].load(), (unsigned long long)outs,
        (unsigned long long)(outs ? c.out_ns.load() / outs / 1000000 : 0), (unsigned long long)ins,
        (unsigned long long)(ins ? c.in_ns.load() / ins / 1000000 : 0), (unsigned long long)(c.in_most_ns.load() / 1000000),
        (unsigned long long)c.failures.load(), (unsigned long long)(sampled ? c.packed.load() * 100 / sampled : 0),
        (unsigned long long)(sampled >> 20), (unsigned long long)c.returned_armed.load(),
        swap_all_pools ? 1 : 0, (unsigned long long)swap_pool_first.load(std::memory_order_relaxed),
        (unsigned long long)swap_pool_end.load(std::memory_order_relaxed), swap_ahead_bytes >> 20,
        swap_pool_largest.load(std::memory_order_relaxed) >> 20);
    if (length > 0) Report(line);
    if (swap_mode.load(std::memory_order_relaxed) == 3) {
        char pack[160];
        const int packed_length = std::snprintf(pack, sizeof(pack),
                                                "EDEN_SWAP_PACK slots=%zu held=%zuMiB not_worth=%llu in_place=%llu tail_refused=%llu pool_first=%llu\n",
                                                out, packed_pages_held.load(std::memory_order_relaxed) * FinePage >> 20,
                                                (unsigned long long)pack_not_worth.load(std::memory_order_relaxed),
                                                (unsigned long long)packed_in_place.load(std::memory_order_relaxed),
                                                (unsigned long long)pack_tail_refused.load(std::memory_order_relaxed),
                                                (unsigned long long)packed_pool_first.load(std::memory_order_relaxed));
        if (packed_length > 0) Report(pack);
    }
    if (pack_shared.load(std::memory_order_relaxed) || pack_on_refusal.load(std::memory_order_relaxed) ||
        pack_pressure_ms.load(std::memory_order_relaxed) != 0) {
        // EDEN_UP_PACK_*: the store (slabs and what their copies hold), the compactor, the pressure
        // passes (slots out, MiB given back, slots armed by the sweep) and the refusals.
        const auto& p = pack_counters;
        const std::uint64_t refusals = p.refusals.load();
        char store[640];
        const int store_length = std::snprintf(store, sizeof(store),
            "EDEN_UP_PACK store slabs=%zu held=%zuMiB copies=%zu copy=%zuMiB packed=%llu from_slot=%llu from_pool=%llu "
            "released=%llu moved=%llu compacted=%llu; pressure passes=%llu out=%llu freed=%lluMiB armed=%llu "
            "pool_largest=%zuMiB; refusals=%llu out=%llu freed=%lluMiB written=%llu short=%llu avg_ms=%llu most_ms=%llu\n",
            pack_store_slabs.load(), pack_store_slabs.load() * (SparseSlot >> 20), pack_store_copies.load(),
            pack_store_bytes.load() >> 20, (unsigned long long)p.packed.load(), (unsigned long long)p.from_slot.load(),
            (unsigned long long)p.from_pool.load(), (unsigned long long)p.released.load(), (unsigned long long)p.moved.load(),
            (unsigned long long)p.compacted.load(), (unsigned long long)p.pressure_passes.load(),
            (unsigned long long)p.pressure_out.load(), (unsigned long long)(p.pressure_freed.load() >> 20),
            (unsigned long long)p.pressure_armed.load(), pack_pool_largest.load() >> 20, (unsigned long long)refusals,
            (unsigned long long)p.refusal_out.load(), (unsigned long long)(p.refusal_freed.load() >> 20),
            (unsigned long long)p.refusal_written.load(), (unsigned long long)p.refusal_short.load(),
            (unsigned long long)(refusals ? p.refusal_ns.load() / refusals / 1000000 : 0),
            (unsigned long long)(p.refusal_most_ns.load() / 1000000));
        if (store_length > 0) Report(store);
    }
}

// The session ends and nothing will answer a touch: armed slots are plain Own slots again. An Out
// slot stays as it is, its bytes gone with the file: the guest is down by now.
void DisarmSlots(SparseRange& range) noexcept {
#ifdef PS5_NATIVE
    const std::size_t slots = (range.end - range.begin.load(std::memory_order_relaxed)) / SparseSlot;
    for (std::size_t slot = 0; slot < slots; ++slot) {
        if (SlotState(range, slot) != SlotProbe) continue;
        LockSlot(range, slot);
        const std::uint8_t state = SlotState(range, slot);
        if (state == SlotProbe &&
            mprotect(reinterpret_cast<void*>(SlotAddress(range, slot)), SparseSlot, PROT_READ | PROT_WRITE) != 0)
            std::abort();
        UnlockSlot(range, slot, state == SlotProbe ? SlotOwn : state);
    }
#else
    (void)range;
#endif
}

#ifdef PS5_NATIVE
// The scanner looks at an armed slot too, as an Own one, for the moment it takes. Without this an
// armed slot could not be given back or made Fine, and arming held on to memory the scanner would
// have freed (run 59 against runs 57 and 61: returns_scan 2 against 65 to 101, 182 MiB more held at
// the same point). A touch in that moment goes uncounted; the slot keeps its age.
bool LendArmedSlot(SparseRange& range, std::size_t slot) noexcept {
    LockSlot(range, slot);
    const std::uint8_t state = SlotState(range, slot);
    if (state != SlotProbe) {
        UnlockSlot(range, slot, state);
        return false;
    }
    if (mprotect(reinterpret_cast<void*>(SlotAddress(range, slot)), SparseSlot, PROT_READ | PROT_WRITE) != 0) std::abort();
    UnlockSlot(range, slot, SlotOwn);
    return true;
}
void TakeArmedSlotBack(SparseRange& range, std::size_t slot, std::uint32_t armed_at) noexcept {
    LockSlot(range, slot);
    const std::uint8_t state = SlotState(range, slot);
    if (state != SlotOwn) { // gone, made Fine, or armed again meanwhile
        UnlockSlot(range, slot, state);
        return;
    }
    if (mprotect(reinterpret_cast<void*>(SlotAddress(range, slot)), SparseSlot, PROT_NONE) != 0) std::abort();
    range.swap[slot] = armed_at;
    UnlockSlot(range, slot, SlotProbe);
}
#endif

std::uint64_t Mix(std::uint64_t hash, std::uint64_t word) noexcept {
    hash ^= word + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
    return hash * 0xff51afd7ed558ccdull;
}
std::uint64_t SampleHash(std::uintptr_t at) noexcept {
    const auto* word = reinterpret_cast<const volatile std::uint64_t*>(at);
    std::uint64_t hash = 0;
    for (std::size_t i = 0; i < SparseSlot / 8; i += SparseSlot / 8 / 64) hash = Mix(hash, word[i] ^ word[i + 1]);
    return hash;
}
std::uint64_t FullHash(std::uintptr_t at, std::size_t bytes) noexcept {
    const auto* word = reinterpret_cast<const volatile std::uint64_t*>(at);
    std::uint64_t hash = 0;
    for (std::size_t i = 0; i < bytes / 8; ++i) hash = Mix(hash, word[i]);
    return hash;
}

void Fence(std::uintptr_t at) noexcept {
    if (mprotect(reinterpret_cast<void*>(at), SparseSlot, PROT_READ) != 0) std::abort();
}
void Unfence(std::uintptr_t at) noexcept {
    if (mprotect(reinterpret_cast<void*>(at), SparseSlot, PROT_READ | PROT_WRITE) != 0) std::abort();
}

// An Own slot that holds only zeroes goes back. Fenced while it is looked at again.
bool ScanZero(SparseRange& range, std::size_t slot) noexcept {
    LockSlot(range, slot);
    if (SlotState(range, slot) != SlotOwn) {
        UnlockSlot(range, slot, SlotState(range, slot));
        return false;
    }
    const std::uintptr_t at = SlotAddress(range, slot);
    Fence(at);
    const bool zero = AllZero(at);
    if (zero) ReturnSlot(range, slot);
    else Unfence(at);
    UnlockSlot(range, slot, zero ? SlotZero : SlotOwn);
    if (zero) L961Add(l961_whole, SparseSlot);
    return zero;
}

// Own slots by the 16 KiB pages of zeroes they hold, for the report (EDEN_CRAM_ZEROS): place 0 none,
// then 1, 2-3, 4-7 ... 64-127. What a lower EDEN_CRAM_FINE_MIN_PAGES could give back, and what the slots that
// wait out a backoff hold. Written by the scanner only; the counts of the last whole pass are kept.
constexpr std::size_t ZeroPlaces = 8;
struct ZeroCounts {
    std::atomic<std::uint32_t> slots[ZeroPlaces]{}, pages[ZeroPlaces]{};
    std::atomic<std::uint32_t> waiting_slots{0}, waiting_pages{0};
};
ZeroCounts zero_counts_now, zero_counts;
void NoteZeroPages(std::size_t zeros, bool waiting) noexcept {
    std::size_t place = 0;
    while (place + 1 < ZeroPlaces && (std::size_t{1} << place) <= zeros) ++place;
    zero_counts_now.slots[place].fetch_add(1, std::memory_order_relaxed);
    zero_counts_now.pages[place].fetch_add(static_cast<std::uint32_t>(zeros), std::memory_order_relaxed);
    if (waiting) {
        zero_counts_now.waiting_slots.fetch_add(1, std::memory_order_relaxed);
        zero_counts_now.waiting_pages.fetch_add(static_cast<std::uint32_t>(zeros), std::memory_order_relaxed);
    }
}
// A quick pass (EDEN_UP_L961_ZERO_TRIM) starts the counts again without keeping them.
void KeepZeroCounts(bool keep = true) noexcept {
    if (!keep) {
        for (std::size_t place = 0; place < ZeroPlaces; ++place) {
            zero_counts_now.slots[place].store(0, std::memory_order_relaxed);
            zero_counts_now.pages[place].store(0, std::memory_order_relaxed);
        }
        zero_counts_now.waiting_slots.store(0, std::memory_order_relaxed);
        zero_counts_now.waiting_pages.store(0, std::memory_order_relaxed);
        return;
    }
    for (std::size_t place = 0; place < ZeroPlaces; ++place) {
        zero_counts.slots[place].store(zero_counts_now.slots[place].exchange(0, std::memory_order_relaxed), std::memory_order_relaxed);
        zero_counts.pages[place].store(zero_counts_now.pages[place].exchange(0, std::memory_order_relaxed), std::memory_order_relaxed);
    }
    zero_counts.waiting_slots.store(zero_counts_now.waiting_slots.exchange(0, std::memory_order_relaxed), std::memory_order_relaxed);
    zero_counts.waiting_pages.store(zero_counts_now.waiting_pages.exchange(0, std::memory_order_relaxed), std::memory_order_relaxed);
}

// An Own slot with many pages of zeroes becomes Fine. A racy count first, then the count again fenced.
// `quick` (EDEN_UP_L961_ZERO_TRIM): a quick pass never ends a backoff. The full pass that T83's count
// ends it on must have run first (fine_after == full passes until then), so a slot made whole is made
// Fine again no sooner than in T83, counted in full passes. fine_after 0: never made whole, no backoff.
bool ScanFine(SparseRange& range, std::size_t slot, bool quick = false) noexcept {
    const std::uintptr_t at = SlotAddress(range, slot);
    std::size_t zeros = 0;
    for (std::size_t page = 0; page < FinePages; ++page) zeros += PageZero(at + page * FinePage);
    const std::uint64_t done = BackoffPasses();
    const std::uint32_t after = range.fine_after[slot];
    const bool waiting = done < after || (quick && after != 0 && done == after);
    NoteZeroPages(zeros, waiting);
    if (waiting || zeros < fine_min_zero_pages) return false;
    LockSlot(range, slot);
    if (SlotState(range, slot) != SlotOwn) {
        UnlockSlot(range, slot, SlotState(range, slot));
        return false;
    }
    Fence(at);
    bool zero[FinePages];
    zeros = 0;
    for (std::size_t page = 0; page < FinePages; ++page) zeros += zero[page] = PageZero(at + page * FinePage);
    bool fined = false;
    if (zeros == FinePages) {
        ReturnSlot(range, slot);
        UnlockSlot(range, slot, SlotZero);
        counters.returns_scan.fetch_add(1, std::memory_order_relaxed);
        L961Add(l961_whole, SparseSlot);
        return true;
    }
    if (zeros >= fine_min_zero_pages) {
        fined = MakeFine(range, slot, zero, FinePages - zeros);
        if (!fined) counters.fine_failures.fetch_add(1, std::memory_order_relaxed);
        else L961Add(l961_fine, zeros * FinePage);
    }
    if (!fined) Unfence(at);
    UnlockSlot(range, slot, fined ? SlotFine : SlotOwn);
    return fined;
}

// Two slots of the same bytes show one block (the console; the host only counts them).
bool ScanShare(SparseRange& range, std::size_t keep, std::size_t other) noexcept {
    const std::size_t first = std::min(keep, other), second = std::max(keep, other);
    LockSlot(range, first);
    LockSlot(range, second);
    const std::uint8_t keep_state = SlotState(range, keep), other_state = SlotState(range, other);
    bool shared = false;
    if ((keep_state == SlotOwn || keep_state == SlotShared) && other_state == SlotOwn) {
        const std::uintptr_t keep_at = SlotAddress(range, keep), other_at = SlotAddress(range, other);
        if (keep_state == SlotOwn) Fence(keep_at);
        Fence(other_at);
        const bool same = std::memcmp(reinterpret_cast<const void*>(keep_at), reinterpret_cast<const void*>(other_at),
                                      SparseSlot) == 0;
#ifdef PS5_NATIVE
        // A block becoming shared counts both slots (the first ShareBlock makes it 1, the second 2); one
        // already shared counts one more. A table that is full shares nothing.
        bool counted = false;
        if (same && ShareBlock(range.physical[keep])) {
            counted = keep_state == SlotShared || ShareBlock(range.physical[keep]);
            if (!counted) (void)UnshareBlock(range.physical[keep]);
        }
        if (counted) {
            if (!MapSlot(other_at, PROT_READ, range.physical[keep])) std::abort();
            if (sceKernelReleaseDirectMemory(range.physical[other], SparseSlot) != 0) std::abort();
            range.physical[other] = range.physical[keep];
            guest_committed.fetch_sub(SparseSlot, std::memory_order_relaxed);
            shared = true;
            L961Add(l961_whole, SparseSlot);
        }
#else
        (void)same;
#endif
        if (!shared) {
            if (keep_state == SlotOwn) Unfence(keep_at);
            Unfence(other_at);
        }
        if (same) counters.dedup.fetch_add(1, std::memory_order_relaxed);
    }
    UnlockSlot(range, second, second == keep ? (shared ? SlotShared : keep_state) : (shared ? SlotShared : other_state));
    UnlockSlot(range, first, first == keep ? (shared ? SlotShared : keep_state) : (shared ? SlotShared : other_state));
    return shared;
}

// `full`: armed slots are lent and looked at too (every pass without EDEN_UP_L961_ZERO_TRIM).
void ScanPass(SparseRange& range, bool full = true) noexcept {
    const auto start = std::chrono::steady_clock::now();
    const std::size_t slots = (range.end - range.begin.load(std::memory_order_relaxed)) / SparseSlot;
    std::vector<std::pair<std::uint64_t, std::size_t>> samples;
    for (std::size_t slot = 0; slot < slots; ++slot) {
        const std::uint8_t state = SlotState(range, slot);
#ifdef PS5_NATIVE
        const std::uint32_t armed_at = range.swap[slot];
        const bool armed = state == SlotProbe && full && LendArmedSlot(range, slot);
#else
        constexpr bool armed = false;
#endif
        if (state != SlotOwn && !armed) continue;
        const std::uintptr_t at = SlotAddress(range, slot);
        if (AllZero(at)) { // a read without the lock: only a hint, checked again fenced
            if (ScanZero(range, slot)) counters.returns_scan.fetch_add(1, std::memory_order_relaxed);
        } else if (fine_requested.load(std::memory_order_relaxed) && ScanFine(range, slot, !full)) {
            // now Fine
        } else if (dedup_requested.load(std::memory_order_relaxed)) {
            samples.emplace_back(SampleHash(at), slot);
        }
#ifdef PS5_NATIVE
        if (armed) TakeArmedSlotBack(range, slot, armed_at);
#endif
    }
    std::sort(samples.begin(), samples.end());
    for (std::size_t i = 0; i < samples.size();) {
        std::size_t j = i + 1;
        while (j < samples.size() && samples[j].first == samples[i].first) ++j;
        if (j - i > 1) {
            std::vector<std::pair<std::uint64_t, std::size_t>> full;
            for (std::size_t k = i; k < j; ++k)
                full.emplace_back(FullHash(SlotAddress(range, samples[k].second), SparseSlot), samples[k].second);
            std::sort(full.begin(), full.end());
            for (std::size_t a = 0; a < full.size();) {
                std::size_t b = a + 1;
                while (b < full.size() && full[b].first == full[a].first) {
                    (void)ScanShare(range, full[a].second, full[b].second);
                    ++b;
                }
                a = b;
            }
        }
        i = j;
    }
    KeepZeroCounts(full);
    counters.passes.fetch_add(1, std::memory_order_relaxed);
    if (full) full_passes.fetch_add(1, std::memory_order_relaxed);
    counters.scan_ns.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count()), std::memory_order_relaxed);
}

void ServiceReport(SparseRange& range, const char* when) noexcept {
    const std::size_t slots = (range.end - range.begin.load(std::memory_order_relaxed)) / SparseSlot;
    std::size_t own = 0, shared = 0, fine = 0, fine_kept = 0, out = 0;
    constexpr std::size_t SlotsPerGiB = (std::size_t{1} << 30) / SparseSlot;
    unsigned held_per_gib[32]{};
    for (std::size_t slot = 0; slot < slots; ++slot) {
        const auto state = SlotState(range, slot);
        if (state != SlotZero && state != SlotOut) ++held_per_gib[std::min<std::size_t>(slot / SlotsPerGiB, 31)];
        own += state == SlotOwn || state == SlotProbe;
        out += state == SlotOut;
        shared += state == SlotShared;
        if (state == SlotFine) {
            ++fine;
            fine_kept += FineKept(range, slot);
        }
    }
    char line[640];
    const int length = std::snprintf(line, sizeof(line),
        "EDEN_CRAM %s held=%zuMiB own=%zu shared=%zu zero=%zu faults=%llu promotions=%llu copies=%llu waits=%llu "
        "returns_clear=%llu returns_free=%llu returns_scan=%llu dedup=%llu reserve_used=%llu failures=%llu "
        "passes=%llu scan_ms=%llu fine=%zu fine_held=%zuMiB fined=%llu unfined=%llu fine_failures=%llu "
        "fine_in_place=%llu tail_refused=%llu reserve=%zuMiB host_transferred=%lluMiB\n",
        when, guest_committed.load(std::memory_order_relaxed) >> 20, own, shared, slots - own - shared - fine - out,
        (unsigned long long)counters.faults.load(), (unsigned long long)counters.promotions.load(),
        (unsigned long long)counters.copies.load(), (unsigned long long)counters.waits.load(),
        (unsigned long long)counters.returns_clear.load(), (unsigned long long)counters.returns_free.load(),
        (unsigned long long)counters.returns_scan.load(), (unsigned long long)counters.dedup.load(),
        (unsigned long long)counters.reserve_used.load(), (unsigned long long)counters.failures.load(),
        (unsigned long long)counters.passes.load(), (unsigned long long)(counters.scan_ns.load() / 1000000),
        fine, fine_kept * FinePage >> 20, (unsigned long long)counters.fined.load(),
        (unsigned long long)counters.unfined.load(), (unsigned long long)counters.fine_failures.load(),
        (unsigned long long)fined_in_place.load(), (unsigned long long)fine_tail_refused.load(),
        reserve_count.load(std::memory_order_relaxed) * (SparseSlot >> 20),
        (unsigned long long)counters.host_reserve_used.load() * (SparseSlot >> 20));
    if (length > 0) Report(line);
    if (StallProbe::On()) { // EDEN_UP_STALL_PROBE: the totals beside each EDEN_CRAM line
        char totals[1024];
        if (StallProbe::TotalsLine(totals, sizeof(totals)) > 0) Report(totals);
    }
    if (l961_zero_trim.load(std::memory_order_relaxed)) {
        char trim[224];
        const int written = std::snprintf(trim, sizeof(trim),
            "EDEN_UP_L961 %s passes=%llu full=%llu busy=%llu gave_back=%lluMiB whole=%lluMiB fine=%lluMiB "
            "wait_s=%llu kernel_zero_filled=%lluMiB\n",
            when, (unsigned long long)l961_passes.load(), (unsigned long long)l961_full.load(),
            (unsigned long long)l961_busy.load(), (unsigned long long)((l961_whole.load() + l961_fine.load()) >> 20),
            (unsigned long long)(l961_whole.load() >> 20), (unsigned long long)(l961_fine.load() >> 20),
            (unsigned long long)l961_interval_s.load(), (unsigned long long)(l961_zero_filled.load() >> 20));
        if (written > 0) Report(trim);
    }
    SwapReport(range);
    GpuReadReport(when); // 0900 (off: nothing)
    // Where the slots lie, per GiB of the guest's memory: the kernel's pools are apart here (the
    // game's at the top, then the applets', then the system's, whose end nx2:gpu's blocks come from).
    char map[256];
    int mapped = std::snprintf(map, sizeof(map), "EDEN_CRAM_MAP slots held per GiB:");
    for (std::size_t gib = 0; gib * SlotsPerGiB < slots && gib < 32 && mapped > 0 &&
                              static_cast<std::size_t>(mapped) < sizeof(map) - 8; ++gib)
        mapped += std::snprintf(map + mapped, sizeof(map) - mapped, " %u", held_per_gib[gib]);
    if (mapped > 0 && static_cast<std::size_t>(mapped) < sizeof(map) - 2) {
        map[mapped++] = '\n';
        map[mapped] = 0;
        Report(map);
    }
    // Who wrote the slots that were promoted: the largest places (offsets in the app's image). A
    // writer's place comes from what lies on its stack, so one path can show as several: the first
    // load of a scene is spread over many places with under 30 slots each.
    constexpr std::size_t TopWriters = 24;
    struct Top { const WriterSite* site; std::uint64_t count; } top[TopWriters]{};
    for (const WriterSite& site : writer_sites) {
        const std::uint64_t count = site.count.load(std::memory_order_relaxed);
        if (site.key.load(std::memory_order_relaxed) == 0 || count <= top[TopWriters - 1].count) continue;
        std::size_t place = TopWriters - 1;
        while (place > 0 && top[place - 1].count < count) {
            top[place] = top[place - 1];
            --place;
        }
        top[place] = {&site, count};
    }
    char writers[640];
    int used = std::snprintf(writers, sizeof(writers),
                             "EDEN_CRAM_WRITERS promotions by code of the app %llu, by other code %llu "
                             "(code the JIT made %llu, system libraries %llu, not known %llu)\n",
                             (unsigned long long)promotions_in_app_code.load(), (unsigned long long)promotions_elsewhere.load(),
                             (unsigned long long)promotions_by_kind[WriterJit].load(),
                             (unsigned long long)promotions_by_kind[WriterLibrary].load(),
                             (unsigned long long)promotions_by_kind[WriterUnknown].load());
    if (used > 0) Report(writers);
    used = std::snprintf(writers, sizeof(writers), "EDEN_CRAM_ZEROS Own slots at the last pass by 16 KiB pages of zeroes:");
    for (std::size_t place = 0; place < ZeroPlaces && used > 0; ++place)
        used += std::snprintf(writers + used, sizeof(writers) - used, " %zu+=%u slots/%uMiB",
                              place == 0 ? std::size_t{0} : std::size_t{1} << (place - 1), zero_counts.slots[place].load(),
                              static_cast<unsigned>(std::uint64_t{zero_counts.pages[place].load()} * FinePage >> 20));
    if (used > 0)
        used += std::snprintf(writers + used, sizeof(writers) - used, "; waiting out a backoff %u slots/%uMiB\n",
                              zero_counts.waiting_slots.load(),
                              static_cast<unsigned>(std::uint64_t{zero_counts.waiting_pages.load()} * FinePage >> 20));
    if (used > 0) Report(writers);
    for (const ClearSite& site : clear_sites) {
        const std::uintptr_t pc = site.pc.load(std::memory_order_acquire);
        if (pc == 0) break;
        if (std::snprintf(writers, sizeof(writers), "EDEN_CRAM_CLEAR_SITE %llu slots in %llu calls: %llx\n",
                          (unsigned long long)site.slots.load(), (unsigned long long)site.calls.load(),
                          (unsigned long long)(pc - 0x400000ull)) > 0)
            Report(writers);
    }
    for (const Top& entry : top) {
        if (entry.count == 0) break;
        int length = std::snprintf(writers, sizeof(writers), "EDEN_CRAM_WRITER %llu:", (unsigned long long)entry.count);
        for (const std::uintptr_t pc : entry.site->pcs) {
            if (length > 0 && static_cast<std::size_t>(length) < sizeof(writers) - 24)
                length += std::snprintf(writers + length, sizeof(writers) - length, " %llx",
                                        (unsigned long long)(pc != 0 ? pc - 0x400000ull : 0));
        }
        if (length > 0 && static_cast<std::size_t>(length) < sizeof(writers) - 2) {
            writers[length++] = '\n';
            writers[length] = 0;
            Report(writers);
        }
    }
}

} // namespace

// What the guest's memory took last, for a graphics allocation the pool refused
// (headless/performance.cpp, ReportPool): the Zero and Fine slots a write made Own, each a whole block
// from the pool, which is what adds to it while the service runs (Shared slots are off unless asked
// for). A slot is a part of the guest's backing, so a range here is a physical range of the guest,
// not a guest address.
extern "C" void eden_guest_memory_growth(void) noexcept {
    SparseRange* const range = service_range.load(std::memory_order_acquire);
    if (range == nullptr) return;
    ServiceReport(*range, "refusal");
    constexpr std::uint64_t WindowMs = 60000;
    constexpr std::size_t RangeSlots = 128; // 256 MiB of backing
    constexpr std::size_t Ranges = 256;
    std::uint16_t by_site[WriterSites + 1] = {}, by_range[Ranges] = {};
    std::size_t last_5s = 0, last_20s = 0, last_60s = 0, by_app = 0, read = 0;
    std::uint64_t oldest_ms = 0;
    const std::uint64_t total = recent_total.load(std::memory_order_acquire), now = MonotonicMs();
    const std::size_t kept = static_cast<std::size_t>(std::min<std::uint64_t>(total, RecentCount));
    for (; read < kept; ++read) {
        const std::uint64_t word = recent_promotions[(total - 1 - read) % RecentCount].load(std::memory_order_relaxed);
        const std::uint64_t taken = word >> RecentTimeShift;
        const std::uint64_t age = now > taken ? now - taken : 0;
        if (age > WindowMs) break;
        oldest_ms = age;
        ++last_60s;
        last_20s += age <= 20000;
        last_5s += age <= 5000;
        by_app += (word >> (RecentSlotBits + RecentSiteBits)) & 1;
        ++by_site[(word >> RecentSlotBits) & ((1u << RecentSiteBits) - 1)];
        ++by_range[std::min<std::size_t>((word & ((1u << RecentSlotBits) - 1)) / RangeSlots, Ranges - 1)];
    }
    constexpr std::size_t SlotMiB = SparseSlot >> 20;
    char line[384];
    int length = std::snprintf(line, sizeof(line),
        "EDEN_GUEST_GROWTH held=%zuMiB: slots taken in the last 5 s %zuMiB, 20 s %zuMiB, 60 s %zuMiB (%zuMiB by code of the "
        "app, %zuMiB by other code); %zu kept, the oldest read %llu ms ago; %llu in the session, %llu given back "
        "(taken is not growth: a slot given back is taken again)\n",
        guest_committed.load(std::memory_order_relaxed) >> 20, last_5s * SlotMiB, last_20s * SlotMiB, last_60s * SlotMiB,
        by_app * SlotMiB, (last_60s - by_app) * SlotMiB, read, (unsigned long long)oldest_ms, (unsigned long long)total,
        (unsigned long long)(counters.returns_clear.load() + counters.returns_free.load() + counters.returns_scan.load()));
    if (length > 0) Report(line);
    // The largest writers and ranges of that minute.
    for (unsigned pick = 0; pick < 6; ++pick) {
        const auto* const top = std::max_element(by_site, by_site + WriterSites + 1);
        if (*top == 0) break;
        const auto site = static_cast<std::size_t>(top - by_site);
        length = std::snprintf(line, sizeof(line), "EDEN_GUEST_GROWTH_WRITER %zuMiB:", *top * SlotMiB);
        if (site == WriterSites) {
            length += std::snprintf(line + length, sizeof(line) - length, " no place left in the table");
        } else {
            for (const std::uintptr_t pc : writer_sites[site].pcs)
                length += std::snprintf(line + length, sizeof(line) - length, " %llx",
                                        (unsigned long long)(pc != 0 ? pc - 0x400000ull : 0));
        }
        std::snprintf(line + length, sizeof(line) - length, "\n");
        Report(line);
        by_site[site] = 0;
    }
    for (unsigned pick = 0; pick < 6; ++pick) {
        const auto* const top = std::max_element(by_range, by_range + Ranges);
        if (*top == 0) break;
        const auto at = static_cast<std::size_t>(top - by_range);
        std::snprintf(line, sizeof(line), "EDEN_GUEST_GROWTH_RANGE backing at 0x%llx: %zuMiB of its %zuMiB\n",
                      (unsigned long long)at * RangeSlots * SparseSlot, *top * SlotMiB, RangeSlots * SlotMiB);
        Report(line);
        by_range[at] = 0;
    }
}

namespace {
// What finer slots would give: 16 KiB pages of the Own slots that hold only zeroes, and those that
// repeat another page (by a full hash). Read once, when the session ends.
void MeasureFinePages(SparseRange& range) noexcept {
    constexpr std::size_t Fine = 16384;
    const std::size_t slots = (range.end - range.begin.load(std::memory_order_relaxed)) / SparseSlot;
    std::vector<std::uint64_t> hashes;
    std::size_t pages = 0, zero = 0;
    for (std::size_t slot = 0; slot < slots; ++slot) {
        if (SlotState(range, slot) != SlotOwn) continue;
        for (std::size_t offset = 0; offset < SparseSlot; offset += Fine) {
            const std::uintptr_t at = SlotAddress(range, slot) + offset;
            ++pages;
            const auto* word = reinterpret_cast<const volatile std::uint64_t*>(at);
            bool is_zero = true;
            for (std::size_t i = 0; i < Fine / 8 && is_zero; ++i) is_zero = word[i] == 0;
            if (is_zero) ++zero;
            else hashes.push_back(FullHash(at, Fine));
        }
    }
    std::sort(hashes.begin(), hashes.end());
    std::size_t repeats = 0;
    for (std::size_t i = 1; i < hashes.size(); ++i) repeats += hashes[i] == hashes[i - 1];
    char line[192];
    const int length = std::snprintf(line, sizeof(line),
        "EDEN_CRAM_FINE pages16k=%zu zero=%zu (%zuMiB) repeats=%zu (%zuMiB)\n", pages, zero, zero * Fine >> 20, repeats,
        repeats * Fine >> 20);
    if (length > 0) Report(line);
}

// EDEN_UP_L961_ZERO_TRIM=1 (zortos293/Lichii#961): one pass, full or quick, and the wait before the
// next one: 5 s after a pass that gave back 64 MiB or more, else twice the last wait, up to the most.
std::chrono::seconds L961Pass(SparseRange& range, std::chrono::seconds interval,
                              std::chrono::steady_clock::time_point& last_full) noexcept {
    const auto start = std::chrono::steady_clock::now();
    const std::chrono::seconds most{l961_max_s.load(std::memory_order_relaxed)};
    const bool full = start - last_full >= most;
    if (full) last_full = start;
    const std::uint64_t whole_before = l961_whole.load(std::memory_order_relaxed);
    const std::uint64_t fine_before = l961_fine.load(std::memory_order_relaxed);
    ScanPass(range, full);
    const std::uint64_t whole = l961_whole.load(std::memory_order_relaxed) - whole_before;
    const std::uint64_t fine = l961_fine.load(std::memory_order_relaxed) - fine_before;
    l961_passes.fetch_add(1, std::memory_order_relaxed);
    if (full) l961_full.fetch_add(1, std::memory_order_relaxed);
    const bool busy = whole + fine >= L961Busy;
    interval = busy ? std::chrono::seconds{L961FastS} : std::min(interval * 2, most);
    l961_interval_s.store(static_cast<std::uint64_t>(interval.count()), std::memory_order_relaxed);
    if (!busy) return interval;
    l961_busy.fetch_add(1, std::memory_order_relaxed);
    const long long ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    char line[224];
    const int length = std::snprintf(line, sizeof(line),
        "EDEN_UP_L961 pass gave back %lluMiB (whole %lluMiB, fine %lluMiB) of held=%zuMiB in %lldms full=%d, "
        "next in %llds; kernel zero-filled %lluMiB so far\n",
        (unsigned long long)((whole + fine) >> 20), (unsigned long long)(whole >> 20), (unsigned long long)(fine >> 20),
        guest_committed.load(std::memory_order_relaxed) >> 20, ms, full ? 1 : 0,
        static_cast<long long>(interval.count()), (unsigned long long)(l961_zero_filled.load() >> 20));
    if (length > 0) Report(line);
    return interval;
}

void ScannerLoop(SparseRange* range) {
    if (::eden_crash_name_thread) ::eden_crash_name_thread("CramScanner");  // crash report, thread dump
    StallProbe::role = StallProbe::Scanner;
    auto next_report = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    // 20 s between passes, or with EDEN_UP_L961_ZERO_TRIM what the passes give back says (L961Pass).
    const bool adaptive = l961_zero_trim.load(std::memory_order_relaxed);
    std::chrono::seconds interval{adaptive ? L961FastS : 20};
    auto last_full = std::chrono::steady_clock::now();
    std::unique_lock lock{scanner_mutex};
    while (!scanner_stop) {
        scanner_wake.wait_for(lock, interval, [] { return scanner_stop; });
        if (scanner_stop) break;
        lock.unlock();
        if (scan_requested.load(std::memory_order_relaxed)) {
            if (adaptive) interval = L961Pass(*range, interval, last_full);
            else ScanPass(*range);
        }
        if (std::chrono::steady_clock::now() >= next_report) {
            ServiceReport(*range, "now");
            next_report = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        }
        lock.lock();
    }
}

// Once: does a write to a slot that shows the zero block come back here, with its address, and
// resume? A refusal keeps the guest's memory committed when it is handed out, as before.
bool ServiceAvailable() noexcept {
    int state = service_state.load(std::memory_order_acquire);
    if (state != 0) return state == 1;
    const char* step = "ok";
    long long fault_ns = -1;
    void* base = nullptr;
    if (!SparseAvailable()) step = "sparse";
    else if (!InstallServiceSignal()) step = "install";
    else if ((base = NewSparseRange(2 * SparseSlot, false)) == nullptr) step = "reserve";
    if (base != nullptr) {
        SparseRange* range = nullptr;
        {
            const std::lock_guard lock{sparse_mutex};
            range = SparseRangeOf(reinterpret_cast<std::uintptr_t>(base));
        }
        if (range == nullptr || !AddStates(*range)) {
            step = "states";
        } else {
            probe_range.store(range, std::memory_order_release);
            auto* word = reinterpret_cast<volatile std::uint64_t*>(static_cast<std::uint8_t*>(base) + SparseSlot + 64);
            const auto before = counters.faults.load(std::memory_order_relaxed);
            probe_expected.store(reinterpret_cast<std::uintptr_t>(word), std::memory_order_release);
            const auto start = std::chrono::steady_clock::now();
            *word = 0x5a17ed5a17ed5a17ull; // faults once: the slot shows the zero block
            fault_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
            probe_expected.store(0, std::memory_order_release);
            if (*word != 0x5a17ed5a17ed5a17ull) step = "value";
            else if (counters.faults.load(std::memory_order_relaxed) != before + 1) step = "fault-count";
            else if (probe_mismatch.load(std::memory_order_relaxed)) step = "fault-address";
            else if (SlotState(*range, 1) != SlotOwn || SlotState(*range, 0) != SlotZero) step = "state";
            else {
                // Back to the zero block, then written again: a slot comes and goes.
                LockSlot(*range, 1);
                ReturnSlot(*range, 1);
                UnlockSlot(*range, 1, SlotZero);
                if (*word != 0) step = "return";
                else {
                    *word = 7;
                    if (*word != 7 || SlotState(*range, 1) != SlotOwn) step = "again";
                }
            }
            if (std::strcmp(step, "ok") == 0) {
                // Fenced and unfenced: read while fenced, written after.
                const std::uintptr_t at = SlotAddress(*range, 1);
                Fence(at);
                if (*word != 7) step = "fence-read";
                Unfence(at);
                *word = 9;
                if (std::strcmp(step, "ok") == 0 && *word != 9) step = "unfence";
            }
            if (std::strcmp(step, "ok") == 0) {
                // Two slots of the same bytes: one block, then copied apart by a write, then the last
                // one takes the block back.
                auto* other = reinterpret_cast<volatile std::uint64_t*>(static_cast<std::uint8_t*>(base) + 64);
                *other = 9;
                const bool shared = ScanShare(*range, 1, 0);
#ifdef PS5_NATIVE
                if (!shared || SlotState(*range, 0) != SlotShared || SlotState(*range, 1) != SlotShared) step = "share";
                else if (*other != 9 || *word != 9) step = "share-read";
                else {
                    *other = 10; // a copy of its own
                    if (*other != 10 || *word != 9 || SlotState(*range, 0) != SlotOwn) step = "share-copy";
                    else {
                        *word = 11; // the last one: the block is its own again
                        if (*word != 11 || *other != 10 || SlotState(*range, 1) != SlotOwn) step = "share-last";
                    }
                }
#else
                if (shared) step = "share-host";
#endif
            }
            if (std::strcmp(step, "ok") == 0) {
                // Slot 1 with one page of data at page 3: Fine (one 16 KiB page kept), read in place,
                // then a write makes it whole again with every byte where it was.
                LockSlot(*range, 1);
                if (SlotState(*range, 1) != SlotZero) ReturnSlot(*range, 1);
                UnlockSlot(*range, 1, SlotZero);
                auto* page3 = reinterpret_cast<volatile std::uint64_t*>(static_cast<std::uint8_t*>(base) + SparseSlot + 3 * FinePage + 8);
                *page3 = 0x0f1e2d3c4b5a6978ull;
                const auto held = guest_committed.load(std::memory_order_relaxed);
                if (!ScanFine(*range, 1) || SlotState(*range, 1) != SlotFine) step = "fine";
                else if (*page3 != 0x0f1e2d3c4b5a6978ull || *word != 0) step = "fine-read";
                else if (held - guest_committed.load(std::memory_order_relaxed) != SparseSlot - FinePage) step = "fine-held";
                else {
                    *word = 13; // page 0: the slot is whole again
                    if (*word != 13 || *page3 != 0x0f1e2d3c4b5a6978ull || SlotState(*range, 1) != SlotOwn ||
                        (range->fine[1] & 0xf00) == 0)
                        step = "fine-write";
                }
            }
#ifdef PS5_NATIVE
            if (std::strcmp(step, "ok") == 0 && swap_mode.load(std::memory_order_relaxed) != 0) {
                char speeds[160] = "";
                swap_refused = SwapCheck(*range, word, speeds, sizeof(speeds));
                char line[256];
                const int length = std::snprintf(line, sizeof(line), "EDEN_SWAP_CHECK ok=%d step=%s file=%s %s\n",
                                                 std::strcmp(swap_refused, "ok") == 0 ? 1 : 0, swap_refused, swap_path, speeds);
                if (length > 0) Report(line);
                CloseSwapFile();
            }
#endif
            for (std::size_t slot = 0; slot < 2; ++slot)
                if (SlotState(*range, slot) != SlotZero) {
                    LockSlot(*range, slot);
                    ReturnSlot(*range, slot);
                    UnlockSlot(*range, slot, SlotZero);
                }
            probe_range.store(nullptr, std::memory_order_release);
        }
        FreeSparsePages(base, 2 * SparseSlot);
    }
    ResetCounters();
    const bool ok = std::strcmp(step, "ok") == 0;
    service_refused = step;
    char line[128];
    const int length = std::snprintf(line, sizeof(line), "EDEN_CRAM_CHECK ok=%d step=%s fault_ns=%lld\n", ok ? 1 : 0,
                                     step, fault_ns);
    if (length > 0) Report(line);
    service_state.store(ok ? 1 : 2, std::memory_order_release);
    return ok;
}

// The guest range starts being a service (when the platform passed the check).
void ServiceStart(SparseRange& range) noexcept {
    // A refusal belongs to the session it ended: a later crash must not report it again.
    refused_address.store(0, std::memory_order_relaxed);
    ReadSwapSwitches();
    if (!service_requested.load(std::memory_order_relaxed) || !ServiceAvailable() || !AddStates(range)) {
        char line[96];
        const int length = std::snprintf(line, sizeof(line), "EDEN_CRAM off: %s\n",
                                         !service_requested.load() ? "switched-off" : service_refused);
        if (length > 0) Report(line);
        return;
    }
    ResetCounters();
    // EDEN_UP_L961_ZERO_TRIM=1 (zortos293/Lichii#961): the scanner's schedule (at L961Busy, above).
    l961_zero_trim.store(Common::PeSwitch("EDEN_UP_L961_ZERO_TRIM"), std::memory_order_relaxed);
    {
        const char* most = std::getenv("EDEN_UP_L961_ZERO_TRIM_MAX_S");
        const unsigned long asked = most != nullptr ? std::strtoul(most, nullptr, 10) : 0;
        l961_max_s.store(asked >= L961FastS && asked <= 300 ? static_cast<unsigned>(asked) : 20u,
                         std::memory_order_relaxed);
    }
    for (auto* counter : {&l961_whole, &l961_fine, &l961_zero_filled, &l961_passes, &l961_full, &l961_busy,
                          &l961_interval_s, &full_passes})
        counter->store(0, std::memory_order_relaxed);
    range.promote = true;
    const char* const free_return = std::getenv("EDEN_CRAM_FREE_RETURN");
    free_return_requested.store(free_return != nullptr && free_return[0] == '1', std::memory_order_relaxed);
    if (const char* pages = std::getenv("EDEN_CRAM_FINE_MIN_PAGES")) {
        const unsigned long asked = std::strtoul(pages, nullptr, 10);
        if (asked >= 1 && asked < FinePages) fine_min_zero_pages = asked;
    }
#ifdef PS5_NATIVE
    if (const char* floor = std::getenv("EDEN_GUEST_RESERVE_FLOOR"))
        reserve_floor.store(std::min<std::size_t>(std::strtoul(floor, nullptr, 10), ReserveBlocks),
                            std::memory_order_relaxed);
    if (const char* mib = std::getenv("EDEN_NX2_CRAM_RESERVE_MIB")) {
        const long asked = std::atol(mib);
        reserve_target.store(asked >= 0 ? std::min<std::size_t>(static_cast<std::size_t>(asked) / (SparseSlot >> 20), ReserveBlocks)
                                        : ReserveBlocks,
                             std::memory_order_relaxed);
    }
    // T178b EDEN_UP_TABLE_RESERVE=<1 to 64> (1: 8 blocks): the tables' own reserve, filled first.
    {
        std::size_t blocks = 0;
        if (const char* value = std::getenv("EDEN_UP_TABLE_RESERVE"); value != nullptr && *value >= '1' && *value <= '9') {
            const unsigned long asked = std::strtoul(value, nullptr, 10);
            blocks = asked == 1 ? TableReserveDefault : std::min<std::size_t>(asked, TableReserveMost);
        }
        table_reserve_target.store(blocks, std::memory_order_relaxed);
        if (blocks != 0) {
            FillTableReserve();
            char line[200];
            const int length = std::snprintf(line, sizeof(line),
                                             "EDEN_UP_TABLE_RESERVE on: %zu blocks for the sparse tables (filled %zu), "
                                             "before every other reserve; table wait %u ms\n",
                                             blocks, table_reserve_count.load(std::memory_order_relaxed), TableWaitMs());
            if (length > 0) Report(line);
        }
    }
    FillReserve(ReserveAtStart);
    ReadJitLendSwitches(); // EDEN_UP_JIT_LEND_UNDER_FLOOR (after the floor)
#endif
    service_range.store(&range, std::memory_order_release);
    Eden::Fastmem::guest_write_fault.store(&HandleGuestWriteFault, std::memory_order_release);
    {
        const std::lock_guard lock{scanner_mutex};
        scanner_stop = false;
    }
    StartStallProbe(); // EDEN_UP_STALL_PROBE
    scanner = std::thread(ScannerLoop, &range);
#ifdef PS5_NATIVE
    keeper = std::thread(ReserveKeeper);
    // A check that failed leaves this off; in measure mode a drive that cannot be written does not.
    if (swap_mode.load(std::memory_order_relaxed) != 0 && std::strcmp(swap_refused, "ok") != 0)
        swap_mode.store(0, std::memory_order_relaxed);
    if (swap_mode.load(std::memory_order_relaxed) == 2 && !OpenSwapFile()) swap_mode.store(1, std::memory_order_relaxed);
#ifndef EDEN_SWAP_LZ4
    if (swap_mode.load(std::memory_order_relaxed) == 3) {
        swap_mode.store(1, std::memory_order_relaxed);
        Report("EDEN_SWAP packing wants LZ4, which this build has not: measuring only\n");
    }
#endif
    packed_pages_held.store(0, std::memory_order_relaxed);
    packed_in_place.store(0, std::memory_order_relaxed);
    packed_pool_first.store(0, std::memory_order_relaxed);
    pack_tail_refused.store(0, std::memory_order_relaxed);
    pack_not_worth.store(0, std::memory_order_relaxed);
    ReadPackSwitches(); // EDEN_UP_PACK_* (the mode is final now)
    ReadGpuReadSwitches(); // 0900 EDEN_UP_GPU_READ_THROUGH, EDEN_UP_FAULT_* (the swap file is open now)
    if (swap_mode.load(std::memory_order_relaxed) != 0) {
        for (auto* counter : {&swap_counters.armed, &swap_counters.touched, &swap_counters.out, &swap_counters.in,
                              &swap_counters.out_ns, &swap_counters.in_ns, &swap_counters.in_most_ns,
                              &swap_counters.failures, &swap_counters.sampled, &swap_counters.packed,
                              &swap_counters.returned_armed})
            counter->store(0, std::memory_order_relaxed);
        for (auto& age : swap_counters.age) age.store(0, std::memory_order_relaxed);
        swapper = std::thread(SwapLoop, &range);
    }
#endif
    char line[128];
    const int length = std::snprintf(line, sizeof(line),
                                     "EDEN_CRAM on: scan=%d dedup=%d fine=%d reserve=%zuMiB fine_min=%zu free_return=%d\n",
                                     scan_requested.load() ? 1 : 0, dedup_requested.load() ? 1 : 0,
                                     fine_requested.load() ? 1 : 0,
                                     reserve_count.load() * (SparseSlot >> 20), fine_min_zero_pages,
                                     free_return_requested.load() ? 1 : 0);
    if (length > 0) Report(line);
    if (l961_zero_trim.load(std::memory_order_relaxed)) {
        const int written = std::snprintf(line, sizeof(line),
                                          "EDEN_UP_L961_ZERO_TRIM on: first=%us fast=%us most=%us busy=%lluMiB\n",
                                          L961FastS, L961FastS, l961_max_s.load(),
                                          (unsigned long long)(L961Busy >> 20));
        if (written > 0) Report(line);
    }
}

void ServiceStop(SparseRange& range) noexcept {
    if (service_range.load(std::memory_order_acquire) != &range) return;
    {
        const std::lock_guard lock{scanner_mutex};
        scanner_stop = true;
    }
    scanner_wake.notify_all();
    if (scanner.joinable()) scanner.join();
#ifdef PS5_NATIVE
    if (keeper.joinable()) keeper.join();
    if (swapper.joinable()) swapper.join();
#endif
    StopStallProbe();
    StopGpuRead(); // 0900: plain copies again before the slots change
    ServiceReport(range, "end");
    MeasureFinePages(range);
    DisarmSlots(range);
    service_range.store(nullptr, std::memory_order_release);
#ifdef PS5_NATIVE
    EmptyReserve();
    table_reserve_target.store(0, std::memory_order_relaxed); // T178b: off until the next start reads it again
    EmptyTableReserve();                                        // (0 held when off)
    CloseSwapFile();
#endif
}

// What a guest range keeps when it is given back: each block once, shared ones too.
void ReleaseServiceBlocks(SparseRange& range) noexcept {
#ifdef PS5_NATIVE
    const std::size_t slots = (range.end - range.begin.load(std::memory_order_relaxed)) / SparseSlot;
    for (std::size_t slot = 0; slot < slots; ++slot) {
        const auto state = SlotState(range, slot);
        if (state == SlotFine) {
            const std::size_t kept = FineKept(range, slot);
            if (kept != 0 && sceKernelReleaseDirectMemory(range.physical[slot], kept * FinePage) != 0) std::abort();
            guest_committed.fetch_sub(kept * FinePage, std::memory_order_relaxed);
        } else if (state == SlotOwn || state == SlotProbe) {
            if (sceKernelReleaseDirectMemory(range.physical[slot], SparseSlot) != 0) std::abort();
        } else if (state == SlotOut && InStore(range.physical[slot])) { // EDEN_UP_PACK_SHARED's store
            DropCopy(StoreSlab(range.physical[slot]), range.swap[slot]);
        } else if (state == SlotOut && range.physical[slot] >= 0) { // packed in memory
            const std::size_t kept = PackedPages(range.swap[slot]);
            if (sceKernelReleaseDirectMemory(range.physical[slot], kept * FinePage) != 0) std::abort();
            guest_committed.fetch_sub(kept * FinePage, std::memory_order_relaxed);
            packed_pages_held.fetch_sub(kept, std::memory_order_relaxed);
        } else if (state == SlotShared && UnshareBlock(range.physical[slot]) == 0) {
            if (sceKernelReleaseDirectMemory(range.physical[slot], SparseSlot) != 0) std::abort();
            guest_committed.fetch_sub(SparseSlot, std::memory_order_relaxed);
        }
        if (state == SlotOwn || state == SlotProbe) guest_committed.fetch_sub(SparseSlot, std::memory_order_relaxed);
        range.owned[slot].store(false, std::memory_order_relaxed);
    }
    if (range.guest && pack_shared.load(std::memory_order_relaxed)) ClosePackStore(); // every copy is gone now
#else
    const std::size_t slots = (range.end - range.begin.load(std::memory_order_relaxed)) / SparseSlot;
    for (std::size_t slot = 0; slot < slots; ++slot) {
        if (SlotState(range, slot) == SlotOwn) guest_committed.fetch_sub(SparseSlot, std::memory_order_relaxed);
        if (SlotState(range, slot) == SlotFine) guest_committed.fetch_sub(FineKept(range, slot) * FinePage, std::memory_order_relaxed);
        range.owned[slot].store(false, std::memory_order_relaxed);
    }
#endif
}
} // namespace

// Whether a slot may be without a mapping (Probe, Out): a system call must not be given a pointer into
// the guest's memory then (tools/eglantine-patches/0125).
extern "C" bool eden_guest_swap_active(void) noexcept {
    return swap_mode.load(std::memory_order_relaxed) != 0;
}

// Where the game's pool lies in the guest's backing (tools/eglantine-patches/0182), when the guest's
// kernel starts. Whole slots only: a slot it shares with the applets' pool stays unarmed.
extern "C" void eden_guest_swap_pool(std::uint64_t offset, std::uint64_t bytes) noexcept {
    swap_pool_first.store((offset + SparseSlot - 1) / SparseSlot, std::memory_order_relaxed);
    swap_pool_end.store((offset + bytes) / SparseSlot, std::memory_order_relaxed);
}

// The slot behind a host address of the guest's memory (core dma_pusher.cpp, EDEN_NX2_JUNK_TRACE):
// its state, and in *ever_out whether it ever went out. -1 for an address outside the guest's
// range or with no service running.
extern "C" int eden_guest_swap_address_state(const void* host, int* ever_out) noexcept {
    if (ever_out != nullptr) *ever_out = 0;
    SparseRange* const range = guest_range.load(std::memory_order_acquire);
    if (range == nullptr || range->state == nullptr) return -1;
    const auto at = reinterpret_cast<std::uintptr_t>(host);
    const std::uintptr_t begin = range->begin.load(std::memory_order_relaxed);
    if (at < begin || at >= range->end) return -1;
    const std::size_t slot = (at - begin) / SparseSlot;
    if (ever_out != nullptr && slot < SwapEverOutSlots)
        *ever_out = static_cast<int>((swap_ever_out[slot / 64].load(std::memory_order_relaxed) >> (slot % 64)) & 1);
    return SlotState(*range, slot);
}

#ifdef PS5_NATIVE
namespace {
// EDEN_UP_RADV_OOM_LEND (eden_guest_memory_reclaim_bounded): cold slots out, lowest in the pool first,
// until `want` bytes went back or 256 slots went; a copy of the loop and the line of
// eden_guest_memory_reclaim (below), which keeps its own text. `cold` is sorted; the caller holds
// swap_out_mutex.
std::size_t ReclaimColdSlots(SparseRange& range, std::size_t want, const std::pair<std::int64_t, std::size_t>* cold,
                             std::size_t count) noexcept {
    std::size_t freed = 0, made = 0;
    const std::uint64_t failed = swap_counters.failures.load(std::memory_order_relaxed);
    for (std::size_t i = 0; i < count && made < 256 && freed < want; ++i) {
        const std::size_t slot = cold[i].second;
        if (SwapOut(range, slot)) {
            ++made;
            // What the slot gave back: all of it to the file, or 2 MiB less its packed pages. Read
            // after the fact, so a touch that brought it back already counts as nothing.
            if (SlotState(range, slot) == SlotOut) {
                const std::int64_t physical = range.physical[slot];
                freed += InStore(physical) ? pack_shared_given // EDEN_UP_PACK_SHARED: 2 MiB, or 0 for a new slab
                         : physical < 0    ? SparseSlot
                                           : SparseSlot - std::min(SparseSlot, PackedPages(range.swap[slot]) * FinePage);
            }
            if (slot < SwapEverOutSlots)
                swap_ever_out[slot / 64].fetch_or(std::uint64_t{1} << (slot % 64), std::memory_order_relaxed);
        } else if (swap_counters.failures.load(std::memory_order_relaxed) != failed) {
            break; // no block for the packed bytes, or the file refused: nothing more now
        }
    }
    char line[128];
    const int length = std::snprintf(line, sizeof(line), "EDEN_SWAP reclaim for graphics: %zu slots, %zu MiB given back (asked %zu MiB)\n",
                                     made, freed >> 20, want >> 20);
    if (length > 0) Report(line);
    return freed;
}

// EDEN_UP_RADV_OOM_LEND (eden_guest_memory_reclaim_bounded): the candidates, the 8192 lowest in the pool
// (a max-heap while they are gathered, 128 KiB of static memory: every slot of a 16 GiB guest span, so
// in practice the same list as the old one, which walks on past slots not worth packing), so nothing is
// asked of the heap on a driver thread with the pool empty; and swap_tick + 1 of the last scan that
// found no cold slot (0: none). Both under swap_out_mutex. A slot turns cold almost only when the tick
// moves on (ArmSlot stamps it with the tick it is armed in). Exceptions, each one slot at a time: the
// scanner's LendArmedSlot/TakeArmedSlotBack (the slot is Own for a moment and comes back with its old
// stamp), and an ArmSlot that read the tick just before the swapper moved it. Verify counts those.
constexpr std::size_t ReclaimCandidates = 8192;
std::pair<std::int64_t, std::size_t> reclaim_cold[ReclaimCandidates];
std::uint64_t reclaim_zero_tick = 0;
// Counters (eden_guest_reserve_lend_stats): relaxed, any thread.
std::atomic<std::uint64_t> reclaim_scans{0}, reclaim_skipped{0}, reclaim_verified{0}, reclaim_missed{0},
    reclaim_capped{0};
} // namespace
#endif

// A graphics allocation found no memory (the port's MemoryAllocator::ReleaseRefused): cold slots go
// out now, on the thread that asked, up to `want` bytes given back or 256 slots, the lowest in the
// pool first. After Aflakon's eden_guest_memory_reclaim (ocarina-cold-memory f0515ee16, patch 1905),
// on this service's swapper: a slot untouched for a tick or more (2 s) is taken, by the mode's way
// out (the file, or LZ4 in memory, in place when asked). EDEN_GUEST_SWAP_RECLAIM=1 turns it on; it
// needs EDEN_GUEST_SWAP=2 or 3. Bytes given back. EDEN_UP_PACK_ON_REFUSAL=1: 64 MiB or more asked is
// PackOnRefusal's, with or without EDEN_GUEST_SWAP_RECLAIM.
extern "C" std::size_t eden_guest_memory_reclaim(std::size_t want) noexcept {
#ifdef PS5_NATIVE
    static const bool enabled = std::getenv("EDEN_GUEST_SWAP_RECLAIM") != nullptr && std::getenv("EDEN_GUEST_SWAP_RECLAIM")[0] == '1';
    const int mode = swap_mode.load(std::memory_order_relaxed);
    SparseRange* const range = service_range.load(std::memory_order_acquire);
    if (pack_on_refusal.load(std::memory_order_relaxed) && want >= PackRefusalLarge && (mode == 2 || mode == 3) &&
        range != nullptr && range->state != nullptr)
        return PackOnRefusal(*range, want);
    if (!enabled || (mode != 2 && mode != 3) || range == nullptr || range->state == nullptr) return 0;
    const std::size_t slots = (range->end - range->begin.load(std::memory_order_relaxed)) / SparseSlot;
    const std::uint32_t now = swap_tick.load(std::memory_order_relaxed);
    std::vector<std::pair<std::int64_t, std::size_t>> cold;
    for (std::size_t slot = 0; slot < slots; ++slot)
        if (SlotState(*range, slot) == SlotProbe && now - range->swap[slot] >= 1) cold.emplace_back(range->physical[slot], slot);
    std::sort(cold.begin(), cold.end());
    const std::lock_guard out_lock{swap_out_mutex};
    std::size_t freed = 0, made = 0;
    const std::uint64_t failed = swap_counters.failures.load(std::memory_order_relaxed);
    for (std::size_t i = 0; i < cold.size() && made < 256 && freed < want; ++i) {
        const std::size_t slot = cold[i].second;
        if (SwapOut(*range, slot)) {
            ++made;
            // What the slot gave back: all of it to the file, or 2 MiB less its packed pages. Read
            // after the fact, so a touch that brought it back already counts as nothing.
            if (SlotState(*range, slot) == SlotOut) {
                const std::int64_t physical = range->physical[slot];
                freed += InStore(physical) ? pack_shared_given // EDEN_UP_PACK_SHARED: 2 MiB, or 0 for a new slab
                         : physical < 0    ? SparseSlot
                                           : SparseSlot - std::min(SparseSlot, PackedPages(range->swap[slot]) * FinePage);
            }
            if (slot < SwapEverOutSlots)
                swap_ever_out[slot / 64].fetch_or(std::uint64_t{1} << (slot % 64), std::memory_order_relaxed);
        } else if (swap_counters.failures.load(std::memory_order_relaxed) != failed) {
            break; // no block for the packed bytes, or the file refused: nothing more now
        }
    }
    char line[128];
    const int length = std::snprintf(line, sizeof(line), "EDEN_SWAP reclaim for graphics: %zu slots, %zu MiB given back (asked %zu MiB)\n",
                                     made, freed >> 20, want >> 20);
    if (length > 0) Report(line);
    return freed;
#else
    (void)want;
    return 0;
#endif
}

// EDEN_UP_RADV_OOM_LEND: eden_guest_memory_reclaim for the driver's out-of-memory callback
// (headless/performance.cpp). The same slots in the same order, but the candidates are gathered under
// swap_out_mutex into a static array (the 8192 lowest in the pool; the old list grew on the heap, and a
// refusal there would end the session inside the driver, the function being noexcept), and a call in
// the same swap tick as a scan that found no cold slot scans nothing (see ReclaimCandidates for the
// one-slot exceptions). At the forest exit (05:37) the ten "0 slots" scans most likely found cold slots
// that would not pack (no block for the packed bytes), so the skip would not have applied there; it
// is for a pool empty with nothing armed for long. `verify` != 0 (EDEN_UP_RADV_OOM_LEND_VERIFY=1):
// that scan runs all the same, and a cold slot it finds is counted as missed. Bytes given back.
extern "C" std::size_t eden_guest_memory_reclaim_bounded(std::size_t want, int verify) noexcept {
#ifdef PS5_NATIVE
    static const bool enabled = std::getenv("EDEN_GUEST_SWAP_RECLAIM") != nullptr && std::getenv("EDEN_GUEST_SWAP_RECLAIM")[0] == '1';
    const int mode = swap_mode.load(std::memory_order_relaxed);
    SparseRange* const range = service_range.load(std::memory_order_acquire);
    if (pack_on_refusal.load(std::memory_order_relaxed) && want >= PackRefusalLarge && (mode == 2 || mode == 3) &&
        range != nullptr && range->state != nullptr)
        return PackOnRefusal(*range, want);
    if (!enabled || (mode != 2 && mode != 3) || range == nullptr || range->state == nullptr) return 0;
    const std::size_t slots = (range->end - range->begin.load(std::memory_order_relaxed)) / SparseSlot;
    const std::lock_guard out_lock{swap_out_mutex};
    const std::uint32_t now = swap_tick.load(std::memory_order_relaxed);
    const bool same_tick = reclaim_zero_tick == static_cast<std::uint64_t>(now) + 1;
    if (same_tick && verify == 0) {
        reclaim_skipped.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    reclaim_scans.fetch_add(1, std::memory_order_relaxed);
    std::size_t count = 0, found = 0;
    for (std::size_t slot = 0; slot < slots; ++slot) {
        if (SlotState(*range, slot) != SlotProbe || now - range->swap[slot] < 1) continue;
        ++found;
        const std::pair<std::int64_t, std::size_t> entry{range->physical[slot], slot};
        if (count < ReclaimCandidates) {
            reclaim_cold[count++] = entry;
            std::push_heap(reclaim_cold, reclaim_cold + count);
        } else if (entry < reclaim_cold[0]) {
            std::pop_heap(reclaim_cold, reclaim_cold + count);
            reclaim_cold[count - 1] = entry;
            std::push_heap(reclaim_cold, reclaim_cold + count);
        }
    }
    reclaim_zero_tick = found == 0 ? static_cast<std::uint64_t>(now) + 1 : 0;
    if (found > count) reclaim_capped.fetch_add(1, std::memory_order_relaxed);
    if (same_tick) {
        reclaim_verified.fetch_add(1, std::memory_order_relaxed);
        if (found != 0) reclaim_missed.fetch_add(1, std::memory_order_relaxed);
    }
    std::sort_heap(reclaim_cold, reclaim_cold + count);
    return ReclaimColdSlots(*range, want, reclaim_cold, count);
#else
    (void)want;
    (void)verify;
    return 0;
#endif
}

// EDEN_UP_RADV_OOM_LEND's counters for the line every 10 s (headless/performance.cpp), in this order:
// lend calls, bytes lent, lent blocks beside the next, lends refused while a guest write waited, lends
// that found nothing above the lowered floor, bounded scans, scans skipped (same tick, none cold), scans
// verified (would have been skipped), of those the ones that found cold slots, scans with more cold slots
// than candidates, reserve blocks held now. At most `count` of them are written; all 0 off the console.
extern "C" void eden_guest_reserve_lend_stats(unsigned long long* values, std::size_t count) noexcept {
#ifdef PS5_NATIVE
    const unsigned long long all[] = {
        oom_lend_calls.load(std::memory_order_relaxed),     oom_lend_bytes.load(std::memory_order_relaxed),
        oom_lend_adjacent.load(std::memory_order_relaxed),  oom_lend_refused_waiting.load(std::memory_order_relaxed),
        oom_lend_none_left.load(std::memory_order_relaxed), reclaim_scans.load(std::memory_order_relaxed),
        reclaim_skipped.load(std::memory_order_relaxed),    reclaim_verified.load(std::memory_order_relaxed),
        reclaim_missed.load(std::memory_order_relaxed),     reclaim_capped.load(std::memory_order_relaxed),
        reserve_count.load(std::memory_order_relaxed),
    };
#else
    const unsigned long long all[11] = {};
#endif
    for (std::size_t i = 0; i < count && i < sizeof(all) / sizeof(all[0]); ++i) values[i] = all[i];
}

// The report now, for a burst of writes that falls between two of the scanner's (headless/performance.cpp).
extern "C" void eden_guest_memory_report(void) noexcept {
    if (SparseRange* const range = service_range.load(std::memory_order_acquire)) ServiceReport(*range, "burst");
}

// ---- What the guest's kernel tells the service ----

// Development switches (dev-settings cram=off, cram_scan=off, cram_dedup=off), before the guest's
// memory is taken.
void SetGuestMemoryService(bool on, bool scan, bool dedup) noexcept {
    service_requested.store(on, std::memory_order_relaxed);
    scan_requested.store(scan, std::memory_order_relaxed);
    dedup_requested.store(dedup, std::memory_order_relaxed);
}

// Settings > Performance > Compact memory (off unless chosen): written slots with many pages of
// zeroes keep only the others. It gives back more memory, and a game that writes such a slot again
// waits for it to be made whole (a fault and a 2 MiB copy, about 0.3 ms).
void SetGuestMemoryFine(bool on) noexcept {
    fine_requested.store(on, std::memory_order_relaxed);
}

// The guest's kernel hands pages out: they are committed now only without the service.
void CommitGuestMemoryOnHandOut(std::uint64_t offset, std::size_t bytes) noexcept {
    SparseRange* const range = guest_range.load(std::memory_order_acquire);
    if (range != nullptr && range->promote) return;
    CommitGuestMemory(offset, bytes);
}

// The guest's kernel fills pages (`at` is the backing at `offset`). With the service, a fill with
// zeroes leaves Zero slots alone and gives whole Own or Shared slots back; anything else is written.
void ClearGuestMemory(std::uint8_t* at, std::uint64_t offset, std::size_t bytes, int fill) noexcept {
    SparseRange* const range = guest_range.load(std::memory_order_acquire);
    if (range == nullptr || !range->promote || fill != 0) {
        if (range != nullptr && !range->promote) CommitGuestMemory(offset, bytes);
        std::memset(at, fill, bytes); // (with the service, a write to a Zero slot promotes it)
        return;
    }
    const std::uint64_t end = offset + bytes;
    std::size_t returned = 0;
    for (std::uint64_t cursor = offset; cursor < end;) {
        const std::size_t slot = static_cast<std::size_t>(cursor / SparseSlot);
        const std::uint64_t slot_end = (cursor / SparseSlot + 1) * SparseSlot;
        const std::uint64_t piece_end = std::min(end, slot_end);
        const bool whole = cursor % SparseSlot == 0 && piece_end == slot_end;
        if (whole) {
            LockSlot(*range, slot);
            const auto state = SlotState(*range, slot);
            if (state != SlotZero) {
                ReturnSlot(*range, slot);
                counters.returns_clear.fetch_add(1, std::memory_order_relaxed);
                ++returned;
            }
            UnlockSlot(*range, slot, SlotZero);
        } else if (SlotState(*range, slot) != SlotZero) {
            std::memset(at + (cursor - offset), 0, static_cast<std::size_t>(piece_end - cursor));
            L961Add(l961_zero_filled, piece_end - cursor);
        }
        cursor = piece_end;
    }
    if (returned != 0) NoteClearSite(reinterpret_cast<std::uintptr_t>(__builtin_return_address(0)), returned);
}

// A page the guest's kernel fills one at a time (optimized allocation): the same rule.
void FillGuestMemory(std::uint8_t* at, int fill, std::size_t bytes) noexcept {
    SparseRange* const range = guest_range.load(std::memory_order_acquire);
    if (range != nullptr && range->promote && fill == 0) {
        const auto begin = range->begin.load(std::memory_order_relaxed);
        const auto address = reinterpret_cast<std::uintptr_t>(at);
        if (address >= begin && address < range->end && SlotState(*range, (address - begin) / SparseSlot) == SlotZero)
            return;
    }
    std::memset(at, fill, bytes);
}

// The guest's kernel freed pages: every slot they cover whole goes back. KPageHeap::Free calls this
// with the pool's lock held, also for what is left of a big block it split, so Zero slots are passed
// over without their lock. Until EDEN_CRAM_FREE_RETURN a freed slot stays Own until the kernel hands
// its pages out again (ClearGuestMemory, returns_clear).
void GuestMemoryFreed(std::uint64_t offset, std::size_t bytes) noexcept {
    SparseRange* const range = guest_range.load(std::memory_order_acquire);
    if (range == nullptr || !range->promote || bytes < SparseSlot ||
        !free_return_requested.load(std::memory_order_relaxed))
        return;
    const std::uint64_t first = (offset + SparseSlot - 1) / SparseSlot;
    const std::uint64_t last = (offset + bytes) / SparseSlot; // exclusive
    for (std::uint64_t slot = first; slot < last; ++slot) {
        if (SlotState(*range, static_cast<std::size_t>(slot)) == SlotZero) continue;
        LockSlot(*range, static_cast<std::size_t>(slot));
        if (SlotState(*range, static_cast<std::size_t>(slot)) != SlotZero) {
            ReturnSlot(*range, static_cast<std::size_t>(slot));
            counters.returns_free.fetch_add(1, std::memory_order_relaxed);
        }
        UnlockSlot(*range, static_cast<std::size_t>(slot), SlotZero);
    }
}

// For the host checks: one scan pass now, and a slot's state (0 Zero, 1 Own, 2 Shared, -1 none).
void GuestMemoryScanNow() noexcept {
    if (SparseRange* const range = service_range.load(std::memory_order_acquire)) ScanPass(*range);
}
int GuestMemorySlotState(std::uint64_t offset) noexcept {
    SparseRange* const range = guest_range.load(std::memory_order_acquire);
    if (range == nullptr || range->state == nullptr) return -1;
    return SlotState(*range, static_cast<std::size_t>(offset / SparseSlot));
}
