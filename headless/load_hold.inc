// SPDX-License-Identifier: GPL-3.0-or-later
// Included once by performance.cpp, at its end (global scope).
//
// EDEN_UP_LOAD_HOLD=<ms> (T178c; off unless set to a number from 500 to 15000): the pre-load hold.
//
// At the load from the field into the area past it, the 12,352 MiB pool runs dry 6-9 s after the
// game's first Native zstd call over the host size limit (svc 0xf2, capacity 0x2029080: the new
// area's archive; once per session in every T175/T177 start on that route). Waiting in the field
// frees almost nothing (pool free total and largest piece flat from arrival to the load; the guest
// swap is the only holder that moves with time). So the hold runs the freeing work at the start of
// the load, in its black screen, with the game's loader held:
//
//   core (patch 0930, svc_nx2_zstd.cpp): the guest thread of that call sleeps in 50 ms guest sleeps
//     (as svcSleepThread: its core runs the game's other threads) before the SVC returns, while
//     eden_load_hold_poll() says so; at most 400 sleeps whatever the front says. Nothing else of
//     the game is held: not the present, not audio, not the GPU threads.
//   front (this file, a thread of its own, every 250 ms):
//     - asks the frame end (patch 0930, RasterizerVulkan::TickFrame) to delete the old area's
//       images unused for 40 frames with nothing to copy back (128 a frame) and to drain as after a
//       refusal (EDEN_UP_PRESSURE_DRAIN's drain: deleted buffers and images back to the driver);
//       needs frames to end (the black screen presents) and EDEN_UP_PRESSURE_DRAIN=1 or 2 for the
//       drain part;
//     - packs cold guest slots out: eden_guest_memory_reclaim_bounded(48 MiB) a step (the swap's
//       reclaim for graphics: armed slots untouched for a swap tick, lowest in the pool first; needs
//       EDEN_GUEST_SWAP=2 or 3 and EDEN_GUEST_SWAP_RECLAIM=1; under 64 MiB a call, so
//       EDEN_UP_PACK_ON_REFUSAL's written-slot sweep is never asked), at most 1536 MiB a hold;
//     - trims the heap (eden_heap_trim_unheld: a running EDEN_UP_HEAP_HOLD does not stop it) at
//       once and every second; at the release, eden_heap_hold_refill as the memory report does.
//   Release: the ledger's free total and the pool's largest piece both at the target
//   (EDEN_UP_LOAD_HOLD_FREE_MIB=<free>[/<largest>], unset 2200/512), or no 16 MiB gained in the
//   last 1.5 s after 3 s, or the time (EDEN_UP_LOAD_HOLD).
//   Not held: a call in the first 30 s of the process, within 60 s of the last hold's end, while a
//   hold runs, or with the target already met (one line each, the first 8).
//
// Lines: "EDEN_UP_LOAD_HOLD begun ..." when it starts, "EDEN_UP_LOAD_HOLD: ..." at the release with
// the waited ms, ledger free and largest piece before and after, and the MiB of each step.
// Unset or 0: eden_load_hold_begin returns 0 at its first test; the frame end reads one relaxed
// atomic (never set). Not a TranslationSwitches name. KEEPS THE STORE.
#ifdef PS5_NATIVE
#include <chrono>
#include <cstdio>

extern "C" void eden_load_hold_gpu_request(unsigned bits) __attribute__((weak));
extern "C" void eden_load_hold_gpu_stats(unsigned long long* values, unsigned count) __attribute__((weak));
extern "C" std::size_t eden_heap_trim_unheld(void) __attribute__((weak));
extern "C" void eden_heap_hold_refill(unsigned long long largest_free) __attribute__((weak));
extern "C" std::size_t eden_guest_memory_reclaim_bounded(std::size_t want, int verify) __attribute__((weak));
extern "C" void eden_guest_memory_reserve(std::size_t* held, std::size_t* target) __attribute__((weak));
extern "C" void eden_direct_ledger(unsigned long long* bytes, unsigned long long* blocks,
                                    unsigned long long* hits, unsigned long long* fallbacks) __attribute__((weak));
extern "C" void eden_vma_usage(unsigned long long* blocks, unsigned long long* used) __attribute__((weak));
extern "C" std::int64_t sceKernelGetDirectMemorySize();
extern "C" std::int32_t sceKernelAvailableDirectMemorySize(std::int64_t, std::int64_t, std::size_t, std::int64_t*,
                                                           std::size_t*);

namespace Eden::LoadHold {
namespace {
constexpr unsigned GpuEvict = 1; // patch 0930's Vulkan::LoadHold::Evict
constexpr unsigned GpuDrain = 2; // ... ::Drain
constexpr long long StepMs = 250;
constexpr long long HeapTrimEveryMs = 1000;
constexpr std::size_t SwapStepBytes = std::size_t{48} << 20;  // under PackRefusalLarge (64 MiB)
constexpr std::size_t SwapMostBytes = std::size_t{1536} << 20;
constexpr unsigned long long GainBytes = 16ull << 20;
constexpr long long GainWindowMs = 1500;
constexpr long long GainAfterMs = 3000;
constexpr long long FirstAfterMs = 30'000;   // since the process started
constexpr long long GapMs = 60'000;          // since the last hold ended
constexpr unsigned long long MiB = 1ull << 20;

long long SteadyMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
const long long process_start_ms = SteadyMs();

struct Config {
    long long max_ms{};
    unsigned long long free_target{};
    unsigned long long largest_target{};
};
const Config& GetConfig() {
    static const Config config = [] {
        Config c{};
        if (const char* value = std::getenv("EDEN_UP_LOAD_HOLD"); value != nullptr && value[0] >= '1' && value[0] <= '9')
            c.max_ms = std::clamp<long long>(std::strtoll(value, nullptr, 10), 500, 15000);
        unsigned long long free_mib = 2200, largest_mib = 512;
        if (const char* value = std::getenv("EDEN_UP_LOAD_HOLD_FREE_MIB"); value != nullptr && value[0] >= '0' && value[0] <= '9') {
            char* end = nullptr;
            free_mib = std::strtoull(value, &end, 10);
            if (end != nullptr && *end == '/') largest_mib = std::strtoull(end + 1, nullptr, 10);
        }
        c.free_target = std::min<unsigned long long>(free_mib, 12352) * MiB;
        c.largest_target = std::min<unsigned long long>(largest_mib, 12352) * MiB;
        if (c.max_ms != 0)
            std::fprintf(stderr, "EDEN_UP_LOAD_HOLD on: at most %lld ms, released at %llu MiB free and a %llu MiB piece\n",
                         c.max_ms, c.free_target / MiB, c.largest_target / MiB);
        return c;
    }();
    return config;
}

struct Pool {
    unsigned long long free{};    // pool size less the ledger's out-of-pool bytes; 0: not known
    unsigned long long largest{}; // the pool's largest free piece; 0: not known
};
Pool ReadPool() {
    Pool pool;
    const std::int64_t total = sceKernelGetDirectMemorySize();
    if (total <= 0) return pool;
    std::int64_t start = 0;
    std::size_t largest = 0;
    if (sceKernelAvailableDirectMemorySize(0, total, 0x4000, &start, &largest) == 0) pool.largest = largest;
    if (eden_direct_ledger) {
        unsigned long long bytes[4]{}, blocks[4]{}, hits = 0, fallbacks = 0;
        eden_direct_ledger(bytes, blocks, &hits, &fallbacks);
        const unsigned long long out = bytes[0] + bytes[1] + bytes[2] + bytes[3];
        pool.free = out < static_cast<unsigned long long>(total) ? static_cast<unsigned long long>(total) - out : 0;
    }
    return pool;
}
bool TargetMet(const Pool& pool, const Config& c) {
    return pool.free >= c.free_target && pool.largest >= c.largest_target;
}

std::mutex mutex;
std::condition_variable wake;
bool started = false;
bool pending = false;             // a hold asked of the worker, not yet taken
std::atomic<int> active{0};       // the core's sleeps go on while 1
long long last_end_ms = 0;        // under mutex
unsigned holds = 0;               // under mutex
unsigned skipped_lines = 0;       // under mutex
unsigned pending_svc = 0;
unsigned long long pending_capacity = 0;
Pool pending_before;

void RunHold(unsigned number, unsigned svc, unsigned long long capacity, Pool before) {
    const Config& c = GetConfig();
    const long long begin_ms = SteadyMs();
    unsigned long long gpu0[5]{}, gpu1[5]{};
    if (eden_load_hold_gpu_stats) eden_load_hold_gpu_stats(gpu0, 5);
    if (eden_load_hold_gpu_request) eden_load_hold_gpu_request(GpuEvict | GpuDrain);
    std::size_t reserve0 = 0, reserve_target = 0, reserve1 = 0;
    if (eden_guest_memory_reserve) eden_guest_memory_reserve(&reserve0, &reserve_target);
    unsigned long long vma_blocks0 = 0, vma_used0 = 0, vma_blocks1 = 0, vma_used1 = 0;
    if (eden_vma_usage) eden_vma_usage(&vma_blocks0, &vma_used0);
    std::fprintf(stderr,
                 "EDEN_UP_LOAD_HOLD begun #%u (svc %#x, capacity %llu MiB): ledger free %llu MiB, largest piece %llu "
                 "MiB, guest reserve %llu MiB; target %llu/%llu MiB, at most %lld ms\n",
                 number, svc, capacity / MiB, before.free / MiB, before.largest / MiB, static_cast<unsigned long long>(reserve0) / MiB,
                 c.free_target / MiB, c.largest_target / MiB, c.max_ms);
    std::size_t swap_bytes = 0, heap_bytes = 0;
    unsigned swap_calls = 0, swap_empty = 0, heap_calls = 0;
    long long last_trim_ms = -HeapTrimEveryMs;
    // The free total at each step of the last GainWindowMs, to see whether anything still comes.
    constexpr std::size_t Window = static_cast<std::size_t>(GainWindowMs / StepMs) + 1;
    unsigned long long recent[Window]{};
    std::size_t steps = 0;
    const char* why = "time";
    Pool now = before;
    for (;;) {
        const long long elapsed = SteadyMs() - begin_ms;
        if (elapsed - last_trim_ms >= HeapTrimEveryMs && eden_heap_trim_unheld) {
            last_trim_ms = elapsed;
            heap_bytes += eden_heap_trim_unheld();
            ++heap_calls;
        }
        if (swap_bytes < SwapMostBytes && eden_guest_memory_reclaim_bounded) {
            const std::size_t given = eden_guest_memory_reclaim_bounded(SwapStepBytes, 0);
            swap_bytes += given;
            ++swap_calls;
            swap_empty += given == 0 ? 1 : 0;
        }
        now = ReadPool();
        recent[steps % Window] = now.free;
        ++steps;
        if (TargetMet(now, c)) {
            why = "target";
            break;
        }
        if (elapsed >= c.max_ms) {
            why = "time";
            break;
        }
        if (elapsed >= GainAfterMs && steps >= Window) {
            const unsigned long long oldest = recent[steps % Window];  // Window - 1 steps ago
            if (now.free < oldest + GainBytes) {
                why = "no gain";
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(StepMs));
    }
    if (eden_load_hold_gpu_request) eden_load_hold_gpu_request(0);
    const long long waited = SteadyMs() - begin_ms;
    {
        // The 60 s gap starts before `active` drops: set after it (as the worker did), an
        // over-limit call in between found no hold running and no gap, and began a second hold
        // right after this one (checker).
        std::scoped_lock lock{mutex};
        last_end_ms = SteadyMs();
    }
    active.store(0, std::memory_order_release); // the loader goes on at its next 50 ms wake
    if (eden_load_hold_gpu_stats) eden_load_hold_gpu_stats(gpu1, 5);
    if (eden_guest_memory_reserve) eden_guest_memory_reserve(&reserve1, &reserve_target);
    if (eden_vma_usage) eden_vma_usage(&vma_blocks1, &vma_used1);
    if (eden_heap_hold_refill && now.largest != 0) eden_heap_hold_refill(now.largest);
    std::fprintf(stderr,
                 "EDEN_UP_LOAD_HOLD: #%u svc %#x released (%s) after %lld ms of %lld; ledger free %llu -> %llu MiB, "
                 "largest piece %llu -> %llu MiB (target %llu/%llu); freed: swap %zu MiB (%u calls, %u gave nothing), "
                 "heap trim %zu MiB (%u), images %llu MiB (%llu images in %llu frames), drain %llu MiB (%llu drains), "
                 "vma blocks %llu -> %llu MiB (in use %llu -> %llu); guest reserve %llu -> %llu MiB\n",
                 number, svc, why, waited, c.max_ms, before.free / MiB, now.free / MiB, before.largest / MiB,
                 now.largest / MiB, c.free_target / MiB, c.largest_target / MiB, swap_bytes >> 20, swap_calls, swap_empty,
                 heap_bytes >> 20, heap_calls, (gpu1[2] - gpu0[2]) / MiB, gpu1[1] - gpu0[1], gpu1[0] - gpu0[0],
                 (gpu1[3] - gpu0[3]) / MiB, gpu1[4] - gpu0[4], vma_blocks0 / MiB, vma_blocks1 / MiB, vma_used0 / MiB,
                 vma_used1 / MiB, static_cast<unsigned long long>(reserve0) / MiB,
                 static_cast<unsigned long long>(reserve1) / MiB);
}

void Worker() {
    Eden::Performance::RegisterWorker("LoadHold");
    for (;;) {
        unsigned number, svc;
        unsigned long long capacity;
        Pool before;
        {
            std::unique_lock lock{mutex};
            wake.wait(lock, [] { return pending; });
            pending = false;
            number = holds;
            svc = pending_svc;
            capacity = pending_capacity;
            before = pending_before;
        }
        RunHold(number, svc, capacity, before); // sets last_end_ms before it releases
    }
}

void SkipLine(const char* why, const Pool& pool) {
    if (skipped_lines >= 8) return;
    ++skipped_lines;
    std::fprintf(stderr, "EDEN_UP_LOAD_HOLD not held (%s): ledger free %llu MiB, largest piece %llu MiB\n", why,
                 pool.free / MiB, pool.largest / MiB);
}
} // namespace
} // namespace Eden::LoadHold

// patch 0930 (svc_nx2_zstd.cpp), on the guest thread that made the over-limit call, inside the SVC:
// 1 when a hold began (the core then sleeps while eden_load_hold_poll() is 1). No wait here.
extern "C" int eden_load_hold_begin(unsigned svc, unsigned long long capacity) {
    namespace LH = Eden::LoadHold;
    const LH::Config& c = LH::GetConfig();
    if (c.max_ms == 0) return 0;
    try {
        const long long now = LH::SteadyMs();
        const LH::Pool pool = LH::ReadPool();
        std::scoped_lock lock{LH::mutex};
        if (LH::active.load(std::memory_order_acquire) != 0 || LH::pending) {
            LH::SkipLine("a hold runs", pool);
            return 0;
        }
        if (now - LH::process_start_ms < LH::FirstAfterMs) {
            LH::SkipLine("under 30 s since the start", pool);
            return 0;
        }
        if (LH::last_end_ms != 0 && now - LH::last_end_ms < LH::GapMs) {
            LH::SkipLine("under 60 s since the last hold", pool);
            return 0;
        }
        if (pool.free == 0 || pool.largest == 0) {
            LH::SkipLine("pool not known", pool);
            return 0;
        }
        if (LH::TargetMet(pool, c)) {
            LH::SkipLine("target already met", pool);
            LH::last_end_ms = now; // as a hold: the next over-limit call of this load is not held either
            return 0;
        }
        if (!LH::started) {
            std::thread(LH::Worker).detach(); // throws when there is no thread or memory: no hold
            LH::started = true;
        }
        ++LH::holds;
        LH::pending = true;
        LH::pending_svc = svc;
        LH::pending_capacity = capacity;
        LH::pending_before = pool;
        LH::active.store(1, std::memory_order_release);
    } catch (...) {
        return 0;
    }
    LH::wake.notify_one();
    return 1;
}

// patch 0930, between the held thread's 50 ms guest sleeps: 1 while the hold goes on.
extern "C" int eden_load_hold_poll(void) {
    return Eden::LoadHold::active.load(std::memory_order_acquire);
}
#endif
