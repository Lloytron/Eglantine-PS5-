#pragma once
// Waits for memory where a refusal would otherwise end the session.
//
// Port of waveydude/Eglantine-PS5- bef232bb72fba49af9416b4937a971b5d8fa6542 ("Memory waits"), which
// follows mihawk-99/PS5_Hookshot aaf80e3 and 9100911 (a native port of the same game). That port
// measured on 10 Nov 2026 that at an area change memory comes back within seconds while one area's
// pages and textures go and the next area's arrive: its game pages gave up after 3 s and survived at
// 10 s, and an operator new that threw std::bad_alloc on its submit thread was gone once it waited.
// Here three places end the session on the first refusal or after a fixed 2 s:
//   - operator new (headless/main.cpp's new handler): EDEN_UP_NEW_WAIT_MS, unset or 0: no wait (off);
//   - the JIT code cache's next 2 MiB (headless/jit-allocator.h): EDEN_UP_JIT_COMMIT_WAIT_MS, unset 2000
//     (his name EDEN_JIT_COMMIT_WAIT_MS is read too, when ours is unset);
//   - a guest write that needs a block (src/guest_memory_service.inc, WaitForBlock):
//     EDEN_UP_GUEST_WRITE_WAIT_MS, unset 2000 (his name EDEN_GUEST_WRITE_WAIT_MS read too).
// Unset, all three behave exactly as before (the two 2000 ms ones are the old 100 steps of 20 ms, and
// print the old lines). A wait holds the waiting thread, never memory. Rules and counters only here;
// tools/test-memory-wait.cpp tests the rules on the host.
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace Eden::MemoryWait {

// A wait in ms from text: unset or not starting with a digit gives `fallback`, a number is held
// within [low, high]. Trailing characters after the digits are ignored, as the other switches do.
inline unsigned ParseMs(const char* text, unsigned fallback, unsigned low, unsigned high) {
    if (text == nullptr || text[0] < '0' || text[0] > '9') return fallback;
    const unsigned long long value = std::strtoull(text, nullptr, 10);
    if (value < low) return low;
    if (value > high) return high;
    return static_cast<unsigned>(value);
}

inline unsigned ReadMs(const char* name, unsigned fallback, unsigned low, unsigned high) {
    return ParseMs(std::getenv(name), fallback, low, high);
}

// Our name first, then the name his env files use; neither set: `fallback`.
inline unsigned ReadMs2(const char* name, const char* his_name, unsigned fallback, unsigned low, unsigned high) {
    const char* text = std::getenv(name);
    if (text == nullptr || text[0] < '0' || text[0] > '9') text = std::getenv(his_name);
    return ParseMs(text, fallback, low, high);
}

// operator new calls its handler after each failed try and tries again when the handler returns; a
// try that succeeds calls nothing. So the handler cannot see a wait end. It tells one wait from the
// next by the request: a call for the same number of bytes within SameRequestGapMs of the last one
// is the same wait going on, and so is any call within EpisodeGapMs.
//
// His first version went by EpisodeGapMs alone, and hung the game (11 Nov 2026, 16 minutes in): with
// the pool empty one try of a 32 MiB request took about 2 s inside the heap (its own heap wait),
// every call came later than 250 ms after the last, each counted as a new wait, and the 5 s limit
// was never reached. The picture stood still for good.
//
// Note for this tree: the byte count comes from eden_heap_last_failure (headless/heap_arenas.inc),
// which is the process's last failure, not this thread's. Another thread's refusal landing between
// this thread's refusal and its handler can show other bytes; after a gap over EpisodeGapMs that
// starts the wait again. The 10 s line counts such restarts ("other size within 60 s").
//
// So (his rule): a same-size shortage within 60 s of a served wait is not waited for; it joins the
// old wait, whose limit is spent, and throws. The handler counts that case apart when the pause is
// longer than one try can take (try_bound_ms): "refused at once", not "gave up".
constexpr std::uint64_t EpisodeGapMs = 250;
// A try can take as long as the heap's own wait, which is at most 60 s (heap_pages.cpp).
constexpr std::uint64_t SameRequestGapMs = 60'000;
// After a wait runs out, this thread's next failures give up at once for this long: memory is not
// coming back, and a caller that copes with the refusal and asks again every frame would otherwise
// stand still for the whole limit each time.
constexpr std::uint64_t CooldownMs = 30'000;
// The heap's empty spaces go back at a wait's start and every TrimEveryMs after.
constexpr std::uint64_t TrimEveryMs = 500;

struct Episode {
    std::uint64_t start_ms = 0;
    std::uint64_t last_ms = 0;
    std::uint64_t last_trim_ms = 0;
    std::uint64_t bytes = 0;
    std::uint64_t cooldown_until_ms = 0;
    bool active = false;
};

enum class Step {
    Start,  // a new wait: log it, trim, wait a step
    Wait,   // the same wait goes on: wait a step
    GiveUp, // no wait (limit 0), the limit is spent, or a wait just ran out here: throw as before
};

// `bytes`: what the failed try asked for.
inline Step Next(Episode& episode, std::uint64_t now_ms, unsigned limit_ms, std::uint64_t bytes = 0) {
    if (limit_ms == 0) {
        episode.active = false;
        return Step::GiveUp;
    }
    if (now_ms < episode.cooldown_until_ms) {
        return Step::GiveUp;
    }
    const std::uint64_t gap = now_ms - episode.last_ms;
    const bool same_wait = episode.active &&
                           (gap <= EpisodeGapMs || (bytes == episode.bytes && gap <= SameRequestGapMs));
    Step step = Step::Wait;
    if (!same_wait) {
        episode.start_ms = now_ms;
        episode.last_trim_ms = now_ms;
        episode.bytes = bytes;
        episode.active = true;
        step = Step::Start;
    }
    episode.last_ms = now_ms;
    if (now_ms - episode.start_ms >= limit_ms) {
        episode.active = false;
        episode.cooldown_until_ms = now_ms + CooldownMs;
        return Step::GiveUp;
    }
    return step;
}

// Whether this step trims the heap: at the start and every TrimEveryMs of the same wait.
inline bool TrimNow(Episode& episode, Step step, std::uint64_t now_ms) {
    if (step == Step::Start) return true;
    if (step == Step::Wait && now_ms - episode.last_trim_ms >= TrimEveryMs) {
        episode.last_trim_ms = now_ms;
        return true;
    }
    return false;
}

// A request no wait can serve gives up at once: his first hybrid build asked operator new for
// 2 TiB out of a damaged table, and ten seconds of waiting would only have delayed that report.
// The largest honest single request seen is a 341 MiB upload buffer.
constexpr std::uint64_t LargestWaitedRequest = std::uint64_t{1} << 30;
inline bool WorthWaitingFor(std::uint64_t bytes) {
    return bytes <= LargestWaitedRequest;
}

// Number of 20 ms steps in a wait of `ms` (at least one).
inline unsigned Steps(unsigned ms) {
    return ms < 20 ? 1u : ms / 20u;
}

// EDEN_UP_NEW_WAIT_MS, set by main once the game's env file is in (0 until then: no wait).
inline std::atomic<unsigned> new_wait_ms{0};
// T130 (ours): the longest one try can take in this run (the heap's own longest wait plus
// EpisodeGapMs), set by main with new_wait_ms. A same-size refusal that the 60 s rule joins to a wait
// after a longer pause than this came after a served wait: the handler counts it as "refused at once"
// (Counters::at_once), not as a wait that ran out.
inline std::atomic<unsigned> try_bound_ms{static_cast<unsigned>(EpisodeGapMs)};

// Our addition: what the new handler did, for the EDEN_UP_NEW_WAIT line every 10 s
// (headless/performance.cpp). Written only while EDEN_UP_NEW_WAIT_MS is on, and only on the failure
// path (the handler runs after a refused operator new, never on a served one).
struct Counters {
    std::atomic<unsigned long long> begun{0};        // Step::Start (one EDEN_NEW_WAIT start line each)
    std::atomic<unsigned long long> gave_up{0};      // a wait ran out (one "gave up" line each)
    std::atomic<unsigned long long> at_once{0};      // T130: refused at once, the same size again within
                                                     // 60 s, after a pause longer than one try (try_bound_ms):
                                                     // a served wait's limit spent by the clock (one
                                                     // "refused at once" line each)
    std::atomic<unsigned long long> steps{0};        // 20 ms sleeps in the handler
    std::atomic<unsigned long long> trims{0};        // heap trims asked by the handler
    std::atomic<unsigned long long> other_size{0};   // a Start while this thread's last wait was open and
                                                     // its last call came within SameRequestGapMs
    std::atomic<unsigned long long> cooldown{0};     // refusals thrown at once in a cooldown
    std::atomic<unsigned long long> too_big{0};      // refusals over LargestWaitedRequest, thrown at once
    std::atomic<unsigned> waiting{0};                // threads inside a 20 ms sleep now
    std::atomic<unsigned> longest_ms{0};             // T130: most ms one wait really slept (20 per step)
};
inline Counters counters;

inline void NoteLongest(std::uint64_t waited_ms) {
    const unsigned value = waited_ms > 0xffffffffull ? 0xffffffffu : static_cast<unsigned>(waited_ms);
    unsigned seen = counters.longest_ms.load(std::memory_order_relaxed);
    while (value > seen &&
           !counters.longest_ms.compare_exchange_weak(seen, value, std::memory_order_relaxed)) {
    }
}

// T132 (ours) EDEN_UP_JIT_COMMIT_RECLAIM=<mode> (headless/jit-allocator.h, commit): what the JIT code
// cache does when the pool refuses its next part. Owner's T130 session, 11 Nov 23:52: one 2 MiB piece of
// CPUCore_1's next 16 MiB step was refused (largest free 1 MiB), the 2 s wait only slept, and the app
// ended with std::abort() while the guest swap was still packing cold memory and the graphics side was
// taking every block that came free.
//   0 or unset: as before, word for word (sleep 20 ms, try again, heap trim at tries 1, 26, 51, 76).
//   1: as 0, and counted: the 10 s line, one line per wait saying how it ended and what mode 2/3 would
//      have done (no reclaim, no clear).
//   2: as 0, and every 100 ms of the wait (tries 1, 6, 11, ...) cold guest memory is packed now, on this
//      thread (eden_guest_memory_reclaim_bounded: no list on the heap; the driver's out-of-memory
//      callback uses it only with EDEN_UP_RADV_OOM_LEND on, which has no console run on record: the
//      owner's run used the unbounded eden_guest_memory_reclaim; its packing part, SwapOut/PackOut, is the
//      well-run one), and the piece is asked for again AT ONCE when that or
//      the heap trim gave something back, before the reserve keeper (100 ms) or a graphics thread takes it.
//      The same limit, then the same abort.
//   3: as 2, and when the limit is spent inside a block compile (jit_impl.inc GetBlock, the only caller
//      that can take a refusal) every core's compiled code is cleared instead of ending the session: the
//      code cache starts again at its beginning, in memory it already has. At most ClearsPerWindow clears
//      in ClearWindowMs; past that, the old abort. Elsewhere (precompiler, batch, constants): the old abort.
// Read once in main.cpp after env.txt and the game's env file, 0 until then.
inline std::atomic<unsigned> jit_reclaim_mode{0};
constexpr unsigned JitReclaimEveryTries = 5;                     // 100 ms
constexpr std::size_t JitReclaimAsk = std::size_t{16} << 20;     // one step; under PackRefusalLarge (64 MiB)
constexpr unsigned ClearsPerWindow = 4;
constexpr std::uint64_t ClearWindowMs = 120'000;
struct JitReclaimCounters {
    std::atomic<unsigned long long> waits{0};          // refused commits that began a wait
    std::atomic<unsigned long long> served_at_once{0}; // served by the ask right after a reclaim or trim
    std::atomic<unsigned long long> served_later{0};   // served by a 20 ms retry
    std::atomic<unsigned long long> cleared{0};        // mode 3: the code was cleared instead of an abort
    std::atomic<unsigned long long> after_clear{0};    // mode 3: another core's clear ended this wait
    std::atomic<unsigned long long> ended{0};          // the limit spent and the session ended (abort)
    std::atomic<unsigned long long> clear_capped{0};   // of those: a clear would have come, the cap said no
    std::atomic<unsigned long long> would_clear{0};    // mode 1: a clear would have come instead of the abort
    std::atomic<unsigned long long> reclaims{0};       // eden_guest_memory_reclaim_bounded calls
    std::atomic<unsigned long long> reclaim_scans{0};  // of those: real scans (each prints "(asked 16 MiB)")
    std::atomic<unsigned long long> reclaim_skipped{0}; // of those: skipped (same swap tick, nothing cold)
    std::atomic<unsigned long long> reclaim_busy{0};   // skipped: another core was reclaiming
    std::atomic<unsigned long long> reclaim_bytes{0};
    std::atomic<unsigned long long> trim_bytes{0};     // what the wait's heap trims gave back (all modes but 0)
    std::atomic<unsigned> longest_ms{0};
};
inline JitReclaimCounters jit_reclaim;
inline std::atomic<bool> jit_reclaiming{false};        // one reclaimer at a time (six cores may wait)
inline std::atomic<unsigned long long> jit_clears{0};  // clears asked by a refused commit (mode 3)
// The JIT thread's side of mode 3, set only by eden_jit_commit_scope (headless/performance.cpp) around
// GetBlock's EnsureMemoryCommitted: 1 while a refused commit may end in a clear. Result: 0 served,
// 1 refused (clear now), 2 another core cleared meanwhile (compile again, no clear).
// Kept per thread in one place, src/memory_pages.cpp (a normal object of common), and reached only
// through these four calls. Not inline thread_locals here: jit-allocator.h is built inside dynarmic,
// the one part built with -flto=thin, and with -femulated-tls the link sees the normal objects' uses
// as __emutls_v.<name>, not <name>, so dynarmic got its own copy that nothing wrote (T132: folded to
// 0, mode 3 never cleared). A call into another object is the pattern already working across that
// boundary (Common::CommitCodeRange). Called only with EDEN_UP_JIT_COMMIT_RECLAIM on or the test
// hook's <where> 4 armed; never on the off path.
int JitClearScope() noexcept;
void SetJitClearScope(int value) noexcept;
int JitCommitResult() noexcept;
void SetJitCommitResult(int value) noexcept;

inline void NoteJitLongest(std::uint64_t waited_ms) {
    const unsigned value = waited_ms > 0xffffffffull ? 0xffffffffu : static_cast<unsigned>(waited_ms);
    unsigned seen = jit_reclaim.longest_ms.load(std::memory_order_relaxed);
    while (value > seen && !jit_reclaim.longest_ms.compare_exchange_weak(seen, value, std::memory_order_relaxed)) {
    }
}

// Mode 3's cap: true when one more clear fits ClearsPerWindow in the last ClearWindowMs (and records it).
// Only the thread whose wait ran out calls it; a mutex-free ring of the last clears' times.
inline bool JitClearAllowed(std::uint64_t now_ms) {
    static std::atomic<std::uint64_t> times[ClearsPerWindow] = {};
    static std::atomic<unsigned> next{0};
    const unsigned slot = next.load(std::memory_order_relaxed) % ClearsPerWindow;
    const std::uint64_t oldest = times[slot].load(std::memory_order_relaxed);
    if (oldest != 0 && now_ms - oldest < ClearWindowMs) return false;
    times[slot].store(now_ms == 0 ? 1 : now_ms, std::memory_order_relaxed);
    next.fetch_add(1, std::memory_order_relaxed);
    return true;
}

} // namespace Eden::MemoryWait
