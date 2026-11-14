#pragma once
// EDEN_UP_GPU_PASS_TIME=1 (off unless set; read once in the Vulkan scheduler's constructor): where
// the host GPU time goes, by render target and by kind of work outside render passes. The worker
// half (timestamps, buckets, the 5 s "EDEN_GPU_PASS" lines) is in gpu_time_probe.inc; this header
// is the recording half: draw and clear counts and the scopes that tag work done outside a render
// pass (uploads, ASTC decodes, copies and blits, downloads, the game's compute). Off: every entry
// point here tests one bool and returns, and no command is recorded.
#include <atomic>
#include <cstdint>

namespace Eden::GpuPass {

// Buckets of work outside the scheduler's render passes, plus the two target buckets. The big
// targets (64 MiB and more) get their own buckets after these, named by size and format.
enum Kind : std::uint32_t {
    Other = 0,       ///< outside render passes, untagged (barriers, query work, presenter, ...)
    UploadBuffer,    ///< the submission's upload command buffer (buffer uploads)
    UploadImage,     ///< Image::UploadMemory: staging to image copies in the main buffer
    Astc,            ///< ASTC decode / ASTC to BCn transcode (and the 3D unswizzle pass)
    Copy,            ///< image copies, blits, conversions, reinterprets, MSAA copies
    Download,        ///< Image::DownloadMemory
    Compute,         ///< the game's compute dispatches
    TargetSmall,     ///< render passes whose largest attachment is under 64 MiB
    TargetOverflow,  ///< big targets past the bucket table (should stay 0)
    FixedKinds,
};

inline bool on = false; // set before the scheduler's worker starts, never changed after
// Recording thread (the GPU thread that records into the scheduler).
inline std::atomic<std::uint32_t> draws{0};
inline std::atomic<std::uint32_t> clears{0};
inline std::uint32_t current_kind = Other;
// RendererVulkan::Composite calls: shown frames, for the per-frame figures.
inline std::atomic<std::uint64_t> presents{0};

inline void CountDraw() noexcept {
    if (on) draws.fetch_add(1, std::memory_order_relaxed);
}
inline void CountClear() noexcept {
    if (on) clears.fetch_add(1, std::memory_order_relaxed);
}
inline void CountPresent() noexcept {
    if (on) presents.fetch_add(1, std::memory_order_relaxed);
}
// Draws are counted after they are recorded (RasterizerVulkan::PrepareDraw after draw_func, and
// after DrawTexture's blit), not at the call's entry: the entry comes before the call requests its
// render pass, so the next count taken (the old pass's end) gave the first draw of every pass to
// the pass before it. Clears likewise (ClearCount, at the end of RasterizerVulkan::Clear); a
// deferred clear becomes the load op of the pass that realizes it, so it goes to that pass
// (load_clears, taken when the scheduler begins its next pass).
inline std::atomic<std::uint32_t> load_clears{0};
struct ClearCount {
    explicit ClearCount(bool deferred_) noexcept : deferred{deferred_} {}
    ~ClearCount() {
        if (!on) return;
        (deferred ? load_clears : clears).fetch_add(1, std::memory_order_relaxed);
    }
    ClearCount(const ClearCount&) = delete;
    ClearCount& operator=(const ClearCount&) = delete;
    bool deferred;
};
inline std::uint32_t TakeLoadClears() noexcept {
    if (!on) return 0;
    return load_clears.exchange(0, std::memory_order_relaxed);
}

struct Counts {
    std::uint32_t draws{};
    std::uint32_t clears{};
};
inline Counts TakeCounts() noexcept {
    if (!on) return {};
    return {draws.exchange(0, std::memory_order_relaxed), clears.exchange(0, std::memory_order_relaxed)};
}

// Worker thread: closes the segment that ran since the last mark (a timestamp, when outside a
// render pass) and makes `kind` the kind of what follows. Set by gpu_time_probe.inc.
inline void (*mark)(void* cmdbuf, std::uint32_t kind, Counts counts, bool enter) = nullptr;

// Tags the scheduler work recorded in its lifetime with `kind`; nests (the outer kind comes back
// at the end). Two recorded marks per scope while on, nothing while off.
template <typename Scheduler>
class Scope {
public:
    Scope(Scheduler& scheduler_, std::uint32_t kind) {
        if (!on) return;
        scheduler = &scheduler_;
        previous = current_kind;
        current_kind = kind;
        Record(kind, true);
    }
    ~Scope() {
        if (scheduler == nullptr) return;
        current_kind = previous;
        Record(previous, false);
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    void Record(std::uint32_t kind, bool enter) {
        scheduler->Record([kind, enter, counts = TakeCounts()](auto cmdbuf) {
            if (mark != nullptr) mark(*cmdbuf, kind, counts, enter);
        });
    }
    Scheduler* scheduler = nullptr;
    std::uint32_t previous = Other;
};

} // namespace Eden::GpuPass
