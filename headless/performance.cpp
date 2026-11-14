// SPDX-License-Identifier: GPL-3.0-or-later
#include "performance.h"
#include "memory_wait.h"
#include "crash_report.h"
#include "stall_watchdog.h"
#include "../src/fastmem.h"
#include "common/cpu_features.h"
#include "common/live_switch.h"
#include "common/pe_switch.h"
#include "common/sparse_large_vector.h"
#include <string_view>
#include <algorithm>
#include <string>
#include <set>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cpuid.h>
#include <cstring>
#include <map>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <semaphore>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>
#include <latch>
#include <time.h>
#ifdef EDEN_DEV_PROFILE
#include <signal.h>
#endif
#ifdef PS5_NATIVE
#include <sys/param.h>
#include <sys/cpuset.h>
#include <pthread_np.h>  // pthread_getthreadid_np (0840, T166 review)
#endif

// dynarmic A64 dispatch-path counters (headless/dynarmic/jit_group_support.inc).
extern "C" void eden_jit_path_counters(unsigned core, unsigned long long* out) __attribute__((weak));
namespace Common {
void GuestMemoryUsage(std::size_t* span, std::size_t* held) noexcept;
// Address space Eden's sparse tables span and the memory they hold (src/memory_pages.cpp).
void SparseUsage(std::size_t* reserved, std::size_t* committed) noexcept;
// The memory behind the JIT's code caches (src/memory_pages.cpp).
void CodeRangeUsage(std::size_t* reserved, std::size_t* committed) noexcept;
} // namespace Common

#ifdef PS5_NATIVE
#include "mkw_native_sample.inc"
#endif

// 0844 (performance.h): null until a driver that offers the count sets it (lead-mesh-41).
extern "C" unsigned long long (*eden_radv_disk_cache_queued)(void) = nullptr;
// 0844 (T172, performance.h): null until the driver sets its drain (lead-mesh-41).
extern "C" bool (*eden_radv_disk_cache_drain)(unsigned timeout_ms) = nullptr;
// T175d (performance.h): null until the driver makes a pipeline cache (lead-mesh-44).
extern "C" unsigned long long (*eden_radv_pipeline_cache_changes)(void) = nullptr;

namespace Eden::Performance {
namespace {
constexpr std::array names{"CPUCore_0", "CPUCore_1", "CPUCore_2", "CPUCore_3",
                           "GPU", "HostTiming", "VSyncThread"};
struct Worker {
    pthread_t thread{};
    clockid_t clock{};
    int clock_error = ENOENT, affinity_error = ENOENT, allowed = 0;
    unsigned long long mask = 0;
    long long wall_ns = 0, mono_ns = 0;
    bool registered = false;
};
std::mutex workers_mutex;
std::array<Worker, names.size()> workers;
#ifdef EDEN_DEV_PROFILE
static_assert(std::atomic<uintptr_t>::is_always_lock_free);
std::array<uintptr_t, 8192> sampled_pcs{};
std::atomic<unsigned> pc_count{};
// Guest core 0 host PCs (JIT code, HLE, memory callbacks), sampled with the GPU thread.
std::array<uintptr_t, 65536> sampled_core_pcs{};
std::atomic<unsigned> core_pc_count{};
unsigned core_pc_reported{};
pthread_t core_sample_thread{};
std::atomic<bool> core_sample_ready{};
unsigned pc_reported{};
bool pc_sampling{};
#if defined(EDEN_DEV_WAIT_CALLERS) && defined(PS5_NATIVE)
// Same firmware6.02 register offsets qualified by the C46 caller sampler.
constexpr size_t runtime_rsp_offset = 248, runtime_rbp_offset = 136;
using CallerChain = std::array<uintptr_t, 8>;
std::array<CallerChain, 8192> sampled_callers{};
// Core0 keeps a chain only for PCs outside our own code: the eboot and the JIT caches name
// themselves by address, and the rest measured at a tenth of its samples, so a bounded array
// covers the run without a 4 MB one.
std::array<CallerChain, 4096> sampled_core_callers{};
std::array<uintptr_t, 4096> core_caller_pcs{};
std::atomic<unsigned> core_caller_count{};
unsigned core_caller_reported{};

CallerChain CaptureCallerChain(uintptr_t rsp, uintptr_t frame, uintptr_t handler_stack) {
    CallerChain callers{};
    // Read only aligned frames in the interrupted stack-top page. No allocation,
    // symbolization or logging in the signal handler. Reject unknown/alternate stacks.
    if (rsp <= 0x10000 || rsp < handler_stack || rsp - handler_stack >= 65536 ||
        (rsp & 4095) > 4096 - 8 * sizeof(uintptr_t)) return callers;
    for (unsigned i = 0; i < callers.size(); ++i) {
        if (frame < rsp || (frame >> 12) != (rsp >> 12) ||
            (frame & (alignof(uintptr_t) - 1)) ||
            (frame & 4095) > 4096 - 2 * sizeof(uintptr_t)) break;
        const auto* chain = reinterpret_cast<const uintptr_t*>(frame);
        callers[i] = chain[1];
        if (chain[0] <= frame || chain[0] - frame > 4096) break;
        frame = chain[0];
    }
    return callers;
}
#endif
// Core 0's A32 blocks in emission order (P0): dynarmic's code cache grows linearly until a full
// clear, so entries stay sorted and a sampled host PC finds its guest block by binary search.
struct JitBlock {
    uintptr_t entry;
    unsigned long long size, location;
};
constexpr std::size_t jit_block_capacity = std::size_t{1} << 20;
JitBlock* core0_blocks = nullptr;
std::atomic<std::size_t> core0_block_count{0};

void PcSignal(int, siginfo_t*, void* context) {
    if (core_sample_ready.load(std::memory_order_acquire) &&
        pthread_equal(pthread_self(), core_sample_thread)) {
        const unsigned core_index = core_pc_count.load(std::memory_order_relaxed);
        if (core_index < sampled_core_pcs.size()) {
            sampled_core_pcs[core_index] = static_cast<const uintptr_t*>(context)[224 / sizeof(uintptr_t)];
            core_pc_count.store(core_index + 1, std::memory_order_release);
#if defined(EDEN_DEV_WAIT_CALLERS) && defined(PS5_NATIVE)
            const uintptr_t pc = sampled_core_pcs[core_index];
            const unsigned caller_index = core_caller_count.load(std::memory_order_relaxed);
            // Library PCs only: eboot and the JIT caches identify themselves by address.
            if (caller_index < core_caller_pcs.size() &&
                (pc < 0x400000 || pc > 0x28c2b30) &&
                (pc < 0x1200000000 || pc >= 0x1300000000)) {
                const auto* words = static_cast<const uintptr_t*>(context);
                char handler_stack_marker;
                sampled_core_callers[caller_index] = CaptureCallerChain(
                    words[runtime_rsp_offset / sizeof(uintptr_t)],
                    words[runtime_rbp_offset / sizeof(uintptr_t)],
                    reinterpret_cast<uintptr_t>(&handler_stack_marker));
                core_caller_pcs[caller_index] = pc;
                core_caller_count.store(caller_index + 1, std::memory_order_release);
            }
#endif
        }
        return;
    }
    const unsigned index = pc_count.load(std::memory_order_relaxed);
    if (index >= sampled_pcs.size()) return;
    // Native firmware 6.02 ABI, qualified by the standalone PC-sampling oracle.
    // SDK FreeBSD ucontext.mc_rip points at the wrong field on this firmware.
    sampled_pcs[index] = static_cast<const uintptr_t*>(context)[224 / sizeof(uintptr_t)];
#if defined(EDEN_DEV_WAIT_CALLERS) && defined(PS5_NATIVE)
    const auto* words = static_cast<const uintptr_t*>(context);
    char handler_stack_marker;
    sampled_callers[index] = CaptureCallerChain(words[runtime_rsp_offset / sizeof(uintptr_t)],
        words[runtime_rbp_offset / sizeof(uintptr_t)], reinterpret_cast<uintptr_t>(&handler_stack_marker));
#endif
    pc_count.store(index + 1, std::memory_order_release);
}
#endif
#ifdef PS5_NATIVE
std::array<unsigned, 5> worker_cpus{};
bool worker_topology_ready = false;
// Physical core (x2APIC above the SMT shift) of each allowed CPU, -1 if unknown.
std::array<int, 64> cpu_core = [] { std::array<int, 64> c{}; c.fill(-1); return c; }();
// CPUs outside guest cores 0-2 and their SMT siblings, and not the GPU thread CPU.
unsigned long long secondary_cpus = 0;
// The process CPU set seen by the topology probe (threads may inherit narrower sets).
cpuset_t topology_allowed{};
bool topology_allowed_valid = false;
std::atomic<bool> placement_secondary{false};
std::atomic<unsigned> secondary_reports{};
// EDEN_UP_GPU_PIN_EXCLUSIVE=1 (PlanGpuExclusive): the shared mask that replaces secondary_cpus, and
// the GPU threads' whole cores. 0 = off. Set per session before the game's threads start.
std::atomic<unsigned long long> exclusive_shared{0}, exclusive_gpu{0};
unsigned long long SharedCpus() {
    const unsigned long long shared = exclusive_shared.load(std::memory_order_relaxed);
    return shared != 0 ? shared : secondary_cpus;
}
void CheckWorkerTopology() {
    cpuset_t original{};
    if (__get_cpuid_max(0, nullptr) < 0xb ||
        cpuset_getaffinity(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, 8, &original)) return;
    std::array<unsigned, 5> cores{};
    unsigned count = 0;
    bool changed = false;
    // Why a probe stopped short (EDEN_WORKER_PROBE): every run on firmware 12.40 reports one
    // distinct core, so nothing is pinned there and the log has never said why.
    unsigned probed = 0, stopped_at = 64, shift = 0, first_id = 0, last_id = 0;
    int refused = 0;
    bool bad_level = false;
    for (unsigned cpu = 0; cpu < 64; ++cpu) {
        if (!CPU_ISSET(cpu, &original)) continue;
        cpuset_t one{};
        CPU_SET(cpu, &one);
        if (cpuset_setaffinity(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, 8, &one)) {
            refused = errno;
            stopped_at = cpu;
            break;
        }
        changed = true;
        unsigned a, b, c, d;
        __cpuid_count(0xb, 0, a, b, c, d);
        // Architectural SMT level: x2APIC ID above the SMT shift identifies
        // the physical core. Do not assume OS CPU numbering matches APIC IDs.
        if (!b || ((c >> 8) & 0xff) != 1 || (a & 31) >= 16) {
            bad_level = true;
            stopped_at = cpu;
            break;
        }
        const unsigned core = d >> (a & 31);
        if (probed++ == 0) first_id = d;
        last_id = d;
        shift = a & 31;
        cpu_core[cpu] = static_cast<int>(core);
        if (count == cores.size() ||
            std::find(cores.begin(), cores.begin() + count, core) != cores.begin() + count) continue;
        cores[count] = core;
        worker_cpus[count++] = cpu;
    }
    if (changed) {
        cpuset_t restored{};
        if (cpuset_setaffinity(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, 8, &original) ||
            cpuset_getaffinity(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, 8, &restored) ||
            std::memcmp(&original, &restored, 8))
            throw std::runtime_error("Cannot restore topology-probe thread affinity");
    }
    unsigned long long allowed_mask = 0;
    for (unsigned cpu = 0; cpu < 64; ++cpu)
        if (CPU_ISSET(cpu, &original)) allowed_mask |= 1ULL << cpu;
    std::printf("EDEN_WORKER_PROBE allowed=%llx probed=%u stopped_at=%u refused=%d bad_level=%d shift=%u "
                "first_id=%u last_id=%u\n", allowed_mask, probed, stopped_at, refused, int(bad_level), shift,
                first_id, last_id);
    // EDEN_PS5_PIN=pairs: when the probe found too few cores, take CPUs 2n and 2n+1 as one core's
    // two threads (the layout tools/check-worker-affinity.py models) and pin as a console that
    // passes the probe does. Off by default: the layout is an assumption, not a reading.
    const char* const pin = std::getenv("EDEN_PS5_PIN");
    if (count != cores.size() && pin != nullptr && std::strcmp(pin, "pairs") == 0) {
        count = 0;
        cpu_core.fill(-1);
        for (unsigned cpu = 0; cpu < 64; ++cpu) {
            if (!CPU_ISSET(cpu, &original)) continue;
            const unsigned core = cpu / 2;
            cpu_core[cpu] = static_cast<int>(core);
            if (count == cores.size() ||
                std::find(cores.begin(), cores.begin() + count, core) != cores.begin() + count) continue;
            cores[count] = core;
            worker_cpus[count++] = cpu;
        }
    }
    worker_topology_ready = count == cores.size();
    topology_allowed = original;
    topology_allowed_valid = true;
    if (worker_topology_ready) {
        for (unsigned cpu = 0; cpu < 64; ++cpu) {
            if (!CPU_ISSET(cpu, &original) || cpu_core[cpu] < 0) continue;
            const bool guest_core = cpu_core[cpu] == static_cast<int>(cores[0]) ||
                cpu_core[cpu] == static_cast<int>(cores[1]) || cpu_core[cpu] == static_cast<int>(cores[2]);
            if (!guest_core && cpu != worker_cpus[3] && cpu != worker_cpus[4]) secondary_cpus |= 1ULL << cpu;
        }
        std::printf("EDEN_WORKER_SECONDARY mask=%llx\n", secondary_cpus);
    }
    std::printf("EDEN_WORKER_TOPOLOGY ready=%d distinct_cores=%u cpus=%u,%u,%u,%u,%u\n",
        worker_topology_ready, count, worker_cpus[0], worker_cpus[1], worker_cpus[2], worker_cpus[3], worker_cpus[4]);
}
// EDEN_PS5_PLACE="CPUCore_4=1/GPUBack=10/VulkanWorker=12,13" (layout experiments; ; or / between entries): the CPUs a
// thread of that name runs on, replacing its pin or the secondary mask. Read on each placement:
// the game's env.txt arrives after the launcher's own threads were placed.
unsigned long long PlacementOverride(const char* name) {
    const char* spec = std::getenv("EDEN_PS5_PLACE");
    if (spec == nullptr || name == nullptr) return 0;
    const std::string_view wanted{name};
    std::string_view rest{spec};
    while (!rest.empty()) {
        const size_t end = rest.find_first_of(";/");
        const std::string_view entry = rest.substr(0, end);
        rest = end == std::string_view::npos ? std::string_view{} : rest.substr(end + 1);
        const size_t eq = entry.find('=');
        if (eq == std::string_view::npos || entry.substr(0, eq) != wanted) continue;
        unsigned long long mask = 0;
        std::string_view cpus = entry.substr(eq + 1);
        while (!cpus.empty()) {
            const size_t comma = cpus.find(',');
            const std::string_view item = cpus.substr(0, comma);
            cpus = comma == std::string_view::npos ? std::string_view{} : cpus.substr(comma + 1);
            unsigned cpu = 0;
            for (const char ch : item) {
                if (ch < '0' || ch > '9') break;
                cpu = cpu * 10 + static_cast<unsigned>(ch - '0');
            }
            if (!item.empty() && cpu < 64) mask |= 1ULL << cpu;
        }
        return mask;
    }
    return 0;
}
// 0840, EDEN_UP_PREPASS_ALL_CORES=1 (off by default; T166 review: it was on unless "0", which
// changed every start with EDEN_UP_SHADER_SET_PREPASS=1; only with that switch): the
// before-play pass's threads (ShaderSetPrepass: its pool and its reporter) get every CPU the
// process is allowed instead of the shared mask, when they are made before the guest starts. In
// run1387 (T162, 13 Nov) its 8 workers shared CPUs 7, 9, 10, 11, 12 while the guest cores (0-5),
// the GPU threads' CPUs (6, 8) were idle: no guest code runs before EDEN_GUEST_START, so the
// guest cores' threads wait, the GPU thread has no guest work, and the boot pipelines are built
// before the pass. The pool is made at the start of the pass and joined at its end, so its
// threads end with the pass. T166 review: 0786 keeps a pool whose builds never returned to the end
// of the session, and those threads (a build that does not return is still running) would keep
// every CPU, the guest cores included. So each widened thread's id is noted, and NoteGuestStart
// puts every one still noted back on the shared mask. A thread that ends is not seen ending, so
// its id stays noted; a later thread that gets the same id is named through RegisterWorker,
// which forgets the id first (ForgetPrepassThread). An unnamed thread that gets a reused id is
// put on the shared mask, which is where unplaced threads are anyway. An EDEN_PS5_PLACE entry for
// ShaderSetPrepass wins.
std::atomic<bool> guest_started{false};
std::atomic<unsigned> prepass_widened{0};
std::atomic<unsigned long long> prepass_mask{0};
std::mutex prepass_threads_mutex;     // guest_started's change, prepass_threads
std::vector<int> prepass_threads;     // ids of widened threads (pthread_getthreadid_np)
std::atomic<bool> prepass_threads_any{false};
void ForgetPrepassThread() {
#ifdef PS5_NATIVE
    if (!prepass_threads_any.load(std::memory_order_relaxed)) return;
    const int self = pthread_getthreadid_np();
    std::scoped_lock lock{prepass_threads_mutex};
    std::erase(prepass_threads, self);
#endif
}
unsigned long long AllowedCpus() {
    if (!topology_allowed_valid) return 0;
    unsigned long long mask = 0;
    for (unsigned cpu = 0; cpu < 64; ++cpu)
        if (CPU_ISSET(cpu, &topology_allowed)) mask |= 1ULL << cpu;
    return mask;
}
unsigned long long PrepassAllCores(const char* name) {
    if (std::strncmp(name, "ShaderSetPrepass", 16) != 0 || guest_started.load(std::memory_order_relaxed) ||
        !Common::PeSwitch("EDEN_UP_SHADER_SET_PREPASS") || !Common::PeSwitch("EDEN_UP_PREPASS_ALL_CORES"))
        return 0;
    const unsigned long long cpus = AllowedCpus();
    if (cpus == 0) return 0;
#ifdef PS5_NATIVE
    {
        // Under the lock NoteGuestStart sets guest_started with: a thread is either noted before
        // the narrowing or not widened at all.
        std::scoped_lock lock{prepass_threads_mutex};
        if (guest_started.load(std::memory_order_relaxed)) return 0;
        prepass_threads.push_back(pthread_getthreadid_np());
        prepass_threads_any.store(true, std::memory_order_relaxed);
    }
#endif
    prepass_widened.fetch_add(1, std::memory_order_relaxed);
    prepass_mask.store(cpus, std::memory_order_relaxed);
    return cpus;
}
// 0842, EDEN_UP_AV1_ALL_CORES=1 (off by default): the movie decoder's thread (nx2:vdec, core
// nx2_gsv.h DecodeLoop) gets every allowed CPU but the GPU threads' (GPU, GPUBack, VulkanWorker and
// EDEN_UP_GPU_PIN_EXCLUSIVE's cores). FFmpeg's libdav1d makes its EDEN_NX2_AV1_THREADS worker
// threads from that thread when it opens a stream (ffmpeg.cpp OpenContext); they are not named
// through this hook and take the CPU set of the thread that made them, so today they share the
// 5-CPU shared mask with everything else that is not placed (the title movie: 18-58 fps across the
// 13 Nov sessions). During the title movie the guest cores' own work is light; during a cutscene
// movie in play the decoder then competes with the guest cores (not measured). The late-dropped
// count is the core's existing line ("nx2:vdecsvc frame N ... K late dropped", every 600 frames and
// at the end of a stream, EDEN_UP_MOVIE_PACE=1). An EDEN_PS5_PLACE entry for nx2:vdec wins.
unsigned long long Av1AllCores(const char* name) {
    if (std::strcmp(name, "nx2:vdec") != 0 || !Common::PeSwitch("EDEN_UP_AV1_ALL_CORES")) return 0;
    const unsigned long long allowed = AllowedCpus();
    unsigned long long gpu = exclusive_gpu.load(std::memory_order_relaxed) | PlacementOverride("GPU") |
                             PlacementOverride("GPUBack") | PlacementOverride("VulkanWorker");
    // The GPU thread's own CPU: its EDEN_PS5_PLACE entry's lowest (as PinWorker), else its pin.
    if (PlacementOverride("GPU") == 0 && worker_topology_ready) gpu |= 1ULL << worker_cpus[4];  // names[4]
    unsigned long long cpus = allowed & ~gpu;
    if (cpus == 0) cpus = allowed;
    std::printf("EDEN_UP_AV1_ALL_CORES: nx2:vdec and the AV1 decoder threads it starts run on mask %llx "
                "(shared mask %llx, GPU threads' CPUs %llx left out)\n", cpus, SharedCpus(), gpu & allowed);
    return cpus;
}
void PlaceSecondary(const char* name) {
    unsigned long long cpus = PlacementOverride(name);
    const unsigned long long gpu = exclusive_gpu.load(std::memory_order_relaxed);
    if (cpus == 0 && placement_secondary.load(std::memory_order_relaxed)) cpus = PrepassAllCores(name);
    if (cpus == 0 && placement_secondary.load(std::memory_order_relaxed)) cpus = Av1AllCores(name);
    if (cpus == 0) {
        if (!placement_secondary.load(std::memory_order_relaxed) || !SharedCpus()) return;
        cpus = SharedCpus();
    } else if (gpu != 0 && std::strcmp(name, "GPUBack") != 0 && (cpus & ~gpu) != 0) {
        // EDEN_UP_GPU_PIN_EXCLUSIVE: a placed thread keeps off the GPU threads' cores, unless it was
        // placed on nothing else.
        cpus &= ~gpu;
    }
    cpuset_t mask{};
    for (unsigned cpu = 0; cpu < 64; ++cpu)
        if (cpus & (1ULL << cpu)) CPU_SET(cpu, &mask);
    const int result = cpuset_setaffinity(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, 8, &mask);
    if (secondary_reports.fetch_add(1, std::memory_order_relaxed) < 64)
        std::printf("EDEN_WORKER_PLACED name=%s mask=%llx error=%d\n", name, cpus, result ? errno : 0);
}
void PinWorker(unsigned index, cpuset_t& affinity) {
    if (index >= worker_cpus.size()) return;
    // EDEN_PS5_PLACE names a CPU itself, so it needs no topology (ready=0 made it do nothing for
    // CPUCore_0-3 and GPU, while it did work for every other thread).
    const unsigned long long over = PlacementOverride(names[index]);
    if (!worker_topology_ready && over == 0) return;
    unsigned cpu = worker_cpus[index];
    if (over != 0) {
        cpu = static_cast<unsigned>(__builtin_ctzll(over));
    }
    if (!CPU_ISSET(cpu, topology_allowed_valid ? &topology_allowed : &affinity)) return;
    cpuset_t one{}, verified{};
    CPU_SET(cpu, &one);
    if (cpuset_setaffinity(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, 8, &one)) {
        std::printf("EDEN_WORKER_PIN name=%s error=%d\n", names[index], errno);
        return;
    }
    if (cpuset_getaffinity(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, 8, &verified) ||
        std::memcmp(&one, &verified, 8)) {
        if (cpuset_setaffinity(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, 8, &affinity))
            throw std::runtime_error("Cannot restore worker affinity");
        return;
    }
    affinity = verified;
    std::printf("EDEN_WORKER_PIN name=%s cpu=%u verified=1\n", names[index], cpu);
}
#endif
#ifdef PS5_NATIVE
// EDEN_UP_CPU_PAIR_PROBE=1 (log only): the round trip of one cache line between every two allowed
// CPUs. CPUID gives every CPU the same x2APIC ID on this firmware (EDEN_WORKER_PROBE first_id=0
// last_id=0), so which CPUs are SMT siblings and which share an L3 is a guess: EDEN_PS5_PIN=pairs
// takes 2n and 2n+1. On Zen 2 siblings answer in tens of ns, two cores of one CCX in about 100 and
// cores of the two CCXs in a few hundred. One line per CPU: the median of 7 x 200 round trips to each
// allowed CPU in that order, in ns (-1: not measured). Once at start, tens of ms for 13 CPUs.
void ProbeCpuPairs(const cpuset_t& allowed) {
    std::vector<unsigned> cpus;
    for (unsigned cpu = 0; cpu < 64; ++cpu)
        if (CPU_ISSET(cpu, &allowed)) cpus.push_back(cpu);
    if (cpus.size() < 2) return;
    const auto pin = [](unsigned cpu) {
        cpuset_t one{};
        CPU_SET(cpu, &one);
        return cpuset_setaffinity(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, 8, &one) == 0;
    };
    // False after ~100 ms; the clock is read only every 4096 spins, never on a quick answer.
    const auto wait_until = [](const auto& done) {
        std::chrono::steady_clock::time_point deadline{};
        for (unsigned spins = 1; !done(); ++spins) {
            if ((spins & 4095) != 0) continue;
            const auto now = std::chrono::steady_clock::now();
            if (deadline == std::chrono::steady_clock::time_point{})
                deadline = now + std::chrono::milliseconds(100);
            else if (now > deadline)
                return false;
        }
        return true;
    };
    alignas(64) std::atomic<unsigned> ball{0};
    alignas(64) std::atomic<unsigned> request{0};
    std::atomic<unsigned> answered{0};
    std::atomic<int> target{-1};
    std::atomic<bool> pinned{false}, stop{false};
    // Same-priority threads are not time-sliced here, so main and the spinning helper must never
    // share a CPU (main would never run again). Main takes cpus[0] and asks for the last CPU before
    // the helper exists: the helper inherits cpus[0], runs once main yields and leaves at once. Main
    // then visits cpus[0..n-2] only, and the helper ends every row on cpus[n-1]. A helper whose move
    // failed yields on every turn.
    unsigned asked = 1;
    bool aborted = !pin(cpus.front());
    target.store(static_cast<int>(cpus.back()), std::memory_order_relaxed);
    request.store(asked, std::memory_order_release);
    // The helper moves to the CPU asked for, then answers every odd ball with the next even one.
    std::thread helper;
    if (!aborted) helper = std::thread([&] {
        unsigned seen = 0;
        while (!stop.load(std::memory_order_acquire)) {
            const unsigned now_asked = request.load(std::memory_order_acquire);
            if (now_asked != seen) {
                seen = now_asked;
                pinned.store(pin(static_cast<unsigned>(target.load(std::memory_order_relaxed))),
                             std::memory_order_relaxed);
                answered.store(now_asked, std::memory_order_release);
                continue;
            }
            const unsigned value = ball.load(std::memory_order_acquire);
            if (value & 1) ball.store(value + 1, std::memory_order_release);
            if (!pinned.load(std::memory_order_relaxed)) std::this_thread::yield();
        }
    });
    // Yield while the helper moves: a helper the OS queued on this CPU still gets to run.
    const auto moved = [&] {
        const auto moved_by = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
        while (answered.load(std::memory_order_acquire) != asked &&
               std::chrono::steady_clock::now() < moved_by)
            std::this_thread::yield();
        return answered.load(std::memory_order_acquire) == asked;
    };
    if (!aborted) aborted = !moved() || !pinned.load(std::memory_order_relaxed);
    std::vector<std::vector<long long>> trip(cpus.size(), std::vector<long long>(cpus.size(), -1));
    for (size_t i = 0; i + 1 < cpus.size() && !aborted; ++i) {
        if (i != 0 && !pin(cpus[i])) continue;
        for (size_t j = i + 1; j < cpus.size() && !aborted; ++j) {
            target.store(static_cast<int>(cpus[j]), std::memory_order_relaxed);
            request.store(++asked, std::memory_order_release);
            if (!moved()) {
                aborted = true;
                break;
            }
            if (!pinned.load(std::memory_order_relaxed)) continue;
            std::array<long long, 7> batches{};
            for (long long& batch : batches) {
                const auto start = std::chrono::steady_clock::now();
                for (unsigned k = 0; k < 200 && !aborted; ++k) {
                    const unsigned sent = ball.load(std::memory_order_relaxed) + 1;
                    ball.store(sent, std::memory_order_release);
                    aborted = !wait_until([&] { return ball.load(std::memory_order_acquire) == sent + 1; });
                }
                batch = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start).count() / 200;
            }
            if (aborted) break;
            std::sort(batches.begin(), batches.end());
            trip[i][j] = trip[j][i] = batches[3];
        }
    }
    stop.store(true, std::memory_order_release);
    if (helper.joinable()) helper.join();
    cpuset_t restored{};
    if (cpuset_setaffinity(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, 8, &allowed) ||
        cpuset_getaffinity(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, 8, &restored) ||
        std::memcmp(&allowed, &restored, 8))
        throw std::runtime_error("Cannot restore pair-probe thread affinity");
    std::string order;
    for (const unsigned cpu : cpus) order += (order.empty() ? "" : ",") + std::to_string(cpu);
    std::printf("EDEN_UP_CPU_PAIR_PROBE cpus=%s aborted=%d\n", order.c_str(), int(aborted));
    for (size_t i = 0; i < cpus.size(); ++i) {
        std::string line;
        for (size_t j = 0; j < cpus.size(); ++j)
            line += (j ? "," : "") + (i == j ? std::string("-") : std::to_string(trip[i][j]));
        std::printf("EDEN_UP_CPU_PAIRS cpu=%u rt_ns=%s\n", cpus[i], line.c_str());
    }
}
// EDEN_UP_GPU_PIN_EXCLUSIVE=1 (zortos293/Lichii #913 by the-cashman, on #769): the GPU front ("GPU")
// and back ("GPUBack") each keep their whole physical core. The shared mask (named threads without a
// pin or placement, and what the main thread hands the threads it starts) becomes the allowed CPUs
// minus the CPU each GPU thread is on and its SMT sibling, minus guest cores 0-2 with their SMT
// siblings (as without the switch) and the one CPU CPUCore_3 and a pinned CPUCore_4/5 are on; a
// placed thread other than GPUBack loses the GPU threads' cores unless nothing
// else is left. Without EDEN_PS5_PIN=pairs too, where nothing else places the other threads. Only GPU
// threads with a known CPU count: the front's pin or placement, GPUBack's placement. Siblings: the
// topology reading when ready, else 2n and 2n+1. Threads a pinned thread starts without naming
// themselves still inherit its CPU, as in #913. Off: nothing changes.
void PlanGpuExclusive(bool enabled) {
    exclusive_shared.store(0, std::memory_order_relaxed);
    exclusive_gpu.store(0, std::memory_order_relaxed);
    if (!enabled || !topology_allowed_valid || !Common::PeSwitch("EDEN_UP_GPU_PIN_EXCLUSIVE")) return;
    unsigned long long allowed = 0;
    for (unsigned cpu = 0; cpu < 64; ++cpu)
        if (CPU_ISSET(cpu, &topology_allowed)) allowed |= 1ULL << cpu;
    const auto core_of = [](unsigned cpu) {
        if (!worker_topology_ready) return static_cast<int>(cpu / 2);
        return cpu_core[cpu] >= 0 ? cpu_core[cpu] : -1 - static_cast<int>(cpu);
    };
    const auto whole_cores = [&](unsigned long long cpus) {
        unsigned long long out = cpus;
        for (unsigned cpu = 0; cpu < 64; ++cpu) {
            if (!(cpus >> cpu & 1)) continue;
            for (unsigned other = 0; other < 64; ++other)
                if ((allowed >> other & 1) && core_of(other) == core_of(cpu)) out |= 1ULL << other;
        }
        return out;
    };
    const auto lowest = [](unsigned long long cpus) { return cpus & (0 - cpus); };
    // As PinWorker: a placement's lowest CPU, else the topology pin.
    const auto pinned = [&](unsigned index) {
        const unsigned long long over = PlacementOverride(names[index]);
        if (over != 0) return lowest(over);
        return worker_topology_ready ? 1ULL << worker_cpus[index] : 0ULL;
    };
    const unsigned long long front = pinned(4) & allowed;
    const unsigned long long back = PlacementOverride("GPUBack") & allowed;
    const unsigned long long gpu = whole_cores(front | back);
    // Guest cores 0-2 keep their whole cores, as the shared mask without the switch keeps them and
    // their SMT siblings clear; CPUCore_3 and a one-CPU CPUCore_4/5 placement keep their own CPU.
    unsigned long long guests = whole_cores(pinned(0) | pinned(1) | pinned(2)) | pinned(3);
    for (const char* name : {"CPUCore_4", "CPUCore_5"}) {
        const unsigned long long over = PlacementOverride(name);
        if (over != 0 && lowest(over) == over) guests |= over;  // one CPU: a pin
    }
    guests &= allowed;
    unsigned long long shared = allowed & ~gpu & ~guests;
    if (shared == 0) shared = allowed & ~gpu;
    std::printf("EDEN_UP_GPU_EXCLUSIVE front=%llx back=%llx reserved=%llx guests=%llx shared=%llx clash=%llx\n",
                front, back, gpu, guests, shared, guests & gpu);
    if (gpu == 0 || shared == 0) return;
    exclusive_gpu.store(gpu, std::memory_order_relaxed);
    exclusive_shared.store(shared, std::memory_order_relaxed);
}
#endif
bool cpu_clocks_valid = false;
bool owner_cpu_clock_valid = false;
std::atomic<unsigned> sample_epoch{};
struct CpuSample {
    unsigned epoch = ~0u;
    long long mono_ns = 0, cpu_ns = -ENOTSUP;
    unsigned long long thread = 0, pc = 0;
    unsigned svc = ~0u, fpcr = 0;
    unsigned long long compilations = 0, compile_ns = 0;
};
std::array<CpuSample, 6> cpu_samples;
// Read/written only by the worker owning this core; samples use workers_mutex.
std::array<unsigned, 6> sampled_epoch{~0u, ~0u, ~0u, ~0u, ~0u, ~0u};
clockid_t process_clock{};
int CurrentCpuClock(clockid_t* clock, bool process = false) {
#ifdef PS5_NATIVE
    // Native titles must use the imported kernel APIs. Firmware rejects direct
    // syscalls from the title. Qualification below rejects unusable CPU clocks.
    if (process) { *clock = CLOCK_PROCESS_CPUTIME_ID; return 0; }
    return pthread_getcpuclockid(pthread_self(), clock);
#else
    return process ? clock_getcpuclockid(0, clock) : pthread_getcpuclockid(pthread_self(), clock);
#endif
}
long long ClockNs(clockid_t clock) {
    timespec time{};
    if (clock_gettime(clock, &time) != 0) return -errno;
    return static_cast<long long>(time.tv_sec) * 1'000'000'000 + time.tv_nsec;
}
void CheckCpuClocks() {
    clockid_t worker_clock{};
    int worker_error = ENOENT;
    const int process_error = CurrentCpuClock(&process_clock, true);
    long long owner_begin = -1, owner_busy = -1, owner_idle = -1;
    std::binary_semaphore ready{0}, proceed{0};
    std::jthread worker([&] {
        worker_error = CurrentCpuClock(&worker_clock);
        ready.release();
        proceed.acquire();
        owner_begin = ClockNs(CLOCK_THREAD_CPUTIME_ID);
        const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(20);
        while (std::chrono::steady_clock::now() < end) std::atomic_signal_fence(std::memory_order_seq_cst);
        owner_busy = ClockNs(CLOCK_THREAD_CPUTIME_ID);
        ready.release();
        proceed.acquire();
        owner_idle = ClockNs(CLOCK_THREAD_CPUTIME_ID);
    });
    ready.acquire();
    const auto cpu_begin = worker_error ? -1 : ClockNs(worker_clock);
    proceed.release();
    ready.acquire();
    const auto cpu_busy = worker_error ? -1 : ClockNs(worker_clock);
    const auto process_idle = process_error ? -1 : ClockNs(process_clock);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const auto cpu_idle = worker_error ? -1 : ClockNs(worker_clock);
    const auto process_end = process_error ? -1 : ClockNs(process_clock);
    proceed.release();
    worker.join();
    const auto own_busy = owner_busy - owner_begin;
    const auto own_idle = owner_idle - owner_busy;
    owner_cpu_clock_valid = owner_begin >= 0 && own_busy >= 5'000'000 &&
        own_busy < 30'000'000 && own_idle >= 0 && own_idle < own_busy / 4 + 1'000'000;
    std::printf("EDEN_PERF_OWNER_CLOCK_CHECK valid=%d busy_ns=%lld idle_ns=%lld\n",
                owner_cpu_clock_valid, own_busy, own_idle);
    const auto busy = cpu_busy - cpu_begin;
    const auto idle = cpu_idle - cpu_busy;
    const auto process_sleep = process_end - process_idle;
    cpu_clocks_valid = !worker_error && !process_error && cpu_begin >= 0 && process_idle >= 0 &&
        busy >= 1'000'000 && idle >= 0 && idle < busy / 4 + 1'000'000 &&
        process_sleep >= 0 && process_sleep < 6'000'000;
    std::printf("EDEN_PERF_CPU_CLOCK_CHECK valid=%d worker_error=%d process_error=%d "
                "worker_clock=%d process_clock=%d busy_ns=%lld idle_ns=%lld process_sleep_ns=%lld\n",
                cpu_clocks_valid, worker_error, process_error, int(worker_clock), int(process_clock),
                busy, idle, process_sleep);
}
}

void ReportOwnThread(const char* name) {
    if (!perf_report.load(std::memory_order_relaxed) || !owner_cpu_clock_valid) return;
    static long long last_cpu = 0, last_mono = 0;
    const long long cpu = ClockNs(CLOCK_THREAD_CPUTIME_ID), mono = ClockNs(CLOCK_MONOTONIC);
    if (last_mono != 0)
        std::printf("EDEN_PERF_THREAD name=%s cpu_ms=%lld wall_ms=%lld\n", name, (cpu - last_cpu) / 1000000LL,
                    (mono - last_mono) / 1000000LL);
    last_cpu = cpu;
    last_mono = mono;
}

void SetSecondaryPlacement(bool enabled) {
#ifdef PS5_NATIVE
    placement_secondary.store(enabled, std::memory_order_relaxed);
    // EDEN_UP_GPU_PIN_EXCLUSIVE: planned per session, once env-<title id>.txt is in. A main thread
    // the last session narrowed gets its whole set back when the new one has no shared mask.
    const bool was_exclusive = exclusive_shared.load(std::memory_order_relaxed) != 0;
    PlanGpuExclusive(enabled);
    const unsigned long long shared = SharedCpus();
    if ((!shared && !was_exclusive) || !topology_allowed_valid) return;
    cpuset_t mask{};
    if (enabled && shared) {
        for (unsigned cpu = 0; cpu < 64; ++cpu)
            if (shared & (1ULL << cpu)) CPU_SET(cpu, &mask);
    } else {
        mask = topology_allowed;
    }
    const int result = cpuset_setaffinity(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, 8, &mask);
    std::printf("EDEN_WORKER_INHERIT enabled=%d mask=%llx error=%d\n", enabled,
                enabled ? shared : 0ULL, result ? errno : 0);
#else
    (void)enabled;
#endif
}

namespace {
// 0841 (HoldBackgroundPass, below): released at the session's end; when the guest started.
std::mutex hold_mutex;
std::condition_variable hold_wake;
bool hold_released = false;
std::optional<std::chrono::steady_clock::time_point> hold_guest_start;
} // namespace

void NoteSessionStart() {
    {
        std::scoped_lock lock{hold_mutex};
        hold_released = false;
        hold_guest_start.reset();
    }
#ifdef PS5_NATIVE
    {
        std::scoped_lock lock{prepass_threads_mutex};
        guest_started.store(false, std::memory_order_relaxed);
        prepass_threads.clear();
        prepass_threads_any.store(false, std::memory_order_relaxed);
    }
    prepass_widened.store(0, std::memory_order_relaxed);
    prepass_mask.store(0, std::memory_order_relaxed);
#endif
}

void NoteGuestStart() {
    {
        std::scoped_lock lock{hold_mutex};
        hold_guest_start = std::chrono::steady_clock::now();
    }
    hold_wake.notify_all();
#ifdef PS5_NATIVE
    std::vector<int> narrow;
    {
        std::scoped_lock lock{prepass_threads_mutex};
        guest_started.store(true, std::memory_order_relaxed);
        narrow.swap(prepass_threads);
        prepass_threads_any.store(false, std::memory_order_relaxed);
    }
    // T166 review: threads still noted (a pool 0786 kept, with a build that did not return) go
    // back on the shared mask, as PlaceSecondary would have placed them. A thread that has ended
    // answers ESRCH and is counted as gone.
    unsigned narrowed = 0, gone = 0, failed = 0;
    int first_error = 0;
    if (!narrow.empty()) {
        unsigned long long cpus = SharedCpus();
        const unsigned long long gpu = exclusive_gpu.load(std::memory_order_relaxed);
        if (gpu != 0 && (cpus & ~gpu) != 0) cpus &= ~gpu;
        cpuset_t mask{};
        for (unsigned cpu = 0; cpu < 64; ++cpu)
            if (cpus & (1ULL << cpu)) CPU_SET(cpu, &mask);
        for (const int id : narrow) {
            if (cpus == 0) break;
            if (cpuset_setaffinity(CPU_LEVEL_WHICH, CPU_WHICH_TID, id, 8, &mask) == 0) {
                ++narrowed;
            } else if (errno == ESRCH) {
                ++gone;
            } else {
                ++failed;
                if (first_error == 0) first_error = errno;
            }
        }
    }
    if (const unsigned widened = prepass_widened.load(std::memory_order_relaxed); widened != 0)
        std::printf("EDEN_UP_PREPASS_ALL_CORES: %u ShaderSetPrepass threads ran on mask %llx before the "
                    "game started (shared mask %llx); none is widened from here; %u still running put "
                    "back on the shared mask, %u ended, %u refused (error %d)\n",
                    widened, prepass_mask.load(std::memory_order_relaxed), SharedCpus(), narrowed, gone,
                    failed, first_error);
#endif
}

void SampleGpuFrame(unsigned frame) {
    std::printf("EDEN_PERF_GPU_FRAME frame=%u mono_ns=%lld cpu_ns=%lld\n", frame,
                ClockNs(CLOCK_MONOTONIC),
                owner_cpu_clock_valid ? ClockNs(CLOCK_THREAD_CPUTIME_ID) : -static_cast<long long>(ENOTSUP));
}

extern "C" void ps5_opengl_heap_snapshot(const char* phase, unsigned iteration);
extern "C" unsigned eden_heap_arenas_created(void) __attribute__((weak));
extern "C" std::size_t eden_heap_committed(void) __attribute__((weak));
extern "C" std::size_t eden_heap_large_held(unsigned* blocks) __attribute__((weak));
extern "C" std::size_t eden_heap_trim(void) __attribute__((weak));
// EDEN_UP_HEAP_HOLD_AT_LOAD (headless/heap_arenas.inc, t174a): the spare again while the pool has room,
// and the counts for its line.
extern "C" void eden_heap_hold_refill(unsigned long long largest_free) __attribute__((weak));
extern "C" int eden_heap_hold_stats(unsigned long long* values, unsigned count) __attribute__((weak));
extern "C" void eden_heap_sites_report(void) __attribute__((weak));
extern "C" void eden_heap_usage(std::size_t* held, std::size_t* in_use, std::size_t* returned) __attribute__((weak));
extern "C" void eden_heap_page_usage(std::size_t* free_pages, std::size_t* idle_arenas, std::size_t* live)
    __attribute__((weak));
extern "C" void eden_heap_growth_report(void) __attribute__((weak));
extern "C" void eden_heap_arena_report(void) __attribute__((weak));
extern "C" void eden_heap_name_thread(const char* name) __attribute__((weak));
extern "C" unsigned eden_guest_memory_wanted(void) __attribute__((weak));
// EDEN_UP_HEAP_WAIT_MS (headless/heap_arenas.inc): pipeline builder requests waiting for memory now,
// and how many waited so far.
extern "C" unsigned eden_heap_wanted(void) __attribute__((weak));
extern "C" void eden_heap_wait_stats(unsigned long long* count, unsigned long long* total_ms, unsigned* longest_ms,
                                     unsigned long long* lost) __attribute__((weak));
// EDEN_UP_HEAP_WAIT_GPU_MS: the same for pipelines a thread builds itself at run time (the GPU thread).
extern "C" void eden_heap_wait_gpu_stats(unsigned long long* count, unsigned long long* total_ms,
                                         unsigned long long* lost) __attribute__((weak));
// EDEN_UP_HEAP_RESCUE: requests about to be refused that an arena served, and those none could.
extern "C" void eden_heap_rescue_stats(unsigned long long* rescued, unsigned long long* missed) __attribute__((weak));
// EDEN_UP_HEAP_STEAL: its mode (0 off) and its counters, in the order of the enum in heap_arenas.inc.
extern "C" int eden_heap_steal_mode_now(void) __attribute__((weak));
extern "C" unsigned eden_heap_steal_stats(unsigned long long* values, unsigned count) __attribute__((weak));
// EDEN_TEST_HEAP_STARVE: test shortages begun on pipeline builder threads (with EDEN_TEST_HEAP_STARVE_GPU=1:
// inside run-time pipeline builds), and requests refused in them.
extern "C" void eden_heap_starve_stats(unsigned long long* begun, unsigned long long* refused) __attribute__((weak));
// T130: both heap test aids' own counts (EDEN_TEST_HEAP_STARVE, EDEN_TEST_HEAP_NO_GROW), whatever the
// steal mode, in the order heap_arenas.inc's eden_heap_test_aid_stats gives them.
extern "C" unsigned eden_heap_test_aid_stats(unsigned long long* values, unsigned count) __attribute__((weak));
extern "C" void eden_guest_memory_reserve(std::size_t* held, std::size_t* target) __attribute__((weak));
extern "C" void eden_guest_memory_report(void) __attribute__((weak));
extern "C" void eden_guest_memory_pool_short(int is_short) __attribute__((weak));
extern "C" void eden_guest_memory_waits(unsigned long long* won, unsigned long long* lost,
                                        unsigned long long* refused) __attribute__((weak));
// EDEN_UP_RADV_OOM_LEND (src/guest_memory_service.inc): the guest reserve's lends to the driver and the
// bounded reclaim's scans, for the line every 10 s.
extern "C" void eden_guest_reserve_lend_stats(unsigned long long* values, std::size_t count) __attribute__((weak));
// EDEN_UP_JIT_COMMIT_AHEAD / EDEN_TEST_JIT_COMMIT_REFUSE (src/memory_pages.cpp): whether either is set,
// and their counters for the line every 10 s.
extern "C" int eden_jit_ahead_active(void) __attribute__((weak));
extern "C" int eden_jit_ahead_stats(unsigned long long* values, std::size_t count) __attribute__((weak));
#ifdef PS5_NATIVE
extern "C" std::int64_t sceKernelGetDirectMemorySize();
extern "C" std::int32_t sceKernelAvailableDirectMemorySize(std::int64_t, std::int64_t, std::size_t, std::int64_t*,
                                                           std::size_t*);
// The allocated region at or after an offset (flag 1: find the next one).
struct DirectMemoryRegion { std::int64_t start; std::int64_t end; std::int32_t type; };
extern "C" std::int32_t sceKernelDirectMemoryQuery(std::int64_t, int, DirectMemoryRegion*, std::size_t);
extern "C" std::int32_t sceKernelAvailableFlexibleMemorySize(std::size_t*);
extern "C" void eden_direct_ledger(unsigned long long* bytes, unsigned long long* blocks,
                                    unsigned long long* hits, unsigned long long* fallbacks) __attribute__((weak));
// EDEN_POOL_LEDGER_EXACT=1 (src/direct_memory.c): 0 when off.
extern "C" int eden_direct_ledger_exact(unsigned long long* pieces, unsigned long long* unmatched,
                                        unsigned long long* unmatched_bytes,
                                        unsigned long long* untracked) __attribute__((weak));
// EDEN_UP_HOST_STACK_KIB (src/host_stacks.c): 0 when off or not in this build.
extern "C" int eden_host_stacks_report(char* line, std::size_t size) __attribute__((weak));
// EDEN_UP_SHADER_SET (video_core/shader_set.cpp, FreeMemoryBytes): the free memory its extraction
// and preparation stop below. This kernel has no vm.stats sysctl counters; the largest free block
// of the pool is what the caches here go by too (RefreshFreeMemory). 0: not known.
extern "C" unsigned long long eden_free_memory_bytes(void) {
    std::int64_t start = 0;
    std::size_t largest = 0;
    const std::int64_t total = sceKernelGetDirectMemorySize();
    if (total > 0 && sceKernelAvailableDirectMemorySize(0, total, 0x4000, &start, &largest) == 0) {
        return largest;
    }
    return 0;
}
// The allocator's blocks and what is in use of them (vulkan_memory_allocator.cpp).
extern "C" void eden_vma_usage(unsigned long long* blocks, unsigned long long* used) __attribute__((weak));
#ifdef EDEN_PS5_RADV
// The driver's own line (tools/patch-radv-memory-report.py), by the name isolate-radv.py gives it.
// Not weak: a driver built without it stops the link here, not the package check (check_package.py).
extern "C" void eden_radv_private_radv_ps5_memory_report_now(void);
#endif
// What is out of the pool by size (src/direct_memory.c) beside what the owners say they hold, and
// what the second pool has left: the flexible size in sce_sys/param.json is decided from this line
// (the same call as ProsperoEden's launch line). A release build imports one more kernel function.
static void ReportLedger() {
    if (!eden_direct_ledger) return;
    unsigned long long bytes[4]{}, blocks[4]{}, hits = 0, fallbacks = 0;
    eden_direct_ledger(bytes, blocks, &hits, &fallbacks);
    std::size_t span = 0, guest = 0, tables = 0, code = 0, reserve = 0, target = 0, flexible = 0;
    ::Common::GuestMemoryUsage(&span, &guest);
    ::Common::SparseUsage(&span, &tables);
    ::Common::CodeRangeUsage(&span, &code);
    if (eden_guest_memory_reserve) eden_guest_memory_reserve(&reserve, &target);
    const bool known = sceKernelAvailableFlexibleMemorySize(&flexible) == 0;
    std::fprintf(stderr, "EDEN_POOL_LEDGER out of the pool: %llu MiB in %llu blocks under 2 MiB, %llu in %llu of 2 MiB, "
                 "%llu in %llu up to 32 MiB, %llu in %llu larger; guest %zu reserve %zu heap %zu jit %zu tables %zu MiB; "
                 "flexible free %lld MiB; guest band hits=%llu fallbacks=%llu\n",
                 bytes[0] >> 20, blocks[0], bytes[1] >> 20, blocks[1], bytes[2] >> 20, blocks[2], bytes[3] >> 20, blocks[3],
                 guest >> 20, reserve >> 20, (eden_heap_committed ? eden_heap_committed() : std::size_t{0}) >> 20,
                 code >> 20, tables >> 20, known ? static_cast<long long>(flexible >> 20) : -1LL, hits, fallbacks);
    // EDEN_POOL_LEDGER_EXACT=1: the counts above took each release off the block it came from. A
    // line of its own, so readers of the line above keep working.
    unsigned long long pieces = 0, unmatched = 0, unmatched_bytes = 0, untracked = 0;
    if (eden_direct_ledger_exact && eden_direct_ledger_exact(&pieces, &unmatched, &unmatched_bytes, &untracked)) {
        std::fprintf(stderr, "EDEN_POOL_LEDGER_EXACT pieces=%llu unmatched=%llu (%llu MiB) untracked=%llu\n", pieces,
                     unmatched, unmatched_bytes >> 20, untracked);
    }
    if (char line[256]; eden_host_stacks_report && eden_host_stacks_report(line, sizeof(line))) {
        std::fprintf(stderr, "%s\n", line);
    }
    // RADV_PS5_MEMORY_REPORT=1: out of the pool, less the owners above, less this, is the driver's
    // own buffers and the rest, every 20 s and not only at a refusal. Without it nothing here runs:
    // this is on the GPU thread, and the question below can refresh the allocator's budget.
    const char* const memory_report = std::getenv("RADV_PS5_MEMORY_REPORT");
    if (memory_report && memory_report[0] == '1') {
        if (eden_vma_usage) {
            unsigned long long vma_blocks = 0, vma_used = 0;
            eden_vma_usage(&vma_blocks, &vma_used);
            std::fprintf(stderr, "EDEN_VMA_USAGE blocks %llu MiB, %llu MiB of it in use\n", vma_blocks >> 20, vma_used >> 20);
        }
#ifdef EDEN_PS5_RADV
        eden_radv_private_radv_ps5_memory_report_now();
#endif
    }
}
#endif

#ifdef PS5_NATIVE
// EDEN_MEMORY_REPORT_THREAD=1: the low-memory burst report and the heap trim run on a thread of
// their own instead of the GPU thread (RefreshFreeMemory below). The report only reads the
// service's slot table, which its own keeper and scanner threads change under it today too, and
// the heap trim takes the allocator's locks like any free; neither needs the GPU thread. One
// worker, started at the first request, never joined: the process ends with _exit.
static bool OffThreadReportsEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_MEMORY_REPORT_THREAD");
        return value != nullptr && value[0] != '0';
    }();
    return enabled;
}
void RegisterWorker(const char* name);
static void WriteMemorySummary(std::size_t largest, bool memory_short);
struct MemorySummaryNumbers {
    bool due{};
    std::size_t largest{};
    bool memory_short{};
};
static void QueueOffThreadReport(bool report, bool trim, MemorySummaryNumbers summary = {}, bool heap = false) {
    struct Pending {
        std::mutex mutex;
        std::condition_variable changed;
        bool report{}, trim{}, started{}, heap{};
        MemorySummaryNumbers summary;
    };
    static Pending* const pending = new Pending;  // outlives every static destructor
    {
        std::scoped_lock lock{pending->mutex};
        pending->report |= report;
        pending->trim |= trim;
        pending->heap |= heap;
        if (summary.due) pending->summary = summary;
        if (!pending->started) {
            pending->started = true;
            std::thread([] {
                RegisterWorker("MemoryReport");
                for (;;) {
                    bool do_report, do_trim, do_heap;
                    MemorySummaryNumbers do_summary;
                    {
                        std::unique_lock lock{pending->mutex};
                        pending->changed.wait(lock, [] {
                            return pending->report || pending->trim || pending->heap || pending->summary.due;
                        });
                        do_report = pending->report;
                        do_trim = pending->trim;
                        do_heap = pending->heap;
                        do_summary = pending->summary;
                        pending->report = pending->trim = pending->heap = false;
                        pending->summary = {};
                    }
                    if (do_report && eden_guest_memory_report) eden_guest_memory_report();
                    if (do_trim) TrimHeap("the pool is short");
                    // EDEN_UP_HEAP_STEAL: the heap's table (arena by arena, space by space) at its brink;
                    // "refused" lets ReportHeap's own limit (30 s, then 2 min) pass it.
                    if (do_heap) ReportHeap("refused: the heap could not grow (EDEN_UP_HEAP_STEAL)");
                    if (do_summary.due) WriteMemorySummary(do_summary.largest, do_summary.memory_short);
                }
            }).detach();
        }
    }
    pending->changed.notify_one();
}
// The pool, the guest's memory and the heap in stderr.log, every 20 s (RefreshFreeMemory).
static void WriteMemorySummary(std::size_t largest, bool memory_short) {
    std::size_t guest_span = 0, guest_held = 0;
    ::Common::GuestMemoryUsage(&guest_span, &guest_held);
    const auto kernel_memory = guest_kernel_memory.load(std::memory_order_relaxed);
    std::fprintf(stderr, "EDEN_MEMORY pool largest free block %zu MiB, short=%d; the guest's memory holds %zu MiB"
                 " (the kernel maps %llu MiB)\n",
                 largest >> 20, int(memory_short), guest_held >> 20,
                 kernel_memory ? kernel_memory() >> 20 : 0ULL);
    ReportHeap("every 20 s");
    if (eden_heap_wait_stats) {
        unsigned long long count = 0, total_ms = 0, lost = 0;
        unsigned longest_ms = 0;
        eden_heap_wait_stats(&count, &total_ms, &longest_ms, &lost);
        if (count != 0)
            std::fprintf(stderr, "EDEN_HEAP_WAIT pipeline builders waited for memory %llu times, %llu ms in all, "
                         "longest %u ms, %llu got none\n", count, total_ms, longest_ms, lost);
    }
    if (eden_heap_wait_gpu_stats) {
        unsigned long long count = 0, total_ms = 0, lost = 0;
        eden_heap_wait_gpu_stats(&count, &total_ms, &lost);
        if (count != 0)
            std::fprintf(stderr, "EDEN_HEAP_WAIT_GPU run-time pipeline builds waited for memory %llu times, %llu ms in "
                         "all, %llu got none\n", count, total_ms, lost);
    }
    if (eden_heap_rescue_stats) {
        unsigned long long rescued = 0, missed = 0;
        eden_heap_rescue_stats(&rescued, &missed);
        if (rescued != 0 || missed != 0)
            std::fprintf(stderr, "EDEN_HEAP_RESCUE requests about to be refused: %llu served by an arena, %llu by none\n",
                         rescued, missed);
    }
    if (eden_heap_starve_stats) {
        unsigned long long begun = 0, refused = 0;
        eden_heap_starve_stats(&begun, &refused);
        if (begun != 0) {
            // Where: pipeline builder threads, or with EDEN_TEST_HEAP_STARVE_GPU=1 the threads inside a
            // run-time pipeline build (the GPU thread's own builds).
            const char* const scope = std::getenv("EDEN_TEST_HEAP_STARVE_GPU");
            const bool gpu_scope = scope != nullptr && scope[0] == '1' && scope[1] == '\0';
            std::fprintf(stderr, "EDEN_TEST_HEAP_STARVE %llu test shortages %s, %llu requests refused in them\n", begun,
                         gpu_scope ? "inside run-time pipeline builds" : "on pipeline builder threads", refused);
        }
    }
    // T130: the test aids' own counts in every 20 s summary while one is set, so a control run (steal
    // off) shows them too: requests seen and refused, and what the steal did with the refusals.
    if (eden_heap_test_aid_stats) {
        unsigned long long aid[9] = {};
        eden_heap_test_aid_stats(aid, 9);
        if (aid[0] != 0 || aid[6] != 0)
            std::fprintf(stderr, "EDEN_TEST_HEAP_AIDS starve every %llu: seen %llu, shortages %llu, refused %llu, "
                         "sent to the steal %llu, served by it %llu; no_grow every %llu: growths seen %llu, refused %llu\n",
                         aid[0], aid[1], aid[2], aid[3], aid[4], aid[5], aid[6], aid[7], aid[8]);
    }
    ReportLedger();
    // EDEN_NX2_POOL_WALK=1 (off unless set): the pool by block size every 20 s (EDEN_REFUSED_POOL,
    // the walk a refusal makes). available_mib is the largest block, the ledger the sum: only
    // this says how much of what is free comes in 2 MiB blocks. Up to 4096 kernel calls, on the
    // GPU thread: not for timing runs. It prints the ledger again, and run-report.py counts
    // each one as a refusal report. Then the writes that had to wait for a block: nothing calls
    // eden_guest_memory_waits today, and the lines WaitForBlock writes go to the kernel log.
    static const bool pool_walk = [] {
        const char* const value = std::getenv("EDEN_NX2_POOL_WALK");
        return value != nullptr && value[0] == '1';
    }();
    if (pool_walk) {
        ReportPool();
        if (eden_guest_memory_waits) {
            unsigned long long won = 0, lost = 0, refused = 0;
            eden_guest_memory_waits(&won, &lost, &refused);
            std::fprintf(stderr, "EDEN_GUEST_WAITS got a block %llu, waited 2 s for nothing %llu, writes refused %llu\n",
                         won, lost, refused);
        }
    }
}
// EDEN_UP_RADV_OOM_LEND: the driver's out-of-memory callbacks while it is on (eden_radv_out_of_memory,
// any driver thread under the callback's mutex): how many, how much the guest reserve lent them, and how
// many ended with at least the asked size rounded up to 2 MiB given back (covered) or not (short).
std::atomic<unsigned long long> radv_oom_callbacks{0}, radv_oom_lent{0}, radv_oom_covered{0}, radv_oom_short{0};
// Their line every 10 s while the switch is on, from RefreshFreeMemory (the GPU thread; one caller
// writes it, the time is taken by a compare-exchange).
static void ReportRadvOomLend(long long now, long long lend) {
    static std::atomic<long long> reported_ns{0};
    long long last = reported_ns.load(std::memory_order_relaxed);
    if (last == 0) {
        reported_ns.compare_exchange_strong(last, now, std::memory_order_relaxed);
        return;
    }
    if (now - last < 10'000'000'000 || !reported_ns.compare_exchange_strong(last, now, std::memory_order_relaxed))
        return;
    unsigned long long guest[11] = {};
    if (eden_guest_reserve_lend_stats) eden_guest_reserve_lend_stats(guest, 11);
    std::fprintf(stderr,
                 "EDEN_UP_RADV_OOM_LEND %lld blocks under the floor: %llu callbacks (%llu covered, %llu short), "
                 "%llu MiB lent (%llu lends, %llu blocks beside the next, %llu refused while a guest write waited, "
                 "%llu found nothing above the lowered floor); reclaim %llu scans, %llu skipped in the same tick, "
                 "%llu verified (%llu would have missed cold slots), %llu over the candidates; reserve %llu MiB\n",
                 lend, radv_oom_callbacks.load(std::memory_order_relaxed),
                 radv_oom_covered.load(std::memory_order_relaxed), radv_oom_short.load(std::memory_order_relaxed),
                 radv_oom_lent.load(std::memory_order_relaxed) >> 20, guest[0], guest[2], guest[3], guest[4], guest[5],
                 guest[6], guest[7], guest[8], guest[9], guest[10] * 2);
}
// EDEN_UP_HEAP_HOLD_AT_LOAD (heap_arenas.inc): from RefreshFreeMemory while it is on, every 10 s when a
// count moved or a hold runs. holds = holds begun (an area change: the texture cache's AreaEvict
// window); left = seconds the one running has left; trims kept = give-backs it refused; spare = MiB held
// ready now; taken = spares committed (MiB); refused = takes the pool said no to; served = growths the
// pool refused and the spare made (bytes asked by the requests that grew), of them in a hold.
static void ReportHeapHold(long long now) {
    static std::atomic<long long> reported_ns{0};
    static std::atomic<unsigned long long> reported_sum{0};
    long long last = reported_ns.load(std::memory_order_relaxed);
    if (last != 0 && now - last < 10'000'000'000) return;
    unsigned long long v[14] = {};
    if (eden_heap_hold_stats(v, 14) == 0) return;
    const unsigned long long sum = v[2] + v[3] + v[4] + v[5] + v[7] + v[8] + v[12] + v[13];
    if (last != 0 && sum == reported_sum.load(std::memory_order_relaxed) && v[1] == 0) return;
    if (!reported_ns.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
    reported_sum.store(sum, std::memory_order_relaxed);
    std::fprintf(stderr,
                 "EDEN_UP_HEAP_HOLD %llu s: holds %llu (%llu begun by an area burst, the rest by a driver "
                 "refusal; running now: %llu s left), trims kept %llu; spare %llu MiB ready, taken %llu (%llu MiB), "
                 "refused by the pool %llu; served %llu growths the pool refused (%llu KiB asked), %llu of them "
                 "in a hold; capped at 3 x %llu s %llu; trims let through in a hold (game, JIT, operator new) %llu\n",
                 v[0], v[2], v[11], v[1], v[3], v[4], v[5], v[6], v[7], v[8], v[9] >> 10, v[10], v[0], v[12], v[13]);
}
// EDEN_UP_NEW_WAIT_MS (headless/memory_wait.h, main.cpp's new handler): from RefreshFreeMemory (every
// 100 ms at most, on the GPU thread) while it is on. Its counters every 10 s, when one has moved. While
// the GPU thread itself is the one waiting this line pauses too; the handler's own start and gave-up
// lines still come. "Not given up" = begun - gave up: served, or still going (waiting now).
static void ReportNewWait(long long now) {
    namespace MemoryWait = Eden::MemoryWait;
    auto& c = MemoryWait::counters;
    static std::atomic<long long> reported_ns{0};
    static std::atomic<unsigned long long> reported_sum{0};
    long long last = reported_ns.load(std::memory_order_relaxed);
    if (last == 0) {
        reported_ns.compare_exchange_strong(last, now, std::memory_order_relaxed);
        return;
    }
    if (now - last < 10'000'000'000) return;
    const unsigned long long begun = c.begun.load(std::memory_order_relaxed);
    const unsigned long long gave_up = c.gave_up.load(std::memory_order_relaxed);
    const unsigned long long steps = c.steps.load(std::memory_order_relaxed);
    const unsigned long long cooldown = c.cooldown.load(std::memory_order_relaxed);
    const unsigned long long too_big = c.too_big.load(std::memory_order_relaxed);
    const unsigned long long at_once = c.at_once.load(std::memory_order_relaxed);
    const unsigned long long builder = c.builder.load(std::memory_order_relaxed);  // 0 unless EDEN_UP_NEW_WAIT_BUILDER_SKIP
    const unsigned long long sum = begun + gave_up + steps + cooldown + too_big + at_once + builder;
    if (sum == reported_sum.load(std::memory_order_relaxed) ||
        !reported_ns.compare_exchange_strong(last, now, std::memory_order_relaxed))
        return;
    reported_sum.store(sum, std::memory_order_relaxed);
    // EDEN_UP_NEW_WAIT_BUILDER_SKIP=1: refusals on pipeline-building threads thrown with no handler wait.
    char builder_text[64] = "";
    if (MemoryWait::builder_skip.load(std::memory_order_relaxed))
        std::snprintf(builder_text, sizeof(builder_text), ", pipeline-building thread %llu", builder);
    std::fprintf(stderr,
                 "EDEN_UP_NEW_WAIT limit=%u ms waits begun %llu, gave up %llu, not given up %llu (other size within "
                 "60 s %llu); 20 ms steps %llu (%llu ms asleep in all), heap trims %llu, longest %u ms asleep, waiting "
                 "now %u; thrown at once: same size again within 60 s %llu, cooldown %llu, over 1 GiB %llu%s\n",
                 MemoryWait::new_wait_ms.load(std::memory_order_relaxed), begun, gave_up,
                 begun >= gave_up ? begun - gave_up : 0ULL, c.other_size.load(std::memory_order_relaxed), steps,
                 steps * 20, c.trims.load(std::memory_order_relaxed), c.longest_ms.load(std::memory_order_relaxed),
                 c.waiting.load(std::memory_order_relaxed), at_once, cooldown, too_big, builder_text);
}
// EDEN_UP_JIT_COMMIT_AHEAD and its test (src/memory_pages.cpp): from RefreshFreeMemory (every 100 ms at
// most, on the GPU thread) while either is set. Its counters every 10 s, when one has moved. "Mapped ahead,
// not yet written" is memory the code caches hold before they need it; "?" when a cache was mapping.
static void ReportJitAhead(long long now) {
    constexpr std::size_t kValues = 19; // 14 counts, then held now, held most, test left, mark, room
    unsigned long long v[kValues] = {};
    static std::atomic<long long> reported_ns{0};
    static std::atomic<unsigned long long> reported_sum{0};
    long long last = reported_ns.load(std::memory_order_relaxed);
    if (last == 0) {
        reported_ns.compare_exchange_strong(last, now, std::memory_order_relaxed);
        return;
    }
    if (now - last < 10'000'000'000) return;
    const int mode = eden_jit_ahead_stats(v, kValues);
    unsigned long long sum = 0;
    for (std::size_t i = 0; i < 14; ++i) sum += v[i];
    if (sum == reported_sum.load(std::memory_order_relaxed) ||
        !reported_ns.compare_exchange_strong(last, now, std::memory_order_relaxed))
        return;
    reported_sum.store(sum, std::memory_order_relaxed);
    char held_now[32];
    if (v[14] == ~0ULL)
        std::snprintf(held_now, sizeof(held_now), "?");
    else
        std::snprintf(held_now, sizeof(held_now), "%llu", v[14] >> 20);
    std::fprintf(stderr,
                 "EDEN_UP_JIT_COMMIT_AHEAD mode=%d mark=%llu%% room=%llu MiB: marks passed %llu; mapped ahead %llu "
                 "blocks (%llu MiB), %llu whole steps; count only, would have mapped %llu steps; no room %llu "
                 "(largest free under the room) %llu (reserve not full) %llu (a write waiting) %llu (pool not "
                 "known); pool refused %llu, lock busy %llu; steps taken at the need %llu (%llu with no room since "
                 "the mark or no mark seen, %llu finished from blocks mapped ahead); mapped ahead, not yet "
                 "written: %s MiB now, %llu MiB most; test refusals %llu, left %llu\n",
                 mode, v[17], v[18] >> 20, v[0], v[2], v[2] * 2, v[3], v[1], v[4], v[5], v[6], v[7], v[8], v[9],
                 v[10], v[11], v[12], held_now, v[15] >> 20, v[13], v[16]);
}
// EDEN_UP_JIT_COMMIT_RECLAIM (headless/memory_wait.h, jit-allocator.h): from RefreshFreeMemory (every
// 100 ms at most, on the GPU thread) while the mode is not 0. Its counters every 10 s, when one has moved.
// The waits' own lines (served / ran out / cleared) come from the waiting thread.
static void ReportJitReclaim(long long now) {
    namespace MemoryWait = Eden::MemoryWait;
    auto& c = MemoryWait::jit_reclaim;
    static std::atomic<long long> reported_ns{0};
    static std::atomic<unsigned long long> reported_sum{0};
    long long last = reported_ns.load(std::memory_order_relaxed);
    if (last == 0) {
        reported_ns.compare_exchange_strong(last, now, std::memory_order_relaxed);
        return;
    }
    if (now - last < 10'000'000'000) return;
    const unsigned long long waits = c.waits.load(std::memory_order_relaxed);
    const unsigned long long reclaims = c.reclaims.load(std::memory_order_relaxed);
    const unsigned long long busy = c.reclaim_busy.load(std::memory_order_relaxed);
    const unsigned long long sum = waits + reclaims + busy;
    if (sum == reported_sum.load(std::memory_order_relaxed) ||
        !reported_ns.compare_exchange_strong(last, now, std::memory_order_relaxed))
        return;
    reported_sum.store(sum, std::memory_order_relaxed);
    std::fprintf(stderr,
                 "EDEN_UP_JIT_COMMIT_RECLAIM mode=%u waits %llu: served at once after a reclaim or trim %llu, by a "
                 "20 ms retry %llu, cleared instead of ending %llu, ended by another core's clear %llu, ended the "
                 "session %llu (clear capped %llu, mode 1 would clear %llu); reclaims %llu (scanned %llu, skipped "
                 "%llu, another core reclaiming %llu) gave %llu MiB, heap trims gave %llu MiB; longest wait %u ms; "
                 "clears so far %llu\n",
                 MemoryWait::jit_reclaim_mode.load(std::memory_order_relaxed), waits,
                 c.served_at_once.load(std::memory_order_relaxed), c.served_later.load(std::memory_order_relaxed),
                 c.cleared.load(std::memory_order_relaxed), c.after_clear.load(std::memory_order_relaxed),
                 c.ended.load(std::memory_order_relaxed), c.clear_capped.load(std::memory_order_relaxed),
                 c.would_clear.load(std::memory_order_relaxed), reclaims,
                 c.reclaim_scans.load(std::memory_order_relaxed), c.reclaim_skipped.load(std::memory_order_relaxed), busy,
                 c.reclaim_bytes.load(std::memory_order_relaxed) >> 20, c.trim_bytes.load(std::memory_order_relaxed) >> 20,
                 c.longest_ms.load(std::memory_order_relaxed), MemoryWait::jit_clears.load(std::memory_order_relaxed));
}
// EDEN_UP_HEAP_STEAL (headless/heap_arenas.inc, eden_heap_steal): from RefreshFreeMemory (every 100 ms
// at most) while it is on. Its counters every 10 s, when one has moved.
// EDEN_UP_HEAP_STEAL_TABLE=1 (off unless set; with EDEN_MEMORY_REPORT_THREAD=1, else no table: it asks
// every arena, too long for the GPU thread): at the first request about to be refused, and then once in
// 30 s at most, the heap's table is queued for the MemoryReport thread. Its own switch because the table
// measures room by taking it (eden_heap_room) and holds the creation lock record by record, so it can
// refuse other threads at the brink: keep it off in both arms of an off-against-count comparison.
static bool HeapStealTable() {
    static const bool on = [] {
        const char* const value = std::getenv("EDEN_UP_HEAP_STEAL_TABLE");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }();
    return on;
}
static void ReportHeapSteal(long long now, int mode) {
    constexpr unsigned kStats = 29;
    unsigned long long v[kStats] = {};
    if (eden_heap_steal_stats(v, kStats) < kStats) return;
    static std::atomic<unsigned long long> tabled_triggers{0};
    static std::atomic<long long> tabled_ns{0};
    if (HeapStealTable() && v[0] != tabled_triggers.load(std::memory_order_relaxed)) {
        tabled_triggers.store(v[0], std::memory_order_relaxed);
        long long last = tabled_ns.load(std::memory_order_relaxed);
        if (OffThreadReportsEnabled() && (last == 0 || now - last >= 30'000'000'000) &&
            tabled_ns.compare_exchange_strong(last, now, std::memory_order_relaxed))
            QueueOffThreadReport(false, false, {}, true);
    }
    static std::atomic<long long> reported_ns{0};
    static std::atomic<unsigned long long> reported_sum{0};
    long long last = reported_ns.load(std::memory_order_relaxed);
    if (last == 0) {
        reported_ns.compare_exchange_strong(last, now, std::memory_order_relaxed);
        return;
    }
    if (now - last < 10'000'000'000) return;
    // Moved: the triggers, what was served or missed, the test's refusals, and the v2 counters.
    unsigned long long sum = 0;
    for (unsigned i = 0; i <= 16; ++i) sum += v[i];
    sum += v[24] + v[25] + v[26] + v[27];
    if (sum == reported_sum.load(std::memory_order_relaxed) ||
        !reported_ns.compare_exchange_strong(last, now, std::memory_order_relaxed))
        return;
    reported_sum.store(sum, std::memory_order_relaxed);
    char fewest[24] = "-";  // no scan yet
    if (v[20] != 0) std::snprintf(fewest, sizeof(fewest), "%llu", v[20] - 1);
    std::fprintf(stderr,
                 "EDEN_UP_HEAP_STEAL mode=%d triggers %llu at a refusal / %llu before a growth; scans %llu; served %llu "
                 "(last arena %llu, empty %llu, partly used %llu; %llu MiB; from spares %llu, kept on the asking thread's "
                 "record %llu), moved to the asking thread %llu (kept as a record's first %llu), none found %llu; last "
                 "arena dropped, a thread is on it %llu; not asked: too big %llu, lock held by this thread %llu, lock "
                 "busy after the wait %llu, lock busy without a wait %llu, after a miss of that size or less %llu, count "
                 "mode within 100 ms %llu; count mode would have served %llu of %llu scans; at the last scan %llu empty "
                 "arenas no thread is on (fewest %s), %llu partly used, %llu spaces of idle arenas only (%llu MiB), %llu "
                 "looked up the slow way; time %llu ms, longest %llu us; test refusals %llu\n",
                 mode, v[0], v[1], v[2], v[3] + v[4] + v[5], v[3], v[4], v[5], v[6] >> 20, v[7], v[26], v[8], v[9],
                 v[10], v[27], v[11], v[12], v[13], v[25], v[14], v[15], v[16], v[2], v[19], fewest, v[21], v[22],
                 v[23], v[28], v[17] / 1000, v[18], v[24]);
}
// The largest free block of direct memory is what the next graphics allocation needs. Looked at
// again every 100 ms; called on the GPU thread by the texture collector (KeepDirtyTextures).
static std::atomic<unsigned long long> largest_free_block{0};
static void RefreshFreeMemory() {
    static std::atomic<long long> checked_ns{0};
    const long long now = NowNs();
    if (const long long last = checked_ns.load(std::memory_order_relaxed); last != 0 && now - last < 100'000'000)
        return;
    checked_ns.store(now, std::memory_order_relaxed);
    std::int64_t start = 0;
    std::size_t largest = 0;
    const std::int64_t total = sceKernelGetDirectMemorySize();
    const bool known = total > 0 && sceKernelAvailableDirectMemorySize(0, total, 0x4000, &start, &largest) == 0;
    std::size_t held = 0, target = 0;
    if (eden_guest_memory_reserve) eden_guest_memory_reserve(&held, &target);
    const PoolView view = ViewOfPool(known, largest, held, target);
    // The pool itself, not view.short_memory: that turns true when the reserve is down to half, and
    // would keep the keeper from ever filling it again. Short under 1 GiB and over only at 1.5 GiB:
    // filling the reserve takes up to 512 MiB, and the largest block must not fall under the mark again.
    static std::atomic<bool> pool_short{false};
    if (known) pool_short.store(largest < kShortMemory ||
                                (pool_short.load(std::memory_order_relaxed) && largest < kShortMemory + kShortMemory / 2),
                                std::memory_order_relaxed);
    if (eden_guest_memory_pool_short) eden_guest_memory_pool_short(pool_short.load(std::memory_order_relaxed));
    largest_free_block.store(known ? (view.free != 0 ? view.free : 1) : 0, std::memory_order_relaxed);
    graphics_memory_short.store(view.short_memory, std::memory_order_relaxed);
    // The pool, the guest's memory and the heap in stderr.log every 20 s, to see what holds it.
    static std::atomic<long long> summary_ns{0};
    if (now - summary_ns.load(std::memory_order_relaxed) >= 20'000'000'000) {
        summary_ns.store(now, std::memory_order_relaxed);
        // Its heap, ledger and kernel queries can block, and the GPU thread must not wait for those
        // (GrowingBlackBeans, issue 21). EDEN_MEMORY_REPORT_THREAD=1 writes it on the report thread.
        if (OffThreadReportsEnabled()) {
            QueueOffThreadReport(false, false, MemorySummaryNumbers{true, largest, view.short_memory});
        } else {
            WriteMemorySummary(largest, view.short_memory);
        }
    }
    // EDEN_UP_RADV_OOM_LEND: its counters every 10 s while it is on (and its first use, before the
    // driver's callback needs it).
    if (const long long lend = ::Common::LiveSwitch::Value<::Common::LiveSwitch::Id::RadvOomLend>(); lend != 0)
        ReportRadvOomLend(now, lend);
    // EDEN_UP_HEAP_STEAL: the table at the heap's brink, and its counters every 10 s.
    if (eden_heap_steal_mode_now && eden_heap_steal_stats)
        if (const int steal = eden_heap_steal_mode_now(); steal != 0) ReportHeapSteal(now, steal);
    // EDEN_UP_NEW_WAIT_MS: the new handler's waits, every 10 s while it is on.
    if (Eden::MemoryWait::new_wait_ms.load(std::memory_order_relaxed) != 0) ReportNewWait(now);
    // EDEN_UP_HEAP_HOLD_AT_LOAD: the spare again when the pool has room (both return at once when off).
    if (eden_heap_hold_stats) {
        if (known && eden_heap_hold_refill) eden_heap_hold_refill(largest);
        ReportHeapHold(now);
    }
    // EDEN_UP_JIT_COMMIT_AHEAD / EDEN_TEST_JIT_COMMIT_REFUSE: their counters every 10 s while either is set.
    if (eden_jit_ahead_active && eden_jit_ahead_stats && eden_jit_ahead_active() != 0) ReportJitAhead(now);
    // EDEN_UP_JIT_COMMIT_RECLAIM: the JIT code cache's waits, every 10 s while it is on.
    if (Eden::MemoryWait::jit_reclaim_mode.load(std::memory_order_relaxed) != 0) ReportJitReclaim(now);
    // The last burst of a session falls between two reports of the guest's memory (every 30 s):
    // who wrote it, while the pool's largest block is nearly gone (every 5 s at most).
    static std::atomic<long long> burst_ns{0};
    const bool burst_due = known && largest < (std::size_t{256} << 20) && eden_guest_memory_report &&
                           now - burst_ns.load(std::memory_order_relaxed) >= 5'000'000'000;
    if (burst_due) burst_ns.store(now, std::memory_order_relaxed);
    // The heap's empty spaces go back to the pool while it is short (every 2 s at most).
    static std::atomic<long long> trimmed_ns{0};
    const bool trim_due = view.short_memory && now - trimmed_ns.load(std::memory_order_relaxed) >= 2'000'000'000;
    if (trim_due) trimmed_ns.store(now, std::memory_order_relaxed);
    // EDEN_MEMORY_REPORT_THREAD=1 (env.txt): both run on their own thread. This function runs on
    // the GPU thread (the texture collector asks for the free memory), and the burst report walks
    // every slot of the guest's memory: with the pool short, run108 had a 120 to 134 ms frame in
    // nearly every 5 s window where the clean stretch had 44 to 52 ms, and the frame rate fell from
    // 27 to 19 to 24. Off: as before, on the calling thread.
    if (burst_due || trim_due) {
        if (OffThreadReportsEnabled()) {
            QueueOffThreadReport(burst_due, trim_due);
        } else {
            if (burst_due) eden_guest_memory_report();
            if (trim_due) TrimHeap("the pool is short");
        }
    }
}
// A write to the guest's memory waiting for a block (src/guest_memory_service.inc, WaitForBlock): the game's own
// memory comes first, so while one waits the caches see memory short and none free, and the texture collector
// evicts at its hardest until the write has its block.
// The same for a pipeline builder's heap request that waits for memory (EDEN_UP_HEAP_WAIT_MS): the
// driver's compiler cannot take a refusal, so the caches give way until it is served.
static bool GuestMemoryWanted() {
    return (eden_guest_memory_wanted && eden_guest_memory_wanted() != 0) || (eden_heap_wanted && eden_heap_wanted() != 0);
}
static bool GraphicsMemoryShort() {
    if (GuestMemoryWanted()) return true;
    RefreshFreeMemory();
    return graphics_memory_short.load(std::memory_order_relaxed);
}
// What the caches' "memory in use" is measured against (performance.h, graphics_memory_free).
static unsigned long long GraphicsMemoryFree() {
    if (GuestMemoryWanted()) return 1; // not 0: the caches read 0 as "not known"
    RefreshFreeMemory();
    return largest_free_block.load(std::memory_order_relaxed);
}
[[maybe_unused]] static const bool graphics_memory_probe_installed =
    (graphics_memory_probe.store(&GraphicsMemoryShort, std::memory_order_relaxed),
     graphics_memory_free.store(&GraphicsMemoryFree, std::memory_order_relaxed), true);
#endif

namespace {
std::mutex hle_mutex;
// Keyed by the service's name pointer (one per service object) and command id.
std::map<std::pair<const char*, unsigned>, std::pair<unsigned long long, unsigned long long>> hle_calls;
} // namespace

void RecordHle(const char* service, unsigned command, long long ns) {
    std::lock_guard lock(hle_mutex);
    auto& entry = hle_calls[{service, command}];
    ++entry.first;
    entry.second += static_cast<unsigned long long>(ns > 0 ? ns : 0);
}

void ReportHeap(const char* why) {
    if (!eden_heap_usage) return;
    std::size_t held = 0, in_use = 0, returned = 0;
    eden_heap_usage(&held, &in_use, &returned);
    unsigned large_blocks = 0;
    const std::size_t large = eden_heap_large_held ? eden_heap_large_held(&large_blocks) : 0;
    std::fprintf(stderr, "[EglantinePS5] heap: holds %zu MiB (%zu MiB of pieces, %zu MiB in %u large blocks), "
                         "%zu MiB of it in use; %zu MiB given back so far (%s)\n",
                 held >> 20, (held - large) >> 20, large >> 20, large_blocks, in_use >> 20, returned >> 20, why);
    // What a heap that gives whole free 2 MiB pages back could return now (heap_arenas.inc).
    if (eden_heap_page_usage) {
        std::size_t free_pages = 0, idle_arenas = 0, live = 0;
        eden_heap_page_usage(&free_pages, &idle_arenas, &live);
        std::fprintf(stderr, "EDEN_HEAP_PAGES %zu MiB of the pieces are whole 2 MiB pages with no block in use "
                             "(%zu MiB of them idle arenas); the blocks in use hold %zu MiB\n",
                     free_pages >> 20, idle_arenas >> 20, live >> 20);
    }
    if (eden_heap_sites_report) eden_heap_sites_report();
    if (eden_heap_growth_report) eden_heap_growth_report();
    // What holds the room, arena by arena: it asks every one of them, so at the first refusals
    // once in 30 s at most, and then once in 2 minutes, to keep the end of a session in the log.
    static std::atomic<long long> arenas_ns{0};
    static std::atomic<unsigned> arena_reports{0};
    const long long now = NowNs();
    long long last = arenas_ns.load(std::memory_order_relaxed);
    const long long gap = arena_reports.load(std::memory_order_relaxed) < 3 ? 30'000'000'000 : 120'000'000'000;
    if (eden_heap_arena_report && std::strstr(why, "refused") && (last == 0 || now - last >= gap) &&
        arenas_ns.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
        arena_reports.fetch_add(1, std::memory_order_relaxed);
        eden_heap_arena_report();
    }
}

std::size_t TrimHeap(const char* why) {
    if (!eden_heap_trim) return 0;
    const std::size_t returned = eden_heap_trim();
    if (returned != 0) {
        char text[160];
        std::snprintf(text, sizeof(text), "gave back %zu MiB: %s", returned >> 20, why);
        ReportHeap(text);
    }
    return returned;
}

// Refusals come in bursts, and a refused staging buffer is asked for twice: the first 32 reports at
// most ten a second, then one every 5 s, so the last moments of a long session are still reported.
bool RefusalReportDue() {
    static std::atomic<unsigned> reports{0};
    static std::atomic<long long> last_ns{0};
    const long long now = NowNs();
    const long long last = last_ns.load(std::memory_order_relaxed);
    const long long gap = reports.load(std::memory_order_relaxed) < 32 ? 100'000'000 : 5'000'000'000;
    if (last != 0 && now - last < gap) return false;
    last_ns.store(now, std::memory_order_relaxed);
    reports.fetch_add(1, std::memory_order_relaxed);
    return true;
}

extern "C" void eden_guest_memory_growth(void) __attribute__((weak));
void ReportPool() {
#ifdef PS5_NATIVE
    // All that is free: the largest block of a range, then of what lies on each side of it. This
    // needs only the call a release build already makes: sceKernelDirectMemoryQuery is used by
    // ReportGpuThread alone, which only a profile build calls, so a release build does not import it.
    const std::int64_t total = sceKernelGetDirectMemorySize();
    struct Range { std::int64_t from, to; };
    std::array<Range, 64> ranges;
    std::size_t pending = 0;
    if (total > 0) ranges[pending++] = {0, total};
    std::size_t free_bytes = 0, largest = 0, from_2_mib = 0, from_16_mib = 0;
    unsigned blocks = 0, asked = 0;
    bool cut_short = false;
    while (pending != 0) {
        const Range range = ranges[--pending];
        if (asked == 4096 || pending + 2 > ranges.size()) {
            cut_short = true;
            break;
        }
        ++asked;
        std::int64_t start = 0;
        std::size_t size = 0;
        if (sceKernelAvailableDirectMemorySize(range.from, range.to, 0x4000, &start, &size) != 0 || size == 0) continue;
        // Whether the kernel clips the block to the range is not documented here: keep it inside.
        const std::int64_t end = std::min(start + static_cast<std::int64_t>(size), range.to);
        start = std::max(start, range.from);
        if (end <= start) continue;
        size = static_cast<std::size_t>(end - start);
        ++blocks;
        free_bytes += size;
        largest = std::max(largest, size);
        if (size >= (std::size_t{2} << 20)) from_2_mib += size;
        if (size >= (std::size_t{16} << 20)) from_16_mib += size;
        // The smaller side is looked at first, which keeps the list short however the blocks lie.
        Range first{range.from, start}, second{end, range.to};
        if (first.to - first.from < second.to - second.from) std::swap(first, second);
        if (first.to - first.from >= 0x4000) ranges[pending++] = first;
        if (second.to - second.from >= 0x4000) ranges[pending++] = second;
    }
    std::size_t held = 0, target = 0;
    if (eden_guest_memory_reserve) eden_guest_memory_reserve(&held, &target);
    const auto kernel_memory = guest_kernel_memory.load(std::memory_order_relaxed);
    std::fprintf(stderr, "EDEN_REFUSED_POOL %lld MiB in all, %s%zu MiB free in %u blocks after %u questions: the largest "
                 "%zu MiB, %zu MiB in blocks of 2 MiB or more, %zu MiB of 16 MiB or more; the guest's reserve holds %zu of "
                 "%zu MiB, %u writes wait for a block; the kernel maps %llu MiB\n",
                 static_cast<long long>(total >> 20), cut_short ? "at least " : "", free_bytes >> 20, blocks, asked,
                 largest >> 20, from_2_mib >> 20, from_16_mib >> 20, held >> 20, target >> 20,
                 eden_guest_memory_wanted ? eden_guest_memory_wanted() : 0u, kernel_memory ? kernel_memory() >> 20 : 0ULL);
    ReportLedger();
#endif
    if (eden_guest_memory_growth) eden_guest_memory_growth();
}

extern "C" void eden_vma_probe(unsigned) __attribute__((weak));
void ReportMemoryPressure(unsigned seconds) {
    if (seconds >= 90 && seconds <= 270 && seconds % 60 == 30 && eden_vma_probe)
        eden_vma_probe(seconds);
    std::size_t tables = 0, table_span = 0, guest = 0, guest_span = 0;
    ::Common::SparseUsage(&table_span, &tables);
    ::Common::GuestMemoryUsage(&guest_span, &guest);
    unsigned blocks = 0;
    const auto large = eden_heap_large_held ? eden_heap_large_held(&blocks) : 0;
    std::printf("MKW_MEMORY seconds=%u tables=%zu table_span=%zu guest=%zu guest_span=%zu heap=%zu large=%zu large_blocks=%u",
        seconds, tables, table_span, guest, guest_span,
        eden_heap_committed ? eden_heap_committed() : std::size_t{0}, large, blocks);
#ifdef PS5_NATIVE
    const auto total = sceKernelGetDirectMemorySize();
    std::int64_t start = 0;
    std::size_t largest = 0;
    const auto status = total > 0 ? sceKernelAvailableDirectMemorySize(0, total, 0x4000, &start, &largest) : -1;
    std::printf(" direct_total=%lld largest_free=%zu query_status=%d", static_cast<long long>(total), largest, int(status));
#endif
    std::printf("\n");
    std::fflush(stdout);
}

void ReportGpuThread(unsigned frame) {
    static unsigned heap_reports = 0;
    if (++heap_reports % 6 == 1) ReportHeap("report");
    // Heap growth and arena use over the run (allocation failures abort the title).
    ps5_opengl_heap_snapshot("vulkan_report", frame);
    {
        // What the heap and Eden's large tables hold of the memory the CPU shares with the GPU:
        // the heap's pieces (they only grow), its large blocks alive now, and the tables' own
        // slots out of the address range they span.
        unsigned large_blocks = 0;
        const std::size_t large = eden_heap_large_held ? eden_heap_large_held(&large_blocks) : 0;
        std::size_t table_span = 0, table_held = 0;
        ::Common::SparseUsage(&table_span, &table_held);
        std::printf("EDEN_PERF_HEAP arenas=%u pieces=%zu large=%zu large_blocks=%u tables=%zu table_span=%zu\n",
                    eden_heap_arenas_created ? eden_heap_arenas_created() : 1u,
                    eden_heap_committed ? eden_heap_committed() : std::size_t{0}, large, large_blocks, table_held,
                    table_span);
    }
    {
        // Cumulative HLE handling time of every service command that has cost at least 1 ms.
        std::lock_guard lock(hle_mutex);
        for (const auto& [key, value] : hle_calls)
            if (value.second >= 1'000'000)
                std::printf("EDEN_DEV_HLE service=%s cmd=%u calls=%llu ns=%llu\n", key.first ? key.first : "?",
                            key.second, value.first, value.second);
    }
#ifdef PS5_NATIVE
    // Direct memory headroom (guest RAM, JIT, RADV and the caches share the pool).
    if (const std::int64_t total = sceKernelGetDirectMemorySize(); total > 0) {
        std::int64_t start = 0;
        std::size_t largest = 0;
        if (sceKernelAvailableDirectMemorySize(0, total, 0x4000, &start, &largest) == 0) {
            // All that is free, against the largest block: how far apart they are tells how much
            // of the pool is holes between allocations. -1: the regions could not be read.
            std::int64_t taken = 0, offset = 0;
            unsigned regions = 0;
            DirectMemoryRegion region{};
            while (offset < total && regions < 8192 &&
                   sceKernelDirectMemoryQuery(offset, 1, &region, sizeof(region)) == 0 && region.end > offset) {
                taken += region.end - region.start;
                offset = region.end;
                ++regions;
            }
            std::printf("EDEN_PERF_DIRECT total=%lld largest_free=%zu free=%lld regions=%u short=%d\n",
                        static_cast<long long>(total), largest,
                        regions != 0 ? static_cast<long long>(total - taken) : -1LL, regions,
                        int(graphics_memory_short.load(std::memory_order_relaxed)));
        }
    }
#endif
    const auto load = [](const Totals& totals, bool calls) {
        return (calls ? totals.calls : totals.nanoseconds).load(std::memory_order_relaxed);
    };
    // Cumulative totals; the analyzer takes deltas between consecutive reports.
    std::printf("EDEN_DEV_GPU frame=%u mono_ns=%lld cpu_ns=%lld idle_ns=%llu dispatch_calls=%llu "
                "dispatch_ns=%llu drain_calls=%llu drain_ns=%llu present_calls=%llu present_ns=%llu "
                "full_calls=%llu full_ns=%llu draws=%llu\n", frame, ClockNs(CLOCK_MONOTONIC),
                owner_cpu_clock_valid ? ClockNs(CLOCK_THREAD_CPUTIME_ID) : -static_cast<long long>(ENOTSUP),
                load(gpu_queue_wait, false), load(gpu_dispatch, true), load(gpu_dispatch, false),
                load(gpu_fence_drain, true), load(gpu_fence_drain, false),
                load(gpu_present_wait, true), load(gpu_present_wait, false),
                load(gpu_queue_full, true), load(gpu_queue_full, false), load(rasterizer_draw, true));
    // What the dispatch is made of: the draw path itself (RasterizerVulkan::PrepareDraw - both
    // cache locks, pipeline configure, dynamic state, the record) against the guest method
    // decoding that surrounds it, plus the dispatch's own three phases. A draw whose
    // EDEN_DEV_DRAW ns is close to EDEN_GPU dispatch_ns is being charged to the rasterizer; one
    // far below it is being charged to the dma pusher.
    std::printf("EDEN_DEV_DRAW calls=%llu ns=%llu\n", load(rasterizer_draw, true),
                load(rasterizer_draw, false));
    std::printf("EDEN_DEV_PUSH lock_ns=%llu push_ns=%llu dispatch_ns=%llu\n",
                load(gpu_sched_lock, false), load(gpu_dma_push, false),
                load(gpu_dma_dispatch, false));
    // Where that draw time went, in order. flush is FlushWork plus the memory flush; pipeline is
    // taking the current pipeline; lock_wait is the wait for both cache mutexes (the hold is what
    // the later phases measure); config is SetEngine plus Configure; state is UpdateDynamicStates
    // and the query, transform-feedback and counter calls; record is draw_func. unattributed is
    // the early-return tails the outer Timer charged but no phase marked, so it should be small,
    // and the _calls columns say how often each phase ran - record below flush is draws that bailed.
    std::printf("EDEN_DEV_RAST flush_ns=%llu pipeline_ns=%llu lock_wait_ns=%llu config_ns=%llu "
                "state_ns=%llu record_ns=%llu unattributed_ns=%llu flush_calls=%llu "
                "pipeline_calls=%llu lock_wait_calls=%llu config_calls=%llu state_calls=%llu "
                "record_calls=%llu\n",
                load(draw_flush, false), load(draw_pipeline, false),
                load(draw_lock_wait, false), load(draw_config, false), load(draw_state, false),
                load(draw_record, false),
                load(rasterizer_draw, false) - (load(draw_flush, false) + load(draw_pipeline, false) +
                                                load(draw_lock_wait, false) +
                                                load(draw_config, false) + load(draw_state, false) +
                                                load(draw_record, false)),
                load(draw_flush, true), load(draw_pipeline, true), load(draw_lock_wait, true),
                load(draw_config, true), load(draw_state, true), load(draw_record, true));
    // Where config's time went. sync is the sampler and image table sync plus the uniform-buffer
    // state; read is the guest-memory handle reads and the view fill; bind is the buffer binds;
    // desc is the descriptor queue, the pushes and the upload barrier; rt is render targets and
    // feedback; draw is ConfigureDraw's compare-and-record. These nest inside config, so
    // unattributed (config less their sum) is the entry clock, SetEngine, the configure_func
    // pointer call and any early return that left the tail unmarked - a negative value means the
    // marks themselves cost more than that.
    std::printf("EDEN_DEV_CFG sync_ns=%llu read_ns=%llu bind_ns=%llu desc_ns=%llu rt_ns=%llu "
                "draw_ns=%llu unattributed_ns=%llu sync_calls=%llu read_calls=%llu "
                "bind_calls=%llu desc_calls=%llu rt_calls=%llu draw_calls=%llu\n",
                load(cfg_sync, false), load(cfg_read, false), load(cfg_bind, false),
                load(cfg_desc, false), load(cfg_rt, false), load(cfg_draw, false),
                load(draw_config, false) - (load(cfg_sync, false) + load(cfg_read, false) +
                                            load(cfg_bind, false) + load(cfg_desc, false) +
                                            load(cfg_rt, false) + load(cfg_draw, false)),
                load(cfg_sync, true), load(cfg_read, true), load(cfg_bind, true),
                load(cfg_desc, true), load(cfg_rt, true), load(cfg_draw, true));
    // Where cfg_bind's time went: tbo is the texture-buffer loop over the enabled stages, update
    // is the index, vertex, transform-feedback and per-stage buffer updates, geometry is the host
    // binds after them. They nest inside cfg_bind, so unattributed (cfg_bind less their sum) is
    // the transform-feedback check, the two pointer-store statements and the clock itself.
    std::printf("EDEN_DEV_BIND tbo_ns=%llu update_ns=%llu geometry_ns=%llu "
                "unattributed_ns=%llu tbo_calls=%llu update_calls=%llu geometry_calls=%llu\n",
                load(bind_tbo, false), load(bind_update, false), load(bind_geometry, false),
                load(cfg_bind, false) - (load(bind_tbo, false) + load(bind_update, false) +
                                         load(bind_geometry, false)),
                load(bind_tbo, true), load(bind_update, true), load(bind_geometry, true));
    // The four calls inside cfg_bind's geometry phase: the index buffer (or the quad substitute),
    // the vertex buffers, transform feedback, and the indirect draw buffers when there is one.
    // They nest inside cfg_bind's geometry, so unattributed is the clock and the two pointer
    // stores around it - not the SynchronizeBuffer calls, which are inside the phases themselves.
    std::printf("EDEN_DEV_GEO index_ns=%llu vertex_ns=%llu xf_ns=%llu indirect_ns=%llu "
                "unattributed_ns=%llu index_calls=%llu vertex_calls=%llu xf_calls=%llu "
                "indirect_calls=%llu\n",
                load(geo_index, false), load(geo_vertex, false), load(geo_xf, false),
                load(geo_indirect, false),
                load(bind_geometry, false) - (load(geo_index, false) + load(geo_vertex, false) +
                                              load(geo_xf, false) + load(geo_indirect, false)),
                load(geo_index, true), load(geo_vertex, true), load(geo_xf, true),
                load(geo_indirect, true));
    // The three inside cfg_bind's update phase: index, vertex and transform-feedback updates; the
    // six-stage uniform, storage and texture loop; and the indirect update when there is one.
    std::printf("EDEN_DEV_UPD state_ns=%llu stage_ns=%llu indirect_ns=%llu unattributed_ns=%llu "
                "state_calls=%llu stage_calls=%llu indirect_calls=%llu\n",
                load(upd_state, false), load(upd_stage, false), load(upd_indirect, false),
                load(bind_update, false) - (load(upd_state, false) + load(upd_stage, false) +
                                            load(upd_indirect, false)),
                load(upd_state, true), load(upd_stage, true), load(upd_indirect, true));
    // Texture-cache GC GPU drains since the last report. The fix these counters watch is the
    // prefetch lookahead (vulkan_gc_downloads.inc): batched drains should cover many images and
    // unbatched ones should be rare, as each unbatched drain is a full pipeline wait on the GPU
    // thread for one eviction.
    std::printf("EDEN_DEV_GC_DRAIN batched_calls=%llu batched_ns=%llu batched_images=%llu "
                "unbatched_calls=%llu unbatched_ns=%llu\n",
                load(gc_batched_drain, true), load(gc_batched_drain, false),
                gc_batched_drain.requested_bytes.load(std::memory_order_relaxed),
                load(gc_unbatched_drain, true), load(gc_unbatched_drain, false));
    std::printf("EDEN_DEV_GUEST cpu_write_calls=%llu cpu_write_ns=%llu cpu_read_calls=%llu cpu_read_ns=%llu "
                "sync_calls=%llu sync_ns=%llu dequeue_calls=%llu dequeue_ns=%llu",
                load(guest_cpu_write, true), load(guest_cpu_write, false),
                load(guest_cpu_read, true), load(guest_cpu_read, false),
                load(guest_sync_wait, true), load(guest_sync_wait, false),
                load(guest_dequeue_wait, true), load(guest_dequeue_wait, false));
    std::printf(" ipc_calls=%llu ipc_ns=%llu", load(guest_ipc_wait, true), load(guest_ipc_wait, false));
    std::printf(" cache_lock_contended=%llu cache_lock_blocked=%llu",
                cache_lock_contended.load(std::memory_order_relaxed), cache_lock_blocked.load(std::memory_order_relaxed));
    std::printf(" fs_file_calls=%llu fs_file_ns=%llu fs_file_bytes=%llu fs_storage_calls=%llu fs_storage_ns=%llu "
                "fs_storage_bytes=%llu",
                load(guest_fs_file, true), load(guest_fs_file, false),
                guest_fs_file_bytes.load(std::memory_order_relaxed),
                load(guest_fs_storage, true), load(guest_fs_storage, false),
                guest_fs_storage_bytes.load(std::memory_order_relaxed));
    for (unsigned core = 0; core < jit_callbacks.size(); ++core)
        std::printf(" r%u=%llu w%u=%llu x%u=%llu", core, jit_callbacks[core].reads.load(std::memory_order_relaxed),
                    core, jit_callbacks[core].writes.load(std::memory_order_relaxed),
                    core, jit_callbacks[core].exclusive_writes.load(std::memory_order_relaxed));
    for (unsigned core = 0; core < 3; ++core)
        std::printf(" idle%u=%llu/%llu/%llu", core, guest_idle[core].calls.load(std::memory_order_relaxed),
                    guest_idle[core].nanoseconds.load(std::memory_order_relaxed) / 1000000,
                    guest_idle[core].sleeps.load(std::memory_order_relaxed));
    std::printf(" cond=");
    for (unsigned slot = 0; slot < render_conditions.size(); ++slot)
        std::printf("%s%llu", slot ? "," : "", render_conditions[slot].load(std::memory_order_relaxed));
    std::printf("\n");
    for (unsigned slot = 0; slot < ipc_threads.size(); ++slot) {
        const auto calls = ipc_threads[slot].calls.load(std::memory_order_relaxed);
        if (calls == 0) continue;
        std::printf("EDEN_DEV_IPC_THREAD slot=%u tid=%llu name=%.23s calls=%llu ns=%llu\n", slot,
                    ipc_threads[slot].tid.load(std::memory_order_relaxed), ipc_threads[slot].name, calls,
                    ipc_threads[slot].ns.load(std::memory_order_relaxed));
    }
    for (unsigned slot = 0; slot < wait_threads.size(); ++slot) {
        const auto calls = wait_threads[slot].calls.load(std::memory_order_relaxed);
        if (calls == 0) continue;
        const auto tid = wait_threads[slot].tid.load(std::memory_order_relaxed);
        const char* name = "";
        for (unsigned islot = 0; islot < ipc_thread_slots; ++islot)
            if (ipc_threads[islot].tid.load(std::memory_order_relaxed) == tid) {
                name = ipc_threads[islot].name;
                break;
            }
        // rsn= is this slot's own wait_ns per ThreadWaitReasonForDebugging, same index order as
        // the histogram line, so a thread's split reads without joining anything by hand.
        std::printf("EDEN_DEV_WAIT_THREAD tid=%llu name=%.23s calls=%llu ns=%llu waker=%llu rsn=", tid,
                    name, calls, wait_threads[slot].ns.load(std::memory_order_relaxed),
                    wait_threads[slot].waker.load(std::memory_order_relaxed));
        for (unsigned reason = 0; reason < wait_reason_slots; ++reason)
            std::printf("%s%llu", reason ? "," : "",
                        wait_threads[slot].reason_ns[reason].load(std::memory_order_relaxed));
        std::printf("\n");
    }
    // tid is the slot index here (performance.h indexes the dispatch table directly by tid).
    for (unsigned tid = 0; tid < dispatch_slots; ++tid) {
        const auto dcalls = dispatch_calls[tid].load(std::memory_order_relaxed);
        if (dcalls == 0) continue;
        std::printf("EDEN_DEV_DISPATCH tid=%u calls=%llu ns=%llu\n", tid, dcalls,
                    dispatch_ns[tid].load(std::memory_order_relaxed));
    }
    std::printf("EDEN_DEV_DISPATCH overflow_calls=%llu\n",
                dispatch_overflow_calls.load(std::memory_order_relaxed));
    // Reason indices are ThreadWaitReasonForDebugging (k_thread.h); named here so a log reads
    // without the source.
    std::printf("EDEN_DEV_WAIT reasons=none,sleep,ipc,sync,condvar,arbitration,suspended,other "
                "overflow_calls=%llu calls=",
                wait_overflow_calls.load(std::memory_order_relaxed));
    for (unsigned reason = 0; reason < wait_reason_slots; ++reason)
        std::printf("%s%llu", reason ? "," : "", wait_reason_calls[reason].load(std::memory_order_relaxed));
    std::printf(" ns=");
    for (unsigned reason = 0; reason < wait_reason_slots; ++reason)
        std::printf("%s%llu", reason ? "," : "", wait_reason_ns[reason].load(std::memory_order_relaxed));
    std::printf("\n");
    std::printf("EDEN_DEV_IPCHIST lt50us=%llu lt500us=%llu lt5ms=%llu ge5ms=%llu overflow_calls=%llu "
                "overflow_ns=%llu\n",
                ipc_hist[0].load(std::memory_order_relaxed), ipc_hist[1].load(std::memory_order_relaxed),
                ipc_hist[2].load(std::memory_order_relaxed), ipc_hist[3].load(std::memory_order_relaxed),
                load(ipc_thread_overflow, true), load(ipc_thread_overflow, false));
    const auto window = Eden::Fastmem::WindowStats();
    std::printf("EDEN_FASTMEM window=%llx pages=%llu chunks=%llu direct_reads=%llu direct_writes=%llu "
                "faults=%llu maps=%llu unmaps=%llu protects=%llu kernel_calls=%llu kernel_ns=%llu failures=%llu\n",
                static_cast<unsigned long long>(window.window), static_cast<unsigned long long>(window.mapped_pages),
                static_cast<unsigned long long>(window.aliased_chunks),
                static_cast<unsigned long long>(window.direct_reads),
                static_cast<unsigned long long>(window.direct_writes),
                static_cast<unsigned long long>(Eden::Fastmem::Faults()),
                static_cast<unsigned long long>(window.map_calls), static_cast<unsigned long long>(window.unmap_calls),
                static_cast<unsigned long long>(window.protect_calls),
                static_cast<unsigned long long>(window.kernel_calls), static_cast<unsigned long long>(window.kernel_ns),
                static_cast<unsigned long long>(window.failures));
    // Guest cores record their owner CPU clocks at their next JIT exit.
    Snapshot();
}

namespace {
// 0841: the background pass beside the game (the core's StartShaderSet, its thread named
// "ShaderSet": 1 build in flight, duty 50%, cap 5000) is held while the game starts up and the
// title movie plays, when EDEN_UP_SHADER_SET_PREPASS=1. In run1387 (T162, 13 Nov), after a pass
// that ended at its time limit, it ran beside the start-up: 2-3 minutes to the first picture
// instead of under one, and about 18 fps in the title movie. The thread waits here, at its first
// line, before it has read or built anything; nothing waits for it but the cache's destructor,
// which main.cpp releases first (ReleaseBackgroundHold). EDEN_UP_PREPASS_BACKGROUND_AFTER_S: the
// seconds after the guest starts that it waits (unset: 0 since the T166 review; 600 was the
// writer's default, not tried); 0: no wait (as before); "never": it
// stays off for the session.
// (Its state is above NoteSessionStart.)

long long BackgroundAfterSeconds() {
    if (!Common::PeSwitch("EDEN_UP_SHADER_SET_PREPASS")) return 0;
    // T166 review: unset is 0 (no wait, as before 0841): the hold is new behaviour, so it is asked
    // for (it was 600 by default with the prepass on). A value that is not a number of seconds or
    // "never" is said and taken as 0, so a typo never keeps the pass off.
    const char* const value = std::getenv("EDEN_UP_PREPASS_BACKGROUND_AFTER_S");
    if (value == nullptr || value[0] == '\0') return 0;
    if (std::strcmp(value, "never") == 0) return -1;
    char* end = nullptr;
    const long long seconds = std::strtoll(value, &end, 10);
    if (end == value || *end != '\0' || seconds < 0) {
        std::printf("EDEN_UP_PREPASS_BACKGROUND_AFTER_S=%s is not a number of seconds or \"never\"; "
                    "ignored (no wait)\n", value);
        return 0;
    }
    return std::min<long long>(seconds, 24LL * 3600);
}

void HoldBackgroundPass() {
    const long long after = BackgroundAfterSeconds();
    if (after == 0) return;
    const auto began = std::chrono::steady_clock::now();
    if (after > 0)
        std::printf("EDEN_UP_PREPASS_BACKGROUND_AFTER_S: the background shader set pass waits until %lld s "
                    "after the game starts\n", after);
    else
        std::printf("EDEN_UP_PREPASS_BACKGROUND_AFTER_S=never: the background shader set pass stays off this "
                    "session\n");
    std::fflush(stdout);
    const char* why = "released at the session's end";
    {
        std::unique_lock lock{hold_mutex};
        while (!hold_released) {
            if (after > 0 && hold_guest_start &&
                std::chrono::steady_clock::now() >= *hold_guest_start + std::chrono::seconds{after}) {
                why = "its time came";
                break;
            }
            hold_wake.wait_for(lock, std::chrono::seconds{1});
        }
    }
    const auto waited = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - began);
    std::printf("EDEN_UP_PREPASS_BACKGROUND_AFTER_S: the background shader set pass goes on after %lld s (%s)\n",
                static_cast<long long>(waited.count()), why);
    std::fflush(stdout);
}
} // namespace

void ReleaseBackgroundHold() {
    {
        std::scoped_lock lock{hold_mutex};
        hold_released = true;
    }
    hold_wake.notify_all();
}

void RegisterWorker(const char* name) {
    // 0840 (T166 review): a thread named here is not a widened before-play thread whose id it may reuse.
#ifdef PS5_NATIVE
    ForgetPrepassThread();
    MkwNativeSample::Register(name);
#endif
    Eden::Crash::NameThread(name);  // a crash report names the thread
    if (eden_heap_name_thread) eden_heap_name_thread(name);  // and the heap's reports
#if defined(EDEN_DEV_PROFILE) && defined(PS5_NATIVE)
    Eden::Stall::NoteThread(name);
#endif
    for (unsigned i = 0; i < names.size(); ++i) {
        if (std::strcmp(name, names[i])) continue;
        Worker worker;
        worker.thread = pthread_self();
#ifdef EDEN_DEV_PROFILE
        if (pc_sampling && (i == 4 || i == pc_sample_core.load())) {
            sigset_t mask;
            sigemptyset(&mask);
            sigaddset(&mask, SIGUSR2);
            if (pthread_sigmask(SIG_UNBLOCK, &mask, nullptr))
                throw std::runtime_error("Cannot enable development PC sampler");
            if (i == pc_sample_core.load()) {
                core_sample_thread = pthread_self();
                core_sample_ready.store(true, std::memory_order_release);
                // dev-settings pc_fast=on: ~500 Hz from 45 s after registration instead of the
                // frontend's 20 Hz polls (from 90 s), for compilation bursts at load. Signals during
                // the boot's first seconds hung it (run radv-zb5-20260930-015220).
                if (pc_fast.load()) {
                    std::thread([] {
                        std::this_thread::sleep_for(std::chrono::seconds(45));
                        while (core_pc_count.load() < sampled_core_pcs.size() &&
                               pthread_kill(core_sample_thread, SIGUSR2) == 0)
                            std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    }).detach();
                }
            }
        }
#endif
        worker.clock_error = CurrentCpuClock(&worker.clock);
        worker.wall_ns = Common::g_wall_clock.GetTimeNS().count();
        worker.mono_ns = ClockNs(CLOCK_MONOTONIC);
#ifdef PS5_NATIVE
        cpuset_t affinity{};
        const int result = cpuset_getaffinity(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1,
                                             sizeof(unsigned long long), &affinity);
#else
        cpu_set_t affinity{};
        const int result = sched_getaffinity(0, sizeof(affinity), &affinity);
#endif
#ifdef PS5_NATIVE
        if (result == 0) PinWorker(i, affinity);
#endif
        worker.affinity_error = result == 0 ? 0 : errno;
        if (!worker.affinity_error) {
            for (unsigned cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
                if (!CPU_ISSET(cpu, &affinity)) continue;
                ++worker.allowed;
                if (cpu < 64) worker.mask |= 1ULL << cpu;
            }
        }
        worker.registered = true;
        std::lock_guard lock(workers_mutex);
        workers[i] = worker;
        return;
    }
#ifdef PS5_NATIVE
    PlaceSecondary(name);
#endif
    // 0841: the background shader set pass (its first line) waits while the game starts.
    if (std::strcmp(name, "ShaderSet") == 0) HoldBackgroundPass();
}

void SampleCpu(unsigned core, unsigned long long thread, unsigned long long pc, unsigned svc, unsigned fpcr) {
    const auto epoch = sample_epoch.load(std::memory_order_relaxed);
    if (sampled_epoch.at(core) == epoch) return;
    sampled_epoch[core] = epoch;
    const CpuSample sample{epoch, ClockNs(CLOCK_MONOTONIC),
        owner_cpu_clock_valid ? ClockNs(CLOCK_THREAD_CPUTIME_ID) : -ENOTSUP, thread, pc, svc, fpcr,
        compilation[core].calls.load(std::memory_order_relaxed),
        compilation[core].nanoseconds.load(std::memory_order_relaxed)};
    std::lock_guard lock(workers_mutex);
    cpu_samples[core] = sample;
}

void Snapshot() {
    // Request a fresh owner-written sample; report the last completed sample with
    // its own timestamp. An idle core may remain stale; never infer zero CPU use.
    sample_epoch.fetch_add(1, std::memory_order_relaxed);
    const auto mono = ClockNs(CLOCK_MONOTONIC);
    std::printf("EDEN_PERF_SAMPLE mono_ns=%lld process_cpu_ns=%lld wall_ns=%lld\n",
                mono, cpu_clocks_valid ? ClockNs(process_clock) : -static_cast<long long>(ENOTSUP),
                static_cast<long long>(Common::g_wall_clock.GetTimeNS().count()));
    std::lock_guard lock(workers_mutex);
#ifdef EDEN_DEV_PROFILE
    std::map<uintptr_t, unsigned> counts;
    const unsigned end = pc_count.load(std::memory_order_acquire);
#if defined(EDEN_DEV_WAIT_CALLERS) && defined(PS5_NATIVE)
    std::map<std::array<uintptr_t, 9>, unsigned> caller_counts;
#endif
    while (pc_reported < end) {
        const unsigned index = pc_reported++;
        ++counts[sampled_pcs[index]];
#if defined(EDEN_DEV_WAIT_CALLERS) && defined(PS5_NATIVE)
        std::array<uintptr_t, 9> key{sampled_pcs[index]};
        std::copy(sampled_callers[index].begin(), sampled_callers[index].end(), key.begin() + 1);
        ++caller_counts[key];
#endif
    }
#if defined(EDEN_DEV_WAIT_CALLERS) && defined(PS5_NATIVE)
    for (const auto& [chain, count] : caller_counts) {
        std::printf("EDEN_PERF_NATIVE_CALLERS mono_ns=%lld pc=%llx count=%u callers=", mono,
                    static_cast<unsigned long long>(chain[0]), count);
        for (unsigned i = 1; i < chain.size(); ++i)
            std::printf("%s%llx", i == 1 ? "" : ",", static_cast<unsigned long long>(chain[i]));
        std::printf("\n");
    }
#endif
    for (const auto& [pc, count] : counts)
        std::printf("EDEN_PERF_NATIVE_PC mono_ns=%lld pc=%llx count=%u\n", mono,
                    static_cast<unsigned long long>(pc), count);
    std::map<uintptr_t, unsigned> core_counts;
    const unsigned core_end = core_pc_count.load(std::memory_order_acquire);
    while (core_pc_reported < core_end) ++core_counts[sampled_core_pcs[core_pc_reported++]];
    for (const auto& [pc, count] : core_counts)
        std::printf("EDEN_PERF_CORE_PC mono_ns=%lld pc=%llx count=%u\n", mono,
                    static_cast<unsigned long long>(pc), count);
#if defined(EDEN_DEV_WAIT_CALLERS) && defined(PS5_NATIVE)
    std::map<std::array<unsigned long long, 9>, unsigned> core_caller_counts;
    const unsigned core_caller_end = core_caller_count.load(std::memory_order_acquire);
    while (core_caller_reported < core_caller_end) {
        const unsigned index = core_caller_reported++;
        std::array<unsigned long long, 9> key{static_cast<unsigned long long>(core_caller_pcs[index])};
        for (unsigned i = 0; i < 8; ++i)
            key[i + 1] = static_cast<unsigned long long>(sampled_core_callers[index][i]);
        ++core_caller_counts[key];
    }
    for (const auto& [chain, count] : core_caller_counts) {
        std::printf("EDEN_PERF_CORE_CALLERS mono_ns=%lld pc=%llx count=%u callers=", mono,
                    chain[0], count);
        for (unsigned i = 1; i < chain.size(); ++i)
            std::printf("%s%llx", i == 1 ? "" : ",", chain[i]);
        std::printf("\n");
    }
#endif
    // The same samples by guest block: location (A32 PC in the low word), block offset, count.
    if (const std::size_t blocks = core0_blocks ? core0_block_count.load(std::memory_order_acquire) : 0) {
        std::map<std::pair<unsigned long long, unsigned long long>, unsigned> guest_counts;
        for (const auto& [pc, count] : core_counts) {
            const JitBlock* begin = core0_blocks;
            const JitBlock* it = std::upper_bound(begin, begin + blocks, pc,
                                                  [](uintptr_t value, const JitBlock& block) { return value < block.entry; });
            if (it == begin) continue;
            --it;
            if (pc - it->entry < it->size) guest_counts[{it->location, pc - it->entry}] += count;
        }
        for (const auto& [key, count] : guest_counts)
            std::printf("EDEN_PERF_GUEST_PC mono_ns=%lld location=%llx host_offset=%llx count=%u\n", mono,
                        key.first, key.second, count);
        // P1: guest and host code of this window's six hottest blocks not dumped before.
        std::map<unsigned long long, unsigned> block_counts;
        for (const auto& [key, count] : guest_counts) block_counts[key.first] += count;
        std::vector<std::pair<unsigned, unsigned long long>> ranked;
        for (const auto& [location, count] : block_counts) ranked.emplace_back(count, location);
        std::sort(ranked.rbegin(), ranked.rend());
        static std::set<unsigned long long> dumped;
        unsigned printed = 0;
        for (const auto& [count, location] : ranked) {
            if (printed == 6) break;
            if (!dumped.insert(location).second) continue;
            ++printed;
            const JitBlock* block = nullptr;
            for (std::size_t i = blocks; i-- > 0;)
                if (core0_blocks[i].location == location) { block = &core0_blocks[i]; break; }
            std::string guest;
            const unsigned long long pc = location & 0xffffffffull;
            for (unsigned long long at = pc; at < pc + 256 && guest_read32; at += 4) {
                unsigned word = 0;
                if (!guest_read32(at, word)) break;
                char text[12];
                std::snprintf(text, sizeof(text), "%08x", word);
                guest += text;
            }
            std::printf("EDEN_PERF_BLOCK_GUEST location=%llx count=%u words=%s\n", location, count, guest.c_str());
            if (block) {
                std::string host;
                const auto* bytes = reinterpret_cast<const unsigned char*>(block->entry);
                for (unsigned long long i = 0; i < block->size && i < 4096; ++i) {
                    char text[4];
                    std::snprintf(text, sizeof(text), "%02x", bytes[i]);
                    host += text;
                }
                std::printf("EDEN_PERF_BLOCK_HOST location=%llx entry=%llx size=%llu bytes=%s\n", location,
                            static_cast<unsigned long long>(block->entry), block->size, host.c_str());
            }
        }
    }
#endif
    for (unsigned i = 0; i < workers.size(); ++i) {
        const auto& worker = workers[i];
        sched_param priority{};
        int policy = -1;
        const int priority_error = worker.registered ?
            pthread_getschedparam(worker.thread, &policy, &priority) : ENOENT;
        const auto cpu = !cpu_clocks_valid ? -static_cast<long long>(ENOTSUP) :
                        worker.clock_error ? -static_cast<long long>(worker.clock_error) :
                                             ClockNs(worker.clock);
        std::printf("EDEN_PERF_WORKER mono_ns=%lld name=%s cpu_ns=%lld clock_id=%d affinity_error=%d "
                    "allowed=%d mask_low64=%llx priority_error=%d policy=%d priority=%d "
                    "registered_wall_ns=%lld registered_mono_ns=%lld\n",
                    mono, names[i], cpu, int(worker.clock), worker.affinity_error, worker.allowed, worker.mask,
                    priority_error, policy, priority.sched_priority, worker.wall_ns, worker.mono_ns);
        if (i < compilation.size()) {
            const auto& sample = cpu_samples[i];
            std::printf("EDEN_PERF_CPU_POINT core=%u phase=%u epoch=%u mono_ns=%lld cpu_ns=%lld thread=%llu pc=%llx svc=%x fpcr=%u compilations=%llu compile_ns=%llu\n",
                        i, unsigned(cpu_state[i].phase.load(std::memory_order_relaxed)), sample.epoch,
                        sample.mono_ns, sample.cpu_ns, sample.thread, sample.pc, sample.svc, sample.fpcr,
                        sample.compilations, sample.compile_ns);
#ifdef EDEN_DEV_PROFILE
            const auto& dups = jit_duplicates[i];
            std::printf("EDEN_PERF_DUPLICATES mono_ns=%lld core=%u first=%llu again=%llu lead_1ms=%llu lead_10ms=%llu "
                        "lead_100ms=%llu lead_1s=%llu lead_10s=%llu lead_more=%llu\n", mono, i,
                        dups[0].load(std::memory_order_relaxed), dups[1].load(std::memory_order_relaxed),
                        dups[2].load(std::memory_order_relaxed), dups[3].load(std::memory_order_relaxed),
                        dups[4].load(std::memory_order_relaxed), dups[5].load(std::memory_order_relaxed),
                        dups[6].load(std::memory_order_relaxed), dups[7].load(std::memory_order_relaxed));
            if (eden_jit_path_counters) {
                unsigned long long path[8]{};
                eden_jit_path_counters(i, path);
                std::printf("EDEN_PERF_JITPATH mono_ns=%lld core=%u runs=%llu thunks=%llu lookups=%llu hits=%llu "
                            "contended=%llu flush_all=%llu flush_locations=%llu far_links=%llu\n", mono, i,
                            path[0], path[1], path[2], path[3], path[4], path[5], path[6], path[7]);
            }
            std::printf("EDEN_PERF_PROGRESS mono_ns=%lld core=%u compilations=%llu compile_ns=%llu evacuations=%llu "
                        "translate_ns=%llu optimize_ns=%llu emit_ns=%llu ranges_ns=%llu\n",
                        mono, i, compilation[i].calls.load(std::memory_order_relaxed),
                        compilation[i].nanoseconds.load(std::memory_order_relaxed),
                        evacuations[i].load(std::memory_order_relaxed),
                        jit_phase_ns[i][0].load(std::memory_order_relaxed), jit_phase_ns[i][1].load(std::memory_order_relaxed),
                        jit_phase_ns[i][2].load(std::memory_order_relaxed), jit_phase_ns[i][3].load(std::memory_order_relaxed));
#else
            std::printf("EDEN_PERF_PROGRESS mono_ns=%lld core=%u compilations=%llu compile_ns=%llu evacuations=%llu\n",
                        mono, i, compilation[i].calls.load(std::memory_order_relaxed),
                        compilation[i].nanoseconds.load(std::memory_order_relaxed),
                        evacuations[i].load(std::memory_order_relaxed));
#endif
        }
    }
    std::fflush(stdout);
}

#ifndef EDEN_DEV_PROFILE
// Release builds: the JIT's block hook (derived a32_interface.cpp) resolves here and does nothing,
// as does the shared JIT's compile-phase hook (headless/dynarmic/jit_impl.inc). Native packages
// cannot import an undefined weak symbol.
extern "C" void eden_jit_block(unsigned, unsigned long long, const void*, unsigned long long) {}
extern "C" void eden_jit_phases(unsigned, unsigned long long, unsigned long long, unsigned long long,
                                unsigned long long) {}
#endif

#ifdef EDEN_DEV_PROFILE
extern "C" void eden_jit_block(unsigned core, unsigned long long location, const void* entry,
                               unsigned long long size) {
    if (core != 0) return;
    if (!core0_blocks) core0_blocks = new JitBlock[jit_block_capacity];
    std::size_t count = core0_block_count.load(std::memory_order_relaxed);
    const auto address = reinterpret_cast<uintptr_t>(entry);
    if (count && address < core0_blocks[count - 1].entry) count = 0; // the cache was cleared
    if (count >= jit_block_capacity) return;
    core0_blocks[count] = JitBlock{address, size, location};
    core0_block_count.store(count + 1, std::memory_order_release);
}

void BeginPcSampling() {
    struct sigaction previous{}, action{};
    if (sigaction(SIGUSR2, nullptr, &previous) || previous.sa_handler != SIG_DFL)
        throw std::runtime_error("Development sampler signal already in use");
    action.sa_sigaction = PcSignal;
    action.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGUSR2, &action, nullptr))
        throw std::runtime_error("Cannot install development PC sampler");
    // Development process lifetime; polling stops before guest-worker shutdown.
    pc_sampling = true;
#if defined(EDEN_DEV_WAIT_CALLERS) && defined(PS5_NATIVE)
    std::printf("EDEN_PERF_CALLER_MODE enabled=1 firmware=6.02 depth=8 page=4096\n");
#endif
    std::printf("EDEN_PERF_PC_ANCHOR address=%llx capacity=%zu\n",
                static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(&PcSignal)), sampled_pcs.size());
    std::printf("EDEN_PERF_LIBC memcpy=%p memset=%p memmove=%p memcmp=%p\n",
                reinterpret_cast<void*>(&std::memcpy), reinterpret_cast<void*>(&std::memset),
                reinterpret_cast<void*>(&std::memmove), reinterpret_cast<void*>(&std::memcmp));
    // Anchors for a sampled PC that falls in a library we hold no symbols for: these bracket
    // the wait primitives the guest cores and the render thread block in.
    std::printf("EDEN_PERF_LIBC2 mutex_lock=%p mutex_unlock=%p cond_wait=%p cond_timedwait=%p "
                "cond_signal=%p cond_broadcast=%p nanosleep=%p clock_gettime=%p sched_yield=%p\n",
                reinterpret_cast<void*>(&pthread_mutex_lock), reinterpret_cast<void*>(&pthread_mutex_unlock),
                reinterpret_cast<void*>(&pthread_cond_wait), reinterpret_cast<void*>(&pthread_cond_timedwait),
                reinterpret_cast<void*>(&pthread_cond_signal), reinterpret_cast<void*>(&pthread_cond_broadcast),
                reinterpret_cast<void*>(&nanosleep), reinterpret_cast<void*>(&clock_gettime),
                reinterpret_cast<void*>(&sched_yield));
}

void PollGpuPc() {
    std::lock_guard lock(workers_mutex);
    if (pc_sampling && workers[4].registered && pc_count.load() < sampled_pcs.size()) {
        const int error = pthread_kill(workers[4].thread, SIGUSR2);
        if (error) throw std::runtime_error("Cannot sample development render thread");
    }
    if (pc_sampling && core_sample_ready.load(std::memory_order_acquire) &&
        core_pc_count.load() < sampled_core_pcs.size())
        pthread_kill(core_sample_thread, SIGUSR2);
}
#endif

extern "C" unsigned eden_heap_arenas_created(void) __attribute__((weak));

void AllocatorCheck() {
    std::printf("EDEN_PERF_HEAP arenas=%u\n", eden_heap_arenas_created ? eden_heap_arenas_created() : 1u);
    constexpr unsigned ops = 50000, live_slots = 1024;
    const auto run = [](unsigned seed) {
        std::vector<void*> live(live_slots, nullptr);
        unsigned x = seed;
        const auto begin = std::chrono::steady_clock::now();
        for (unsigned i = 0; i < ops; ++i) {
            x = x * 1664525u + 1013904223u;
            void*& slot = live[x % live_slots];
            std::free(slot);
            slot = std::malloc(16 + ((x >> 16) & 511));
        }
        const auto end = std::chrono::steady_clock::now();
        for (void* pointer : live) std::free(pointer);
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    };
    std::printf("EDEN_PERF_ALLOC threads=1 pairs=%u ns_per_pair=%.1f\n", ops, run(1) / double(ops));
    for (unsigned threads : {2u, 4u, 8u}) {
        std::vector<long long> elapsed(threads);
        std::latch start{static_cast<std::ptrdiff_t>(threads)};
        std::vector<std::thread> pool;
        for (unsigned index = 0; index < threads; ++index)
            pool.emplace_back([&, index] { start.arrive_and_wait(); elapsed[index] = run(index + 7); });
        for (auto& thread : pool) thread.join();
        long long worst = 0, sum = 0;
        for (auto value : elapsed) { worst = std::max(worst, value); sum += value; }
        std::printf("EDEN_PERF_ALLOC threads=%u pairs=%u ns_per_pair_mean=%.1f ns_per_pair_worst=%.1f\n",
                    threads, ops, sum / double(threads) / ops, worst / double(ops));
    }
}

void PlatformChecks() {
#ifdef PS5_NATIVE
    CheckWorkerTopology();
    if (topology_allowed_valid && Common::PeSwitch("EDEN_UP_CPU_PAIR_PROBE")) ProbeCpuPairs(topology_allowed);
#endif
    CheckCpuClocks();
#ifdef EDEN_DEV_PROFILE
    AllocatorCheck();
#endif
    using clock = std::chrono::steady_clock;
    const auto ns = [](auto start) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - start).count();
    };
    std::printf("EDEN_PERF_CLOCK leaf_max=%x invariant=%d tsc_hz=%llu native=%d\n",
                __get_cpuid_max(0, nullptr), Common::g_cpu_caps.invariant_tsc,
                static_cast<unsigned long long>(Common::g_cpu_caps.tsc_frequency),
                Common::g_wall_clock.IsNative());
    unsigned long long sum = 0;
    auto start = clock::now();
    for (unsigned i = 0; i < 100'000; ++i) sum += Common::g_wall_clock.GetCNTPCT();
    std::printf("EDEN_PERF_CLOCK_READ calls=100000 elapsed_ns=%lld checksum=%llu\n",
                static_cast<long long>(ns(start)), sum);
    // Bound the diagnostic overhead comparison; per-block timers use the
    // already-qualified fenced counter, with the upstream steady-clock fallback.
    start = clock::now();
    for (unsigned i = 0; i < 1000; ++i) sum += clock::now().time_since_epoch().count();
    const auto system_clock_ns = ns(start);
    start = clock::now();
    for (unsigned i = 0; i < 1000; ++i) sum += Common::g_wall_clock.GetTimeNS().count();
    std::printf("EDEN_PERF_DIAGNOSTIC_CLOCK calls=1000 system_ns=%lld counter_ns=%lld checksum=%llu\n",
                static_cast<long long>(system_clock_ns), static_cast<long long>(ns(start)), sum);
    for (const unsigned requested_us : {1000u, 16667u, 50000u}) {
        const auto wall = Common::g_wall_clock.GetTimeNS();
        start = clock::now();
        std::this_thread::sleep_for(std::chrono::microseconds(requested_us));
        const auto mono_delta = ns(start);
        const auto wall_delta = (Common::g_wall_clock.GetTimeNS() - wall).count();
        if (mono_delta <= 0 || wall_delta <= 0) throw std::runtime_error("Nonmonotonic clock");
        if (std::abs(wall_delta - mono_delta) > std::max<decltype(mono_delta)>(2'000'000, mono_delta / 20))
            throw std::runtime_error("Guest clock disagrees with monotonic time");
        std::printf("EDEN_PERF_SLEEP requested_us=%u mono_ns=%lld wall_ns=%lld\n",
                    requested_us, static_cast<long long>(mono_delta), static_cast<long long>(wall_delta));
    }
    constexpr std::size_t bytes = 16 * 1024 * 1024;
    auto* memory = static_cast<unsigned long long*>(Common::AllocateMemoryPages(bytes));
    if (!memory) throw std::bad_alloc{};
    struct Release {
        void* memory;
        ~Release() { Common::FreeMemoryPages(memory, bytes); }
    } release{memory};
    constexpr std::size_t words = bytes / sizeof(*memory);
    for (std::size_t i = 0; i < words; ++i) memory[i] = i;
    const volatile auto* reads = memory;
    sum = 0;
    start = clock::now();
    for (unsigned repeat = 0; repeat < 4; ++repeat)
        for (std::size_t i = 0; i < words; ++i) sum += reads[i];
    if (sum != 4ULL * words * (words - 1) / 2) throw std::runtime_error("Memory read checksum");
    std::printf("EDEN_PERF_MEMORY pattern=stream bytes=%zu elapsed_ns=%lld checksum=%llu\n",
                bytes * 4, static_cast<long long>(ns(start)), sum);
    // Full-cycle LCG, one node per cache line. Compare a small warm working set
    // with the same dependent-load loop across the larger owned mapping.
    for (const std::size_t nodes : {std::size_t{512}, bytes / 64}) {
        for (std::size_t i = 0; i < nodes; ++i) memory[i * 8] = (i * 5 + 1) & (nodes - 1);
        std::size_t index = 0;
        constexpr unsigned steps = 1'048'576;
        start = clock::now();
        for (unsigned i = 0; i < steps; ++i) index = reads[index * 8];
        if (index != 0) throw std::runtime_error("Memory chain checksum");
        std::printf("EDEN_PERF_MEMORY pattern=dependent working_bytes=%zu steps=%u elapsed_ns=%lld checksum=%zu\n",
                    nodes * 64, steps, static_cast<long long>(ns(start)), index);
    }
    std::fflush(stdout);
}
} // namespace Eden::Performance

// EDEN_JIT_CACHE_MIB=<core 0>,<core 1>,<cores 2 to 5>: the Switch 2 code caches' sizes in MiB, read
// once as the first JIT starts (headless/CMakeLists.txt, jit_cache_sizes); 0 or a missing field keeps
// the built-in size. A cache is address space committed 16 MiB at a time as it fills, so a larger
// one costs only what is used; the most is 1 GiB (memory_pages.cpp CodeSteps). Ocarina of Time's
// worker caches reached 96 of 96 MiB by 11 minutes (runs 160 and 162, EDEN_JIT_MEMORY); one more
// block then clears the whole group (EDEN_JIT_PRESSURE in heap.log). Aflakon's 36-minute session
// ran with 128 MiB on cores 3 to 5 (ocarina-cold-memory f0515ee16).
extern "C" unsigned eden_jit_cache_bytes(unsigned core, unsigned fallback) {
    static const std::array<unsigned, 3> mib = [] {
        std::array<unsigned, 3> sizes{};
        const char* value = std::getenv("EDEN_JIT_CACHE_MIB");
        for (std::size_t i = 0; value != nullptr && i < sizes.size(); ++i) {
            char* end = nullptr;
            const unsigned long asked = std::strtoul(value, &end, 10);
            sizes[i] = static_cast<unsigned>(std::min<unsigned long>(asked, 1024));
            if (end == nullptr || *end != ',') break;
            value = end + 1;
        }
        if (sizes[0] != 0 || sizes[1] != 0 || sizes[2] != 0)
            std::fprintf(stderr, "EDEN_JIT_CACHE_MIB core0=%u core1=%u cores2to5=%u (0: as built)\n", sizes[0], sizes[1], sizes[2]);
        return sizes;
    }();
    const unsigned asked = core == 0 ? mib[0] : core == 1 ? mib[1] : core < 6 ? mib[2] : 0;
    return asked != 0 ? asked << 20 : fallback;
}

// dynarmic A64 hooks (weak there): whole-cache evacuations and block compilation time.
extern "C" void eden_jit_evacuation(unsigned core) {
    if (core < Eden::Performance::evacuations.size())
        Eden::Performance::evacuations[core].fetch_add(1, std::memory_order_relaxed);
}
#ifdef EDEN_DEV_PROFILE
// Development builds only (the per-phase and duplicate counters exist there).
extern "C" void eden_jit_phases(unsigned core, unsigned long long translate, unsigned long long optimize,
                                unsigned long long emit, unsigned long long location) {
    if (core >= Eden::Performance::jit_phase_ns.size()) return;
    auto& phases = Eden::Performance::jit_phase_ns[core];
    phases[0].fetch_add(translate, std::memory_order_relaxed);
    phases[1].fetch_add(optimize, std::memory_order_relaxed);
    phases[2].fetch_add(emit, std::memory_order_relaxed);
    if (!Eden::Performance::jit_duplicate_tracking.load(std::memory_order_relaxed)) return;
    // Which cores compiled each location, and when the first one did.
    struct First { unsigned cores; long long ns; };
    static std::mutex mutex;
    static auto* compiled = [] {
        auto* map = new std::unordered_map<unsigned long long, First>;
        map->reserve(1u << 21);
        return map;
    }();
    const long long now = Common::g_wall_clock.GetTimeNS().count();
    unsigned index = 0;
    {
        std::lock_guard lock{mutex};
        auto [it, inserted] = compiled->try_emplace(location, First{1u << core, now});
        if (!inserted) {
            if (it->second.cores & (1u << core)) {
                index = 1;
            } else {
                it->second.cores |= 1u << core;
                const long long lead = now - it->second.ns;
                index = 2;
                for (long long bound = 1'000'000; index < 7 && lead >= bound; bound *= 10) ++index;
            }
        }
    }
    Eden::Performance::jit_duplicates[core][index].fetch_add(1, std::memory_order_relaxed);
}
extern "C" void eden_jit_ranges(unsigned core, unsigned long long ns) {
    if (core < Eden::Performance::jit_phase_ns.size())
        Eden::Performance::jit_phase_ns[core][3].fetch_add(ns, std::memory_order_relaxed);
}
#endif
// The shared JIT's compile timers (headless/dynarmic/jit_impl.inc): the qualified invariant-TSC
// clock, about 26 ns per read against about 0.9 µs for the steady clock's system call.
extern "C" unsigned long long eden_jit_clock_ns() {
    return static_cast<unsigned long long>(Common::g_wall_clock.GetTimeNS().count());
}
extern "C" void eden_jit_compile(unsigned core, unsigned long long ns) {
    if (core >= Eden::Performance::compilation.size()) return;
    Eden::Performance::compilation[core].calls.fetch_add(1, std::memory_order_relaxed);
    Eden::Performance::compilation[core].nanoseconds.fetch_add(ns, std::memory_order_relaxed);
}
// EDEN_UP_JIT_COMMIT_RECLAIM (headless/dynarmic/jit_impl.inc, weak there): 2 asks whether the switch is
// on (each JIT's constructor, once); 1 opens the scope around GetBlock's EnsureMemoryCommitted, on the
// compiling thread; 0 closes it and returns what the commit left there (memory_wait.h JitCommitResult).
extern "C" int eden_jit_refuse_in_compile(void) __attribute__((weak)); // the shared test hook, <where> 4
extern "C" int eden_jit_commit_scope(int op) {
    namespace MemoryWait = Eden::MemoryWait;
    // The scope also opens for EDEN_TEST_JIT_COMMIT_REFUSE's <where> 4 alone: with the switch at 0 nothing
    // reads what the commit leaves there (0 always), so GetBlock goes on as before.
    if (op == 2)
        return MemoryWait::jit_reclaim_mode.load(std::memory_order_relaxed) != 0 ||
                       (eden_jit_refuse_in_compile && eden_jit_refuse_in_compile() != 0)
                   ? 1
                   : 0;
    if (op == 1) {
        MemoryWait::SetJitClearScope(1);
        MemoryWait::SetJitCommitResult(0);
        return 0;
    }
    const int result = MemoryWait::JitCommitResult();
    MemoryWait::SetJitClearScope(0);
    MemoryWait::SetJitCommitResult(0);
    return result;
}

#ifdef PS5_NATIVE
// The driver found no memory for one of its buffers (tools/patch-radv-series/lead-mesh-22): give the
// pool what can go at once, then the driver tries once more. Its own buffers (command buffers,
// uploads) never reach the Vulkan allocator's refusal path (MemoryAllocator::ReleaseRefused), so
// without this the session ended at vkEndCommandBuffer (Ocarina on 13.60, crash-20261108-222452).
// Any thread, inside the driver: only memory work here, nothing that calls back into Vulkan.
extern "C" std::size_t eden_guest_memory_reclaim(std::size_t want) __attribute__((weak));
// EDEN_UP_HEAP_HOLD_AT_LOAD (headless/heap_arenas.inc): a hold begins or goes on; 1 when it began now.
extern "C" int eden_heap_hold_begin(void) __attribute__((weak));
// EDEN_UP_RADV_OOM_LEND=<blocks> (src/guest_memory_service.inc): the reclaim without a list on the heap
// and without a second scan in the same swap tick when the last found nothing cold, and the guest
// reserve lent down to that many blocks under its floor. Death at the forest exit (05:37, T120): eleven
// refusals of 1.4 to 15.3 MiB in about a second, each with 0 to 64 MiB given back, while the reserve sat
// at its floor of 64 MiB that nothing here could reach.
extern "C" std::size_t eden_guest_memory_reclaim_bounded(std::size_t want, int verify) __attribute__((weak));
extern "C" std::size_t eden_guest_reserve_lend_quiet(std::size_t bytes, std::size_t below_floor) __attribute__((weak));
extern "C" std::uint64_t eden_radv_out_of_memory(std::uint64_t bytes) {
    static std::mutex one_at_a_time;
    const std::lock_guard lock{one_at_a_time};
    Eden::Performance::graphics_refusals.fetch_add(1, std::memory_order_relaxed);
    // Live, read once per call (not a hot path); 0 is the code as it was.
    const std::uint64_t lend =
        static_cast<std::uint64_t>(::Common::LiveSwitch::Value<::Common::LiveSwitch::Id::RadvOomLend>());
    const std::uint64_t want = std::clamp<std::uint64_t>(bytes * 4, std::uint64_t{64} << 20, std::uint64_t{512} << 20);
    // EDEN_UP_HEAP_HOLD_AT_LOAD (heap_arenas.inc): a refused request of 64 MiB or more is an area's big
    // image, which comes before the texture cache's burst; the heap's pieces stay from here on (the trim
    // below gives nothing back while the hold runs). run1399: 304,132 KiB refused, the heap's 32 MiB given
    // here, a 4 MiB heap request refused 1.6 s later (its crash log 975-1015).
    if (bytes >= (std::uint64_t{64} << 20) && eden_heap_hold_begin && eden_heap_hold_begin() != 0)
        std::fprintf(stderr, "EDEN_UP_HEAP_HOLD begun: the driver was refused %llu MiB; the heap keeps its pieces\n",
                     static_cast<unsigned long long>(bytes >> 20));
    std::uint64_t freed = Eden::Performance::TrimHeap("the driver found no memory");
    if (lend != 0 && eden_guest_memory_reclaim_bounded) {
        static const bool verify = Common::PeSwitch("EDEN_UP_RADV_OOM_LEND_VERIFY");
        if (freed < bytes)
            freed += eden_guest_memory_reclaim_bounded(static_cast<std::size_t>(want), verify ? 1 : 0);
    } else if (freed < bytes && eden_guest_memory_reclaim) {
        freed += eden_guest_memory_reclaim(static_cast<std::size_t>(want));
    }
    // The driver asks once more, for one range of `bytes` or in 2 MiB pieces: the asked size rounded up
    // to 2 MiB and one block more, less what already went back.
    std::uint64_t lent = 0;
    if (lend != 0) {
        constexpr std::uint64_t block = std::uint64_t{2} << 20;
        const std::uint64_t rounded = (bytes + block - 1) & ~(block - 1);
        if (freed < bytes && freed < rounded + block && eden_guest_reserve_lend_quiet) {
            lent = eden_guest_reserve_lend_quiet(static_cast<std::size_t>(rounded + block - freed),
                                                 static_cast<std::size_t>(lend));
            freed += lent;
        }
        Eden::Performance::radv_oom_callbacks.fetch_add(1, std::memory_order_relaxed);
        Eden::Performance::radv_oom_lent.fetch_add(lent, std::memory_order_relaxed);
        (freed >= rounded ? Eden::Performance::radv_oom_covered : Eden::Performance::radv_oom_short)
            .fetch_add(1, std::memory_order_relaxed);
    }
    static std::atomic<unsigned> logged{0};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 64) {
        if (lend != 0)
            std::fprintf(stderr,
                         "EDEN_RADV_OUT_OF_MEMORY %llu KiB asked: %llu MiB given back, lent %llu MiB of the guest "
                         "reserve (down to %llu blocks under its floor)\n",
                         static_cast<unsigned long long>(bytes >> 10), static_cast<unsigned long long>(freed >> 20),
                         static_cast<unsigned long long>(lent >> 20), static_cast<unsigned long long>(lend));
        else
            std::fprintf(stderr, "EDEN_RADV_OUT_OF_MEMORY %llu KiB asked: %llu MiB given back\n",
                         static_cast<unsigned long long>(bytes >> 10), static_cast<unsigned long long>(freed >> 20));
    }
    return freed;
}
#endif

#ifdef PS5_NATIVE
extern "C" std::int64_t sceKernelGetDirectMemorySize();
extern "C" std::int32_t sceKernelAvailableDirectMemorySize(std::int64_t, std::int64_t, std::size_t, std::int64_t*,
                                                           std::size_t*);
// How many times the driver found no memory (graphics_refusals; eden_radv_out_of_memory above). Core
// patch 0940 (EDEN_UP_TC_OOM_SPARE_FRAMES) stops sparing images for 600 frames when it moves. Weak there.
extern "C" unsigned long long eden_graphics_refusals(void) {
    return Eden::Performance::graphics_refusals.load(std::memory_order_relaxed);
}
// Nonzero while a game write or a pipeline builder's heap request waits for a block (as
// GuestMemoryWanted above). Core patch 0941 ends EDEN_UP_TC_OOM_SPARE_FRAMES' sparing for 600
// frames when it reads one: the game's memory comes first. Weak there.
extern "C" unsigned eden_tc_game_memory_wanted(void) {
    using Eden::Performance::eden_guest_memory_wanted;
    using Eden::Performance::eden_heap_wanted;
    return ((eden_guest_memory_wanted && eden_guest_memory_wanted() != 0) ||
            (eden_heap_wanted && eden_heap_wanted() != 0))
               ? 1u
               : 0u;
}
// The largest free block of the pool in bytes, 0 when unknown, looked at again every 20 ms at most:
// pipeline builds ask it before they start (core patch 0554, waveydude's 0915,
// EDEN_PIPELINE_MEMORY_GATE_MIB), from any builder thread. Only the kernel query and two atomics:
// no lock, no allocation, no log. Asked only while the gate is on. From waveydude/Eglantine-PS5-
// commit b9f52d98e8; changed: a known empty pool reads as 1 byte, not 0.
extern "C" unsigned long long eden_pool_largest_free(void) {
    static std::atomic<long long> checked_ns{0};
    static std::atomic<unsigned long long> largest_seen{0};
    const long long now = ::Eden::Performance::NowNs();
    const long long last = checked_ns.load(std::memory_order_relaxed);
    if (last != 0 && now - last < 20'000'000) return largest_seen.load(std::memory_order_relaxed);
    checked_ns.store(now, std::memory_order_relaxed);
    std::int64_t start = 0;
    std::size_t largest = 0;
    const std::int64_t total = sceKernelGetDirectMemorySize();
    // 0 tells the gate "unknown" and it never waits; an empty pool is the case it is for, so a
    // known largest block of 0 is given as 1 byte (not in waveydude's text). A failed query stays 0.
    const bool known =
        total > 0 && sceKernelAvailableDirectMemorySize(0, total, 0x4000, &start, &largest) == 0;
    const unsigned long long value = known ? std::max<unsigned long long>(largest, 1) : 0;
    largest_seen.store(value, std::memory_order_relaxed);
    return value;
}
#endif

#include "ordered_read_trace.inc"
// EDEN_UP_LOAD_HOLD (T178c, off unless set): the pre-load hold at an area load (core patch 0930).
#include "load_hold.inc"
