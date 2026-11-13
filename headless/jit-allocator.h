// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstddef>
#include <cstdint>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <xbyak/xbyak.h>
#include "memory_wait.h"
// The heap's empty pieces back to the pool (headless/heap_arenas.inc); absent in host tools.
extern "C" std::size_t eden_heap_trim(void) __attribute__((weak));
// EDEN_UP_JIT_COMMIT_RECLAIM=2/3: cold guest memory packed now (src/guest_memory_service.inc; 0 without
// EDEN_GUEST_SWAP=2/3 and EDEN_GUEST_SWAP_RECLAIM=1); absent in host tools.
extern "C" std::size_t eden_guest_memory_reclaim_bounded(std::size_t want, int verify) __attribute__((weak));
// Its scan and skip counts (values 5 and 6, src/guest_memory_service.inc), to tell a real scan from a call
// that returned at once; absent in host tools.
extern "C" void eden_guest_reserve_lend_stats(unsigned long long* values, std::size_t count) __attribute__((weak));
namespace Common {
void* AllocateMemoryPages(std::size_t) noexcept;
void FreeMemoryPages(void*) noexcept;
void* MapExecutableAlias(void*, std::size_t) noexcept;
// A cache that takes its memory as it fills (src/memory_pages.cpp).
void* ReserveCodeRange(std::size_t, void** executable) noexcept;
bool CommitCodeRange(const void* executable, std::size_t bytes) noexcept;
bool FreeCodeRange(void* executable) noexcept;
}
// Separate RW/NX and RX views avoid changing page permissions per compiled block.
// Unsupported native JIT allocation falls back to the qualified owned W^X path.
inline Xbyak::Allocator* EdenJitAllocator() {
    struct Allocator final : Xbyak::Allocator {
        struct Mapping { void* writable; std::size_t size; bool grows; };
        std::mutex mutex;
        std::unordered_map<std::uint8_t*, Mapping> mappings;
        std::uint8_t* alloc(std::size_t size) override {
#ifdef EDEN_JIT_ALIAS_NATIVE
            auto* diagnostics = stdout; // Native stdout is the separate heap log.
#else
            auto* diagnostics = stderr; // Host stdout is the strict lifecycle receipt.
#endif
            const long page = sysconf(_SC_PAGESIZE);
            if (!size || page <= 0 || size > std::numeric_limits<std::size_t>::max() - page)
                return nullptr;
            const auto span = (size + page - 1) & ~(std::size_t(page) - 1);
            void* executable = MAP_FAILED;
            void* writable = MAP_FAILED;
            // A large cache is address space, and memory comes behind it as the JIT fills it
            // (commit below). Its first step now: the code generator writes at once.
            void* grown_executable = nullptr;
            if (void* const grown = Common::ReserveCodeRange(size, &grown_executable)) {
                bool kept = false;
                if (Common::CommitCodeRange(grown_executable, 1)) {
                    try {
                        std::lock_guard lock(mutex);
                        mappings.emplace(static_cast<std::uint8_t*>(grown_executable), Mapping{grown, size, true});
                        kept = true;
                    } catch (...) {
                    }
                }
                if (kept) {
                    std::fprintf(diagnostics, "EDEN_JIT_ALIAS bytes=%zu active=1 code=%p grows=1\n", size,
                                 grown_executable);
                    return static_cast<std::uint8_t*>(grown_executable);
                }
                if (!Common::FreeCodeRange(grown_executable)) std::abort();
            }
#ifdef EDEN_JIT_ALIAS_NATIVE
            writable = Common::AllocateMemoryPages(size);
            if (!writable) return nullptr;
            executable = Common::MapExecutableAlias(writable, size);
            if (!executable) {
                std::fprintf(diagnostics, "EDEN_JIT_ALIAS bytes=%zu active=0 errno=%d\n", span, errno);
                return static_cast<std::uint8_t*>(writable);
            }
#else
            const int executable_fd = memfd_create("eden-jit", MFD_CLOEXEC);
            if (executable_fd >= 0 && ftruncate(executable_fd, span) == 0) {
                executable = mmap(nullptr, span, PROT_READ | PROT_EXEC, MAP_SHARED, executable_fd, 0);
                writable = mmap(nullptr, span, PROT_READ | PROT_WRITE, MAP_SHARED, executable_fd, 0);
            }
            if (executable_fd >= 0 && close(executable_fd) != 0) std::abort();
#endif
            if (executable != MAP_FAILED && executable && writable != MAP_FAILED && writable && executable != writable) {
                auto* pointer = static_cast<std::uint8_t*>(executable);
                try {
                    std::lock_guard lock(mutex);
                    mappings.emplace(pointer, Mapping{writable, span, false});
                    std::fprintf(diagnostics, "EDEN_JIT_ALIAS bytes=%zu active=1\n", span);
                    return pointer;
                } catch (...) {
                    // Release both views before reporting allocation failure.
                    if (munmap(executable, span) != 0) std::abort();
#ifdef EDEN_JIT_ALIAS_NATIVE
                    Common::FreeMemoryPages(writable);
#else
                    if (munmap(writable, span) != 0) std::abort();
#endif
                    return nullptr;
                }
            }
#ifdef EDEN_JIT_ALIAS_NATIVE
            std::abort(); // Distinct views are required by the direct mapper.
#else
            if (executable != MAP_FAILED && executable && munmap(executable, span) != 0) std::abort();
            if (writable != MAP_FAILED && writable && writable != executable && munmap(writable, span) != 0) std::abort();
            std::fprintf(diagnostics, "EDEN_JIT_ALIAS bytes=%zu active=0 errno=%d\n", span, errno);
            return static_cast<std::uint8_t*>(Common::AllocateMemoryPages(size));
#endif
        }
        std::uint8_t* writableAddress(std::uint8_t* pointer) override {
            std::lock_guard lock(mutex);
            const auto entry = mappings.find(pointer);
            return entry == mappings.end() ? pointer : static_cast<std::uint8_t*>(entry->second.writable);
        }
        // Memory behind the first `bytes` of a cache: asked before the JIT writes that far.
        void commit(std::uint8_t* pointer, std::size_t bytes) override {
            if (Common::CommitCodeRange(pointer, bytes)) return;
            // Neither the pool nor the guest's reserve had a block: the heap's empty pieces go back,
            // and this core waits while the graphics caches, which see the pool short, give memory
            // back (a burst can take the pool for a moment): up to EDEN_UP_JIT_COMMIT_WAIT_MS (his
            // name EDEN_JIT_COMMIT_WAIT_MS is read too), 2 s unset (memory_wait.h; in his runs a wait
            // that ran out in a later area was this one at 2 s). Unset: the old 100 steps and the old
            // line, word for word. Read at the first wait, long after the env files are in.
            static const unsigned wait_ms =
                Eden::MemoryWait::ReadMs2("EDEN_UP_JIT_COMMIT_WAIT_MS", "EDEN_JIT_COMMIT_WAIT_MS", 2000, 20, 60000);
            const int attempts = static_cast<int>(Eden::MemoryWait::Steps(wait_ms));
            if (wait_ms == 2000)
                std::fputs("JIT code cache: no memory for its next part; waiting for some\n", stderr);
            else
                std::fprintf(stderr, "JIT code cache: no memory for its next part; waiting up to %u ms for some\n",
                             wait_ms);
            // EDEN_UP_JIT_COMMIT_RECLAIM (memory_wait.h): 0 is the loop below, as it was.
            if (const unsigned mode = Eden::MemoryWait::jit_reclaim_mode.load(std::memory_order_relaxed); mode != 0) {
                reclaim_wait(pointer, bytes, attempts, mode);
                return;
            }
            for (int attempt = 1; attempt <= attempts; ++attempt) {
                if (attempt % 25 == 1 && eden_heap_trim) eden_heap_trim();
                usleep(20000);
                if (Common::CommitCodeRange(pointer, bytes)) {
                    std::fprintf(stderr, "JIT code cache: memory found after %d ms\n", attempt * 20);
                    return;
                }
            }
            std::fputs("JIT code cache: the console has no memory for its next part\n", stderr);
            std::abort();
        }
        // EDEN_UP_JIT_COMMIT_RECLAIM=1..3 (memory_wait.h). Runs on the thread whose commit was refused: a
        // guest core inside GetBlock (holding only its own eden_compile_mutex, state Lookup, so a clear
        // by another core does not wait for it), or the precompiler (holding that core's
        // eden_compile_mutex and marked Compiling). code_mutex is NOT held here: CommitCodeRange has
        // returned. What it calls: eden_heap_trim (as the old loop) and eden_guest_memory_reclaim_bounded
        // (swap_out_mutex, then slot locks; static candidate list, no heap request; called from driver
        // threads with the pool empty only with EDEN_UP_RADV_OOM_LEND on, never yet on the console: read
        // as safe from the code, not from a run). Neither takes a lock a guest core holds while it
        // waits here, and no graphics or driver call is made from here (the texture cache is GPU
        // thread only; the driver's locks may be held by a thread that is inside the out-of-memory
        // callback, which itself takes swap_out_mutex).
        void reclaim_wait(std::uint8_t* pointer, std::size_t bytes, int attempts, unsigned mode) {
            namespace MW = Eden::MemoryWait;
            auto& c = MW::jit_reclaim;
            c.waits.fetch_add(1, std::memory_order_relaxed);
            const unsigned long long clears_before = MW::jit_clears.load(std::memory_order_acquire);
            const bool may_clear = mode == 3 && MW::JitClearScope() != 0;
            std::size_t trimmed = 0, reclaimed = 0;
            unsigned reclaims = 0, scans = 0;
            const auto served = [&](int ms, bool at_once) {
                (at_once ? c.served_at_once : c.served_later).fetch_add(1, std::memory_order_relaxed);
                c.trim_bytes.fetch_add(trimmed, std::memory_order_relaxed);
                c.reclaim_bytes.fetch_add(reclaimed, std::memory_order_relaxed);
                MW::NoteJitLongest(static_cast<std::uint64_t>(ms));
                std::fprintf(stderr, "JIT code cache: memory found after %d ms\n", ms);
                std::fprintf(stderr,
                             "EDEN_UP_JIT_COMMIT_RECLAIM mode=%u served after %d ms (%s): heap trims gave %zu MiB, "
                             "%u guest reclaims (%u scanned) gave %zu MiB\n",
                             mode, ms, at_once ? "the ask right after a reclaim" : "a 20 ms retry", trimmed >> 20,
                             reclaims, scans, reclaimed >> 20);
            };
            for (int attempt = 1; attempt <= attempts; ++attempt) {
                bool fresh = false;
                if (attempt % 25 == 1 && eden_heap_trim) {
                    const std::size_t given = eden_heap_trim();
                    trimmed += given;
                    fresh = given != 0;
                }
                if (mode >= 2 && attempt % static_cast<int>(MW::JitReclaimEveryTries) == 1 &&
                    eden_guest_memory_reclaim_bounded) {
                    if (!MW::jit_reclaiming.exchange(true, std::memory_order_acquire)) {
                        // Scans before and after: a call can return 0 at once (EDEN_GUEST_SWAP_RECLAIM
                        // unset, swap mode not 2/3, or the same-tick skip), which is no scan at all. The
                        // counters are global: with EDEN_UP_RADV_OOM_LEND on, a driver scan in between
                        // would count here too.
                        unsigned long long before[7] = {}, after[7] = {};
                        if (eden_guest_reserve_lend_stats) eden_guest_reserve_lend_stats(before, 7);
                        const std::size_t given = eden_guest_memory_reclaim_bounded(MW::JitReclaimAsk, 0);
                        if (eden_guest_reserve_lend_stats) eden_guest_reserve_lend_stats(after, 7);
                        MW::jit_reclaiming.store(false, std::memory_order_release);
                        ++reclaims;
                        c.reclaims.fetch_add(1, std::memory_order_relaxed);
                        if (after[5] != before[5]) {
                            ++scans;
                            c.reclaim_scans.fetch_add(1, std::memory_order_relaxed);
                        } else if (after[6] != before[6]) {
                            c.reclaim_skipped.fetch_add(1, std::memory_order_relaxed);
                        }
                        reclaimed += given;
                        fresh = fresh || given != 0;
                    } else {
                        c.reclaim_busy.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                // Something came back on this thread: ask now, before the keeper's next 100 ms fill or
                // a graphics thread's next refusal takes it.
                if (mode >= 2 && fresh && Common::CommitCodeRange(pointer, bytes)) {
                    served((attempt - 1) * 20, true);
                    return;
                }
                usleep(20000);
                if (may_clear && MW::jit_clears.load(std::memory_order_acquire) != clears_before) {
                    // Another core's clear reset this cache too: what this commit was for is gone.
                    c.after_clear.fetch_add(1, std::memory_order_relaxed);
                    MW::SetJitCommitResult(2);
                    std::fprintf(stderr, "EDEN_UP_JIT_COMMIT_RECLAIM mode=3 wait ended after %d ms by another "
                                         "core's clear; this block is compiled again\n", attempt * 20);
                    return;
                }
                if (Common::CommitCodeRange(pointer, bytes)) {
                    served(attempt * 20, false);
                    return;
                }
            }
            c.trim_bytes.fetch_add(trimmed, std::memory_order_relaxed);
            c.reclaim_bytes.fetch_add(reclaimed, std::memory_order_relaxed);
            MW::NoteJitLongest(static_cast<std::uint64_t>(attempts) * 20);
            timespec now{};
            clock_gettime(CLOCK_MONOTONIC, &now);
            const std::uint64_t now_ms = static_cast<std::uint64_t>(now.tv_sec) * 1000 + now.tv_nsec / 1000000;
            if (may_clear && MW::JitClearAllowed(now_ms)) {
                c.cleared.fetch_add(1, std::memory_order_relaxed);
                const unsigned long long clear = MW::jit_clears.fetch_add(1, std::memory_order_acq_rel) + 1;
                MW::SetJitCommitResult(1);
                std::fprintf(stderr,
                             "EDEN_UP_JIT_COMMIT_RECLAIM mode=3 nothing came back in %d ms (heap %zu MiB, %u reclaims "
                             "(%u scanned) %zu MiB): every core's compiled code is cleared instead of ending the session "
                             "(clear %llu)\n",
                             attempts * 20, trimmed >> 20, reclaims, scans, reclaimed >> 20, clear);
                return;
            }
            c.ended.fetch_add(1, std::memory_order_relaxed);
            if (may_clear) {
                c.clear_capped.fetch_add(1, std::memory_order_relaxed);
                std::fprintf(stderr, "EDEN_UP_JIT_COMMIT_RECLAIM mode=3 %u clears in the last %llu s already: no "
                                     "more, the session ends as before\n",
                             MW::ClearsPerWindow, static_cast<unsigned long long>(MW::ClearWindowMs / 1000));
            } else if (mode == 1 && MW::JitClearScope() != 0) {
                c.would_clear.fetch_add(1, std::memory_order_relaxed);
            }
            std::fprintf(stderr,
                         "EDEN_UP_JIT_COMMIT_RECLAIM mode=%u ran out after %d ms: heap trims gave %zu MiB, %u guest "
                         "reclaims (%u scanned) gave %zu MiB; %s\n",
                         mode, attempts * 20, trimmed >> 20, reclaims, scans, reclaimed >> 20,
                         mode == 1 ? (MW::JitClearScope() != 0 ? "mode 3 would clear the compiled code here"
                                                               : "outside a block compile: mode 3 would end it too")
                                   : "ending the session");
            std::fputs("JIT code cache: the console has no memory for its next part\n", stderr);
            std::abort();
        }
        void free(std::uint8_t* pointer) override {
            {
                std::lock_guard lock(mutex);
                const auto entry = mappings.find(pointer);
                if (entry != mappings.end()) {
                    const auto mapping = entry->second;
                    if (mapping.grows) {
                        if (!Common::FreeCodeRange(pointer)) std::abort();
                        mappings.erase(entry);
                        return;
                    }
                    if (munmap(pointer, mapping.size) != 0) std::abort();
#ifdef EDEN_JIT_ALIAS_NATIVE
                    Common::FreeMemoryPages(mapping.writable);
#else
                    if (munmap(mapping.writable, mapping.size) != 0) std::abort();
#endif
                    mappings.erase(entry);
                    return;
                }
            }
            Common::FreeMemoryPages(pointer);
        }
        bool useProtect() const override { return false; }
    };
    static Allocator allocator;
    return &allocator;
}
