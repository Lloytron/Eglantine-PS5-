// SPDX-License-Identifier: GPL-3.0-or-later
// EglantinePS5: compiled A64 blocks shared by the JITs of one guest process.
//
// Every guest core keeps its own JIT (Jit::Impl), emitter and code region, and compiles into its
// own region in parallel with the other cores; the blocks are published here by location and any
// core runs any core's blocks. Generated code therefore reads everything that belongs to one core
// (callbacks, TPIDR registers, exclusive-monitor slots, fast dispatch table, JIT object) through
// the JIT state register instead of embedding it (a64_jitstate.h eden_* fields, callback.cpp).
//
// A direct branch to another block is a site whose 32-bit displacement (or, for return stack
// pushes, 64-bit immediate) is naturally aligned: linking it when the destination is compiled
// and unlinking it when the destination is invalidated are single atomic stores through the
// region's writable view, safe while other cores execute the site. An unlinked site targets a
// stub that stores the destination PC and returns to the dispatcher.
//
// All members are guarded by `mutex` except the clear protocol's atomics (see a64_interface.cpp).
//
// Precompilation: `history` lists every location published, in order and across clears: the block
// list a later session compiles ahead of the game (headless/jit_list.h). A host thread that is no
// guest core compiles such a block into a member's region through `precompile` (jit_impl.inc,
// EdenPrecompile); `precompile_mutex` keeps the members alive while it does.
//
// Site shards (EDEN_UP_JIT_SITE_SHARDS, after waveydude/Eglantine-PS5- a657c6b): the table of
// sites waiting for a target can be split into SiteShardCount tables picked by a hash of the
// target. A table that is full grows by asking for a whole new array at once (old and new held
// together while it moves); one table for every target needed a single 65,011,712-byte request
// late in a session, which the pool could not serve. Split in 128, each growth asks for about
// 1/128 of that. Off (unset or 0): every target uses table 0, which is the old single table.
#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

#include <boost/icl/interval_set.hpp>

#include "common/common_types.h"
#include "common/container/unordered_map.h"
#include "dynarmic/backend/block_range_information.h"
#include "dynarmic/ir/location_descriptor.h"

// Whether `history` is kept (headless/jit_list.h turns it off without a block list).
extern "C" bool eden_jit_keep_history;

namespace Dynarmic::Backend::X64 {

// Guards a group's maps. Critical sections are short (a lookup, or registering one block) and
// the guest cores run on dedicated host CPUs, so waiters spin instead of sleeping in the kernel.
class JitGroupLock {
public:
    void lock() noexcept {
        for (unsigned spins = 0;;) {
            if (!locked.exchange(true, std::memory_order_acquire))
                return;
            while (locked.load(std::memory_order_relaxed)) {
                if (++spins < 4096)
                    __builtin_ia32_pause();
                else
                    std::this_thread::yield();
            }
        }
    }
    bool try_lock() noexcept {
        return !locked.load(std::memory_order_relaxed) && !locked.exchange(true, std::memory_order_acquire);
    }
    void unlock() noexcept {
        locked.store(false, std::memory_order_release);
    }

private:
    std::atomic<bool> locked{false};
};

// EDEN_UP_JIT_SITE_SHARDS counters for the 10 s line, all groups together. Written under a
// group's lock, only with the switch on (counts only: no clock is read under the lock); read
// without a lock (JitGroup::LogSiteShards, which also keeps the 10 s time).
struct JitSiteShardStats {
    std::atomic<unsigned long long> keys_peak{0};        // most targets one group held at once
    std::atomic<unsigned long long> table_keys_peak{0};  // most targets one table held at once
    std::atomic<unsigned long long> grows{0};            // table growths (a new array each; each table's first one counts too)
    std::atomic<unsigned long long> grow_bytes_max{0};   // largest growth request, bytes
    std::atomic<unsigned long long> grow_bytes_last{0};  // the latest growth request, bytes
    std::atomic<unsigned long long> grows_over_2mib{0};  // growths the heap steal cannot serve
    std::atomic<long long> next_log_ns{0};
};

class JitGroup {
public:
    struct Block {
        const void* entrypoint;
        std::size_t size;
    };
    enum class SiteKind : u8 {
        Rel32At2,  // jz/jg rel32: 0F 8x rel32
        Rel32At1,  // jmp rel32: E9 rel32
        Imm64At2,  // mov rcx, imm64: 48 B9 imm64
    };
    struct Site {
        const u8* code;      // executable address of the instruction
        u8* write;           // the same bytes in the region's writable view
        const u8* unlinked;  // destination while the target block is not compiled
        SiteKind kind;
    };
    // A site emitted (unlinked) in a block that is not registered yet, with its target.
    struct PendingSite {
        IR::LocationDescriptor target;
        Site site;
    };

    JitGroupLock mutex;
    // Links left unlinked because the target was out of rel32 reach (development report).
    static inline std::atomic<unsigned long long> far_links{0};

    const Block* Find(IR::LocationDescriptor location) const {
        const auto it = blocks.find(location);
        return it == blocks.end() ? nullptr : &it->second;
    }

    // Points a site at `target`, or at its unlinked destination when `target` is null or out of
    // rel32 reach.
    static void Link(const Site& site, const void* target) {
        const u8* destination = static_cast<const u8*>(target);
        switch (site.kind) {
        case SiteKind::Rel32At2:
        case SiteKind::Rel32At1: {
            const std::size_t opcode = site.kind == SiteKind::Rel32At2 ? 2 : 1;
            const auto displacement = [&](const u8* to) {
                return static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(to)) -
                       static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(site.code + opcode + 4));
            };
            std::int64_t value = destination ? displacement(destination) : 0;
            if (destination && value != static_cast<std::int32_t>(value))
                far_links.fetch_add(1, std::memory_order_relaxed);
            if (!destination || value != static_cast<std::int32_t>(value))
                value = displacement(site.unlinked);
            __atomic_store_n(reinterpret_cast<std::uint32_t*>(site.write + opcode),
                             static_cast<std::uint32_t>(static_cast<std::int32_t>(value)), __ATOMIC_RELEASE);
            break;
        }
        case SiteKind::Imm64At2:
            __atomic_store_n(reinterpret_cast<std::uint64_t*>(site.write + 2),
                             reinterpret_cast<std::uint64_t>(destination ? destination : site.unlinked),
                             __ATOMIC_RELEASE);
            break;
        }
    }

    // Publishes a block compiled for `location` covering guest bytes [first, last]: links its own
    // sites to the blocks already compiled, links every site waiting for it, and returns true.
    // Returns false and leaves `result` at the block another core published first (the new
    // block is then never entered, so its sites are dropped).
    bool Register(IR::LocationDescriptor location, Block block, u64 first, u64 last,
                  const std::vector<PendingSite>& own_sites, Block& result) {
        const auto [it, inserted] = blocks.try_emplace(location, block);
        result = it->second;
        if (!inserted)
            return false;
        if (eden_jit_keep_history && history.size() < history_limit)
            history.push_back(location.Value());
        ranges.AddRange(boost::icl::discrete_interval<u64>::closed(first, last), location);
        for (const PendingSite& pending : own_sites) {
            if (const auto target = blocks.find(pending.target); target != blocks.end())
                Link(pending.site, target->second.entrypoint);
            SiteMap& table = SitesFor(pending.target);
            if (site_mode == 0) {
                table[pending.target].push_back(pending.site);
                continue;
            }
            const std::size_t keys = table.size();
            const std::size_t capacity = table.bucket_count();
            table[pending.target].push_back(pending.site);
            CountSite(table, keys, capacity);
        }
        const auto& target_sites = SitesFor(location);
        if (const auto waiting = target_sites.find(location); waiting != target_sites.end())
            for (const Site& site : waiting->second)
                Link(site, block.entrypoint);
        return true;
    }

    // Forgets the blocks overlapping `invalid`, unlinks every site that targets them and returns
    // their locations. Their code stays in place (a core may still be executing it) until the
    // next full clear.
    std::vector<u64> Invalidate(const boost::icl::interval_set<u64>& invalid) {
        std::vector<u64> invalidated;
        ++invalidations;
        for (const auto& location : ranges.InvalidateRanges(invalid)) {
            if (blocks.erase(location) == 0)
                continue;
            invalidated.push_back(location.Value());
            const auto& target_sites = SitesFor(location);
            if (const auto waiting = target_sites.find(location); waiting != target_sites.end())
                for (const Site& site : waiting->second)
                    Link(site, nullptr);
        }
        return invalidated;
    }

    // With every member quiescent: forget all blocks and sites (their regions are reset too).
    // Off, tables 1..127 are never used and stay empty (no memory behind them).
    void Clear() {
        ++invalidations;
        blocks.clear();
        if (site_mode == 1) {
            for (auto& shard : sites)
                shard.clear();
        } else {
            sites[0].clear();
        }
        site_keys = 0;
        ranges.ClearCache();
    }

    // EDEN_UP_JIT_SITE_SHARDS, read once per process (the first group asks; env.txt is in by
    // then, groups are made when the game's JIT starts). 0 / unset: off. 1: sites split into
    // SiteShardCount tables by a hash of the target. 2: one table as when off, counted only.
    // 1 and 2 print an EDEN_UP_JIT_SITE_SHARDS line every 10 s (LogSiteShards).
    static int SiteShardMode() {
        static const int mode = [] {
            const char* const text = std::getenv("EDEN_UP_JIT_SITE_SHARDS");
            const int value = text && (text[0] == '1' || text[0] == '2') && text[1] == '\0' ? text[0] - '0' : 0;
            if (value != 0)
                std::fprintf(stderr, "EDEN_UP_JIT_SITE_SHARDS mode %d: %s\n", value,
                             value == 1 ? "jump-site table split in 128 by target" : "one jump-site table, count only");
            return value;
        }();
        return mode;
    }

    static inline JitSiteShardStats site_stats;

    // Prints the 10 s line when one is due. Called with no group lock held and only with the
    // switch on (jit_impl.inc GetBlock, before a compilation, with its TSC clock `now_ns`), so
    // neither the clock read nor a slow log write ever happens under the spin lock. The first
    // call only sets the clock; one core wins each 10 s (compare-exchange).
    static void LogSiteShards(long long now_ns) {
        long long next = site_stats.next_log_ns.load(std::memory_order_relaxed);
        if (next == 0) {
            site_stats.next_log_ns.compare_exchange_strong(next, now_ns + 10'000'000'000LL, std::memory_order_relaxed);
            return;
        }
        if (now_ns < next ||
            !site_stats.next_log_ns.compare_exchange_strong(next, now_ns + 10'000'000'000LL, std::memory_order_relaxed))
            return;
        std::fprintf(stderr,
                     "EDEN_UP_JIT_SITE_SHARDS mode=%d tables=%zu keys_peak=%llu table_keys_peak=%llu grows=%llu "
                     "grow_kib_max=%llu grow_kib_last=%llu grows_over_2mib=%llu\n",
                     SiteShardMode(), SiteShardMode() == 1 ? SiteShardCount : std::size_t{1},
                     site_stats.keys_peak.load(std::memory_order_relaxed),
                     site_stats.table_keys_peak.load(std::memory_order_relaxed),
                     site_stats.grows.load(std::memory_order_relaxed),
                     site_stats.grow_bytes_max.load(std::memory_order_relaxed) / 1024,
                     site_stats.grow_bytes_last.load(std::memory_order_relaxed) / 1024,
                     site_stats.grows_over_2mib.load(std::memory_order_relaxed));
    }

    // Precompilation (see the top of this file). The functions take a member. From `precompile`,
    // -2 means its region is at the reserve kept for the core's own compilations and -3 that the
    // core is compiling there now (it never waits behind a precompilation that could wait).
    // `invalidations` counts range invalidations and clears: a precompiled block is dropped when
    // one happened while it was compiled, because the precompiler reads guest code the game is
    // not about to run and the game may be mapping it at that moment.
    static constexpr std::size_t history_limit = 4'000'000;
    std::vector<u64> history;
    u64 invalidations = 0;
    int (*precompile)(void* member, u64 location) = nullptr;
    std::size_t (*space)(void* member) = nullptr;
    std::mutex precompile_mutex;

    // Members (Jit::Impl, type-erased) and the clear protocol.
    std::vector<void*> members;
    std::atomic<bool> clearing{false};
    std::atomic<u64> epoch{0};
    std::mutex park_mutex;
    std::condition_variable parked;
    const void* key = nullptr;

private:
    using SiteMap = ::Common::unordered_map<IR::LocationDescriptor, std::vector<Site>>;
    static constexpr std::size_t SiteShardCount = 128;

    // waveydude's shard choice, unchanged when on. Off and in count mode: table 0, the old
    // single table (same type, same key hash).
    SiteMap& SitesFor(IR::LocationDescriptor target) {
        if (site_mode != 1)
            return sites[0];
        std::size_t hash = 0;
        ::Common::HashCombine(hash, target.Value());
        return sites[hash % SiteShardCount];
    }

    // Bytes one table asks the allocator for at `capacity` (boost::unordered_flat_map: 15 slots
    // per 16-byte group, plus a sentinel group, rounded up to whole elements). For the old
    // single table at 2^17 groups this gives 65,011,712: the request in the crash report.
    static unsigned long long SiteTableBytes(std::size_t capacity) {
        if (capacity == 0)
            return 0;
        const unsigned long long element = sizeof(SiteMap::value_type);
        const unsigned long long groups = (capacity + 1) / 15;
        const unsigned long long bytes = element * capacity + 16 * (groups + 1) - 1;
        return (bytes + element - 1) / element * element;
    }

    // After one insert into `table` (with the switch on): targets held, and a growth when the
    // table's capacity changed. Under the group's lock.
    void CountSite(const SiteMap& table, std::size_t keys_before, std::size_t capacity_before) {
        const auto raise = [](std::atomic<unsigned long long>& peak, unsigned long long value) {
            unsigned long long seen = peak.load(std::memory_order_relaxed);
            while (value > seen && !peak.compare_exchange_weak(seen, value, std::memory_order_relaxed)) {
            }
        };
        if (table.size() != keys_before) {
            ++site_keys;
            raise(site_stats.keys_peak, site_keys);
            raise(site_stats.table_keys_peak, table.size());
        }
        if (table.bucket_count() != capacity_before) {
            const unsigned long long bytes = SiteTableBytes(table.bucket_count());
            site_stats.grows.fetch_add(1, std::memory_order_relaxed);
            site_stats.grow_bytes_last.store(bytes, std::memory_order_relaxed);
            raise(site_stats.grow_bytes_max, bytes);
            if (bytes > 2ull * 1024 * 1024)
                site_stats.grows_over_2mib.fetch_add(1, std::memory_order_relaxed);
        }
    }

    ::Common::unordered_map<IR::LocationDescriptor, Block> blocks;
    // Split the flat table so a growth replaces one shard rather than allocating a second copy
    // of the entire table at once (waveydude: the single table needed 62 MiB to grow).
    std::array<SiteMap, SiteShardCount> sites;
    const int site_mode = SiteShardMode();
    unsigned long long site_keys = 0;  // targets this group holds (counted with the switch on)
    BlockRangeInformation<u64> ranges;
};

}  // namespace Dynarmic::Backend::X64
