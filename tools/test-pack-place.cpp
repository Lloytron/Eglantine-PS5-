// SPDX-License-Identifier: GPL-3.0-or-later
// Host test for src/pack_place.h (EDEN_GUEST_SWAP_PACK_IN_PLACE, PackOut in src/guest_memory_service.inc).
//   c++ -std=c++20 -O2 -fsanitize=address,undefined -Isrc tools/test-pack-place.cpp -o /tmp/pp && /tmp/pp
// The pool is modelled in 16 KiB pages with the lowest room that fits (what the console does,
// src/direct_memory.c): guest slots are 2 MiB blocks at 2 MiB steps, a packed slot keeps `kept` pages.
// The pool starts full of guest slots, as at the memory wall of the W-13 to W-19 sessions.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "pack_place.h"

using namespace Eden::PackPlace;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

namespace {
constexpr int SlotPages = 128;
struct Pool {
    std::vector<char> used;
    explicit Pool(int slots) : used(static_cast<size_t>(slots) * SlotPages, 0) {}
    int Take(int pages, int align) {
        const int n = static_cast<int>(used.size());
        for (int at = 0; at + pages <= n;) {
            const void* hit = std::memchr(used.data() + at, 1, static_cast<size_t>(pages));
            if (hit == nullptr) {
                std::fill_n(used.begin() + at, pages, 1);
                return at;
            }
            const int taken = static_cast<int>(static_cast<const char*>(hit) - used.data());
            at = (taken + 1 + align - 1) / align * align;
        }
        return -1;
    }
    void Give(int at, int pages) { std::fill_n(used.begin() + at, pages, 0); }
    int FreePages() const { return static_cast<int>(std::count(used.begin(), used.end(), 0)); }
    int WholeFreeSlots() const {
        int whole = 0;
        for (size_t slot = 0; slot * SlotPages < used.size(); ++slot)
            whole += std::all_of(used.begin() + slot * SlotPages, used.begin() + (slot + 1) * SlotPages,
                                 [](char page) { return page == 0; });
        return whole;
    }
};
struct Result { int free_mib, whole_mib, in_place, own_block, not_packed, unpack_refused; };

// `slots` guest slots fill the pool; the swapper keeps about 60% packed while touches bring slots back.
Result Run(Mode mode, unsigned seed, int slots = 512, int steps = 20000) {
    std::mt19937 random{seed};
    Pool pool{slots};
    struct Slot { int at = -1, kept = 0; };
    std::vector<Slot> guest(static_cast<size_t>(slots));
    for (Slot& slot : guest) slot.at = pool.Take(SlotPages, SlotPages);
    std::vector<int> own(static_cast<size_t>(slots)), out;
    for (int i = 0; i < slots; ++i) own[static_cast<size_t>(i)] = i;
    Result result{};
    std::normal_distribution<double> packs_to{64.0, 18.0}; // about half, as the W-19 run packed
    const auto pick = [&](std::vector<int>& from) {
        const size_t index = random() % from.size();
        const int id = from[index];
        from[index] = from.back();
        from.pop_back();
        return id;
    };
    for (int step = 0; step < steps; ++step) {
        const bool pack = static_cast<int>(out.size()) < slots * 6 / 10 && (out.empty() || random() % 100 < 55);
        if (pack) {
            const int id = pick(own);
            Slot& slot = guest[static_cast<size_t>(id)];
            const int kept = std::clamp(static_cast<int>(packs_to(random)), 8, 112);
            int block = -1;
            if (AsksPoolFirst(true, mode)) block = pool.Take(kept, 1);
            switch (Choose(true, mode, block >= 0)) {
            case Place::OwnBlock:
                pool.Give(slot.at, SlotPages);
                slot = Slot{block, kept};
                ++result.own_block;
                out.push_back(id);
                break;
            case Place::InPlace:
                CHECK(block < 0); // nothing taken from the pool is left behind
                pool.Give(slot.at + kept, SlotPages - kept);
                slot.kept = kept;
                ++result.in_place;
                out.push_back(id);
                break;
            case Place::None:
                CHECK(block < 0);
                ++result.not_packed;
                own.push_back(id);
                break;
            }
        } else {
            const int id = pick(out);
            Slot& slot = guest[static_cast<size_t>(id)];
            const int block = pool.Take(SlotPages, SlotPages);
            if (block < 0) { // the guest touched a packed slot and the pool has no 2 MiB block
                ++result.unpack_refused;
                out.push_back(id);
                continue;
            }
            pool.Give(slot.at, slot.kept);
            slot = Slot{block, 0};
            own.push_back(id);
        }
    }
    result.free_mib = pool.FreePages() / 64;
    result.whole_mib = pool.WholeFreeSlots() * 2;
    return result;
}
} // namespace

int main() {
    // The setting. "1" keeps its old meaning (any value starting with 1); only "2" is the new mode.
    CHECK(ModeOf(nullptr) == Mode::Own);
    CHECK(ModeOf("") == Mode::Own);
    CHECK(ModeOf("0") == Mode::Own);
    CHECK(ModeOf("1") == Mode::InPlace);
    CHECK(ModeOf("2") == Mode::PoolFirst);
    CHECK(ModeOf("20") == Mode::Own);
    CHECK(ModeOf("3") == Mode::Own);
    CHECK(ModeOf("on") == Mode::Own);

    // The choice, every case.
    for (const Mode mode : {Mode::Own, Mode::InPlace, Mode::PoolFirst})
        for (const bool gave : {false, true}) CHECK(Choose(false, mode, gave) == Place::None);
    CHECK(AsksPoolFirst(true, Mode::Own) && AsksPoolFirst(true, Mode::PoolFirst));
    CHECK(!AsksPoolFirst(true, Mode::InPlace)); // =1 asks the pool for nothing, as before
    CHECK(!AsksPoolFirst(false, Mode::Own) && !AsksPoolFirst(false, Mode::PoolFirst));
    CHECK(Choose(true, Mode::Own, true) == Place::OwnBlock);
    CHECK(Choose(true, Mode::Own, false) == Place::None);       // as before: the slot waits
    CHECK(Choose(true, Mode::InPlace, false) == Place::InPlace);
    CHECK(Choose(true, Mode::PoolFirst, true) == Place::OwnBlock);
    CHECK(Choose(true, Mode::PoolFirst, false) == Place::InPlace); // packing never stops for memory

    for (const unsigned seed : {1u, 2u, 3u, 4u}) {
        const Result in_place = Run(Mode::InPlace, seed);
        const Result pool_first = Run(Mode::PoolFirst, seed);
        const Result own = Run(Mode::Own, seed);
        std::printf("seed %u  in place: %d MiB free, %d whole, %d unpacks refused | pool first: %d MiB free, %d whole, "
                    "%d in place of %d, %d unpacks refused | own block only: %d not packed\n",
                    seed, in_place.free_mib, in_place.whole_mib, in_place.unpack_refused, pool_first.free_mib,
                    pool_first.whole_mib, pool_first.in_place, pool_first.in_place + pool_first.own_block,
                    pool_first.unpack_refused, own.not_packed);
        // What the consoles showed with =1: hundreds of MiB free and no whole block among them.
        CHECK(in_place.free_mib > 200 && in_place.whole_mib == 0);
        CHECK(in_place.unpack_refused > 0);
        // =2: most of what is free is whole 2 MiB blocks, and about as much is free.
        CHECK(pool_first.whole_mib * 10 >= pool_first.free_mib * 7);
        CHECK(pool_first.free_mib * 10 >= in_place.free_mib * 9);
        // With the pool full the first slots can only go in place; that is rare afterwards.
        CHECK(pool_first.in_place >= 1);
        CHECK(pool_first.in_place * 20 < pool_first.own_block);
        CHECK(pool_first.not_packed == 0); // packing never stopped for memory
        // Without the fallback a full pool packs nothing at all (run 161).
        CHECK(own.own_block == 0 && own.not_packed > 0 && own.free_mib == 0);
    }
    if (failures == 0) std::puts("PASS");
    return failures != 0;
}
