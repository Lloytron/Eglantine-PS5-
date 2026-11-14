// SPDX-License-Identifier: GPL-3.0-or-later
// 0870, from waveydude (Wave Ocarina 20, a20de88).
// Where a packed guest slot's bytes go (src/guest_memory_service.inc, PackOut). Header only and free
// of the PS5 SDK, so tools/test-pack-place.cpp tests it with a model of the pool on any host.
//
// EDEN_GUEST_SWAP_PACK_IN_PLACE:
//   unset, 0  a block of its own from the pool; the slot's 2 MiB block goes back whole. With no block
//             in the pool for the bytes the slot is not packed.
//   1         always in the front of the slot's own block; the block's tail goes back. No block is
//             asked of the pool, but the tail is a hole under 2 MiB in the middle of the pool that no
//             guest slot, heap piece or graphics block (all whole 2 MiB pages) can take: W-13 to W-19
//             ended with 430 to 1,340 MiB of the pool free and its largest free block 1 to 8 MiB.
//   2         a block of its own first (the copies fill the holes, whole 2 MiB blocks go back); in
//             place only when the pool has no block for the bytes, so packing never stops for memory.
#pragma once

namespace Eden::PackPlace {

enum class Mode : int { Own = 0, InPlace = 1, PoolFirst = 2 };
enum class Place { None, OwnBlock, InPlace };

constexpr Mode ModeOf(const char* value) noexcept {
    if (value == nullptr) return Mode::Own;
    return value[0] == '1' ? Mode::InPlace : value[0] == '2' && value[1] == '\0' ? Mode::PoolFirst : Mode::Own;
}

// A block of the slot's own is asked of the pool before anything else.
constexpr bool AsksPoolFirst(bool worth, Mode mode) noexcept {
    return worth && mode != Mode::InPlace;
}

// `mode`: the setting while the console lets a block's tail go back on its own (asked once a
// session), else Mode::Own. `pool_gave_block`: the pool had a block for the packed bytes (asked only
// when AsksPoolFirst).
constexpr Place Choose(bool worth, Mode mode, bool pool_gave_block) noexcept {
    if (!worth) return Place::None;
    if (mode == Mode::InPlace) return Place::InPlace;
    if (pool_gave_block) return Place::OwnBlock;
    return mode == Mode::PoolFirst ? Place::InPlace : Place::None;
}

} // namespace Eden::PackPlace
