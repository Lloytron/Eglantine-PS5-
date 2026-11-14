# Preserve the upstream container; repair its throwing-constructor contract.
file(READ "${SLOT_VECTOR_INPUT}" slot_source)
set(old_insert "[[nodiscard]] SlotId insert(Args&&... args) noexcept {\n        const u32 index = FreeValueIndex();\n        new (&values[index].object) T(std::forward<Args>(args)...);")
string(FIND "${slot_source}" "${old_insert}" insert_offset)
if(insert_offset LESS 0)
    message(FATAL_ERROR "Pinned SlotVector insertion changed")
endif()
string(REPLACE "${old_insert}"
    "[[nodiscard]] SlotId insert(Args&&... args) {\n        const u32 index = FreeValueIndex();\n        try {\n            new (&values[index].object) T(std::forward<Args>(args)...);\n        } catch (...) {\n            free_list.push_back(index); // Restores the popped slot without allocating.\n            throw;\n        }"
    slot_source "${slot_source}")
# t174a (13 Nov 2026): the table's growth may fail too. Upstream marks FreeValueIndex and Reserve
# noexcept, so a refused doubling (run1395: the image table 2048 -> 4096 slots, one 10,354,696 byte
# operator new[] on GPUBack) went to std::terminate before any caller could see it. Without noexcept
# the std::bad_alloc reaches insert's caller; core patch 0890 makes Reserve allocate everything
# before it moves a slot (nothing has changed when it throws), and 0892 lets the image and buffer
# refusal paths take it (TextureCacheRuntime::SurvivesImageRefusal, BufferCache::JoinBuffer).
# Elsewhere it still ends the program, as before (no caller there catches it).
foreach(pinned
        "[[nodiscard]] u32 FreeValueIndex() noexcept {"
        "void Reserve(size_t new_capacity) noexcept {")
    string(FIND "${slot_source}" "${pinned}" pinned_offset)
    if(pinned_offset LESS 0)
        message(FATAL_ERROR "Pinned SlotVector growth changed: ${pinned}")
    endif()
    string(REPLACE " noexcept {" " {" unpinned "${pinned}")
    string(REPLACE "${pinned}" "${unpinned}" slot_source "${slot_source}")
endforeach()
file(WRITE "${SLOT_VECTOR_OUTPUT}.in" "${slot_source}")
configure_file("${SLOT_VECTOR_OUTPUT}.in" "${SLOT_VECTOR_OUTPUT}" COPYONLY)
