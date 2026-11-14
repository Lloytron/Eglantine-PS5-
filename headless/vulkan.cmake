if(NOT PS5_NATIVE)
    message(FATAL_ERROR "EDEN_PS5_VULKAN requires PS5_NATIVE")
endif()
# Eden's Lossless Scaling frame generation. Upstream gates it behind ENABLE_LSFG,
# which cmake_dependent_option pins to the "ANDROID" condition (CMakeLists.txt:
# "Only Android"), so -DENABLE_LSFG=ON alone is silently forced back OFF. Enable
# it here rather than with -DANDROID=ON: that would also flip ENABLE_OPENGL and
# other options OFF, and the console build needs the OpenGL renderer too.
# On in every build: without the user's own Lossless.dll it does nothing, and the launcher shows
# its settings greyed out (Services::frame_gen_state).
option(EDEN_PS5_FRAMEGEN "Build Eden's Lossless Scaling frame generation (LSFG)" ON)
execute_process(COMMAND python3 "${PORT_ROOT}/tools/prepare-vulkan-port.py"
    "${PROJECT_SOURCE_DIR}" "${PORT_BUILD_DIR}" COMMAND_ERROR_IS_FATAL ANY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${PORT_ROOT}/tools/prepare-vulkan-port.py"
    "${EDEN_PORT_DIR}/transient_images.inc"
    "${EDEN_PORT_DIR}/gpu_time_probe.inc"
    "${EDEN_PORT_DIR}/vulkan_hud.vert" "${EDEN_PORT_DIR}/vulkan_hud.frag"
    "${EDEN_PORT_DIR}/vulkan_hud_prepare.inc" "${EDEN_PORT_DIR}/vulkan_hud_draw.inc"
    "${EDEN_PORT_DIR}/vulkan_loading.inc")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${EDEN_PORT_DIR}/vulkan_download_batch.inc"
    "${EDEN_PORT_DIR}/vulkan_gc_downloads.inc")
list(REMOVE_ITEM video_sources vulkan_common/vulkan_device.cpp renderer_vulkan/vk_scheduler.cpp
    vulkan_common/vulkan_memory_allocator.cpp renderer_vulkan/vk_staging_buffer_pool.cpp)
list(APPEND video_sources "${PORT_BUILD_DIR}/vulkan_device.cpp" "${PORT_BUILD_DIR}/vulkan_scheduler.cpp"
    "${PORT_BUILD_DIR}/vulkan_memory_allocator.cpp" "${PORT_BUILD_DIR}/vulkan_staging_buffer_pool.cpp")
# Split the dispatch timer into its three phases (prepare-vulkan-port.py).
list(REMOVE_ITEM video_sources control/scheduler.cpp)
list(APPEND video_sources "${PORT_BUILD_DIR}/control_scheduler.cpp")
list(REMOVE_ITEM video_sources vulkan_common/vulkan_instance.cpp vulkan_common/vulkan_wrapper.cpp
    vulkan_common/vulkan_library.cpp vulkan_common/vulkan_surface.cpp
    renderer_vulkan/vk_rasterizer.cpp renderer_vulkan/vk_blit_screen.cpp
    renderer_vulkan/present/window_adapt_pass.cpp renderer_vulkan/renderer_vulkan.cpp
    renderer_vulkan/vk_present_manager.cpp renderer_vulkan/vk_swapchain.cpp
    renderer_vulkan/vk_query_cache.cpp)
list(APPEND video_sources "${PORT_BUILD_DIR}/vulkan_instance.cpp" "${PORT_BUILD_DIR}/vulkan_wrapper.cpp"
    "${PORT_BUILD_DIR}/vulkan_rasterizer.cpp" "${PORT_BUILD_DIR}/vulkan_blit_screen.cpp"
    "${PORT_BUILD_DIR}/vulkan_window_adapt_pass.cpp" "${PORT_BUILD_DIR}/vulkan_renderer.cpp"
    "${PORT_BUILD_DIR}/vulkan_library.cpp" "${EDEN_PORT_DIR}/vulkan_surface.cpp" "${EDEN_PORT_DIR}/vulkan_allocator.cpp"
    "${PORT_BUILD_DIR}/vulkan_present_manager.cpp" "${PORT_BUILD_DIR}/vulkan_swapchain.cpp"
    "${PORT_BUILD_DIR}/vulkan_query_cache.cpp")
foreach(name vk_fence_manager vk_buffer_cache vk_texture_cache vk_descriptor_buffer vk_multi_range_buffer vk_compute_pass vk_graphics_pipeline vk_compute_pipeline vk_pipeline_cache)
    list(REMOVE_ITEM video_sources renderer_vulkan/${name}.cpp)
    list(APPEND video_sources "${PORT_BUILD_DIR}/${name}_cost.cpp")
endforeach()

# Scheduler is embedded in RendererVulkan. Existing depfiles may still name the
# upstream header, so invalidate consumers when introducing its port override.
set_property(SOURCE "${PORT_BUILD_DIR}/video_core.cpp" TARGET_DIRECTORY video_core APPEND PROPERTY OBJECT_DEPENDS
    "${PORT_BUILD_DIR}/include/video_core/renderer_vulkan/vk_scheduler.h"
    "${PORT_BUILD_DIR}/include/video_core/vulkan_common/vulkan_memory_allocator.h")
foreach(video_source IN LISTS video_sources)
    if(NOT IS_ABSOLUTE "${video_source}")
        set(video_source "${PROJECT_SOURCE_DIR}/src/video_core/${video_source}")
    endif()
    set_property(SOURCE "${video_source}" TARGET_DIRECTORY video_core APPEND PROPERTY OBJECT_DEPENDS
        "${PORT_BUILD_DIR}/include/video_core/renderer_vulkan/vk_scheduler.h"
        "${PORT_BUILD_DIR}/include/video_core/vulkan_common/vulkan_memory_allocator.h")
endforeach()
target_include_directories(video_core BEFORE PUBLIC "${PORT_BUILD_DIR}/include")
target_include_directories(video_core BEFORE PUBLIC "${PORT_BUILD_DIR}/vulkan-cache")
target_include_directories(video_core PRIVATE "${PORT_BUILD_DIR}" "${EDEN_PORT_DIR}")
# The descriptor-buffer writer override (null descriptors): rebuild its callers once,
# since existing depfiles still name the upstream header.
set_property(SOURCE
    "${PORT_BUILD_DIR}/vk_graphics_pipeline_cost.cpp" "${PORT_BUILD_DIR}/vk_compute_pipeline_cost.cpp"
    "${PORT_BUILD_DIR}/vk_pipeline_cache_cost.cpp" "${PORT_BUILD_DIR}/vulkan_rasterizer.cpp"
    TARGET_DIRECTORY video_core APPEND PROPERTY OBJECT_DEPENDS
    "${PORT_BUILD_DIR}/include/video_core/renderer_vulkan/pipeline_helper.h")
# PresentManager is embedded by value in RendererVulkan. Rebuild consumers whose
# old depfiles still name the upstream header when adding the failure field.
set_property(SOURCE
    "${PROJECT_SOURCE_DIR}/src/video_core/video_core.cpp"
    "${PROJECT_SOURCE_DIR}/src/video_core/renderer_vulkan/vk_turbo_mode.cpp"
    "${PROJECT_SOURCE_DIR}/src/video_core/renderer_vulkan/present/frame_gen.cpp"
    "${PORT_BUILD_DIR}/vulkan_renderer.cpp" "${PORT_BUILD_DIR}/vulkan_blit_screen.cpp"
    "${PORT_BUILD_DIR}/vulkan_rasterizer.cpp" "${PORT_BUILD_DIR}/vulkan_window_adapt_pass.cpp"
    TARGET_DIRECTORY video_core APPEND PROPERTY OBJECT_DEPENDS
    "${PORT_BUILD_DIR}/include/video_core/renderer_vulkan/vk_present_manager.h"
    # FrameGen is embedded by value in RendererVulkan too. The port's copy of its header adds the
    # member that reports generation once per session (tools/prepare-vulkan-port.py).
    "${PORT_BUILD_DIR}/include/video_core/renderer_vulkan/present/frame_gen.h")
# Existing depfiles name the original header; rebuild all filter consumers once
# when introducing the override so the shared struct layout stays consistent.
set_property(SOURCE
    "${PROJECT_SOURCE_DIR}/src/video_core/renderer_opengl/present/layer.cpp"
    "${PORT_BUILD_DIR}/renderer_opengl.cpp"
    "${PROJECT_SOURCE_DIR}/src/video_core/renderer_opengl/gl_blit_screen.cpp"
    "${PROJECT_SOURCE_DIR}/src/video_core/renderer_vulkan/present/layer.cpp"
    TARGET_DIRECTORY video_core APPEND PROPERTY OBJECT_DEPENDS
    "${PORT_BUILD_DIR}/include/video_core/present.h")
# BlitScreen stores WindowAdaptPass by value through unique_ptr; all consumers
# must see the extended resource layout, including cached objects predating it.
set_property(SOURCE
    "${PROJECT_SOURCE_DIR}/src/video_core/renderer_vulkan/present/filters.cpp"
    TARGET_DIRECTORY video_core APPEND PROPERTY OBJECT_DEPENDS
    "${PORT_BUILD_DIR}/include/video_core/renderer_vulkan/present/window_adapt_pass.h")
# A newly generated override is not in old depfiles until the sources rebuild.
# Track it explicitly for the API wrappers and allocator/device setup callers.
set_property(SOURCE
    "${PROJECT_SOURCE_DIR}/src/video_core/vulkan_common/vulkan_wrapper.cpp"
    "${PROJECT_SOURCE_DIR}/src/video_core/vulkan_common/vulkan_device.cpp"
    TARGET_DIRECTORY video_core APPEND PROPERTY OBJECT_DEPENDS
    "${PORT_BUILD_DIR}/include/video_core/vulkan_common/vulkan_wrapper.h")
target_compile_definitions(video_core PRIVATE PS5_NATIVE=1)
# Keep incompatible Mesa/PSBC globals private to the Vulkan archives.
if(EDEN_VULKAN_DRIVER STREQUAL "RADV")
    set(radv_archive "${PORT_ROOT}/build/radv-isolated/libvulkan_radeon.ps5.a")
    if(NOT EXISTS "${radv_archive}")
        message(FATAL_ERROR "Build and isolate the pinned RADV release archive first")
    endif()
    target_link_libraries(video_core PRIVATE "${radv_archive}"
        "${PORT_ROOT}/build/stubs/libSceAgcDriver.so"
        "${sdk}/target/lib/libSceSysmodule.so")
else()
set(vk_isolated "${PORT_ROOT}/build/vulkan-isolated")
foreach(archive libps5vk.a libpsbc.a)
    if(NOT EXISTS "${vk_isolated}/${archive}")
        message(FATAL_ERROR "Stage isolated Vulkan compiler archives before building")
    endif()
endforeach()
target_link_libraries(video_core PRIVATE "${vk_isolated}/libps5vk.a" "${vk_isolated}/libpsbc.a"
    "${PORT_ROOT}/build/stubs/libSceAgcDriver.so"
    "${sdk}/target/lib/libSceSysmodule.so")
endif()

# Lossless Scaling frame generation (EDEN_PS5_FRAMEGEN above). These are the
# sources upstream adds under "if (ENABLE_LSFG)" in src/video_core/CMakeLists.txt;
# paths are relative to src/video_core, like everything else in video_sources.
if(EDEN_PS5_FRAMEGEN)
    list(APPEND video_sources
        frame_gen/lossless_dll.cpp
        frame_gen/lsfg_translate.cpp
        # The port's copies: prepare-vulkan-port.py adds the frame generation diagnostics to
        # these two (tools/prepare-vulkan-port.py), so the generated files replace the originals.
        "${PORT_BUILD_DIR}/vulkan_frame_gen.cpp"
        "${PORT_BUILD_DIR}/vulkan_lsfg_shaders.cpp"
        renderer_vulkan/present/fsr3_frame_gen.cpp
        renderer_vulkan/present/frame_gen_pacer.cpp
        renderer_vulkan/present/lsfg_alpha.cpp
        renderer_vulkan/present/lsfg_beta.cpp
        renderer_vulkan/present/lsfg_chain.cpp
        renderer_vulkan/present/lsfg_common.cpp
        renderer_vulkan/present/lsfg_delta.cpp
        renderer_vulkan/present/lsfg_gamma.cpp
        renderer_vulkan/present/lsfg_generate.cpp
        renderer_vulkan/present/lsfg_mipmaps.cpp)
    # Eden's own gate for those sources. PUBLIC, exactly as upstream declares it
    # (src/video_core/CMakeLists.txt): the flag changes members of RendererVulkan
    # and PresentManager, so every consumer of their headers must see the same
    # layout. The port's generated overrides of those headers carry the same
    # #ifdef blocks, so the whole target stays consistent.
    target_compile_definitions(video_core PUBLIC HAS_LSFG)
    # The port's frontend gate, so headless/main.cpp can ask for frame generation
    # (and say so in the log). eden-headless is created after this file is
    # included, so a directory-level definition reaches it.
    add_compile_definitions(EDEN_PS5_FRAMEGEN=1)
endif()

# EDEN_UP_SHADER_PREP_SOURCE_STAMP (core patch 0596, off by default): the translation id, made at
# build time from the translator as compiled (tools/gen-translation-id.py, after waveydude's
# generator). Ninja reruns it when any hashed file changes (its depfile); it rewrites the header
# only when the id changes. When it cannot account for an input it writes an empty id, and the
# core keeps today's stamp. Hashes only: nothing of it goes into the app but a 32-hex id.
set(eden_tid_header "${PORT_BUILD_DIR}/include/video_core/renderer_vulkan/translation_id.h")
set(eden_tid_context "${PORT_BUILD_DIR}/translation_context.txt")
set(eden_tid_sirit_imported FALSE)
set(eden_tid_sirit_location "")
set(eden_tid_sirit_source "")
if(YUZU_USE_BUNDLED_SIRIT)
    # sirit-ci: a prebuilt library; its file bytes stand for its sources.
    set(eden_tid_sirit_imported TRUE)
    if(DEFINED CPM_PACKAGE_sirit-ci_SOURCE_DIR)
        file(GLOB_RECURSE eden_tid_sirit_libs "${CPM_PACKAGE_sirit-ci_SOURCE_DIR}/*.a")
        list(SORT eden_tid_sirit_libs)
        list(LENGTH eden_tid_sirit_libs eden_tid_sirit_count)
        if(eden_tid_sirit_count EQUAL 1)
            set(eden_tid_sirit_location "${eden_tid_sirit_libs}")
        endif()
    endif()
elseif(DEFINED CPM_PACKAGE_sirit_SOURCE_DIR)
    set(eden_tid_sirit_source "${CPM_PACKAGE_sirit_SOURCE_DIR}")
endif()
set(eden_tid_spirv "")
if(DEFINED CPM_PACKAGE_SPIRV-Headers_SOURCE_DIR AND
   IS_DIRECTORY "${CPM_PACKAGE_SPIRV-Headers_SOURCE_DIR}/include")
    set(eden_tid_spirv "${CPM_PACKAGE_SPIRV-Headers_SOURCE_DIR}/include")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" eden_tid_config)
# file(GENERATE) evaluates the text once per enabled language (C, CXX, ASM). The target options
# carry $<COMPILE_LANGUAGE:...> parts (-fno-rtti for CXX only, the C,CXX warnings), so the texts
# differ and the generate step fails ("written multiple times with different content"). CONDITION
# keeps only the CXX evaluation, the language the translator is compiled in: one write, the same
# text on every generate of the same tree.
file(GENERATE OUTPUT "${eden_tid_context}" CONTENT
"compiler=${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} ${CMAKE_CXX_COMPILER_TARGET}
flags=${CMAKE_CXX_FLAGS} | ${CMAKE_CXX_FLAGS_${eden_tid_config}} | ${CMAKE_BUILD_TYPE}
shader_recompiler_definitions=$<TARGET_PROPERTY:shader_recompiler,COMPILE_DEFINITIONS>
shader_recompiler_options=$<TARGET_PROPERTY:shader_recompiler,COMPILE_OPTIONS>
video_core_definitions=$<TARGET_PROPERTY:video_core,COMPILE_DEFINITIONS>
video_core_options=$<TARGET_PROPERTY:video_core,COMPILE_OPTIONS>
include_dirs=$<TARGET_PROPERTY:video_core,INCLUDE_DIRECTORIES>
sirit_source=${eden_tid_sirit_source}
sirit_imported=${eden_tid_sirit_imported}
sirit_location=${eden_tid_sirit_location}
spirv_headers=${eden_tid_spirv}
driver=${EDEN_VULKAN_DRIVER}
" CONDITION $<COMPILE_LANGUAGE:CXX>)
add_custom_command(OUTPUT "${eden_tid_header}"
    COMMAND python3 "${PORT_ROOT}/tools/gen-translation-id.py"
        --source "${PROJECT_SOURCE_DIR}" --port-build "${PORT_BUILD_DIR}"
        --context "${eden_tid_context}" --out "${eden_tid_header}"
        --depfile "${PORT_BUILD_DIR}/translation_id.d"
        --inventory "${PORT_BUILD_DIR}/translation_inputs.txt"
    DEPENDS "${PORT_ROOT}/tools/gen-translation-id.py" "${eden_tid_context}"
    DEPFILE "${PORT_BUILD_DIR}/translation_id.d"
    COMMENT "EDEN translation id (EDEN_UP_SHADER_PREP_SOURCE_STAMP)"
    VERBATIM)
add_custom_target(eden_translation_id DEPENDS "${eden_tid_header}")
add_dependencies(video_core eden_translation_id)
set_property(SOURCE "${PROJECT_SOURCE_DIR}/src/video_core/renderer_vulkan/vk_pipeline_prep.cpp"
    TARGET_DIRECTORY video_core APPEND PROPERTY OBJECT_DEPENDS "${eden_tid_header}")
