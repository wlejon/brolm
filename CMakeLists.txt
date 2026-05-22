cmake_minimum_required(VERSION 3.24)
project(brolm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/MP>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    )
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
endif()

# --- GPU backend selection ---
#
# brolm composes brotensor ops — it ships no GPU kernels of its own, so it
# enables no CUDA/Metal language. It only forwards the backend choice so a
# standalone GPU build resolves brotensor's CUDA/Metal backend. Built inside
# bro, brotensor is already a target (added transitively) and the
# if(NOT TARGET ...) guards below reuse it, backend and all.
option(BROTENSOR_WITH_CUDA  "Build the CUDA GPU backend"  OFF)
option(BROTENSOR_WITH_METAL "Build the Metal GPU backend" OFF)

# --- Siblings: bromath (header-only) + brotensor (tensor + ops) ---
#
# Prefer standalone repos at ../<name>; fall back to third_party/ submodules.
# Matches the pattern documented in bro/docs/multi-repo-workflow.md.

set(BROMATH_DIR "${CMAKE_SOURCE_DIR}/../bromath" CACHE PATH
    "Path to standalone bromath repo (set to a missing path to force submodule fallback)")
if(NOT TARGET bromath)
    if(EXISTS "${BROMATH_DIR}/CMakeLists.txt")
        add_subdirectory("${BROMATH_DIR}" "${CMAKE_BINARY_DIR}/bromath" EXCLUDE_FROM_ALL)
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/third_party/bromath/CMakeLists.txt")
        add_subdirectory(third_party/bromath EXCLUDE_FROM_ALL)
    else()
        message(FATAL_ERROR
            "bromath not found at ${BROMATH_DIR} or third_party/bromath. "
            "Clone github.com/wlejon/bromath next to this repo, or override with -DBROMATH_DIR=...")
    endif()
endif()

set(BROTENSOR_DIR "${CMAKE_SOURCE_DIR}/../brotensor" CACHE PATH
    "Path to standalone brotensor repo (set to a missing path to force submodule fallback)")
if(NOT TARGET brotensor)
    if(EXISTS "${BROTENSOR_DIR}/CMakeLists.txt")
        add_subdirectory("${BROTENSOR_DIR}" "${CMAKE_BINARY_DIR}/brotensor" EXCLUDE_FROM_ALL)
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/third_party/brotensor/CMakeLists.txt")
        add_subdirectory(third_party/brotensor EXCLUDE_FROM_ALL)
    else()
        message(FATAL_ERROR
            "brotensor not found at ${BROTENSOR_DIR} or third_party/brotensor. "
            "Clone the brotensor repo next to this repo, or override with -DBROTENSOR_DIR=...")
    endif()
endif()

# --- brolm library ---

add_library(brolm STATIC)
add_library(brolm::brolm ALIAS brolm)

target_include_directories(brolm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(brolm PUBLIC cxx_std_20)
target_link_libraries(brolm PUBLIC
    bromath::bromath
    brotensor::brotensor
)

if(MSVC)
    target_compile_options(brolm PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(brolm PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
endif()

target_sources(brolm PRIVATE
    src/version.cpp
    src/tokenizer_clip.cpp
    src/tokenizer_t5.cpp
    src/qwen_tokenizer.cpp
    src/clip.cpp
    src/t5.cpp
    src/clip_image.cpp
    src/clip_score.cpp
)

# --- Install / export ---
#
# Off by default: brolm is consumed as a sibling repo via the same
# add_subdirectory pattern bro uses for its other libraries, not via
# find_package.
option(BROLM_INSTALL "Generate install targets for brolm" OFF)
if(BROLM_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    install(TARGETS brolm
        EXPORT brolmTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    install(EXPORT brolmTargets
        FILE brolmTargets.cmake
        NAMESPACE brolm::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/brolm
    )

    set(_cfg_in "${CMAKE_CURRENT_BINARY_DIR}/brolmConfig.cmake.in")
    file(WRITE "${_cfg_in}"
        "@PACKAGE_INIT@\n"
        "include(\"\${CMAKE_CURRENT_LIST_DIR}/brolmTargets.cmake\")\n"
    )
    configure_package_config_file("${_cfg_in}"
        "${CMAKE_CURRENT_BINARY_DIR}/brolmConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/brolm
    )
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/brolmConfig.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/brolm
    )
endif()

# --- Tests ---
option(BROLM_TESTS "Build brolm tests" ON)
if(BROLM_TESTS AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_testing()
    add_subdirectory(tests)
endif()
