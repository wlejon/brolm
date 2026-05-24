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
    src/detail/byte_level_bpe.cpp
    src/tokenizer_clip.cpp
    src/tokenizer_t5.cpp
    src/qwen_tokenizer.cpp
    src/whisper_tokenizer.cpp
    src/clip.cpp
    src/t5.cpp
    src/qwen.cpp
    src/qwen_generate.cpp
    src/clip_image.cpp
    src/clip_score.cpp
    src/alignment_adapter.cpp
    src/qwen35_config.cpp
    src/qwen35_tokenizer.cpp
    src/qwen35_preprocessor.cpp
    src/qwen35_vision.cpp
    src/qwen35_text.cpp
    src/qwen35_vl.cpp
)

# --- Install ---
#
# Off by default: brolm is consumed as a sibling repo via the same
# add_subdirectory pattern bro uses for its other libraries, not via
# find_package.
#
# When enabled, this installs only the static library and the public headers.
# It deliberately generates no CMake package config (brolmConfig.cmake): brolm
# links bromath/brotensor as EXCLUDE_FROM_ALL add_subdirectory targets that are
# not themselves installed or exported. An install(EXPORT) set that references
# them fails generation outright ("target brolm ... requires target brotensor
# that is not in any export set"), and a package config would in turn have to
# find_dependency() those siblings, neither of which ships an installed config.
# Consume brolm with add_subdirectory(), not find_package().
option(BROLM_INSTALL "Install the brolm static library and public headers" OFF)
if(BROLM_INSTALL)
    include(GNUInstallDirs)
    install(TARGETS brolm
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

# --- Tests ---
option(BROLM_TESTS "Build brolm tests" ON)
if(BROLM_TESTS AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_testing()
    add_subdirectory(tests)
endif()

# --- Tools (ad-hoc drivers, not run by ctest) ---
option(BROLM_TOOLS "Build brolm ad-hoc tool drivers" ON)
if(BROLM_TOOLS AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_subdirectory(tools)
endif()
