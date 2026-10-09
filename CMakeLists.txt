cmake_minimum_required(VERSION 3.24)

# Pre-1.0: siblings vendor this repo via add_subdirectory and build it from
# source, so the version is a pin point for a tag, not an ABI promise.
project(brolm VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Coverage. Scoped to a top-level build: a consumer that add_subdirectory()s
# brolm has no business having its own build instrumented from in here.
option(BROLM_COVERAGE "Instrument the build for gcov coverage (GCC/Clang)" OFF)
if(BROLM_COVERAGE AND PROJECT_IS_TOP_LEVEL)
    if(MSVC)
        message(FATAL_ERROR
            "BROLM_COVERAGE requires GCC or Clang (gcov instrumentation); MSVC is unsupported.")
    endif()
    add_compile_options(--coverage -O0 -g)
    add_link_options(--coverage)
    # Lets tests drop wall-clock bounds that instrumentation cannot meet.
    add_compile_definitions(BROLM_COVERAGE_BUILD=1)
    message(STATUS "brolm: coverage instrumentation ON (-O0)")
endif()

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
# Vulkan compute backend (brotensor's docs/vulkan.md), the AMD GPU path.
# brolm needs no GLSL of its own, so this only forwards it.
option(BROTENSOR_WITH_VULKAN "Build the Vulkan GPU backend" OFF)

# --- Dependencies: bromath (header-only), bronze (+ brass), brotensor, broimage ---
#
# Each resolves through cmake/bro_deps.cmake, like every repo in the bro
# ecosystem: an existing target (a superbuild such as bro already added it, and
# its pins win) first; otherwise the ../<name> working tree beside the top-level
# project; otherwise the commit pinned here, fetched at configure. Leaves first,
# so broimage finds bromath, brass and brotensor already there.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/bro_deps.cmake)

bro_dependency(bromath GITHUB wlejon/bromath TARGET bromath::bromath)

# bronze, the JavaScript runtime the api binds. It brings brass, its code
# generator, which brolm's JIT paths link too. The api is bound across bronze's
# C++ embed boundary, so it has to build against the bronze the surrounding
# tree builds against.
bro_dependency(bronze GITHUB wlejon/bronze TARGET bronze_runtime_shared
               OPTIONS BRONZE_BUILD_SHARED_RUNTIME=ON BRONZE_BUILD_TESTS=OFF)

bro_dependency(brotensor GITHUB wlejon/brotensor)
bro_dependency(broimage GITHUB wlejon/broimage)


# --- brolm library ---

add_library(brolm STATIC)
add_library(brolm::brolm ALIAS brolm)

target_include_directories(brolm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(brolm PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)
target_compile_features(brolm PUBLIC cxx_std_20)
target_link_libraries(brolm PUBLIC
    bromath::bromath
    brotensor::brotensor
    broimage::broimage
)

# brass arrives with bronze (above).
if(TARGET brass)
    target_link_libraries(brolm PUBLIC brass)
    target_compile_definitions(brolm PUBLIC BROLM_HAS_BRASS_JIT=1)
endif()

if(MSVC)
    target_compile_options(brolm PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(brolm PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
endif()

target_sources(brolm PRIVATE
    src/version.cpp
    src/grammar.cpp
    src/grammar/dfa_compiler.cpp
    src/grammar/vocab_indexer.cpp
    src/grammar/grammar_jit.cpp
    src/sampler.cpp
    src/detail/byte_level_bpe.cpp
    src/detail/unicode.cpp
    src/detail/spm_unigram.cpp
    src/detail/spm_bpe.cpp
    src/detail/hf_bpe.cpp
    src/detail/spm_normalizer.cpp
    src/detail/dense_decoder.cpp
    src/detail/generate.cpp
    src/detail/profile.cpp
    src/tokenizer_clip.cpp
    src/tokenizer_t5.cpp
    src/tokenizer_nllb.cpp
    src/nllb_config.cpp
    src/detail/nllb_ops.cpp
    src/nllb_encoder.cpp
    src/nllb_decoder.cpp
    src/nllb_beam.cpp
    src/nllb.cpp
    src/qwen_tokenizer.cpp
    src/llama3_tokenizer.cpp
    src/gemma_tokenizer.cpp
    src/mistral_tokenizer.cpp
    src/mistral3_config.cpp
    src/mistral3_text.cpp
    src/mistral3_generate.cpp
    src/mistral3_vision.cpp
    src/mistral3_projector.cpp
    src/mistral3_preprocessor.cpp
    src/mistral3_vl.cpp
    src/whisper_tokenizer.cpp
    src/clip.cpp
    src/t5.cpp
    src/llm2vec.cpp
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
    src/qwen35_text_weights.cpp
    src/qwen35_vl.cpp
    src/gemma2_config.cpp
    src/gemma2.cpp
    src/qwen3vl_config.cpp
    src/qwen3vl_tokenizer.cpp
    src/qwen3vl_vision.cpp
    src/qwen3vl_text.cpp
    src/qwen3vl_vl.cpp
    src/qwen3vl_prompt.cpp
    src/modernbert_config.cpp
    src/modernbert.cpp
    src/laya_config.cpp
    src/laya_tokenizer.cpp
    src/laya_sequence.cpp
    src/laya.cpp
    src/laya_batch.cpp
    src/laya_questions.cpp
    src/laya_scheduler.cpp
    src/laya_grad.cpp
    src/laya_grad_encoder.cpp
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

# --- Standalone Bronze JavaScript API (before tests, which add a test against it) ---
option(BROLM_ENABLE_API "Build Bronze JavaScript API" ON)
if(BROLM_ENABLE_API)
    add_subdirectory(src/api)
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

