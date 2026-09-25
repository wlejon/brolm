function(brolm_add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE brolm)
    if(MSVC)
        target_compile_options(${name} PRIVATE /W4)
    else()
        target_compile_options(${name} PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME ${name} COMMAND ${name})
endfunction()

brolm_add_test(brolm_test_smoke test_smoke.cpp)
brolm_add_test(brolm_test_tokenizer_clip test_tokenizer_clip.cpp)
brolm_add_test(brolm_test_tokenizer_t5   test_tokenizer_t5.cpp)
brolm_add_test(brolm_test_tokenizer_nllb test_tokenizer_nllb.cpp)
brolm_add_test(brolm_test_nllb_config    test_nllb_config.cpp)
brolm_add_test(brolm_test_nllb_encoder   test_nllb_encoder.cpp)
brolm_add_test(brolm_test_nllb_decoder   test_nllb_decoder.cpp)
brolm_add_test(brolm_test_nllb_beam      test_nllb_beam.cpp)
brolm_add_test(brolm_test_nllb           test_nllb.cpp)
brolm_add_test(brolm_test_qwen_tokenizer test_qwen_tokenizer.cpp)
brolm_add_test(brolm_test_qwen_tokenizer_unicode test_qwen_tokenizer_unicode.cpp)
# HF-oracle fixture lives in tests/ref/; the reference tokenizer.json files
# are looked up in the sibling checkouts (or BROLM_OMNIVOICE_DIR /
# BROLM_QWEN_TTS_DIR) and the comparison skips when absent.
target_compile_definitions(brolm_test_qwen_tokenizer_unicode PRIVATE
    BROLM_TEST_REF_DIR="${CMAKE_CURRENT_SOURCE_DIR}/ref"
    BROLM_SIBLING_DIR="${PROJECT_SOURCE_DIR}/..")
brolm_add_test(brolm_test_llama3_tokenizer test_llama3_tokenizer.cpp)
brolm_add_test(brolm_test_gemma_tokenizer test_gemma_tokenizer.cpp)
brolm_add_test(brolm_test_mistral_tokenizer test_mistral_tokenizer.cpp)
brolm_add_test(brolm_test_mistral3_config   test_mistral3_config.cpp)
brolm_add_test(brolm_test_mistral3_text     test_mistral3_text.cpp)
brolm_add_test(brolm_test_mistral3_generate test_mistral3_generate.cpp)
brolm_add_test(brolm_test_mistral3_vision   test_mistral3_vision.cpp)
brolm_add_test(brolm_test_mistral3_projector test_mistral3_projector.cpp)
brolm_add_test(brolm_test_mistral3_preprocessor test_mistral3_preprocessor.cpp)
brolm_add_test(brolm_test_mistral3_vl       test_mistral3_vl.cpp)
brolm_add_test(brolm_test_mistral3_mmproj   test_mistral3_mmproj.cpp)
brolm_add_test(brolm_test_mistral3_gguf     test_mistral3_gguf.cpp)
brolm_add_test(brolm_test_whisper_tokenizer test_whisper_tokenizer.cpp)
brolm_add_test(brolm_test_clip           test_clip.cpp)
brolm_add_test(brolm_test_t5             test_t5.cpp)
brolm_add_test(brolm_test_llm2vec        test_llm2vec.cpp)
brolm_add_test(brolm_test_quantize_t5    test_quantize_t5.cpp)
brolm_add_test(brolm_test_clip_image     test_clip_image.cpp)
brolm_add_test(brolm_test_clip_score     test_clip_score.cpp)
brolm_add_test(brolm_test_qwen           test_qwen.cpp)
brolm_add_test(brolm_test_qwen_generate  test_qwen_generate.cpp)
brolm_add_test(brolm_test_qwen_gguf      test_qwen_gguf.cpp)
brolm_add_test(brolm_test_qwen_gguf_q8   test_qwen_gguf_q8.cpp)
brolm_add_test(brolm_test_alignment_adapter test_alignment_adapter.cpp)
brolm_add_test(brolm_test_qwen35_config     test_qwen35_config.cpp)
brolm_add_test(brolm_test_qwen35_tokenizer  test_qwen35_tokenizer.cpp)
brolm_add_test(brolm_test_qwen35_preprocessor test_qwen35_preprocessor.cpp)
brolm_add_test(brolm_test_qwen35_vision       test_qwen35_vision.cpp)
brolm_add_test(brolm_test_qwen35_text         test_qwen35_text.cpp)
brolm_add_test(brolm_test_qwen35_vl            test_qwen35_vl.cpp)
brolm_add_test(brolm_test_gemma2               test_gemma2.cpp)
brolm_add_test(brolm_test_qwen3vl_config       test_qwen3vl_config.cpp)
brolm_add_test(brolm_test_qwen3vl_tokenizer    test_qwen3vl_tokenizer.cpp)
brolm_add_test(brolm_test_qwen3vl_preprocessor test_qwen3vl_preprocessor.cpp)
brolm_add_test(brolm_test_qwen3vl_vision       test_qwen3vl_vision.cpp)
brolm_add_test(brolm_test_qwen3vl_text         test_qwen3vl_text.cpp)
brolm_add_test(brolm_test_qwen3vl_vl           test_qwen3vl_vl.cpp)
brolm_add_test(brolm_test_pipeline_multigpu    test_pipeline_multigpu.cpp)
brolm_add_test(brolm_test_qwen35_pipeline_multigpu test_qwen35_pipeline_multigpu.cpp)
brolm_add_test(brolm_test_grammar_sampler         test_grammar_sampler.cpp)
brolm_add_test(brolm_test_grammar_jit             test_grammar_jit.cpp)
target_include_directories(brolm_test_grammar_jit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
brolm_add_test(brolm_test_laya test_laya.cpp)
# Numeric parity against the PyTorch reference (tests/ref/gen_laya_parity.py
# writes the fixture); weights from LAYA_MODEL_DIR or the ../laya sibling.
brolm_add_test(brolm_test_laya_parity test_laya_parity.cpp)
# Request scheduler behaviour (priority, split across GPUs, deadlines,
# shutdown); same weights lookup, skips without them.
brolm_add_test(brolm_test_laya_scheduler test_laya_scheduler.cpp)
target_compile_definitions(brolm_test_laya_scheduler PRIVATE BROLM_SIBLING_DIR="${PROJECT_SOURCE_DIR}/..")
# Soft-token input and its gradient (LayaGrad): soft rows equal to the token
# embeddings reproduce the text answer; the backward matches directional
# finite differences of the forward. Same weights lookup, skips without them.
brolm_add_test(brolm_test_laya_grad test_laya_grad.cpp)
target_compile_definitions(brolm_test_laya_grad PRIVATE BROLM_SIBLING_DIR="${PROJECT_SOURCE_DIR}/..")
target_compile_definitions(brolm_test_laya_parity PRIVATE
    BROLM_TEST_REF_DIR="${CMAKE_CURRENT_SOURCE_DIR}/ref"
    BROLM_SIBLING_DIR="${PROJECT_SOURCE_DIR}/..")

# Bronze JavaScript binding (brolm_api): bro.lm mount points and the
# constructors on globalThis, every loader rejecting no args / a non-string /
# a nonexistent path with a throw, tokenizer construction from a missing
# file. No weights, no CUDA.
if(BROLM_ENABLE_API AND TARGET brolm_api)
    add_executable(brolm_test_api test_lm_api.cpp)
    target_link_libraries(brolm_test_api PRIVATE brolm_api bronze_runtime_shared)
    # Weights-gated sections look under <repo>/weights and skip when absent.
    target_compile_definitions(brolm_test_api PRIVATE BROLM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
    if(MSVC)
        target_compile_options(brolm_test_api PRIVATE /W4)
    else()
        target_compile_options(brolm_test_api PRIVATE -Wall -Wextra)
    endif()
    # bronze's shared runtime is built into BRONZE_SHARED_RUNTIME_DIR, not
    # beside this executable, and the PE loader only looks beside the .exe
    # (0xc0000135 + a modal "dll was not found" otherwise). Stage it next to
    # the test as a post-build step and put both dirs on ctest's PATH.
    if(WIN32)
        add_custom_command(TARGET brolm_test_api POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:bronze_runtime_shared>
                $<TARGET_FILE_DIR:brolm_test_api>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_LINKER_FILE:bronze_runtime_shared>
                $<TARGET_FILE_DIR:brolm_test_api>
        )
    endif()
    # It runs JavaScript, which brass compiles to machine code for the target;
    # on one brass has no code generator for (BRASS_HOST_BACKEND, decided in
    # brass's CMakeLists; x86_64 and AArch64 have one) that code cannot run. The
    # executable still builds there, which is the check that the embed links;
    # the test is not registered rather than registered to bus-error.
    if(NOT BRASS_HOST_BACKEND)
        message(STATUS "brolm: brolm_test_api not registered (brass has no code generator for this target)")
    else()
        add_test(NAME brolm_test_api COMMAND brolm_test_api)
        # The same binding test under bronze's GC stress oracle: a collection
        # on every allocation, from-space poisoned afterwards, so a Value held
        # across an allocating embed call crashes or reads garbage here
        # instead of passing by luck. `ctest -LE gcstress` skips it.
        add_test(NAME brolm_test_api_gcstress COMMAND brolm_test_api)
        set_tests_properties(brolm_test_api_gcstress PROPERTIES LABELS gcstress)
        if(WIN32)
            set_tests_properties(brolm_test_api PROPERTIES
                ENVIRONMENT "PATH=$<TARGET_FILE_DIR:bronze_runtime_shared>;$<TARGET_FILE_DIR:brolm_test_api>;$ENV{PATH}")
            set_tests_properties(brolm_test_api_gcstress PROPERTIES
                ENVIRONMENT "BRONZE_GC_STRESS=1;BRONZE_GC_POISON=1;PATH=$<TARGET_FILE_DIR:bronze_runtime_shared>;$<TARGET_FILE_DIR:brolm_test_api>;$ENV{PATH}")
        else()
            set_tests_properties(brolm_test_api_gcstress PROPERTIES
                ENVIRONMENT "BRONZE_GC_STRESS=1;BRONZE_GC_POISON=1")
        endif()
    endif()
endif()
