#pragma once

#include <brass/codegen/grammar_builder.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace brolm {

class VocabIndexer;

// ─── Compiled DFA Transition Table ───────────────────────────────────────────

struct CompiledDfa {
    int32_t num_states = 0;
    int32_t start_state = 0;
    std::vector<int32_t> transition_table; // size = num_states * 256
    std::vector<bool> is_accept_state;     // size = num_states

    bool is_valid() const noexcept {
        return num_states > 0 &&
               transition_table.size() == static_cast<size_t>(num_states) * 256 &&
               is_accept_state.size() == static_cast<size_t>(num_states);
    }

    int32_t step(int32_t state, uint8_t byte_val) const noexcept {
        if (state < 0 || state >= num_states) return -1;
        return transition_table[static_cast<size_t>(state) * 256 + byte_val];
    }
};

// ─── Grammar Kernel Compilation & Disk Cache Statistics ─────────────────────

struct GrammarKernelCacheStats {
    size_t cache_hits = 0;
    size_t cache_misses = 0;
    size_t smt_guards_eliminated = 0;
    size_t smt_queries_solved = 0;
    size_t superopt_masks_elided = 0;
    size_t superopt_extensions_elided = 0;
    size_t superopt_dags_synthesized = 0;
    size_t superopt_constants_synthesized = 0;
    size_t superopt_lir_windows_optimized = 0;
    size_t mir_functions_verified_equivalent = 0;
    bool loaded_from_disk_cache = false;
    std::string cache_file_path;
};

// ─── JIT Grammar Engine ──────────────────────────────────────────────────────

class JitGrammar {
public:
    brass::codegen::LogitMaskFn mask_fn = nullptr;
    brass::codegen::DfaScanStringFn scan_fn = nullptr;
    int32_t current_state_ = 0;

    explicit JitGrammar(CompiledDfa dfa);
    JitGrammar(CompiledDfa dfa, const std::vector<std::string>& vocab);
    JitGrammar(const JitGrammar& other);
    JitGrammar& operator=(const JitGrammar& other);
    JitGrammar(JitGrammar&& other) noexcept;
    JitGrammar& operator=(JitGrammar&& other) noexcept;
    ~JitGrammar();

    // Advances current_state_ by consuming text using scan_fn (or scalar fallback).
    // Returns true on success, false if the sequence was rejected (state becomes -1).
    bool accept(std::string_view text);

    // Checks if appending text is valid from current_state_ without modifying state.
    bool can_accept(std::string_view text) const;

    // Checks if current_state_ is an accepting state.
    bool is_accepted() const;

    // Resets current_state_ back to dfa_.start_state.
    void reset();

    // Sets logits of all invalid next tokens to -INFINITY using JIT vectorized mask_fn.
    // When is_accepted() is false, eos_id is also masked out.
    void mask_logits(float* logits, int vocab_size,
                     const std::vector<std::string>& vocab_tokens,
                     int eos_id = -1) const;

    // Accessors
    const CompiledDfa& dfa() const noexcept { return dfa_; }
    int32_t current_state() const noexcept { return current_state_; }
    bool is_jit_compiled() const noexcept { return mask_fn != nullptr && scan_fn != nullptr; }
    bool is_loaded_from_disk_cache() const noexcept { return loaded_from_disk_cache_; }
    brass::codegen::DfaFilterTokensFn filter_fn() const noexcept { return filter_fn_; }

    // Optional precomputation of token validity masks across all states
    void precompute_vocab(const std::vector<std::string>& vocab_tokens);

    // Direct access to internal VocabIndexer (for testing/introspection)
    VocabIndexer* vocab_indexer() const noexcept { return vocab_indexer_.get(); }

    // Persistent on-disk AOT kernel cache introspection and test hooks
    static GrammarKernelCacheStats kernel_cache_stats();
    static std::string kernel_cache_path();
    static void reset_kernel_cache_for_testing();

private:
    CompiledDfa dfa_;
    mutable std::unique_ptr<VocabIndexer> vocab_indexer_;
    mutable std::vector<uint64_t> scratch_mask_;

    brass::codegen::KernelFunction mask_kfn_;
    brass::codegen::KernelFunction scan_kfn_;
    brass::codegen::KernelFunction filter_kfn_;
    std::shared_ptr<void> aot_lib_;
    brass::codegen::DfaFilterTokensFn filter_fn_ = nullptr;
    bool loaded_from_disk_cache_ = false;

    void compile_kernels();
};

}  // namespace brolm
