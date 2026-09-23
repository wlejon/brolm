#pragma once

#include "brolm/grammar_jit.h"
#include <brass/codegen/grammar_builder.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace brolm {

class VocabIndexer {
public:
    VocabIndexer() = default;
    explicit VocabIndexer(const std::vector<std::string>& vocab,
                          brass::codegen::DfaFilterTokensFn filter_fn = nullptr);

    // Set or update vocabulary and clear mask cache
    void set_vocab(const std::vector<std::string>& vocab,
                   brass::codegen::DfaFilterTokensFn filter_fn = nullptr);

    // Get the valid token bitmask for a given DFA state.
    // Computed lazily and cached per state.
    const std::vector<uint64_t>& get_valid_mask(int32_t state_id, const CompiledDfa& dfa);

    // Precomputes bitmasks for all states in the DFA
    void precompute_all(const CompiledDfa& dfa);

    // Clears the cached state bitmasks
    void clear_cache();

    uint64_t vocab_size() const noexcept { return num_tokens_; }
    size_t num_words() const noexcept { return num_words_; }
    bool has_vocab() const noexcept { return num_tokens_ > 0; }

    const std::vector<uint32_t>& token_offsets() const noexcept { return token_offsets_; }
    const std::string& token_bytes() const noexcept { return token_bytes_; }

private:
    uint64_t num_tokens_ = 0;
    size_t num_words_ = 0;

    // Packed contiguous token representation for SIMD / fast scanning
    std::vector<uint32_t> token_offsets_;
    std::string token_bytes_;

    brass::codegen::DfaFilterTokensFn filter_fn_ = nullptr;

    // Cache: state_id -> valid_mask (num_words words)
    std::unordered_map<int32_t, std::vector<uint64_t>> mask_cache_;
    std::vector<uint64_t> empty_mask_;
    // Bit set for every token with text. A token with no text (a special
    // without a literal form) leaves the DFA where it was, so a byte scan
    // calls it valid in every state; it would pass the mask and never
    // advance the grammar. Every state mask is ANDed with this.
    std::vector<uint64_t> has_text_;
};

}  // namespace brolm
