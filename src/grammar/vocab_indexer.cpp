#include "grammar/vocab_indexer.h"

#include <algorithm>
#include <cstring>

namespace brolm {

VocabIndexer::VocabIndexer(const std::vector<std::string>& vocab,
                           brass::codegen::DfaFilterTokensFn filter_fn) {
    set_vocab(vocab, filter_fn);
}

void VocabIndexer::set_vocab(const std::vector<std::string>& vocab,
                             brass::codegen::DfaFilterTokensFn filter_fn) {
    filter_fn_ = filter_fn;
    num_tokens_ = vocab.size();
    num_words_ = (num_tokens_ + 63) / 64;
    empty_mask_.assign(num_words_, 0ULL);
    mask_cache_.clear();

    token_offsets_.clear();
    token_bytes_.clear();
    token_offsets_.reserve(num_tokens_ + 1);

    size_t total_bytes = 0;
    for (const auto& tok : vocab) {
        total_bytes += tok.size();
    }
    token_bytes_.reserve(total_bytes);

    has_text_.assign(num_words_, 0ULL);
    token_offsets_.push_back(0);
    for (size_t i = 0; i < vocab.size(); ++i) {
        const auto& tok = vocab[i];
        token_bytes_ += tok;
        token_offsets_.push_back(static_cast<uint32_t>(token_bytes_.size()));
        if (!tok.empty()) has_text_[i / 64] |= (1ULL << (i % 64));
    }
}

const std::vector<uint64_t>& VocabIndexer::get_valid_mask(int32_t state_id, const CompiledDfa& dfa) {
    if (state_id < 0 || state_id >= dfa.num_states || num_tokens_ == 0) {
        return empty_mask_;
    }

    auto it = mask_cache_.find(state_id);
    if (it != mask_cache_.end()) {
        return it->second;
    }

    std::vector<uint64_t> mask(num_words_, 0ULL);

    if (filter_fn_ && !token_offsets_.empty()) {
        filter_fn_(
            dfa.transition_table.data(),
            state_id,
            token_offsets_.data(),
            reinterpret_cast<const uint8_t*>(token_bytes_.data()),
            num_tokens_,
            mask.data()
        );
    } else {
        // Fast DFA scanning fallback
        for (uint64_t i = 0; i < num_tokens_; ++i) {
            uint32_t start = token_offsets_[i];
            uint32_t end = token_offsets_[i + 1];
            int32_t s = state_id;

            for (uint32_t j = start; j < end; ++j) {
                uint8_t b = static_cast<uint8_t>(token_bytes_[j]);
                s = dfa.transition_table[static_cast<size_t>(s) * 256 + b];
                if (s < 0) break;
            }

            if (s >= 0) {
                mask[i / 64] |= (1ULL << (i % 64));
            }
        }
    }

    for (size_t w = 0; w < num_words_; ++w) mask[w] &= has_text_[w];

    auto [insert_it, _] = mask_cache_.emplace(state_id, std::move(mask));
    return insert_it->second;
}

void VocabIndexer::precompute_all(const CompiledDfa& dfa) {
    if (!dfa.is_valid() || num_tokens_ == 0) return;
    for (int32_t s = 0; s < dfa.num_states; ++s) {
        get_valid_mask(s, dfa);
    }
}

void VocabIndexer::clear_cache() {
    mask_cache_.clear();
}

}  // namespace brolm
