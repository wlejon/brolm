#pragma once

// The Hugging Face `tokenizers` BPE model (tokenizer.json "model": {"type":
// "BPE"}), id-based and mirrored step for step: a word's characters become
// symbols (a character missing from the vocab falls back to its "<0xNN>" byte
// tokens when byte_fallback is on, else to the unk token, fused across a run
// when fuse_unk is on), then merges apply lowest rank first, leftmost first on
// ties, from a priority queue whose stale entries are skipped exactly as
// tokenizers' Word::merge_all skips them. ignore_merges looks the whole word
// up first. No continuing_subword_prefix / end_of_word_suffix / dropout
// (inference, and no Laya checkpoint uses them — load() rejects them).
//
// Pre-tokenization is the caller's: this encodes one already-split word, in
// the vocabulary's own character space (GPT-2 byte-level unicode for a
// ByteLevel tokenizer, raw UTF-8 with U+2581 metaspaces for a Metaspace one).

#include "brolm/detail/json.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace brolm::detail::hfbpe {

struct StringHash {
    using is_transparent = void;
    std::size_t operator()(std::string_view s) const noexcept { return std::hash<std::string_view>{}(s); }
};
using Vocab = std::unordered_map<std::string, int32_t, StringHash, std::equal_to<>>;

class Model {
public:
    // From tokenizer.json's "model" object. Throws std::runtime_error on a
    // non-BPE model, an unsupported option, or a merge whose parts or result
    // are missing from the vocab (as tokenizers does).
    void load(const json::Value& model);

    // Append the ids of one pre-tokenized word.
    void encode_word(std::string_view word, std::vector<int32_t>& out) const;

    // Model-vocabulary id of `token`, or -1. (Added tokens live outside the
    // model, as in tokenizers: the caller matches them before encode_word.)
    int32_t id(std::string_view token) const;

    std::size_t vocab_size() const { return vocab_.size(); }
    std::size_t merge_count() const { return merges_.size(); }
    bool byte_fallback() const { return byte_fallback_; }

private:
    struct Merge {
        int32_t rank;
        int32_t new_id;
    };
    static uint64_t pair_key(int32_t a, int32_t b) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b);
    }

    Vocab vocab_;
    std::unordered_map<uint64_t, Merge> merges_;
    int32_t byte_ids_[256];  // "<0xNN>" ids (-1 when absent)
    int32_t unk_id_ = -1;
    bool byte_fallback_ = false;
    bool fuse_unk_ = false;
    bool ignore_merges_ = false;
};

}  // namespace brolm::detail::hfbpe
