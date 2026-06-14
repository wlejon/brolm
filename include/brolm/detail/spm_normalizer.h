#pragma once

// SentencePiece "Precompiled" charsmap normalizer.
//
// HuggingFace tokenizer.json files for SentencePiece models (NLLB, T5, …) carry
// a `Precompiled` normalizer whose `precompiled_charsmap` is SentencePiece's
// serialized normalization map: a Darts-clone double-array trie over UTF-8 byte
// sequences plus a blob of null-terminated replacement strings. Normalizing a
// string means, at each position, taking the LONGEST trie key that is a prefix
// of the remaining input and emitting its replacement (≈ NFKC plus SPM's custom
// folds: full-width forms, ligatures, circled digits, control-char stripping);
// positions with no trie key are copied one UTF-8 character at a time.
//
// This is the exact algorithm from SentencePiece's Normalizer::NormalizePrefix
// + Darts::DoubleArray::commonPrefixSearch, reimplemented so brolm needs no
// SentencePiece dependency. Shared across SPM-family tokenizers.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace brolm::detail::spm {

class PrecompiledNormalizer {
public:
    PrecompiledNormalizer() = default;

    // Build from the RAW (already base64-decoded) precompiled_charsmap bytes.
    // An empty/too-short blob yields an empty (identity) normalizer.
    explicit PrecompiledNormalizer(std::string_view charsmap);

    // True when no usable charsmap was loaded — callers should then skip
    // normalization (treat the text as-is).
    bool empty() const { return units_.empty(); }

    // Return the SentencePiece-normalized form of `input`.
    std::string normalize(std::string_view input) const;

private:
    // Longest trie key that is a prefix of key[0..key_len); on success sets
    // `out_len` (matched byte length) and `out_val` (offset into normalized_).
    bool longest_match(const std::uint8_t* key, std::size_t key_len,
                       std::uint32_t& out_len, std::uint32_t& out_val) const;

    std::vector<std::uint32_t> units_;   // Darts-clone double-array
    std::string normalized_;             // null-terminated replacement blob
};

}  // namespace brolm::detail::spm
