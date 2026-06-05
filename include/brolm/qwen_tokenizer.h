#pragma once

// Qwen3 byte-level BPE tokenizer (GPT-2 family).
//
// Loads a Hugging Face GPT-2-format vocab.json + merges.txt pair. Produces a
// variable-length sequence of int32 token IDs by running byte-level BPE over
// the input, the same algorithm family as brolm::clip::Tokenizer but with the
// GPT-2/Qwen conventions:
//   - Case-preserving: no lowercasing.
//   - No </w> end-of-word marker (CLIP appends one; GPT-2/Qwen do not).
//   - Spaces are preserved and byte-encoded: a space (0x20) maps to the
//     byte-level unicode char 'Ġ' (U+0120). A word following a space is one
//     pre-token, Ġ-prefixed (" word"), distinct from "word" at string start.
//   - Special tokens (<|endoftext|>, <|im_start|>, <|im_end|>, ...) are
//     matched verbatim in the input and emitted as their single vocab id
//     *before* byte-level BPE runs on the surrounding text.
//
// Pre-tokenization is ASCII-focused (mirrors clip::Tokenizer): runs of
// letters, single digits, runs of other non-whitespace punctuation, and
// contraction splits ('s, 't, 're, 've, 'm, 'll, 'd). Unlike CLIP, leading
// whitespace is NOT dropped — it is folded into the following pre-token via
// the GPT-2 leading-space (Ġ) convention, so " hello" is one pre-token.
// Non-ASCII bytes still flow through the byte-level encoding correctly but are
// lumped into the "punctuation run" category rather than the Unicode
// letter/digit categories HF's full regex uses. In practice this matches HF
// behavior on English and code prompts, which is the target use case. Full
// Unicode-property regex pre-tokenization is out of scope.

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "brolm/detail/byte_level_bpe.h"

namespace brotensor::gguf { class File; }

namespace brolm::qwen {

class Tokenizer {
public:
    // Load a HF GPT-2-format vocab.json + merges.txt. Throws std::runtime_error
    // on I/O or parse failure. `extra_special_tokens` are added to the built-in
    // Qwen3 special-token set (<|endoftext|>, <|im_start|>, <|im_end|>); each
    // must be present in vocab.json or it is ignored.
    static Tokenizer load(const std::string& vocab_json_path,
                          const std::string& merges_txt_path,
                          const std::vector<std::string>& extra_special_tokens = {});

    // Build a tokenizer from a Qwen3 GGUF file's metadata. Reads:
    //   tokenizer.ggml.tokens   — array of vocab strings (id = index)
    //   tokenizer.ggml.merges   — array of "a b" merge pairs (priority order)
    //   tokenizer.ggml.token_type    — optional array of i32; type 3 (control)
    //                                  ids are registered as special tokens
    //   tokenizer.ggml.bos/eos/padding_token_id — optional u32
    // The standard Qwen3 specials (<|endoftext|>, <|im_start|>, <|im_end|>) are
    // auto-registered by name when present in the vocab. `extra_special_tokens`
    // works the same way as in load().
    static Tokenizer from_gguf(
        const brotensor::gguf::File& f,
        const std::vector<std::string>& extra_special_tokens = {});

    // Encode `text` into int32 token IDs via byte-level BPE. Substrings that
    // exactly match a registered special token are emitted atomically as that
    // token's single id; the text around them is BPE-encoded normally.
    //
    // Qwen3 has NO beginning-of-sequence (BOS) token. `add_special` is a
    // documented near-no-op hook: when true, the end-of-text token
    // (<|endoftext|>) is appended as an EOS marker if it exists in the vocab.
    // When false (default) nothing is appended.
    std::vector<int32_t> encode(std::string_view text, bool add_special = false) const;

    // Inverse of encode: ids -> vocab pieces -> byte-decode -> UTF-8 string.
    // Special-token ids decode to their literal string form. Unknown ids are
    // skipped. decode(encode(s)) round-trips for ASCII text.
    std::string decode(const std::vector<int32_t>& ids) const;

    // Render a ChatML conversation: for each (role, content) pair emit
    // "<|im_start|>role\ncontent<|im_end|>\n". When `add_generation_prompt` is
    // true a trailing "<|im_start|>assistant\n" is appended to cue generation.
    // The result is plain text — feed it to encode().
    std::string apply_chat_template(
        const std::vector<std::pair<std::string, std::string>>& messages,
        bool add_generation_prompt = true) const;

    std::size_t vocab_count() const { return vocab_.size(); }
    std::size_t merge_count() const { return merge_ranks_.size(); }

    // Register `token` as an atomic special with the given `id`. Updates the
    // forward / inverse special-token maps used by encode() and decode(); the
    // token does NOT need to exist in vocab.json (Qwen3.5 carries its
    // multimodal / control specials only in tokenizer_config.json's
    // added_tokens_decoder, with ids above max(vocab.json)). Re-registering
    // the same string with a different id overwrites the previous binding.
    // Recognises "<|endoftext|>", "<|im_start|>", and "<|im_end|>" and
    // updates the cached endoftext_id_ / im_start_id_ / im_end_id_ so the
    // matching accessors return the new ids.
    void register_special_token(const std::string& token, int32_t id);

    // Key special-token ids, or -1 if absent from the vocab.
    // Convention: eos_id() is <|im_end|> (the ChatML turn terminator Qwen3
    // generates to end an assistant turn); endoftext_id() is <|endoftext|>.
    int eos_id() const { return im_end_id_; }
    int im_start_id() const { return im_start_id_; }
    int im_end_id() const { return im_end_id_; }
    int endoftext_id() const { return endoftext_id_; }

private:
    Tokenizer() = default;

    // BPE-merge one pre-token (already byte-encoded into the GPT-2 unicode
    // space) into a list of vocab strings.
    std::vector<std::string> bpe_(const std::string& token) const;

    // Byte-encode a UTF-8 pre-token, BPE-merge it, look the pieces up in
    // vocab_, and append the resulting ids to `out`.
    void encode_piece_(std::string_view piece, std::vector<int32_t>& out) const;

    std::unordered_map<std::string, int32_t> vocab_;
    // id -> token string, for decode().
    std::unordered_map<int32_t, std::string> id_to_token_;
    // Key: "first\x01second" (\x01 cannot appear in a byte-encoded token).
    std::unordered_map<std::string, int32_t> merge_ranks_;
    // Special tokens matched verbatim before BPE / rendered literally on decode.
    brolm::detail::bpe::SpecialTokens specials_;
    // Byte (0..255) -> UTF-8 of the GPT-2 byte-level unicode mapping.
    std::string byte_to_unicode_[256];
    // Inverse: GPT-2 unicode codepoint -> original byte, for decode().
    std::unordered_map<uint32_t, unsigned char> unicode_to_byte_;

    int im_start_id_   = -1;
    int im_end_id_     = -1;
    int endoftext_id_  = -1;
};

}  // namespace brolm::qwen
