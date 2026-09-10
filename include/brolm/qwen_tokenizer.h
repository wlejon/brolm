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
// Pre-tokenization reproduces the Hugging Face `tokenizers` Split regex of
// the Qwen2/Qwen3 tokenizer.json exactly, over code points with real Unicode
// properties (brolm/detail/unicode.h):
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}
//   | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// so CJK, Cyrillic, Arabic, Indic, Thai, emoji and every other script get the
// same pre-token boundaries — and therefore the same ids — as HF. Text is NFC-
// normalized first, as the Qwen2 tokenizer.json's normalizer (and the slow
// Qwen2Tokenizer) do. Unlike CLIP, leading whitespace is NOT dropped — it is
// folded into the following pre-token via the GPT-2 leading-space (Ġ)
// convention, so " hello" is one pre-token. Invalid UTF-8 bytes are treated as
// "other" characters and round-trip through the byte-level encoding.
//
// A tokenizer.json's own normalizer / pre_tokenizer blocks are honoured on
// load: Llama-3's file (no normalizer, \p{N}{1,3}) selects those variants;
// vocab.json + merges.txt and GGUF loads use the Qwen2 conventions above.

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

    // Build a tokenizer from a Hugging Face unified `tokenizer.json` (the
    // "fast"/`tokenizers` container). Reads the byte-level-BPE model block:
    //   model.vocab   — object { "token": id, ... }
    //   model.merges  — array of "a b" strings, or ["a","b"] pairs (newer HF)
    //   added_tokens  — array of { "id", "content", "special" }; each is added
    //                   to the vocab and registered as an atomic special
    //                   matched before BPE, whatever its "special" flag (HF
    //                   extracts every added token verbatim; the flag only
    //                   governs skip_special_tokens on decode).
    //   normalizer    — an NFC normalizer (Qwen2/Qwen3) turns on NFC
    //                   normalization in encode(); null (Llama-3) turns it off.
    //   pre_tokenizer — the Split regex's digit rule (\p{N} vs \p{N}{1,3})
    //                   selects single-digit or up-to-three-digit pre-tokens.
    // This is the same GPT-2 byte-level BPE as load(); only the container
    // differs. Models that ship a single tokenizer.json instead of the
    // vocab.json + merges.txt pair — Llama-3 among them — load through here.
    // `extra_special_tokens` are registered in addition to the added_tokens
    // specials, if present in the vocab. Throws std::runtime_error on I/O or
    // parse failure, or if the model block is not a byte-level BPE.
    static Tokenizer from_tokenizer_json(
        const std::string& tokenizer_json_path,
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
    // skipped. decode(encode(s)) == s for any byte string that is already
    // NFC-normalized (all ASCII, all precomposed text, and invalid UTF-8
    // alike); input that is not in NFC comes back in NFC form, exactly as it
    // does through HF.
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

    // Pre-tokenization conventions in effect: whether encode() NFC-normalizes
    // its input, and the longest digit run one \p{N} pre-token may hold (1
    // for Qwen2/Qwen3, 3 for Llama-3). Set from a tokenizer.json's normalizer
    // and pre_tokenizer blocks; the other loaders use the Qwen2 defaults.
    bool normalizes_nfc() const { return normalize_nfc_; }
    int digit_run_max() const { return digit_run_max_; }

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

    // Qwen2 defaults; from_tokenizer_json overrides from the file.
    bool normalize_nfc_ = true;
    int digit_run_max_  = 1;
};

}  // namespace brolm::qwen
