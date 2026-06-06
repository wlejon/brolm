#pragma once

// Mistral "Tekken" tokenizer (tiktoken family).
//
// Mistral 3.x models (Nemo, Small 3.1, Pixtral, ...) ship a tekken.json — a
// tiktoken-style byte-level BPE, NOT the GPT-2/HF vocab.json+merges.txt form
// used by brolm::qwen and NOT the SentencePiece Unigram used by brolm::t5.
// The differences that matter:
//   - Tokens are RAW bytes (base64-encoded in tekken.json); there is no GPT-2
//     byte->visible-unicode remapping. A leading space is a literal 0x20 byte.
//   - There is no merges.txt: the merge order is implied by the vocabulary,
//     so BPE merges the adjacent pair whose concatenation has the lowest id
//     (ids are monotonic in BPE rank). See bpe::tiktoken_encode_piece.
//   - The id space reserves the first `num_special_tokens` ids (1000 for
//     Mistral) for special tokens; a regular vocab token of rank r gets the
//     final id r + num_special_tokens. Unfilled special slots are padded with
//     "<SPECIAL_n>" placeholders so every id in [0, num_special_tokens) maps.
//
// Special tokens (<s>, </s>, [INST], [/INST], [SYSTEM_PROMPT], [IMG], ...) are
// matched verbatim before BPE and rendered literally on decode — the shared
// brolm::detail::bpe scaffolding (SpecialTokens + encode_with_specials) does
// this, exactly as for qwen. Newer tekken.json files carry a special_tokens
// list; Mistral's (e.g. Mistral-Small-3.1) omit it, so the loader falls back to
// the canonical Tekken v7 table (ids 0..19: <unk> <s> </s> [INST] ... ).
//
// Pre-tokenization implements Tekken's config.pattern faithfully over ASCII —
// case-aware letter splitting (Lu*Ll+ | Lu+Ll*, so "HelloWorld" -> "Hello",
// "World"), single digits (\p{N}), optional-space + punctuation runs, and the
// whitespace/newline rules — so token boundaries (and thus ids, given the exact
// merge + vocab) match mistral-common on ASCII text. There is NO contraction
// rule (Tekken's pattern has none: "don't" -> "don" + "'" + "t"). Non-ASCII
// bytes (>=0x80) are treated as "other" (punctuation class), so multibyte-letter
// scripts are not yet split on Unicode letter boundaries — that needs \p{L}
// property tables (future work). Tokenization stays lossless and round-trips.

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "brolm/detail/byte_level_bpe.h"

namespace brolm::mistral {

class Tokenizer {
public:
    // Load a Mistral tekken.json. Reads the vocab (base64 raw-byte tokens +
    // ranks), the special_tokens list, and config.default_num_special_tokens /
    // config.default_vocab_size for the id math. Throws std::runtime_error on
    // I/O or parse failure.
    static Tokenizer load(const std::string& tekken_json_path);

    // Encode `text` into int32 token ids via tiktoken byte-level BPE.
    // Substrings matching a registered special token are emitted atomically as
    // that token's id; the text around them is BPE-encoded normally.
    //
    // When `add_special` is true the BOS token (<s>) is prepended — Mistral
    // prefixes BOS at the tokenizer level. (apply_chat_template already emits a
    // leading <s>, so call encode() on its output with add_special=false.)
    std::vector<int32_t> encode(std::string_view text, bool add_special = false) const;

    // Inverse of encode: special ids render as their literal string; regular
    // ids concatenate their raw token bytes. decode(encode(s)) round-trips for
    // any byte string (the vocab contains every single byte).
    std::string decode(const std::vector<int32_t>& ids) const;

    // Render a Mistral instruction conversation as plain text (feed to
    // encode() with add_special=false — the leading <s> is included):
    //   <s>[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][INST]{user}[/INST]{assistant}</s>...
    // "system" messages emit a [SYSTEM_PROMPT] block, "user" an [INST] block,
    // "assistant" the content followed by </s>; any other role is emitted bare.
    // `add_generation_prompt` is accepted for signature parity with the other
    // tokenizers; Mistral needs no extra cue (the model generates after the
    // final [/INST]).
    std::string apply_chat_template(
        const std::vector<std::pair<std::string, std::string>>& messages,
        bool add_generation_prompt = true) const;

    // Register `token` as an atomic special with id `id` (need not be in the
    // vocab). Re-registering overwrites. Updates the cached <s>/</s>/[INST]/...
    // accessor ids when the matching token is (re)registered.
    void register_special_token(const std::string& token, int32_t id);

    // Number of regular (non-special) vocab tokens.
    std::size_t vocab_count() const { return id_to_bytes_.size(); }
    // Number of registered special tokens (includes <SPECIAL_n> fillers).
    std::size_t special_count() const { return specials_.size(); }
    // Total id space: regular tokens + reserved special slots.
    int vocab_size() const { return vocab_size_; }
    int num_special_tokens() const { return num_special_tokens_; }

    // Key special-token ids, or -1 if absent from the tokenizer.
    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }
    int unk_id() const { return unk_id_; }
    int pad_id() const { return pad_id_; }
    int inst_id() const { return inst_id_; }
    int inst_end_id() const { return inst_end_id_; }
    int img_id() const { return img_id_; }
    int img_break_id() const { return img_break_id_; }
    int img_end_id() const { return img_end_id_; }

    // Generic lookup for any special token string, or -1 if not registered.
    int special_id(const std::string& token) const {
        return specials_.id_for_token(token);
    }

private:
    Tokenizer() = default;

    // Pre-tokenize a special-free span and tiktoken-encode each piece.
    void encode_span_(std::string_view span, std::vector<int32_t>& out) const;
    // Resolve the cached named-special accessor ids from specials_.
    void resolve_named_ids_();

    // Regular-token vocab: raw-byte string -> id (id = rank + num_special).
    std::unordered_map<std::string, int32_t> byte_to_id_;
    // id -> raw bytes, for decode() (regular tokens only).
    std::unordered_map<int32_t, std::string> id_to_bytes_;
    // Special tokens, matched before BPE and rendered literally on decode.
    brolm::detail::bpe::SpecialTokens specials_;

    int num_special_tokens_ = 0;
    int vocab_size_ = 0;

    int bos_id_ = -1, eos_id_ = -1, unk_id_ = -1, pad_id_ = -1;
    int inst_id_ = -1, inst_end_id_ = -1;
    int img_id_ = -1, img_break_id_ = -1, img_end_id_ = -1;
};

}  // namespace brolm::mistral
