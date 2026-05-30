#pragma once

// Whisper byte-level BPE tokenizer (GPT-2 family + Whisper specials).
//
// Whisper's tokenizer is the GPT-2 byte-level BPE — same algorithm family as
// brolm::qwen::Tokenizer — extended with a fixed set of special tokens that
// frame the transcript and carry control signals to the decoder:
//
//   <|endoftext|>          — sequence terminator
//   <|startoftranscript|>  — leads every decode
//   <|en|>, <|zh|>, ...    — language tag (99 of them); pick one per utterance
//   <|transcribe|>         — task: ASR in the source language
//   <|translate|>          — task: ASR + translation to English
//   <|notimestamps|>       — request plain text without timestamp tokens
//   <|nospeech|>           — emitted when the audio has no speech
//   <|0.00|> .. <|30.00|>  — timestamp tokens at 0.02s granularity
//
// The decoder always begins with a prompt prefix built from those tokens:
//   <|startoftranscript|> <|lang|> <|task|> [<|notimestamps|>]
//
// All special tokens live in vocab.json as plain entries (Whisper does not
// split them across an added_tokens.json), so we auto-register every vocab
// entry whose string is of the form "<|...|>" as a special — no hard-coded
// table needed. Helpers below expose the well-known ids and decode the
// timestamp tokens back to seconds.
//
// Pre-tokenization, byte<->unicode mapping, and the BPE merge loop are
// inherited from brolm::detail::bpe; whisper-specific logic is just the
// special-token set and the prompt/timestamp helpers.

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace brotensor::gguf { class File; }

namespace brolm::whisper {

class Tokenizer {
public:
    // Load a HF Whisper vocab.json + merges.txt. Every vocab entry of the
    // form "<|...|>" is auto-registered as an atomic special. Throws
    // std::runtime_error on I/O or parse failure.
    //
    // Upstream openai/whisper-* checkpoints split the ~1600 "<|...|>" specials
    // out of vocab.json into a separate added_tokens.json (same {string: id}
    // shape). Pass its path as `added_tokens_json_path` to merge those entries
    // into the vocab before the special-token scan, so an UNMODIFIED upstream
    // checkout tokenizes correctly. Empty (the default) loads vocab.json alone
    // — the layout produced by the older convert-whisper.py merge step.
    static Tokenizer load(const std::string& vocab_json_path,
                          const std::string& merges_txt_path,
                          const std::string& added_tokens_json_path = "");

    // Build a tokenizer from a Whisper GGUF file's metadata. Reads
    // tokenizer.ggml.{tokens, merges, token_type} the same way the Qwen3
    // loader does. Every "<|...|>" vocab entry is auto-registered as a
    // special and the well-known Whisper ids (sot, eos, language tags,
    // timestamps) are looked up by string. Throws on missing required
    // metadata keys.
    static Tokenizer from_gguf(const brotensor::gguf::File& f);

    // Encode `text` into int32 token IDs. Substrings that exactly match a
    // registered special token are emitted as that token's single id; the
    // text around them is BPE-encoded normally.
    //
    // `add_special` is independent of the Whisper decoder prompt — when true,
    // a trailing <|endoftext|> is appended (if it exists). Use build_prompt()
    // to construct the decode prefix.
    std::vector<int32_t> encode(std::string_view text, bool add_special = false) const;

    // Inverse of encode. Special-token ids decode to their literal "<|...|>"
    // form; `skip_special` drops them instead. Unknown ids are skipped.
    std::string decode(const std::vector<int32_t>& ids,
                       bool skip_special = false) const;

    // Build the Whisper decoder prompt:
    //   <|startoftranscript|> <|lang|> <|task|> [<|notimestamps|>]
    //
    // `language` is an ISO-639-1 code ("en", "zh", ...) — the function looks
    // up <|language|> in the vocab. `task` is "transcribe" or "translate".
    // Throws std::runtime_error if any required token is missing.
    std::vector<int32_t> build_prompt(std::string_view language,
                                      std::string_view task,
                                      bool with_timestamps = true) const;

    // Look up an arbitrary token string. Returns -1 if not in vocab.
    int token_to_id(std::string_view tok) const;

    // Well-known special-token ids, or -1 if absent.
    int eos_id()              const { return endoftext_id_; }     // <|endoftext|>
    int sot_id()              const { return sot_id_; }            // <|startoftranscript|>
    int no_speech_id()        const { return no_speech_id_; }      // <|nospeech|>
    int no_timestamps_id()    const { return no_timestamps_id_; }  // <|notimestamps|>
    int transcribe_id()       const { return transcribe_id_; }
    int translate_id()        const { return translate_id_; }

    // Timestamp tokens occupy a contiguous id range [first_timestamp_id,
    // last_timestamp_id]. Returns -1 / -1 if the vocab has none. Each token
    // is "<|N.NN|>" with N rounded to two decimals at 0.02s granularity from
    // 0.00 up to 30.00 inclusive (1501 tokens in the standard Whisper vocab).
    int first_timestamp_id() const { return first_timestamp_id_; }
    int last_timestamp_id()  const { return last_timestamp_id_; }

    // True iff `id` is within the timestamp range. timestamp_seconds returns
    // the floating-point seconds value (0.02 * (id - first_timestamp_id));
    // behaviour is undefined when is_timestamp(id) is false.
    bool  is_timestamp(int32_t id) const;
    float timestamp_seconds(int32_t id) const;

    std::size_t vocab_count() const { return vocab_.size(); }
    std::size_t merge_count() const { return merge_ranks_.size(); }

private:
    Tokenizer() = default;

    // Byte-encode `piece`, run BPE, look up in vocab, push to `out`.
    void encode_piece_(std::string_view piece, std::vector<int32_t>& out) const;

    std::unordered_map<std::string, int32_t> vocab_;
    std::unordered_map<int32_t, std::string> id_to_token_;
    std::unordered_map<std::string, int32_t> merge_ranks_;
    std::unordered_map<std::string, int32_t> special_tokens_;
    std::unordered_map<int32_t, std::string> special_ids_;
    std::string                              byte_to_unicode_[256];
    std::unordered_map<uint32_t, unsigned char> unicode_to_byte_;

    int endoftext_id_       = -1;
    int sot_id_             = -1;
    int no_speech_id_       = -1;
    int no_timestamps_id_   = -1;
    int transcribe_id_      = -1;
    int translate_id_       = -1;
    int first_timestamp_id_ = -1;
    int last_timestamp_id_  = -1;
};

}  // namespace brolm::whisper
