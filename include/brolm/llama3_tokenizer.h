#pragma once

// Llama-3 tokenizer.
//
// Llama-3 uses the same GPT-2-style byte-level BPE as Qwen3 / GPT-2, but ships
// a single Hugging Face `tokenizer.json` (the "fast" container) rather than the
// vocab.json + merges.txt pair, and its control tokens live in that file's
// `added_tokens` block at ids 128000..128255 (base vocab is 128000 merges).
//
// This is a thin wrapper around `brolm::qwen::Tokenizer`: the byte-level BPE
// machinery is shared verbatim (loaded via qwen::Tokenizer::from_tokenizer_json,
// which auto-registers the added_tokens specials), and this class only resolves
// the Llama-3 control tokens as named accessors, prepends the begin-of-text
// token on encode, and renders the Llama-3 header/turn chat template.
//
// Pre-tokenization is inherited from qwen::Tokenizer — the ASCII-focused GPT-2
// approximation, which matches HF behaviour on English and code (the LLM2Vec /
// ARDY use case). Llama-3's tiktoken-style Unicode-property regex is not
// reproduced; full-fidelity parity on non-English input is out of scope.

#include "brolm/qwen_tokenizer.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace brolm::llama3 {

class Tokenizer {
public:
    // Load a Llama-3 `tokenizer.json`. The added_tokens specials it carries are
    // registered by qwen::Tokenizer::from_tokenizer_json; this resolves the
    // named control-token ids on top. Throws std::runtime_error on I/O or parse
    // failure.
    static Tokenizer load(const std::string& tokenizer_json_path);

    // Encode `text` into int32 ids via byte-level BPE. When `add_bos` is true
    // (the default) the <|begin_of_text|> token is prepended — Llama-3 / LLM2Vec
    // always lead a sequence with BOS. Substrings matching a registered special
    // are emitted atomically.
    std::vector<int32_t> encode(std::string_view text, bool add_bos = true) const;

    // Inverse of encode. Special ids render to their literal string form.
    std::string decode(const std::vector<int32_t>& ids) const {
        return inner_.decode(ids);
    }

    // Render a Llama-3 chat conversation:
    //   <|begin_of_text|>
    //   (<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>)*
    // and, when `add_generation_prompt` is true, a trailing
    //   <|start_header_id|>assistant<|end_header_id|>\n\n
    // The result is plain text — feed it to encode() with add_bos=false (the
    // template already carries BOS).
    std::string apply_chat_template(
        const std::vector<std::pair<std::string, std::string>>& messages,
        bool add_generation_prompt = true) const;

    std::size_t vocab_count() const { return inner_.vocab_count(); }
    std::size_t merge_count() const { return inner_.merge_count(); }

    // ── Named control-token ids (−1 if absent from the loaded vocab) ─────────
    int bos_id()          const { return bos_id_; }            // <|begin_of_text|>
    int eos_id()          const { return eot_id_; }            // turn terminator
    int end_of_text_id()  const { return end_of_text_id_; }    // <|end_of_text|>
    int eot_id()          const { return eot_id_; }            // <|eot_id|>
    int start_header_id() const { return start_header_id_; }
    int end_header_id()   const { return end_header_id_; }

    // Underlying byte-level-BPE tokenizer, for callers needing the raw surface.
    const qwen::Tokenizer& inner() const { return inner_; }

private:
    explicit Tokenizer(qwen::Tokenizer&& inner);

    void resolve_named_ids_();

    qwen::Tokenizer inner_;

    int bos_id_          = -1;
    int end_of_text_id_  = -1;
    int eot_id_          = -1;
    int start_header_id_ = -1;
    int end_header_id_   = -1;
};

}  // namespace brolm::llama3
