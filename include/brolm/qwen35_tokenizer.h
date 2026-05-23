#pragma once

// Qwen3.5 VLM tokenizer.
//
// Qwen3.5 uses the same GPT-2-style byte-level BPE as Qwen3 (vocab.json +
// merges.txt) — only the vocab is larger (248320 entries) and the
// special-token table grows to include the multimodal tokens (vision, image,
// video, audio) and the assistant-tool tokens.
//
// This header is a thin wrapper around `brolm::qwen::Tokenizer`: the BPE
// machinery is the same, we just register the full Qwen3.5 special-token set
// up front and expose the multimodal IDs as named accessors so callers don't
// have to look them up by string.

#include "brolm/qwen_tokenizer.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brolm::qwen35 {

class Tokenizer {
public:
    // Load a Qwen3.5 checkpoint's `vocab.json` + `merges.txt`. The full
    // special-token set (see `kAllSpecialTokens()`) is registered in addition
    // to whatever is already in `extra_special_tokens`. Tokens absent from
    // the vocab are silently ignored — older / trimmed vocabs are tolerated.
    static Tokenizer load(const std::string& vocab_json_path,
                          const std::string& merges_txt_path,
                          const std::vector<std::string>& extra_special_tokens = {});

    // The HF "added_tokens" set for Qwen3.5 — pulled verbatim from the
    // tokenizer_config.json shipped with Qwen/Qwen3.5-0.8B (33 entries). Same
    // list across the Qwen3.5 size lineup; the trimmed-vocab tolerance in
    // load() keeps this robust to future additions.
    static const std::vector<std::string>& kAllSpecialTokens();

    // ── Pass-through to the underlying BPE encoder/decoder ────────────────
    std::vector<int32_t> encode(std::string_view text,
                                bool add_special = false) const {
        return inner_.encode(text, add_special);
    }
    std::string decode(const std::vector<int32_t>& ids) const {
        return inner_.decode(ids);
    }
    std::string apply_chat_template(
        const std::vector<std::pair<std::string, std::string>>& messages,
        bool add_generation_prompt = true) const {
        return inner_.apply_chat_template(messages, add_generation_prompt);
    }
    std::size_t vocab_count() const { return inner_.vocab_count(); }
    std::size_t merge_count() const { return inner_.merge_count(); }

    // ── Named special-token IDs (−1 if absent from the loaded vocab) ──────
    int eos_id()                const { return inner_.eos_id(); }       // <|im_end|>
    int endoftext_id()          const { return inner_.endoftext_id(); }
    int im_start_id()           const { return inner_.im_start_id(); }
    int im_end_id()             const { return inner_.im_end_id(); }

    int vision_start_id()       const { return vision_start_id_; }
    int vision_end_id()         const { return vision_end_id_; }
    int vision_pad_id()         const { return vision_pad_id_; }
    int image_pad_id()          const { return image_pad_id_; }
    int video_pad_id()          const { return video_pad_id_; }

    int think_open_id()         const { return think_open_id_; }
    int think_close_id()        const { return think_close_id_; }
    int tool_call_open_id()     const { return tool_call_open_id_; }
    int tool_call_close_id()    const { return tool_call_close_id_; }

private:
    explicit Tokenizer(qwen::Tokenizer&& inner);

    // Resolve every named-token id against the loaded vocab once at load
    // time, so accessors are O(1).
    void resolve_named_ids_();

    qwen::Tokenizer inner_;

    int vision_start_id_    = -1;
    int vision_end_id_      = -1;
    int vision_pad_id_      = -1;
    int image_pad_id_       = -1;
    int video_pad_id_       = -1;
    int think_open_id_      = -1;
    int think_close_id_     = -1;
    int tool_call_open_id_  = -1;
    int tool_call_close_id_ = -1;
};

}  // namespace brolm::qwen35
