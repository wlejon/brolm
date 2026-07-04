#pragma once

// Qwen3-VL tokenizer.
//
// Qwen3-VL's vocab.json holds only the 151643 byte-level-BPE merges — every
// special token (including <|endoftext|>, <|im_start|>, <|im_end|>, not just
// the multimodal ones) lives ABOVE that range, in tokenizer_config.json's
// `added_tokens_decoder`, and must be injected at explicit ids. Named
// accessors are exposed for the four multimodal specials that vision/VL code
// needs directly: <|vision_start|>, <|vision_end|>, <|image_pad|>,
// <|video_pad|>.
//
// This header is a thin wrapper around `brolm::qwen::Tokenizer`, exactly the
// pattern `brolm::qwen35::Tokenizer` uses: the BPE machinery is shared, we
// just inject the extra multimodal ids at load time and expose them as named
// accessors so callers don't have to look them up by string.

#include "brolm/qwen_tokenizer.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brolm::qwen3vl {

class Tokenizer {
public:
    // Load a Qwen3-VL checkpoint's `vocab.json` + `merges.txt`. The full
    // added-tokens table (see file comment) is registered at its documented
    // ids in addition to whatever is already in `extra_special_tokens`.
    static Tokenizer load(const std::string& vocab_json_path,
                          const std::string& merges_txt_path,
                          const std::vector<std::string>& extra_special_tokens = {});

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
    int eos_id()       const { return inner_.eos_id(); }       // <|im_end|>
    int endoftext_id() const { return inner_.endoftext_id(); }
    int im_start_id()  const { return inner_.im_start_id(); }
    int im_end_id()    const { return inner_.im_end_id(); }

    int vision_start_id() const { return vision_start_id_; }
    int vision_end_id()   const { return vision_end_id_; }
    int image_pad_id()    const { return image_pad_id_; }
    int video_pad_id()    const { return video_pad_id_; }

private:
    explicit Tokenizer(qwen::Tokenizer&& inner);

    qwen::Tokenizer inner_;

    int vision_start_id_ = -1;
    int vision_end_id_   = -1;
    int image_pad_id_    = -1;
    int video_pad_id_    = -1;
};

}  // namespace brolm::qwen3vl
