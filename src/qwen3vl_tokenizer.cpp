#include "brolm/qwen3vl_tokenizer.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace brolm::qwen3vl {

namespace {

// (token, id) pairs from Qwen/Qwen3-VL-4B-Instruct's `tokenizer_config.json`
// (`added_tokens_decoder`). These ids live ABOVE vocab.json (which the base
// Qwen3 tokenizer tops out below), so they are not discoverable via the
// inner tokenizer's vocab lookup — they must be injected with explicit IDs.
// <|endoftext|>/<|im_start|>/<|im_end|> are already inside vocab.json and
// auto-registered by qwen::Tokenizer::load; not repeated here.
struct SpecialEntry {
    const char* token;
    int32_t     id;
};

const std::vector<SpecialEntry>& kSpecialTable() {
    static const std::vector<SpecialEntry> kTable = {
        {"<|vision_start|>", 151652},
        {"<|vision_end|>",   151653},
        {"<|image_pad|>",    151655},
        {"<|video_pad|>",    151656},
    };
    return kTable;
}

}  // namespace

Tokenizer Tokenizer::load(const std::string& vocab_json_path,
                          const std::string& merges_txt_path,
                          const std::vector<std::string>& extra_special_tokens) {
    qwen::Tokenizer inner = qwen::Tokenizer::load(vocab_json_path,
                                                  merges_txt_path,
                                                  extra_special_tokens);
    for (const SpecialEntry& e : kSpecialTable()) {
        inner.register_special_token(e.token, e.id);
    }
    return Tokenizer(std::move(inner));
}

Tokenizer::Tokenizer(qwen::Tokenizer&& inner) : inner_(std::move(inner)) {
    auto find_id = [](const char* name) -> int {
        for (const SpecialEntry& e : kSpecialTable()) {
            if (std::string(e.token) == name) return e.id;
        }
        return -1;
    };
    vision_start_id_ = find_id("<|vision_start|>");
    vision_end_id_   = find_id("<|vision_end|>");
    image_pad_id_    = find_id("<|image_pad|>");
    video_pad_id_    = find_id("<|video_pad|>");
}

}  // namespace brolm::qwen3vl
