#include "brolm/qwen3vl_tokenizer.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace brolm::qwen3vl {

namespace {

// (token, id) pairs from Qwen/Qwen3-VL-4B-Instruct's `tokenizer_config.json`
// (`added_tokens_decoder`). vocab.json tops out at 151643 (verified against
// the downloaded checkpoint: vocab_count == 151643) — EVERY special token,
// including <|endoftext|>/<|im_start|>/<|im_end|>, lives above that range
// and must be injected with explicit IDs; none are discoverable via the
// inner tokenizer's vocab lookup.
struct SpecialEntry {
    const char* token;
    int32_t     id;
};

const std::vector<SpecialEntry>& kSpecialTable() {
    static const std::vector<SpecialEntry> kTable = {
        {"<|endoftext|>",         151643},
        {"<|im_start|>",          151644},
        {"<|im_end|>",            151645},
        {"<|object_ref_start|>",  151646},
        {"<|object_ref_end|>",    151647},
        {"<|box_start|>",         151648},
        {"<|box_end|>",           151649},
        {"<|quad_start|>",        151650},
        {"<|quad_end|>",          151651},
        {"<|vision_start|>",      151652},
        {"<|vision_end|>",        151653},
        {"<|vision_pad|>",        151654},
        {"<|image_pad|>",         151655},
        {"<|video_pad|>",         151656},
        {"<tool_call>",           151657},
        {"</tool_call>",          151658},
        {"<|fim_prefix|>",        151659},
        {"<|fim_middle|>",        151660},
        {"<|fim_suffix|>",        151661},
        {"<|fim_pad|>",           151662},
        {"<|repo_name|>",         151663},
        {"<|file_sep|>",          151664},
        {"<tool_response>",       151665},
        {"</tool_response>",      151666},
        {"<think>",               151667},
        {"</think>",              151668},
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
