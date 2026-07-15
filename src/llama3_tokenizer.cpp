#include "brolm/llama3_tokenizer.h"

#include <string>
#include <utility>
#include <vector>

namespace brolm::llama3 {

Tokenizer::Tokenizer(qwen::Tokenizer&& inner) : inner_(std::move(inner)) {}

Tokenizer Tokenizer::load(const std::string& tokenizer_json_path) {
    // from_tokenizer_json auto-registers the added_tokens specials (the
    // <|begin_of_text|> / <|eot_id|> / header tokens Llama-3 carries at
    // 128000..128255), so no extra names are needed here.
    qwen::Tokenizer inner =
        qwen::Tokenizer::from_tokenizer_json(tokenizer_json_path, /*extras=*/{});
    Tokenizer t(std::move(inner));
    t.resolve_named_ids_();
    return t;
}

void Tokenizer::resolve_named_ids_() {
    // A registered special matches verbatim before BPE, so encoding its literal
    // string yields exactly its single id; an unregistered token BPE-splits into
    // several pieces (or none), which we report as absent (-1). No new inner API
    // is needed and the ids come straight from the loaded tokenizer.json.
    auto find_id = [this](const char* tok) -> int {
        const std::vector<int32_t> ids = inner_.encode(tok, /*add_special=*/false);
        return ids.size() == 1 ? ids[0] : -1;
    };
    bos_id_          = find_id("<|begin_of_text|>");
    end_of_text_id_  = find_id("<|end_of_text|>");
    eot_id_          = find_id("<|eot_id|>");
    start_header_id_ = find_id("<|start_header_id|>");
    end_header_id_   = find_id("<|end_header_id|>");
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, bool add_bos) const {
    std::vector<int32_t> ids;
    if (add_bos && bos_id_ >= 0) ids.push_back(bos_id_);
    const std::vector<int32_t> body = inner_.encode(text, /*add_special=*/false);
    ids.insert(ids.end(), body.begin(), body.end());
    return ids;
}

std::string Tokenizer::apply_chat_template(
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool add_generation_prompt) const {
    std::string out = "<|begin_of_text|>";
    for (const auto& [role, content] : messages) {
        out += "<|start_header_id|>";
        out += role;
        out += "<|end_header_id|>\n\n";
        out += content;
        out += "<|eot_id|>";
    }
    if (add_generation_prompt) {
        out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
    }
    return out;
}

}  // namespace brolm::llama3
