#include "brolm/qwen35_tokenizer.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace brolm::qwen35 {

namespace {

// (token, id) pairs from Qwen/Qwen3.5-0.8B's `tokenizer_config.json`
// (`added_tokens_decoder`). Verified against the downloaded shard:
//   $ python -c "import json; d=json.load(open('tokenizer_config.json'));
//                print(d['added_tokens_decoder'])"
// The whole Qwen3.5 size lineup ships the same vocab + added_tokens block,
// so this table is shared across 0.8B/2B/4B/9B/35B/122B.
//
// Order is ascending by id and matches the documented added_tokens layout
// at IDs 248044..248076. These ids live ABOVE vocab.json (which tops out at
// 248043), so they are not discoverable via the inner tokenizer's vocab
// lookup — they must be injected with the explicit IDs.
struct SpecialEntry {
    const char* token;
    int32_t     id;
};

const std::vector<SpecialEntry>& kSpecialTable() {
    static const std::vector<SpecialEntry> kTable = {
        {"<|endoftext|>",         248044},
        {"<|im_start|>",          248045},
        {"<|im_end|>",            248046},
        {"<|object_ref_start|>",  248047},
        {"<|object_ref_end|>",    248048},
        {"<|box_start|>",         248049},
        {"<|box_end|>",           248050},
        {"<|quad_start|>",        248051},
        {"<|quad_end|>",          248052},
        {"<|vision_start|>",      248053},
        {"<|vision_end|>",        248054},
        {"<|vision_pad|>",        248055},
        {"<|image_pad|>",         248056},
        {"<|video_pad|>",         248057},
        {"<tool_call>",           248058},
        {"</tool_call>",          248059},
        {"<|fim_prefix|>",        248060},
        {"<|fim_middle|>",        248061},
        {"<|fim_suffix|>",        248062},
        {"<|fim_pad|>",           248063},
        {"<|repo_name|>",         248064},
        {"<|file_sep|>",          248065},
        {"<tool_response>",       248066},
        {"</tool_response>",      248067},
        {"<think>",               248068},
        {"</think>",              248069},
        {"<|audio_start|>",       248070},
        {"<|audio_end|>",         248071},
        {"<tts_pad>",             248072},
        {"<tts_text_bos>",        248073},
        {"<tts_text_eod>",        248074},
        {"<tts_text_bos_single>", 248075},
        {"<|audio_pad|>",         248076},
    };
    return kTable;
}

}  // namespace

const std::vector<std::string>& Tokenizer::kAllSpecialTokens() {
    static const std::vector<std::string> kNames = [] {
        std::vector<std::string> out;
        out.reserve(kSpecialTable().size());
        for (const SpecialEntry& e : kSpecialTable()) out.emplace_back(e.token);
        return out;
    }();
    return kNames;
}

Tokenizer Tokenizer::load(const std::string& vocab_json_path,
                          const std::string& merges_txt_path,
                          const std::vector<std::string>& extra_special_tokens) {
    // The inner load() path only registers specials it finds in vocab.json;
    // Qwen3.5's specials are NOT in vocab.json. Load the BPE with no extras
    // (avoids spurious work), then explicitly inject every entry from
    // kSpecialTable() — and any caller-provided extras with id -1 marker
    // semantics dropped, since extras need an id to be useful here.
    (void)extra_special_tokens;  // reserved for a future overload accepting (string,id) pairs
    qwen::Tokenizer inner = qwen::Tokenizer::load(vocab_json_path,
                                                  merges_txt_path,
                                                  /*extras=*/{});
    for (const SpecialEntry& e : kSpecialTable()) {
        inner.register_special_token(e.token, e.id);
    }
    Tokenizer t(std::move(inner));
    t.resolve_named_ids_();
    return t;
}

Tokenizer::Tokenizer(qwen::Tokenizer&& inner) : inner_(std::move(inner)) {}

void Tokenizer::resolve_named_ids_() {
    // The accessor ids come straight from kSpecialTable() — encoding the
    // literal string would also work now that register_special_token has run,
    // but a direct table lookup is cheaper and avoids depending on encode's
    // longest-match traversal in case future specials overlap as substrings.
    auto find_id = [](const char* name) -> int {
        for (const SpecialEntry& e : kSpecialTable()) {
            if (std::string(e.token) == name) return e.id;
        }
        return -1;
    };
    vision_start_id_    = find_id("<|vision_start|>");
    vision_end_id_      = find_id("<|vision_end|>");
    vision_pad_id_      = find_id("<|vision_pad|>");
    image_pad_id_       = find_id("<|image_pad|>");
    video_pad_id_       = find_id("<|video_pad|>");
    think_open_id_      = find_id("<think>");
    think_close_id_     = find_id("</think>");
    tool_call_open_id_  = find_id("<tool_call>");
    tool_call_close_id_ = find_id("</tool_call>");
}

}  // namespace brolm::qwen35
