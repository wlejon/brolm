#include "brolm/tokenizer_nllb.h"

#include "brolm/detail/json.h"

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace brolm::nllb {

namespace json = ::brolm::detail::json;

// ─── load ──────────────────────────────────────────────────────────────────

Tokenizer Tokenizer::load(const std::string& tokenizer_json_path) {
    std::ifstream f(tokenizer_json_path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("nllb::Tokenizer: cannot open '" +
                                 tokenizer_json_path + "'");
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();

    json::Value root = json::parse(text);  // throws on malformed input
    const json::Value* model = root.find("model");
    if (!model || !model->is_object()) {
        throw std::runtime_error("nllb::Tokenizer: tokenizer.json has no "
                                 "'model' object");
    }
    const json::Value* vocab = model->find("vocab");
    if (!vocab || !vocab->is_array()) {
        throw std::runtime_error("nllb::Tokenizer: model has no Unigram "
                                 "'vocab' array");
    }

    Tokenizer t;

    // Track special ids by content, preferring the added_tokens declaration
    // (parsed below) over a model.vocab hit.
    int bos = -1, pad = -1, eos = -1, unk = -1;
    auto note_special = [&](const std::string& piece, int32_t id) {
        if      (piece == "<s>")    bos = id;
        else if (piece == "<pad>")  pad = id;
        else if (piece == "</s>")   eos = id;
        else if (piece == "<unk>")  unk = id;
    };

    // --- base Unigram pieces (id = array index) -----------------------------
    const auto& entries = vocab->as_array();
    if (entries.empty())
        throw std::runtime_error("nllb::Tokenizer: empty vocab");
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const auto& pair = entries[i];
        if (!pair.is_array() || pair.as_array().size() < 2)
            throw std::runtime_error("nllb::Tokenizer: malformed vocab entry");
        const std::string& piece = pair.as_array()[0].as_string();
        const int32_t id = static_cast<int32_t>(i);
        t.model_.add(piece, id, pair.as_array()[1].as_number());
        note_special(piece, id);
    }
    t.model_.finalize();

    // --- added tokens: specials + FLORES-200 language codes -----------------
    //
    // These are decode-only (never Viterbi-matched) and live at their declared
    // ids — for NLLB the language codes sit above the base vocab (256001+).
    // A language code is any added token that is not a "<...>" bracket-special.
    const json::Value* added = root.find("added_tokens");
    if (added && added->is_array()) {
        for (const auto& e : added->as_array()) {
            if (!e.is_object()) continue;
            const json::Value* cv = e.find("content");
            const json::Value* iv = e.find("id");
            if (!cv || !cv->is_string() || !iv) continue;
            const std::string content = cv->as_string();
            const int32_t id = static_cast<int32_t>(iv->as_number());

            t.model_.add_decode_only(content, id);
            t.added_.emplace(content, id);
            t.special_ids_.insert(id);
            note_special(content, id);
            if (!content.empty() && content.front() != '<')
                t.lang_ids_.emplace(content, id);
        }
    }

    // unk_id from the Unigram model (index into vocab) takes priority; fall
    // back to the <unk> content hit, then the conventional id.
    t.unk_id_ = model->get_int("unk_id", (unk >= 0) ? unk : 3);
    t.bos_id_ = (bos >= 0) ? bos : 0;
    t.pad_id_ = (pad >= 0) ? pad : 1;
    t.eos_id_ = (eos >= 0) ? eos : 2;
    return t;
}

// ─── public framing ─────────────────────────────────────────────────────────

int Tokenizer::lang_id(const std::string& code) const {
    auto it = lang_ids_.find(code);
    if (it == lang_ids_.end())
        throw std::runtime_error("nllb::Tokenizer: unknown language code '" +
                                 code + "'");
    return it->second;
}

std::vector<int32_t> Tokenizer::tokenize(std::string_view text) const {
    return model_.tokenize(text, unk_id_);
}

std::vector<int32_t> Tokenizer::encode_source(std::string_view text,
                                              const std::string& src_lang) const {
    std::vector<int32_t> out;
    out.push_back(lang_id(src_lang));
    const std::vector<int32_t> pieces = model_.tokenize(text, unk_id_);
    out.insert(out.end(), pieces.begin(), pieces.end());
    out.push_back(eos_id_);
    return out;
}

std::vector<int32_t> Tokenizer::decoder_start(const std::string& tgt_lang) const {
    // decoder_start_token_id (= </s>) then the forced target-language BOS.
    return {eos_id_, lang_id(tgt_lang)};
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids,
                              bool skip_special) const {
    return model_.decode(ids, skip_special ? &special_ids_ : nullptr);
}

}  // namespace brolm::nllb
