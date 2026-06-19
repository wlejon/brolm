#include "brolm/gemma_tokenizer.h"

#include "brolm/detail/json.h"

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace brolm::gemma {

namespace json = ::brolm::detail::json;
namespace bpe = ::brolm::detail::bpe;

// ─── load ──────────────────────────────────────────────────────────────────

Tokenizer Tokenizer::load(const std::string& tokenizer_json_path) {
    std::ifstream f(tokenizer_json_path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("gemma::Tokenizer: cannot open '" +
                                 tokenizer_json_path + "'");
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();

    json::Value root = json::parse(text);  // throws on malformed input
    const json::Value* model = root.find("model");
    if (!model || !model->is_object()) {
        throw std::runtime_error("gemma::Tokenizer: tokenizer.json has no "
                                 "'model' object");
    }
    const std::string type = model->get_string("type", "");
    if (type != "BPE") {
        throw std::runtime_error("gemma::Tokenizer: expected a BPE model, got '" +
                                 type + "'");
    }

    const json::Value* vocab = model->find("vocab");
    if (!vocab || !vocab->is_object()) {
        throw std::runtime_error("gemma::Tokenizer: model has no BPE 'vocab' "
                                 "object");
    }
    const json::Value* merges = model->find("merges");
    if (!merges || !merges->is_array()) {
        throw std::runtime_error("gemma::Tokenizer: model has no 'merges' array");
    }

    Tokenizer t;

    // Gemma: ByteFallback on, no prefix space. fuse_unk follows the file
    // (default false for Gemma — byte-fallback covers everything anyway).
    t.model_.set_add_prefix_space(false);
    t.model_.set_byte_fallback(model->get_bool("byte_fallback", true));
    t.model_.set_fuse_unk(model->get_bool("fuse_unk", false));

    int bos = -1, eos = -1, pad = -1, unk = -1;
    auto note_special = [&](const std::string& piece, int32_t id) {
        if      (piece == "<bos>")  bos = id;
        else if (piece == "<eos>")  eos = id;
        else if (piece == "<pad>")  pad = id;
        else if (piece == "<unk>")  unk = id;
    };

    // --- base BPE vocab (piece -> id) ---------------------------------------
    const auto& members = vocab->as_object();
    if (members.empty())
        throw std::runtime_error("gemma::Tokenizer: empty vocab");
    for (const auto& m : members) {
        const int32_t id = static_cast<int32_t>(m.second.as_number());
        t.model_.add(m.first, id);
        note_special(m.first, id);
    }

    // --- merge ranks ("a b" or ["a","b"], priority = list order) ------------
    const auto& mg = merges->as_array();
    for (std::size_t i = 0; i < mg.size(); ++i) {
        std::string left, right;
        if (mg[i].is_array()) {
            const auto& pr = mg[i].as_array();
            if (pr.size() < 2) continue;
            left = pr[0].as_string();
            right = pr[1].as_string();
        } else if (mg[i].is_string()) {
            const std::string& s = mg[i].as_string();
            const std::size_t sp = s.find(' ');
            if (sp == std::string::npos) continue;
            left = s.substr(0, sp);
            right = s.substr(sp + 1);
        } else {
            continue;
        }
        t.model_.add_merge(left, right, static_cast<int32_t>(i));
    }

    // --- added tokens: specials + control tokens ----------------------------
    //
    // Matched verbatim in the input (before BPE) and rendered literally on
    // decode. Most are already in model.vocab; add_decode_only keeps any that
    // are not (so decode can still render them) without making them mergeable.
    const json::Value* added = root.find("added_tokens");
    if (added && added->is_array()) {
        for (const auto& e : added->as_array()) {
            if (!e.is_object()) continue;
            const json::Value* cv = e.find("content");
            const json::Value* iv = e.find("id");
            if (!cv || !cv->is_string() || !iv) continue;
            const std::string content = cv->as_string();
            const int32_t id = static_cast<int32_t>(iv->as_number());
            t.specials_.add(content, id);
            if (!t.model_.has(content)) t.model_.add_decode_only(content, id);
            note_special(content, id);
        }
    }

    t.model_.finalize();

    // Defaults: <pad>=0, <eos>=1, <bos>=2, <unk>=3.
    t.pad_id_ = (pad >= 0) ? pad : 0;
    t.eos_id_ = (eos >= 0) ? eos : 1;
    t.bos_id_ = (bos >= 0) ? bos : 2;
    t.unk_id_ = (unk >= 0) ? unk : 3;
    return t;
}

// ─── encode / tokenize / decode ──────────────────────────────────────────────

std::vector<int32_t> Tokenizer::tokenize(std::string_view text) const {
    std::vector<int32_t> out;
    bpe::encode_with_specials(
        text, specials_,
        [this](std::string_view span, std::vector<int32_t>& o) {
            const auto ids = model_.tokenize(span, unk_id_);
            o.insert(o.end(), ids.begin(), ids.end());
        },
        out);
    return out;
}

std::vector<int32_t> Tokenizer::encode(std::string_view text,
                                       bool add_bos, bool add_eos) const {
    std::vector<int32_t> out;
    if (add_bos) out.push_back(bos_id_);
    const auto ids = tokenize(text);
    out.insert(out.end(), ids.begin(), ids.end());
    if (add_eos) out.push_back(eos_id_);
    return out;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    return model_.decode(ids);
}

}  // namespace brolm::gemma
