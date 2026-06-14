#include "brolm/tokenizer_t5.h"

#include "brolm/detail/json.h"

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace brolm::t5 {

namespace json = ::brolm::detail::json;

// ─── load ──────────────────────────────────────────────────────────────────

Tokenizer Tokenizer::load(const std::string& tokenizer_json_path) {
    std::ifstream f(tokenizer_json_path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("t5::Tokenizer: cannot open '" +
                                 tokenizer_json_path + "'");
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();

    json::Value root = json::parse(text);  // throws on malformed input
    const json::Value* model = root.find("model");
    if (!model || !model->is_object()) {
        throw std::runtime_error("t5::Tokenizer: tokenizer.json has no "
                                 "'model' object");
    }
    const json::Value* vocab = model->find("vocab");
    if (!vocab || !(vocab->is_array() || vocab->is_object())) {
        throw std::runtime_error("t5::Tokenizer: model has no 'vocab' "
                                 "array or object");
    }

    Tokenizer t;
    t.unk_id_ = model->get_int("unk_id", 2);

    int eos_id = -1;
    int pad_id = -1;

    auto add_piece = [&](const std::string& piece, int32_t id, double score) {
        t.model_.add(piece, id, score);
        if (piece == "</s>") eos_id = id;
        if (piece == "<pad>") pad_id = id;
    };

    if (vocab->is_array()) {
        // SentencePiece Unigram: vocab is [[piece, log-prob], ...], id = index.
        const auto& entries = vocab->as_array();
        if (entries.empty())
            throw std::runtime_error("t5::Tokenizer: empty vocab");
        for (std::size_t i = 0; i < entries.size(); ++i) {
            const auto& pair = entries[i];
            if (!pair.is_array() || pair.as_array().size() < 2)
                throw std::runtime_error("t5::Tokenizer: malformed vocab entry");
            add_piece(pair.as_array()[0].as_string(),
                      static_cast<int32_t>(i), pair.as_array()[1].as_number());
        }
    } else {
        // HF BPE (SentencePiece exported as BPE, e.g. NeMo Parakeet): vocab is
        // a {piece: id} object with no per-piece scores. encode()/tokenize()'s
        // Unigram Viterbi is not meaningful here (scores are 0), but decode()
        // — the inverse id->piece metaspace join the ASR drivers need — is
        // exact, which is the supported operation for this vocab shape.
        const auto& members = vocab->as_object();
        if (members.empty())
            throw std::runtime_error("t5::Tokenizer: empty vocab");
        for (const auto& m : members)
            add_piece(m.first, static_cast<int32_t>(m.second.as_number()), 0.0);
    }

    t.model_.finalize();
    t.eos_id_ = (eos_id >= 0) ? eos_id : 1;
    t.pad_id_ = (pad_id >= 0) ? pad_id : 0;
    return t;
}

// ─── tokenize / decode (delegated to the shared Unigram core) ───────────────

std::vector<int32_t> Tokenizer::tokenize(std::string_view text) const {
    return model_.tokenize(text, unk_id_);
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    return model_.decode(ids);
}

// ─── encode ──────────────────────────────────────────────────────────────────

std::vector<int32_t> Tokenizer::encode(std::string_view text,
                                       int max_length) const {
    std::vector<int32_t> ids = tokenize(text);

    std::vector<int32_t> out;
    if (max_length <= 0) return out;
    out.reserve(static_cast<std::size_t>(max_length));

    // Content fills up to max_length-1 slots; eos always takes the last used
    // slot. If content + eos overflows, truncate the content.
    const std::size_t content_cap =
        static_cast<std::size_t>(max_length) - 1;
    const std::size_t n =
        (ids.size() > content_cap) ? content_cap : ids.size();
    for (std::size_t i = 0; i < n; ++i) out.push_back(ids[i]);

    out.push_back(eos_id_);
    while (out.size() < static_cast<std::size_t>(max_length)) {
        out.push_back(pad_id_);
    }
    return out;
}

}  // namespace brolm::t5
