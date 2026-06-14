#include "brolm/tokenizer_nllb.h"

#include "brolm/detail/byte_level_bpe.h"
#include "brolm/detail/json.h"

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace brolm::nllb {

namespace json = ::brolm::detail::json;
namespace bpe = ::brolm::detail::bpe;

namespace {
// SentencePiece metaspace marker U+2581 ('▁').
const std::string kMeta = "\xE2\x96\x81";
}  // namespace

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
    const std::string type = model->get_string("type", "");
    if (type != "BPE") {
        throw std::runtime_error("nllb::Tokenizer: expected a BPE model, got '" +
                                 type + "'");
    }

    const json::Value* vocab = model->find("vocab");
    if (!vocab || !vocab->is_object()) {
        throw std::runtime_error("nllb::Tokenizer: model has no BPE 'vocab' "
                                 "object");
    }
    const json::Value* merges = model->find("merges");
    if (!merges || !merges->is_array()) {
        throw std::runtime_error("nllb::Tokenizer: model has no 'merges' array");
    }

    Tokenizer t;

    int bos = -1, pad = -1, eos = -1, unk = -1;
    auto note_special = [&](const std::string& piece, int32_t id) {
        if      (piece == "<s>")    bos = id;
        else if (piece == "<pad>")  pad = id;
        else if (piece == "</s>")   eos = id;
        else if (piece == "<unk>")  unk = id;
    };

    // --- base BPE vocab (piece -> id) ---------------------------------------
    const auto& members = vocab->as_object();
    if (members.empty())
        throw std::runtime_error("nllb::Tokenizer: empty vocab");
    t.vocab_.reserve(members.size() * 2);
    t.vocab_inv_.reserve(members.size() * 2);
    for (const auto& m : members) {
        const int32_t id = static_cast<int32_t>(m.second.as_number());
        t.vocab_.emplace(m.first, id);
        t.vocab_inv_.emplace(id, m.first);
        note_special(m.first, id);
    }

    // --- merge ranks ("a b" / ["a","b"] -> "a\x01b" -> rank) ----------------
    const auto& mg = merges->as_array();
    t.merge_ranks_.reserve(mg.size() * 2);
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
        std::string key = left;
        key += '\x01';
        key += right;
        t.merge_ranks_.emplace(std::move(key), static_cast<int32_t>(i));
    }

    // --- added tokens: specials + FLORES-200 language codes -----------------
    //
    // Decode-only — inserted explicitly via the framing helpers, never produced
    // by BPE. A language code is any added token that is not a "<...>" special.
    const json::Value* added = root.find("added_tokens");
    if (added && added->is_array()) {
        for (const auto& e : added->as_array()) {
            if (!e.is_object()) continue;
            const json::Value* cv = e.find("content");
            const json::Value* iv = e.find("id");
            if (!cv || !cv->is_string() || !iv) continue;
            const std::string content = cv->as_string();
            const int32_t id = static_cast<int32_t>(iv->as_number());

            t.added_.emplace(content, id);
            t.added_inv_.emplace(id, content);
            t.special_ids_.insert(id);
            note_special(content, id);
            if (!content.empty() && content.front() != '<')
                t.lang_ids_.emplace(content, id);
        }
    }

    t.unk_id_ = (unk >= 0) ? unk : 3;
    t.bos_id_ = (bos >= 0) ? bos : 0;
    t.pad_id_ = (pad >= 0) ? pad : 1;
    t.eos_id_ = (eos >= 0) ? eos : 2;
    return t;
}

// ─── encoding ────────────────────────────────────────────────────────────────

void Tokenizer::encode_text_(std::string_view text,
                             std::vector<int32_t>& out) const {
    // Metaspace pre-tokenization: prepend a leading marker (add_prefix_space)
    // and replace every space with the marker, then split so each word begins
    // with the marker. The BPE merge loop runs on the resulting Unicode
    // codepoints directly (SentencePiece pieces, not byte-level encoded).
    std::string ms = kMeta;
    for (char c : text) {
        if (c == ' ') ms += kMeta;
        else          ms += c;
    }

    std::size_t i = 0;
    while (i < ms.size()) {
        std::size_t next = ms.find(kMeta, i + kMeta.size());
        if (next == std::string::npos) next = ms.size();
        const std::string word = ms.substr(i, next - i);
        i = next;

        const std::vector<std::string> pieces =
            bpe::bpe_merge(word, merge_ranks_, /*append_end_of_word=*/false);
        for (const auto& p : pieces) {
            auto it = vocab_.find(p);
            if (it != vocab_.end()) {
                out.push_back(it->second);
            } else if (out.empty() || out.back() != unk_id_) {
                // fuse_unk: collapse a run of unknown symbols into one <unk>.
                out.push_back(unk_id_);
            }
        }
    }
}

std::vector<int32_t> Tokenizer::tokenize(std::string_view text) const {
    std::vector<int32_t> out;
    encode_text_(text, out);
    return out;
}

std::vector<int32_t> Tokenizer::encode_source(std::string_view text,
                                              const std::string& src_lang) const {
    std::vector<int32_t> out;
    out.push_back(lang_id(src_lang));
    encode_text_(text, out);
    out.push_back(eos_id_);
    return out;
}

std::vector<int32_t> Tokenizer::decoder_start(const std::string& tgt_lang) const {
    // decoder_start_token_id (= </s>) then the forced target-language BOS.
    return {eos_id_, lang_id(tgt_lang)};
}

// ─── decoding ────────────────────────────────────────────────────────────────

std::string Tokenizer::decode(const std::vector<int32_t>& ids,
                              bool skip_special) const {
    std::string pieces;
    for (int32_t id : ids) {
        if (skip_special && special_ids_.count(id)) continue;
        auto it = vocab_inv_.find(id);
        if (it != vocab_inv_.end()) {
            pieces += it->second;
            continue;
        }
        auto ai = added_inv_.find(id);
        if (ai != added_inv_.end()) pieces += ai->second;
    }

    // Metaspace decode: marker -> space, then drop the single leading space the
    // add_prefix_space step introduced.
    std::string out;
    out.reserve(pieces.size());
    std::size_t i = 0;
    while (i < pieces.size()) {
        if (pieces.compare(i, kMeta.size(), kMeta) == 0) {
            out += ' ';
            i += kMeta.size();
        } else {
            out += pieces[i];
            ++i;
        }
    }
    if (!out.empty() && out.front() == ' ') out.erase(out.begin());
    return out;
}

// ─── language codes ──────────────────────────────────────────────────────────

int Tokenizer::lang_id(const std::string& code) const {
    auto it = lang_ids_.find(code);
    if (it == lang_ids_.end())
        throw std::runtime_error("nllb::Tokenizer: unknown language code '" +
                                 code + "'");
    return it->second;
}

}  // namespace brolm::nllb
