// LayaTokenizer: loading and encode() — the tokenizer.json pipeline of a Laya
// checkpoint (see laya_tokenizer.h). Sequence building lives in
// laya_sequence.cpp.

#include "brolm/laya_tokenizer.h"

#include "brolm/detail/byte_level_bpe.h"
#include "brolm/detail/hf_bpe.h"
#include "brolm/detail/json.h"
#include "brolm/detail/unicode.h"

#include <filesystem>
#include <functional>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace brolm::laya {

namespace bpe = brolm::detail::bpe;
namespace j   = brolm::detail::json;
namespace uni = brolm::detail::unicode;

namespace {

[[noreturn]] void fail_tok(const std::string& msg) {
    throw std::runtime_error("laya::LayaTokenizer: " + msg);
}

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail_tok("cannot open '" + path + "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

constexpr std::string_view kMetaspace = "\xE2\x96\x81";  // U+2581

bool is_other(uint32_t cp) {
    return !uni::is_white_space(cp) && !uni::is_letter(cp) && !uni::is_number(cp);
}

// The GPT-2 ByteLevel split ModernBERT's tokenizer.json declares
// (ByteLevel, use_regex=true):
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// Contractions are case-sensitive, there is no digit-run limit, and a
// punctuation run does not swallow trailing newlines (all three differ from
// the Qwen2/Llama-3 pattern).
template <class Emit>
void byte_level_split(std::string_view text, Emit&& emit_piece) {
    std::vector<uint32_t> cps;
    std::vector<std::size_t> offs;
    cps.reserve(text.size());
    offs.reserve(text.size() + 1);
    for (std::size_t i = 0; i < text.size();) {
        offs.push_back(i);
        cps.push_back(uni::decode_utf8(text, i));
    }
    offs.push_back(text.size());
    const std::size_t n = cps.size();
    auto emit = [&](std::size_t a, std::size_t b) { emit_piece(text.substr(offs[a], offs[b] - offs[a])); };

    std::size_t i = 0;
    while (i < n) {
        const uint32_t c = cps[i];
        if (c == '\'' && i + 1 < n) {
            const uint32_t d = cps[i + 1];
            if (d == 's' || d == 't' || d == 'm' || d == 'd') {
                emit(i, i + 2);
                i += 2;
                continue;
            }
            if (i + 2 < n) {
                const uint32_t e = cps[i + 2];
                if ((d == 'r' && e == 'e') || (d == 'v' && e == 'e') || (d == 'l' && e == 'l')) {
                    emit(i, i + 3);
                    i += 3;
                    continue;
                }
            }
        }
        // ` ?\p{L}+`, ` ?\p{N}+`, ` ?[^\s\p{L}\p{N}]+`
        bool matched = false;
        for (int cls = 0; cls < 3 && !matched; ++cls) {
            auto in_cls = [cls](uint32_t cp) {
                return cls == 0 ? uni::is_letter(cp) : cls == 1 ? uni::is_number(cp) : is_other(cp);
            };
            std::size_t k = i + (c == ' ' ? 1 : 0);
            if (k < n && in_cls(cps[k])) {
                ++k;
                while (k < n && in_cls(cps[k])) ++k;
                emit(i, k);
                i = k;
                matched = true;
            }
        }
        if (matched) continue;
        // Only whitespace reaches here. \s+(?!\S): the whole run at end of
        // text; otherwise the run minus its last code point, which is left for
        // the next token (a ' ' joins it, anything else falls to \s+ alone).
        std::size_t run_end = i + 1;
        while (run_end < n && uni::is_white_space(cps[run_end])) ++run_end;
        if (run_end == n || run_end - i == 1) {
            emit(i, run_end);
            i = run_end;
        } else {
            emit(i, run_end - 1);
            i = run_end - 1;
        }
    }
}

// The code point ending just before byte `end` (UTF-8), and its start.
uint32_t prev_codepoint(std::string_view s, std::size_t end, std::size_t& start) {
    start = end - 1;
    while (start > 0 && (static_cast<unsigned char>(s[start]) & 0xC0) == 0x80) --start;
    std::size_t i = start;
    return uni::decode_utf8(s, i);
}

std::string token_content(const j::Value* v) {
    if (!v) return {};
    if (v->is_string()) return v->as_string();
    if (v->is_object()) return v->get_string("content", "");
    return {};
}

}  // namespace

struct LayaTokenizer::Model {
    brolm::detail::hfbpe::Model bpe;
    bpe::SpecialTokens added;                 // every added token, matched on the raw text
    std::unordered_map<int32_t, uint8_t> strip;  // added-token id -> 1 lstrip | 2 rstrip

    // Normalizer steps in order: NFC, or Replace(from -> to).
    struct Norm {
        bool nfc = false;
        std::string from, to;
    };
    std::vector<Norm> norms;

    // ByteLevel
    std::string byte_to_unicode[256];
    bool byte_level_prefix_space = false;
    // Metaspace
    enum class Prepend { Always, First, Never } prepend = Prepend::Always;
    bool split = true;
};

struct LayaTokenizer::PieceCache {
    static constexpr std::size_t kMaxEntries = 1 << 16;
    std::mutex mu;
    std::unordered_map<std::string, std::vector<int32_t>> ids;
};

LayaTokenizer::LayaTokenizer() = default;

std::size_t LayaTokenizer::vocab_size() const { return model_ ? model_->bpe.vocab_size() : 0; }

LayaTokenizer LayaTokenizer::load(const std::string& tokenizer_json_path) {
    LayaTokenizer t;
    t.cache_ = std::make_shared<PieceCache>();
    auto m = std::make_shared<Model>();

    j::Value root;
    try {
        root = j::parse(slurp(tokenizer_json_path));
    } catch (const std::runtime_error& e) {
        if (std::string(e.what()).rfind("laya::", 0) == 0) throw;
        fail_tok(std::string("json parse: ") + e.what());
    }
    if (!root.is_object()) fail_tok("root is not a JSON object");

    const j::Value* model = root.find("model");
    if (!model || !model->is_object()) fail_tok("missing 'model' object");
    m->bpe.load(*model);

    // Added tokens: matched verbatim before normalization (all Laya
    // checkpoints' added tokens are normalized=false).
    std::unordered_map<std::string, int32_t> added_ids;
    if (const j::Value* at = root.find("added_tokens"); at && at->is_array()) {
        for (const auto& tok : at->as_array()) {
            const j::Value* content = tok.find("content");
            const j::Value* idv = tok.find("id");
            if (!content || !content->is_string() || !idv || !idv->is_number()) continue;
            const int32_t id = static_cast<int32_t>(idv->as_number());
            m->added.add(content->as_string(), id);
            added_ids[content->as_string()] = id;
            const uint8_t s = (tok.get_bool("lstrip", false) ? 1 : 0) | (tok.get_bool("rstrip", false) ? 2 : 0);
            if (s) m->strip[id] = s;
        }
    }

    // Normalizer.
    std::function<void(const j::Value&)> add_norm = [&](const j::Value& n) {
        if (n.is_null()) return;
        const std::string type = n.get_string("type", "");
        if (type == "NFC") {
            m->norms.push_back(Model::Norm{true, {}, {}});
        } else if (type == "Replace") {
            const j::Value* pat = n.find("pattern");
            const j::Value* s = pat ? pat->find("String") : nullptr;
            if (!s || !s->is_string()) fail_tok("Replace normalizer: only a String pattern is supported");
            m->norms.push_back(Model::Norm{false, s->as_string(), n.get_string("content", "")});
        } else if (type == "Sequence") {
            if (const j::Value* list = n.find("normalizers"); list && list->is_array())
                for (const auto& e : list->as_array()) add_norm(e);
        } else {
            fail_tok("unsupported normalizer '" + type + "'");
        }
    };
    if (const j::Value* n = root.find("normalizer"); n) add_norm(*n);

    // Pre-tokenizer.
    const j::Value* pre = root.find("pre_tokenizer");
    if (!pre || !pre->is_object()) fail_tok("missing 'pre_tokenizer'");
    const std::string pre_type = pre->get_string("type", "");
    if (pre_type == "ByteLevel") {
        t.kind_ = Kind::ByteLevel;
        if (!pre->get_bool("use_regex", true)) fail_tok("ByteLevel pre-tokenizer without use_regex is not supported");
        m->byte_level_prefix_space = pre->get_bool("add_prefix_space", false);
        bpe::build_byte_to_unicode(m->byte_to_unicode);
    } else if (pre_type == "Metaspace") {
        t.kind_ = Kind::Metaspace;
        if (pre->get_string("replacement", std::string(kMetaspace)) != kMetaspace)
            fail_tok("Metaspace pre-tokenizer: only the U+2581 replacement is supported");
        std::string scheme = pre->get_string("prepend_scheme", "");
        if (scheme.empty()) scheme = pre->get_bool("add_prefix_space", true) ? "always" : "never";
        m->prepend = scheme == "always" ? Model::Prepend::Always
                   : scheme == "first"  ? Model::Prepend::First
                                        : Model::Prepend::Never;
        m->split = pre->get_bool("split", true);
    } else {
        fail_tok("unsupported pre_tokenizer '" + pre_type + "'");
    }

    // Special-token roles from tokenizer_config.json (the reference reads
    // tok.cls_token_id etc. through it).
    const std::filesystem::path cfg_path =
        std::filesystem::path(tokenizer_json_path).parent_path() / "tokenizer_config.json";
    std::string cls = "[CLS]", sep = "[SEP]", pad = "[PAD]", mask = "[MASK]", unk = "[UNK]";
    if (std::filesystem::exists(cfg_path)) {
        const j::Value cfg = j::parse(slurp(cfg_path.string()));
        auto role = [&](const char* key, std::string& out) {
            const std::string s = token_content(cfg.find(key));
            if (!s.empty()) out = s;
        };
        role("cls_token", cls);
        role("sep_token", sep);
        role("pad_token", pad);
        role("mask_token", mask);
        role("unk_token", unk);
    }
    auto id_of = [&](const std::string& tok, bool required) -> int32_t {
        if (const auto it = added_ids.find(tok); it != added_ids.end()) return it->second;
        const int32_t id = m->bpe.id(tok);
        if (id < 0 && required) fail_tok("special token '" + tok + "' is not in the vocabulary");
        return id;
    };
    t.cls_id_ = id_of(cls, true);
    t.sep_id_ = id_of(sep, true);
    t.pad_id_ = id_of(pad, true);
    t.mask_id_ = id_of(mask, true);
    t.unk_id_ = id_of(unk, false);
    t.mask_token_ = mask;
    t.model_ = std::move(m);
    return t;
}

void LayaTokenizer::encode_piece_cached_(std::string_view piece, std::vector<int32_t>& out) const {
    const Model& m = *model_;
    auto run = [&](std::vector<int32_t>& dst) {
        if (kind_ == Kind::ByteLevel) {
            std::string mapped;
            mapped.reserve(piece.size() * 2);
            for (unsigned char c : piece) mapped += m.byte_to_unicode[c];
            m.bpe.encode_word(mapped, dst);
        } else {
            m.bpe.encode_word(piece, dst);
        }
    };
    if (!cache_) {
        run(out);
        return;
    }
    const std::string key(piece);
    {
        std::lock_guard<std::mutex> lock(cache_->mu);
        const auto it = cache_->ids.find(key);
        if (it != cache_->ids.end()) {
            out.insert(out.end(), it->second.begin(), it->second.end());
            return;
        }
    }
    const std::size_t at = out.size();
    run(out);
    std::lock_guard<std::mutex> lock(cache_->mu);
    if (cache_->ids.size() >= PieceCache::kMaxEntries) cache_->ids.clear();
    cache_->ids.emplace(key, std::vector<int32_t>(out.begin() + static_cast<std::ptrdiff_t>(at), out.end()));
}

// One span between added tokens: normalize, pre-tokenize, BPE each piece.
// `at_start`: the span begins the input (Metaspace prepend_scheme "first").
void LayaTokenizer::encode_span_(std::string_view span, bool at_start, std::vector<int32_t>& out) const {
    const Model& m = *model_;
    std::string buf;
    for (const Model::Norm& n : m.norms) {
        if (n.nfc) {
            if (!uni::is_nfc(span)) {
                buf = uni::nfc(span);
                span = buf;
            }
        } else if (!n.from.empty() && span.find(n.from) != std::string_view::npos) {
            std::string r;
            r.reserve(span.size() + span.size() / 2);
            std::size_t pos = 0;
            for (std::size_t hit; (hit = span.find(n.from, pos)) != std::string_view::npos; pos = hit + n.from.size()) {
                r.append(span.substr(pos, hit - pos));
                r += n.to;
            }
            r.append(span.substr(pos));
            buf = std::move(r);
            span = buf;
        }
    }
    if (span.empty()) return;

    if (kind_ == Kind::ByteLevel) {
        std::string prefixed;
        if (m.byte_level_prefix_space && span.front() != ' ') {
            prefixed = " " + std::string(span);
            span = prefixed;
        }
        byte_level_split(span, [&](std::string_view p) { encode_piece_cached_(p, out); });
        return;
    }

    // Metaspace: ' ' -> U+2581, prepend one when the span does not start
    // with it, then split before every U+2581 (MergedWithNext: a run of
    // metaspaces leaves each but the last as a piece of its own).
    std::string s;
    s.reserve(span.size() + span.size() / 2 + 3);
    for (char c : span) {
        if (c == ' ') s += kMetaspace;
        else s += c;
    }
    const bool starts_ms = s.compare(0, kMetaspace.size(), kMetaspace) == 0;
    if (!starts_ms && (m.prepend == Model::Prepend::Always || (m.prepend == Model::Prepend::First && at_start)))
        s.insert(0, kMetaspace);
    const std::string_view sv = s;
    if (!m.split) {
        encode_piece_cached_(sv, out);
        return;
    }
    std::size_t start = 0;
    while (start < sv.size()) {
        std::size_t next = sv.find(kMetaspace, start + (sv.compare(start, kMetaspace.size(), kMetaspace) == 0
                                                            ? kMetaspace.size() : 1));
        if (next == std::string_view::npos) next = sv.size();
        encode_piece_cached_(sv.substr(start, next - start), out);
        start = next;
    }
}

std::vector<int32_t> LayaTokenizer::encode(std::string_view text) const {
    if (!model_) fail_tok("encode: no tokenizer loaded");
    const Model& m = *model_;
    std::vector<int32_t> ids;
    ids.reserve(text.size() / 2 + 4);

    // Added tokens first, leftmost-longest, then the spans between them.
    // lstrip / rstrip tokens also swallow the whitespace beside them.
    std::size_t span_start = 0, i = 0;
    while (i < text.size()) {
        const auto hit = m.added.match(text, i);
        if (hit.first == 0) {
            ++i;
            continue;
        }
        std::size_t span_end = i, stop = i + hit.first;
        if (const auto it = m.strip.find(hit.second); it != m.strip.end()) {
            if (it->second & 1) {
                while (span_end > span_start) {
                    std::size_t cs = 0;
                    if (!uni::is_white_space(prev_codepoint(text, span_end, cs))) break;
                    span_end = cs;
                }
            }
            if (it->second & 2) {
                while (stop < text.size()) {
                    std::size_t k = stop;
                    if (!uni::is_white_space(uni::decode_utf8(text, k))) break;
                    stop = k;
                }
            }
        }
        if (span_end > span_start) encode_span_(text.substr(span_start, span_end - span_start), span_start == 0, ids);
        ids.push_back(hit.second);
        i = span_start = stop;
    }
    if (span_start < text.size()) encode_span_(text.substr(span_start), span_start == 0, ids);
    return ids;
}

}  // namespace brolm::laya
