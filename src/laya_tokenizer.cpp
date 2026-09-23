#include "brolm/laya_tokenizer.h"
#include "brolm/laya.h"

#include "brolm/detail/json.h"
#include "brolm/detail/unicode.h"

#include <algorithm>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>

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

std::string replace_all(std::string str, const std::string& from, const std::string& to) {
    std::size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
    return str;
}

bool is_other(uint32_t cp) {
    return !uni::is_white_space(cp) && !uni::is_letter(cp) && !uni::is_number(cp);
}

// The GPT-2 ByteLevel split ModernBERT's tokenizer.json declares
// (ByteLevel, use_regex=true, add_prefix_space=false):
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// Contractions are case-sensitive, there is no digit-run limit, and a
// punctuation run does not swallow trailing newlines (all three differ from
// the Qwen2/Llama-3 pattern).
std::vector<std::string_view> pre_tokenize(std::string_view text) {
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

    std::vector<std::string_view> pieces;
    auto emit = [&](std::size_t a, std::size_t b) {
        pieces.push_back(text.substr(offs[a], offs[b] - offs[a]));
    };

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
                if ((d == 'r' && e == 'e') || (d == 'v' && e == 'e') ||
                    (d == 'l' && e == 'l')) {
                    emit(i, i + 3);
                    i += 3;
                    continue;
                }
            }
        }

        // ?\p{L}+
        {
            std::size_t j_idx = i;
            if (c == ' ') ++j_idx;
            if (j_idx < n && uni::is_letter(cps[j_idx])) {
                std::size_t k = j_idx + 1;
                while (k < n && uni::is_letter(cps[k])) ++k;
                emit(i, k);
                i = k;
                continue;
            }
        }

        // ?\p{N}+
        {
            std::size_t j_idx = i;
            if (c == ' ') ++j_idx;
            if (j_idx < n && uni::is_number(cps[j_idx])) {
                std::size_t k = j_idx + 1;
                while (k < n && uni::is_number(cps[k])) ++k;
                emit(i, k);
                i = k;
                continue;
            }
        }

        // ?[^\s\p{L}\p{N}]+
        {
            std::size_t j_idx = i;
            if (c == ' ') ++j_idx;
            if (j_idx < n && is_other(cps[j_idx])) {
                std::size_t k = j_idx + 1;
                while (k < n && is_other(cps[k])) ++k;
                emit(i, k);
                i = k;
                continue;
            }
        }

        // Only whitespace reaches here (every other code point is L, N or
        // "other", all matched above).
        std::size_t run_end = i + 1;
        while (run_end < n && uni::is_white_space(cps[run_end])) ++run_end;

        // \s+(?!\S): the whole run at end of text; otherwise the run minus its
        // last code point, which is left for the next token (a ' ' joins it,
        // anything else falls to \s+ as a single code point).
        if (run_end == n) {
            emit(i, run_end);
            i = run_end;
        } else if (run_end - i > 1) {
            emit(i, run_end - 1);
            i = run_end - 1;
        } else {
            emit(i, run_end);
            i = run_end;
        }
    }
    return pieces;
}

}  // namespace

struct LayaTokenizer::PieceCache {
    static constexpr std::size_t kMaxEntries = 1 << 16;
    std::mutex mu;
    std::unordered_map<std::string, std::vector<int32_t>> ids;
};

void LayaTokenizer::encode_piece_cached_(std::string_view piece, std::vector<int32_t>& out) const {
    if (!cache_) {
        bpe::encode_piece(piece, byte_to_unicode_, vocab_, merge_ranks_, /*append_end_of_word=*/false, out);
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
    bpe::encode_piece(piece, byte_to_unicode_, vocab_, merge_ranks_, /*append_end_of_word=*/false, out);
    std::lock_guard<std::mutex> lock(cache_->mu);
    if (cache_->ids.size() >= PieceCache::kMaxEntries) cache_->ids.clear();
    cache_->ids.emplace(key, std::vector<int32_t>(out.begin() + static_cast<std::ptrdiff_t>(at), out.end()));
}

LayaTokenizer LayaTokenizer::load(const std::string& tokenizer_json_path) {
    LayaTokenizer t;
    t.cache_ = std::make_shared<PieceCache>();
    std::unordered_map<uint32_t, unsigned char> inv;
    bpe::build_byte_unicode_maps(t.byte_to_unicode_, inv);

    j::Value root;
    try {
        root = j::parse(slurp(tokenizer_json_path));
    } catch (const std::exception& e) {
        fail_tok(std::string("json parse: ") + e.what());
    }
    if (!root.is_object()) fail_tok("root is not a JSON object");

    const j::Value* model = root.find("model");
    if (!model || !model->is_object()) fail_tok("missing 'model' object");

    const j::Value* vocab = model->find("vocab");
    if (!vocab || !vocab->is_object()) fail_tok("missing 'model.vocab' object");

    for (const auto& [tok, idv] : vocab->as_object()) {
        if (idv.is_number()) {
            t.vocab_.emplace(tok, static_cast<int32_t>(idv.as_number()));
        }
    }

    const j::Value* merges = model->find("merges");
    if (merges && merges->is_array()) {
        const auto& arr = merges->as_array();
        for (std::size_t i = 0; i < arr.size(); ++i) {
            std::string a, b;
            if (arr[i].is_array()) {
                const auto& pair = arr[i].as_array();
                if (pair.size() == 2 && pair[0].is_string() && pair[1].is_string()) {
                    a = pair[0].as_string();
                    b = pair[1].as_string();
                }
            } else if (arr[i].is_string()) {
                const std::string& line = arr[i].as_string();
                const auto sp = line.find(' ');
                if (sp != std::string::npos) {
                    a = line.substr(0, sp);
                    b = line.substr(sp + 1);
                }
            }
            if (!a.empty() && !b.empty()) {
                t.merge_ranks_.emplace(a + '\x01' + b, static_cast<int32_t>(i));
            }
        }
    }

    if (const j::Value* at = root.find("added_tokens"); at && at->is_array()) {
        for (const auto& tok : at->as_array()) {
            if (!tok.is_object()) continue;
            const j::Value* content = tok.find("content");
            const j::Value* idv = tok.find("id");
            if (content && content->is_string() && idv && idv->is_number()) {
                const std::string& s = content->as_string();
                const int32_t id = static_cast<int32_t>(idv->as_number());
                t.vocab_[s] = id;
                t.specials_.add(s, id);
            }
        }
    }

    t.specials_.add("[CLS]", kClsTokenId);
    t.specials_.add("[SEP]", kSepTokenId);
    t.specials_.add("[PAD]", kPadTokenId);
    t.specials_.add("[MASK]", kMaskTokenId);
    t.specials_.add("[UNK]", kUnkTokenId);

    return t;
}

std::vector<int32_t> LayaTokenizer::encode(std::string_view text) const {
    std::vector<int32_t> ids;
    ids.reserve(text.size());

    bpe::encode_with_specials(
        text, specials_,
        [this](std::string_view span, std::vector<int32_t>& out) {
            std::string normalized;
            if (normalize_nfc_ && !uni::is_nfc(span)) {
                normalized = uni::nfc(span);
                span = normalized;
            }
            for (const auto p : pre_tokenize(span)) encode_piece_cached_(p, out);
        },
        ids);

    return ids;
}

std::vector<std::string> LayaTokenizer::render_options(const LayaQuestion& q) {
    std::vector<std::string> opts;
    if (q.type == "choice") {
        opts.reserve(q.criteria_choice.size());
        for (const auto& [k, v] : q.criteria_choice) {
            if (v.empty()) {
                opts.push_back(k);
            } else {
                opts.push_back(k + ": " + v);
            }
        }
        return opts;
    }
    if (q.type == "score") {
        opts.reserve(q.criteria_score.size());
        for (std::size_t i = 0; i < q.criteria_score.size(); ++i) {
            opts.push_back("level " + std::to_string(i) + ": " + q.criteria_score[i]);
        }
        return opts;
    }
    // noul
    std::string f_crit = q.criteria_noul_false.empty() ? "no, the statement does not hold" : q.criteria_noul_false;
    std::string t_crit = q.criteria_noul_true.empty() ? "yes, the statement holds" : q.criteria_noul_true;
    return {"false: " + f_crit, "true: " + t_crit};
}

std::string python_json_spacing(std::string_view json) {
    std::string out;
    out.reserve(json.size() + json.size() / 8);
    bool in_str = false;
    for (std::size_t i = 0; i < json.size(); ++i) {
        const char c = json[i];
        out.push_back(c);
        if (in_str) {
            if (c == '\\' && i + 1 < json.size()) {
                out.push_back(json[++i]);
            } else if (c == '"') {
                in_str = false;
            }
            continue;
        }
        if (c == '"') {
            in_str = true;
        } else if (c == ',' || c == ':') {
            out.push_back(' ');
            // tolerate input that is already spaced
            while (i + 1 < json.size() &&
                   (json[i + 1] == ' ' || json[i + 1] == '\n' || json[i + 1] == '\t' || json[i + 1] == '\r')) {
                ++i;
            }
        }
    }
    return out;
}

std::vector<int32_t> LayaTokenizer::encode_state(const std::string& state_json_or_text) const {
    return encode(replace_all(state_json_or_text, "[MASK]", " "));
}

SequenceResult LayaTokenizer::build_sequence(const std::string& state_json_or_text,
                                            const LayaQuestion& q,
                                            int max_len,
                                            int head_max_len,
                                            bool truncate_left) const {
    return build_sequence_ids(encode_state(state_json_or_text), q, max_len, head_max_len,
                              truncate_left);
}

SequenceResult LayaTokenizer::build_sequence_ids(const std::vector<int32_t>& state_ids,
                                                 const LayaQuestion& q,
                                                 int max_len,
                                                 int head_max_len,
                                                 bool truncate_left) const {
    const std::vector<std::string> opts = render_options(q);
    const std::string ins = replace_all(q.instructions, "[MASK]", " ");
    const std::string head_text = q.type + " question: " + ins;
    std::vector<int32_t> head_ids = encode(head_text);

    std::vector<std::vector<int32_t>> opt_ids;
    opt_ids.reserve(opts.size());
    for (const auto& opt : opts) {
        std::string opt_text = " " + replace_all(opt, "[MASK]", " ");
        std::vector<int32_t> piece = encode(opt_text);
        if (piece.size() > 48) piece.resize(48);
        std::vector<int32_t> cur;
        cur.reserve(1 + piece.size());
        cur.push_back(kMaskTokenId);
        cur.insert(cur.end(), piece.begin(), piece.end());
        opt_ids.push_back(std::move(cur));
    }

    int total_opt_len = 0;
    for (const auto& o : opt_ids) total_opt_len += static_cast<int>(o.size());
    int opt_budget = head_max_len - total_opt_len;

    if (opt_budget < 16) {
        const int n_opts = std::max(1, static_cast<int>(opt_ids.size()));
        const int per = std::max(4, (head_max_len - 16) / n_opts);
        for (auto& o : opt_ids) {
            if (static_cast<int>(o.size()) > per) o.resize(static_cast<std::size_t>(per));
        }
        total_opt_len = 0;
        for (const auto& o : opt_ids) total_opt_len += static_cast<int>(o.size());
        opt_budget = head_max_len - total_opt_len;
    }

    const int head_budget = std::max(8, opt_budget);
    if (static_cast<int>(head_ids.size()) > head_budget) {
        head_ids.resize(static_cast<std::size_t>(head_budget));
    }

    std::vector<int32_t> ids;
    ids.reserve(static_cast<std::size_t>(max_len));
    ids.push_back(kClsTokenId);
    ids.insert(ids.end(), head_ids.begin(), head_ids.end());
    ids.push_back(kSepTokenId);

    std::vector<int32_t> markers;
    markers.reserve(opt_ids.size());
    for (const auto& o : opt_ids) {
        markers.push_back(static_cast<int32_t>(ids.size()));
        ids.insert(ids.end(), o.begin(), o.end());
    }
    ids.push_back(kSepTokenId);

    const int room = std::max(0, max_len - static_cast<int>(ids.size()) - 1);
    // Reference: `st[-room:] if truncate_left else st[:room]`. Python's
    // st[-0:] is the whole list, so a left-truncated state with no room left
    // is kept whole and cut by the final ids[:max_len] below; mirrored as is.
    const int n_st = static_cast<int>(state_ids.size());
    auto st_begin = state_ids.begin();
    auto st_end = state_ids.end();
    if (n_st > room) {
        if (!truncate_left) {
            st_end = state_ids.begin() + room;
        } else if (room > 0) {
            st_begin = state_ids.end() - room;
        }
    }
    ids.insert(ids.end(), st_begin, st_end);
    ids.push_back(kSepTokenId);

    if (static_cast<int>(ids.size()) > max_len) {
        ids.resize(static_cast<std::size_t>(max_len));
    }

    std::vector<int32_t> valid_markers;
    for (int32_t m : markers) {
        if (m < max_len) valid_markers.push_back(m);
    }
    return {std::move(ids), std::move(valid_markers)};
}

}  // namespace brolm::laya
