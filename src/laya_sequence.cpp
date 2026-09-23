// Laya sequence building: the reference rl_common.build_sequence / render_options
// over a LayaTokenizer, and the json.dumps re-spacer for JS-serialised states.

#include "brolm/laya.h"
#include "brolm/laya_tokenizer.h"

#include <algorithm>
#include <string>

namespace brolm::laya {

namespace {

std::string replace_all(std::string str, const std::string& from, const std::string& to) {
    if (from.empty()) return str;
    std::size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
    return str;
}

}  // namespace

std::vector<std::string> LayaTokenizer::render_options(const LayaQuestion& q) {
    std::vector<std::string> opts;
    if (q.type == "choice") {
        opts.reserve(q.criteria_choice.size());
        for (const auto& [k, v] : q.criteria_choice) opts.push_back(v.empty() ? k : k + ": " + v);
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
    const std::string f = q.criteria_noul_false.empty() ? "no, the statement does not hold" : q.criteria_noul_false;
    const std::string t = q.criteria_noul_true.empty() ? "yes, the statement holds" : q.criteria_noul_true;
    return {"false: " + f, "true: " + t};
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
    return encode(replace_all(state_json_or_text, mask_token_, " "));
}

SequenceResult LayaTokenizer::build_sequence(const std::string& state_json_or_text, const LayaQuestion& q, int max_len,
                                            int head_max_len, bool truncate_left) const {
    return build_sequence_ids(encode_state(state_json_or_text), q, max_len, head_max_len, truncate_left);
}

SequenceResult LayaTokenizer::build_sequence_ids(const std::vector<int32_t>& state_ids, const LayaQuestion& q,
                                                 int max_len, int head_max_len, bool truncate_left) const {
    const std::vector<std::string> opts = render_options(q);
    const std::string ins = replace_all(q.instructions, mask_token_, " ");
    std::vector<int32_t> head_ids = encode(q.type + " question: " + ins);

    std::vector<std::vector<int32_t>> opt_ids;
    opt_ids.reserve(opts.size());
    for (const auto& opt : opts) {
        std::vector<int32_t> piece = encode(" " + replace_all(opt, mask_token_, " "));
        if (piece.size() > 48) piece.resize(48);
        std::vector<int32_t> cur;
        cur.reserve(1 + piece.size());
        cur.push_back(mask_id_);
        cur.insert(cur.end(), piece.begin(), piece.end());
        opt_ids.push_back(std::move(cur));
    }

    auto total = [&] {
        int n = 0;
        for (const auto& o : opt_ids) n += static_cast<int>(o.size());
        return n;
    };
    int opt_budget = head_max_len - total();
    if (opt_budget < 16) {  // too many / too long options: shrink every option text evenly
        const int n_opts = std::max(1, static_cast<int>(opt_ids.size()));
        const int per = std::max(4, (head_max_len - 16) / n_opts);
        for (auto& o : opt_ids) {
            if (static_cast<int>(o.size()) > per) o.resize(static_cast<std::size_t>(per));
        }
        opt_budget = head_max_len - total();
    }
    const int head_budget = std::max(8, opt_budget);
    if (static_cast<int>(head_ids.size()) > head_budget) head_ids.resize(static_cast<std::size_t>(head_budget));

    std::vector<int32_t> ids;
    ids.reserve(static_cast<std::size_t>(std::max(0, max_len)));
    ids.push_back(cls_id_);
    ids.insert(ids.end(), head_ids.begin(), head_ids.end());
    ids.push_back(sep_id_);

    std::vector<int32_t> markers;
    markers.reserve(opt_ids.size());
    for (const auto& o : opt_ids) {
        markers.push_back(static_cast<int32_t>(ids.size()));
        ids.insert(ids.end(), o.begin(), o.end());
    }
    ids.push_back(sep_id_);

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
    ids.push_back(sep_id_);
    if (static_cast<int>(ids.size()) > max_len) ids.resize(static_cast<std::size_t>(max_len));

    std::vector<int32_t> valid_markers;
    for (int32_t m : markers) {
        if (m < max_len) valid_markers.push_back(m);
    }
    return {std::move(ids), std::move(valid_markers)};
}

}  // namespace brolm::laya
