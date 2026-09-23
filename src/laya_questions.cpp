// Jev-shaped question parsing for the Laya decision model — the C++ twin of
// the reference RLAgent._to_internal + render_options input handling.

#include "brolm/laya.h"

#include "brolm/detail/json.h"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace brolm::laya {

namespace j = brolm::detail::json;

namespace {

[[noreturn]] void fail_q(const std::string& msg) {
    throw std::runtime_error("laya::parse_questions_json: " + msg);
}

void dump_string(const std::string& s, bool ensure_ascii, std::string& out) {
    out.push_back('"');
    for (std::size_t i = 0; i < s.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        switch (c) {
            case '"':  out += "\\\""; continue;
            case '\\': out += "\\\\"; continue;
            case '\n': out += "\\n";  continue;
            case '\r': out += "\\r";  continue;
            case '\t': out += "\\t";  continue;
            case '\b': out += "\\b";  continue;
            case '\f': out += "\\f";  continue;
            default: break;
        }
        char buf[16];
        if (c < 0x20) {
            std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
            out += buf;
        } else if (c < 0x80 || !ensure_ascii) {
            out.push_back(static_cast<char>(c));
        } else {
            // Decode one UTF-8 sequence and emit \uXXXX (surrogate pair above
            // the BMP), as Python's json.dumps(ensure_ascii=True) does.
            uint32_t cp = 0;
            int extra = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : 1;
            cp = c & (0x3F >> extra);
            for (int k = 0; k < extra && i + 1 < s.size(); ++k) {
                cp = (cp << 6) | (static_cast<unsigned char>(s[++i]) & 0x3F);
            }
            if (cp >= 0x10000) {
                cp -= 0x10000;
                std::snprintf(buf, sizeof(buf), "\\u%04x\\u%04x",
                              0xD800 + (cp >> 10), 0xDC00 + (cp & 0x3FF));
            } else {
                std::snprintf(buf, sizeof(buf), "\\u%04x", cp);
            }
            out += buf;
        }
    }
    out.push_back('"');
}

// Python json.dumps(value, ensure_ascii=..., separators=(", ", ": ")).
void dump(const j::Value& v, bool ensure_ascii, std::string& out) {
    switch (v.type()) {
        case j::Type::Null:   out += "null"; return;
        case j::Type::Bool:   out += v.as_bool() ? "true" : "false"; return;
        case j::Type::Number: {
            const double d = v.as_number();
            char buf[64];
            if (std::floor(d) == d && std::fabs(d) < 1e15) {
                std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(d));
                out += buf;
            } else {
                auto r = std::to_chars(buf, buf + sizeof(buf), d);
                out.append(buf, r.ptr);
            }
            return;
        }
        case j::Type::String: dump_string(v.as_string(), ensure_ascii, out); return;
        case j::Type::Array: {
            out.push_back('[');
            bool first = true;
            for (const auto& e : v.as_array()) {
                if (!first) out += ", ";
                first = false;
                dump(e, ensure_ascii, out);
            }
            out.push_back(']');
            return;
        }
        case j::Type::Object: {
            out.push_back('{');
            bool first = true;
            for (const auto& [k, e] : v.as_object()) {
                if (!first) out += ", ";
                first = false;
                dump_string(k, ensure_ascii, out);
                out += ": ";
                dump(e, ensure_ascii, out);
            }
            out.push_back('}');
            return;
        }
    }
}

// A criterion value as option text: strings verbatim, null -> "" (no
// description), anything structured as JSON.
std::string criterion_text(const j::Value& v) {
    if (v.is_string()) return v.as_string();
    if (v.is_null()) return "";
    std::string s;
    dump(v, /*ensure_ascii=*/false, s);
    return s;
}

}  // namespace

std::vector<LayaQuestion> parse_questions_json(const std::string& json_text) {
    j::Value root;
    try {
        root = j::parse(json_text);
    } catch (const std::exception& e) {
        fail_q(e.what());
    }
    if (!root.is_object()) fail_q("questions must be a JSON object {id: question}");

    std::vector<LayaQuestion> out;
    for (const auto& [qid, def] : root.as_object()) {
        if (!def.is_object()) fail_q("question '" + qid + "' is not an object");
        LayaQuestion q;
        q.id = qid;
        q.type = def.get_string("type", "");
        if (q.type != "choice" && q.type != "score" && q.type != "noul") {
            fail_q("question '" + qid + "': type must be choice, score or noul");
        }
        const j::Value* ins = def.find("instructions");
        if (!ins) fail_q("question '" + qid + "': missing instructions");
        if (ins->is_string()) {
            q.instructions = ins->as_string();
        } else {
            dump(*ins, /*ensure_ascii=*/true, q.instructions);  // json.dumps defaults
        }

        const j::Value* crit = def.find("criteria");
        if (q.type == "choice") {
            if (!crit || !(crit->is_object() || crit->is_array())) {
                fail_q("question '" + qid + "': choice criteria must be an object or a list");
            }
            if (crit->is_array()) {
                for (const auto& e : crit->as_array()) {
                    q.criteria_choice.emplace_back(criterion_text(e), "");
                }
            } else {
                for (const auto& [k, v] : crit->as_object()) {
                    q.criteria_choice.emplace_back(k, criterion_text(v));
                }
            }
        } else if (q.type == "score") {
            if (!crit || !(crit->is_object() || crit->is_array())) {
                fail_q("question '" + qid + "': score criteria must be a list");
            }
            if (crit->is_array()) {
                for (const auto& e : crit->as_array()) q.criteria_score.push_back(criterion_text(e));
            } else {
                for (const auto& [k, v] : crit->as_object()) q.criteria_score.push_back(k);
            }
        } else if (crit && crit->is_object()) {
            if (const j::Value* f = crit->find("false")) q.criteria_noul_false = criterion_text(*f);
            if (const j::Value* t = crit->find("true")) q.criteria_noul_true = criterion_text(*t);
        }
        out.push_back(std::move(q));
    }
    return out;
}

}  // namespace brolm::laya
