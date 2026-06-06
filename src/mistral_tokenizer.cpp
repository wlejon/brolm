#include "brolm/mistral_tokenizer.h"

#include "brolm/detail/byte_level_bpe.h"

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brolm::mistral {

namespace bpe = brolm::detail::bpe;

// ─── tekken.json reader ─────────────────────────────────────────────────────
//
// tekken.json is a single object whose interesting keys are:
//   "vocab":          [ { "rank": int, "token_bytes": "<base64>", ... }, ... ]
//   "special_tokens": [ { "rank": int, "token_str": "<s>", ... }, ... ] | null
//   "config":         { "default_num_special_tokens": int,
//                       "default_vocab_size": int, "pattern": "...", ... }
// The vocab array is large (~130k entries), so rather than build a full JSON
// DOM we stream it with a purpose-built scanner that keeps only rank +
// token_bytes (vocab), rank + token_str (specials), and the two config counts,
// and skips every other value — including the multimodal/audio sub-configs and
// the per-token "token_str" strings we don't need. The scanner is tolerant of
// key order.

namespace {

struct Reader {
    const char* p;
    const char* end;

    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error("tekken.json: " + msg);
    }

    void skip_ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
    }
    bool eat(char c) {
        skip_ws();
        if (p < end && *p == c) { ++p; return true; }
        return false;
    }
    void expect(char c) {
        if (!eat(c)) fail(std::string("expected '") + c + "'");
    }
    char peek() {
        skip_ws();
        if (p >= end) fail("unexpected end of input");
        return *p;
    }

    std::string parse_string() {
        skip_ws();
        if (p >= end || *p != '"') fail("expected string");
        ++p;
        std::string out;
        while (p < end && *p != '"') {
            if (*p == '\\' && p + 1 < end) {
                char e = p[1];
                p += 2;
                switch (e) {
                    case '"':  out += '"';  break;
                    case '\\': out += '\\'; break;
                    case '/':  out += '/';  break;
                    case 'n':  out += '\n'; break;
                    case 't':  out += '\t'; break;
                    case 'r':  out += '\r'; break;
                    case 'b':  out += '\b'; break;
                    case 'f':  out += '\f'; break;
                    case 'u': {
                        if (p + 4 > end) fail("truncated \\u escape");
                        uint32_t cp = 0;
                        for (int i = 0; i < 4; ++i) {
                            char c = p[i];
                            int v = (c >= '0' && c <= '9') ? c - '0'
                                  : (c >= 'a' && c <= 'f') ? c - 'a' + 10
                                  : (c >= 'A' && c <= 'F') ? c - 'A' + 10
                                  : -1;
                            if (v < 0) fail("bad hex in \\u escape");
                            cp = (cp << 4) | static_cast<uint32_t>(v);
                        }
                        p += 4;
                        bpe::encode_codepoint_utf8(cp, out);
                        break;
                    }
                    default: fail("unsupported escape");
                }
            } else {
                out += *p++;
            }
        }
        if (p >= end) fail("unterminated string");
        ++p;
        return out;
    }

    // Parse a JSON number, returning it as a double (callers needing ints cast).
    double parse_number() {
        skip_ws();
        const char* start = p;
        if (p < end && (*p == '-' || *p == '+')) ++p;
        while (p < end && ((*p >= '0' && *p <= '9') || *p == '.' ||
                           *p == 'e' || *p == 'E' || *p == '+' || *p == '-')) {
            ++p;
        }
        if (p == start) fail("expected number");
        return std::stod(std::string(start, p));
    }

    // Skip any JSON value (object/array/string/number/true/false/null).
    void skip_value() {
        char c = peek();
        if (c == '{') {
            ++p;
            if (eat('}')) return;
            do {
                parse_string();   // key
                expect(':');
                skip_value();
            } while (eat(','));
            expect('}');
        } else if (c == '[') {
            ++p;
            if (eat(']')) return;
            do {
                skip_value();
            } while (eat(','));
            expect(']');
        } else if (c == '"') {
            parse_string();
        } else if (c == 't' || c == 'f') {
            // true / false
            while (p < end && *p >= 'a' && *p <= 'z') ++p;
        } else if (c == 'n') {
            while (p < end && *p >= 'a' && *p <= 'z') ++p;
        } else {
            parse_number();
        }
    }
};

struct VocabEntry {
    int rank;
    std::string bytes;  // raw bytes (already base64-decoded)
};

struct ParsedTekken {
    std::vector<VocabEntry> vocab;
    std::vector<std::pair<int, std::string>> specials;  // (rank, token_str)
    int num_special_tokens = 0;  // 0 = unset
    int vocab_size = 0;          // 0 = unset
};

void parse_vocab(Reader& r, std::vector<VocabEntry>& out) {
    r.expect('[');
    if (r.eat(']')) return;
    do {
        r.expect('{');
        VocabEntry e{-1, {}};
        bool have_rank = false, have_bytes = false;
        if (!r.eat('}')) {
            do {
                std::string key = r.parse_string();
                r.expect(':');
                if (key == "rank") {
                    e.rank = static_cast<int>(r.parse_number());
                    have_rank = true;
                } else if (key == "token_bytes") {
                    e.bytes = bpe::base64_decode(r.parse_string());
                    have_bytes = true;
                } else {
                    r.skip_value();
                }
            } while (r.eat(','));
            r.expect('}');
        }
        if (have_rank && have_bytes) out.push_back(std::move(e));
    } while (r.eat(','));
    r.expect(']');
}

void parse_specials(Reader& r, std::vector<std::pair<int, std::string>>& out) {
    // special_tokens may be null.
    if (r.peek() == 'n') { r.skip_value(); return; }
    r.expect('[');
    if (r.eat(']')) return;
    do {
        r.expect('{');
        int rank = -1;
        std::string tok;
        bool have_rank = false, have_str = false;
        if (!r.eat('}')) {
            do {
                std::string key = r.parse_string();
                r.expect(':');
                if (key == "rank") {
                    rank = static_cast<int>(r.parse_number());
                    have_rank = true;
                } else if (key == "token_str") {
                    tok = r.parse_string();
                    have_str = true;
                } else {
                    r.skip_value();
                }
            } while (r.eat(','));
            r.expect('}');
        }
        if (have_rank && have_str) out.emplace_back(rank, std::move(tok));
    } while (r.eat(','));
    r.expect(']');
}

void parse_config(Reader& r, ParsedTekken& t) {
    r.expect('{');
    if (r.eat('}')) return;
    do {
        std::string key = r.parse_string();
        r.expect(':');
        if (key == "default_num_special_tokens") {
            t.num_special_tokens = static_cast<int>(r.parse_number());
        } else if (key == "default_vocab_size") {
            t.vocab_size = static_cast<int>(r.parse_number());
        } else {
            r.skip_value();
        }
    } while (r.eat(','));
    r.expect('}');
}

// Canonical Tekken v7 special-token table, positions = ids. Mistral's
// tekken.json (e.g. Mistral-Small-3.1) omits the special_tokens list entirely —
// the table is carried out-of-band (mistral-common hardcodes it; the HF export
// mirrors it in tokenizer_config.json's added_tokens_decoder). We use this when
// the file provides no special_tokens; the remaining slots up to
// num_special_tokens are padded with "<SPECIAL_n>" placeholders.
const char* const kDefaultSpecials[] = {
    "<unk>",            // 0
    "<s>",              // 1
    "</s>",             // 2
    "[INST]",           // 3
    "[/INST]",          // 4
    "[AVAILABLE_TOOLS]",  // 5
    "[/AVAILABLE_TOOLS]", // 6
    "[TOOL_RESULTS]",   // 7
    "[/TOOL_RESULTS]",  // 8
    "[TOOL_CALLS]",     // 9
    "[IMG]",            // 10
    "<pad>",            // 11
    "[IMG_BREAK]",      // 12
    "[IMG_END]",        // 13
    "[PREFIX]",         // 14
    "[MIDDLE]",         // 15
    "[SUFFIX]",         // 16
    "[SYSTEM_PROMPT]",  // 17
    "[/SYSTEM_PROMPT]", // 18
    "[TOOL_CONTENT]",   // 19
};

ParsedTekken parse_tekken(const std::string& text) {
    Reader r{text.data(), text.data() + text.size()};
    ParsedTekken t;
    r.expect('{');
    if (!r.eat('}')) {
        do {
            std::string key = r.parse_string();
            r.expect(':');
            if (key == "vocab") {
                parse_vocab(r, t.vocab);
            } else if (key == "special_tokens") {
                parse_specials(r, t.specials);
            } else if (key == "config") {
                parse_config(r, t);
            } else {
                r.skip_value();
            }
        } while (r.eat(','));
        r.expect('}');
    }
    return t;
}

// ─── pre-tokenization (faithful to Tekken's config.pattern, ASCII) ──────────
//
// Tekken's split regex (config.pattern), ordered alternatives:
//   1. [^\r\n\p{L}\p{N}]? \p{Lu}* \p{Ll}+        letters, lower-ending
//   2. [^\r\n\p{L}\p{N}]? \p{Lu}+ \p{Ll}*        letters, upper-only/Title
//   3. \p{N}                                     a single digit
//   4.  ?[^\s\p{L}\p{N}]+ [\r\n/]*               optional-space + punctuation
//   5. \s*[\r\n]+                                whitespace ending in newline
//   6. \s+(?!\S)                                 trailing whitespace
//   7. \s+                                       remaining whitespace
// We implement these exactly over ASCII (bit-for-bit token boundaries, hence
// id parity given the exact merge + vocab). Non-ASCII bytes (>=0x80) are
// treated as "other": not letter/digit/space, so they flow through the
// punctuation alternative. That keeps everything lossless and round-tripping,
// but does not yet split multibyte-letter scripts on Unicode letter
// boundaries — that needs \p{L}/\p{Lu}/\p{Ll} property tables (future work).
// There is deliberately NO contraction rule: Tekken's pattern has none, so
// "don't" tokenizes as "don" + "'" + "t".

inline bool pt_upper(unsigned char c)  { return c >= 'A' && c <= 'Z'; }
inline bool pt_lower(unsigned char c)  { return c >= 'a' && c <= 'z'; }
inline bool pt_letter(unsigned char c) { return pt_upper(c) || pt_lower(c); }
inline bool pt_digit(unsigned char c)  { return c >= '0' && c <= '9'; }
inline bool pt_space(unsigned char c)  {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}
// The optional leading char of the letter alternatives: not a letter, not a
// digit, not CR/LF (so it may be a space, tab, punctuation, or non-ASCII byte).
inline bool pt_letter_lead(unsigned char c) {
    return !pt_letter(c) && !pt_digit(c) && c != '\r' && c != '\n';
}
// A char of the punctuation alternative's [^\s\p{L}\p{N}] class.
inline bool pt_punct(unsigned char c) {
    return !pt_space(c) && !pt_letter(c) && !pt_digit(c);
}

// Length of the single Tekken pre-token beginning at text[i] (always >= 1).
std::size_t match_token(std::string_view text, std::size_t i) {
    const std::size_t n = text.size();
    auto at = [&](std::size_t k) { return static_cast<unsigned char>(text[k]); };

    // alt 1: [^\r\n L N]? Lu* Ll+   (optional lead tried greedily, i.e. first)
    for (int use_lead = 1; use_lead >= 0; --use_lead) {
        std::size_t p = i;
        if (use_lead) { if (p < n && pt_letter_lead(at(p))) ++p; else continue; }
        std::size_t q = p;
        while (q < n && pt_upper(at(q))) ++q;
        const std::size_t low = q;
        while (q < n && pt_lower(at(q))) ++q;
        if (q > low) return q - i;            // had >= 1 lowercase
    }
    // alt 2: [^\r\n L N]? Lu+ Ll*
    for (int use_lead = 1; use_lead >= 0; --use_lead) {
        std::size_t p = i;
        if (use_lead) { if (p < n && pt_letter_lead(at(p))) ++p; else continue; }
        std::size_t q = p;
        const std::size_t up = q;
        while (q < n && pt_upper(at(q))) ++q;
        if (q > up) {                          // had >= 1 uppercase
            while (q < n && pt_lower(at(q))) ++q;
            return q - i;
        }
    }
    // alt 3: a single digit.
    if (pt_digit(at(i))) return 1;
    // alt 4: ' '? [^\s L N]+ [\r\n/]*   (the optional lead is a literal space).
    {
        std::size_t p = i;
        if (at(p) == ' ' && p + 1 < n && pt_punct(at(p + 1))) ++p;
        const std::size_t s = p;
        while (p < n && pt_punct(at(p))) ++p;
        if (p > s) {
            while (p < n && (at(p) == '\r' || at(p) == '\n' || at(p) == '/')) ++p;
            return p - i;
        }
    }
    // Whitespace alternatives. text[i] is whitespace here; find the run [i, R).
    std::size_t R = i;
    while (R < n && pt_space(at(R))) ++R;
    // alt 5: \s*[\r\n]+ — if the run holds a newline, take through the last one.
    std::size_t last_nl = std::string::npos;
    for (std::size_t k = i; k < R; ++k) {
        if (at(k) == '\r' || at(k) == '\n') last_nl = k;
    }
    if (last_nl != std::string::npos) return last_nl + 1 - i;
    // Pure spaces/tabs. alt 6: \s+(?!\S) leaves the last char to fold into the
    // following token when more text follows; alt 7: \s+ takes the rest.
    if (R == n) return R - i;                  // trailing run -> whole run
    if (R - i > 1) return (R - 1) - i;         // leave last space for folding
    return 1;                                  // lone space before non-space
}

std::vector<std::string> pre_tokenize(std::string_view text) {
    std::vector<std::string> pieces;
    std::size_t i = 0;
    while (i < text.size()) {
        std::size_t len = match_token(text, i);
        if (len == 0) len = 1;  // defensive: match_token always returns >= 1
        pieces.emplace_back(text.substr(i, len));
        i += len;
    }
    return pieces;
}

}  // namespace

// ─── Tokenizer ──────────────────────────────────────────────────────────────

void Tokenizer::resolve_named_ids_() {
    bos_id_       = specials_.id_for_token("<s>");
    eos_id_       = specials_.id_for_token("</s>");
    unk_id_       = specials_.id_for_token("<unk>");
    pad_id_       = specials_.id_for_token("<pad>");
    inst_id_      = specials_.id_for_token("[INST]");
    inst_end_id_  = specials_.id_for_token("[/INST]");
    img_id_       = specials_.id_for_token("[IMG]");
    img_break_id_ = specials_.id_for_token("[IMG_BREAK]");
    img_end_id_   = specials_.id_for_token("[IMG_END]");
}

Tokenizer Tokenizer::load(const std::string& tekken_json_path) {
    std::ifstream f(tekken_json_path, std::ios::binary);
    if (!f) throw std::runtime_error(
        "tekken.json: cannot open '" + tekken_json_path + "'");
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();

    ParsedTekken parsed = parse_tekken(text);

    Tokenizer t;
    t.num_special_tokens_ =
        parsed.num_special_tokens > 0 ? parsed.num_special_tokens : 1000;
    // inner = number of regular tokens actually used.
    int inner = parsed.vocab_size > 0
                    ? parsed.vocab_size - t.num_special_tokens_
                    : static_cast<int>(parsed.vocab.size());
    if (inner < 0) inner = 0;
    t.vocab_size_ = parsed.vocab_size > 0
                        ? parsed.vocab_size
                        : t.num_special_tokens_ + inner;

    // Regular tokens: id = rank + num_special_tokens, capped at the inner size.
    t.byte_to_id_.reserve(static_cast<std::size_t>(inner));
    t.id_to_bytes_.reserve(static_cast<std::size_t>(inner));
    for (auto& e : parsed.vocab) {
        if (e.rank < 0 || e.rank >= inner) continue;
        const int32_t id = e.rank + t.num_special_tokens_;
        t.byte_to_id_.emplace(e.bytes, id);
        t.id_to_bytes_.emplace(id, std::move(e.bytes));
    }

    // Special tokens occupy ids [0, num_special_tokens). Newer tekken.json files
    // list them; Mistral's omit the list, so fall back to the canonical Tekken
    // table. Either way, register what we have, then pad the remaining slots
    // with "<SPECIAL_n>" so every id in the reserved range maps.
    if (parsed.specials.empty()) {
        const int n = static_cast<int>(sizeof(kDefaultSpecials) /
                                       sizeof(kDefaultSpecials[0]));
        for (int i = 0; i < n && i < t.num_special_tokens_; ++i) {
            parsed.specials.emplace_back(i, kDefaultSpecials[i]);
        }
    }
    std::vector<bool> filled(static_cast<std::size_t>(t.num_special_tokens_), false);
    for (auto& [rank, tok] : parsed.specials) {
        if (rank < 0 || rank >= t.num_special_tokens_) continue;
        t.specials_.add(tok, rank);
        filled[static_cast<std::size_t>(rank)] = true;
    }
    for (int i = 0; i < t.num_special_tokens_; ++i) {
        if (filled[static_cast<std::size_t>(i)]) continue;
        t.specials_.add("<SPECIAL_" + std::to_string(i) + ">", i);
    }

    t.resolve_named_ids_();
    return t;
}

void Tokenizer::register_special_token(const std::string& token, int32_t id) {
    specials_.add(token, id);
    resolve_named_ids_();
}

void Tokenizer::encode_span_(std::string_view span,
                             std::vector<int32_t>& out) const {
    for (const auto& p : pre_tokenize(span)) {
        bpe::tiktoken_encode_piece(p, byte_to_id_, out);
    }
}

std::vector<int32_t> Tokenizer::encode(std::string_view text,
                                       bool add_special) const {
    std::vector<int32_t> ids;
    ids.reserve(text.size() + 1);

    if (add_special && bos_id_ >= 0) ids.push_back(bos_id_);

    bpe::encode_with_specials(
        text, specials_,
        [this](std::string_view span, std::vector<int32_t>& out) {
            encode_span_(span, out);
        },
        ids);
    return ids;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int32_t id : ids) {
        if (const std::string* sp = specials_.token_for_id(id)) {
            out += *sp;
            continue;
        }
        auto it = id_to_bytes_.find(id);
        if (it != id_to_bytes_.end()) out += it->second;
        // Unknown id: skip.
    }
    return out;
}

std::string Tokenizer::apply_chat_template(
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool /*add_generation_prompt*/) const {
    std::string out = "<s>";
    for (const auto& [role, content] : messages) {
        if (role == "system") {
            out += "[SYSTEM_PROMPT]";
            out += content;
            out += "[/SYSTEM_PROMPT]";
        } else if (role == "user") {
            out += "[INST]";
            out += content;
            out += "[/INST]";
        } else if (role == "assistant") {
            out += content;
            out += "</s>";
        } else {
            out += content;
        }
    }
    return out;
}

}  // namespace brolm::mistral
