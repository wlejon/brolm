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

// ─── pre-tokenization (Tekken / cl100k-family, ASCII approximation) ─────────

std::vector<std::string> pre_tokenize(std::string_view text) {
    std::vector<std::string> pieces;
    std::size_t i = 0;
    while (i < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[i]);

        if (c == '\'') {
            if (bpe::starts_with(text, i, "'re")) { pieces.emplace_back("'re"); i += 3; continue; }
            if (bpe::starts_with(text, i, "'ve")) { pieces.emplace_back("'ve"); i += 3; continue; }
            if (bpe::starts_with(text, i, "'ll")) { pieces.emplace_back("'ll"); i += 3; continue; }
            if (bpe::starts_with(text, i, "'s"))  { pieces.emplace_back("'s");  i += 2; continue; }
            if (bpe::starts_with(text, i, "'t"))  { pieces.emplace_back("'t");  i += 2; continue; }
            if (bpe::starts_with(text, i, "'m"))  { pieces.emplace_back("'m");  i += 2; continue; }
            if (bpe::starts_with(text, i, "'d"))  { pieces.emplace_back("'d");  i += 2; continue; }
        }

        std::size_t start = i;
        // A single leading space folds into a following letter/punctuation run,
        // but NOT into a digit run (cl100k digits have no leading-space form).
        if (c == ' ') {
            bool fold = false;
            if (i + 1 < text.size()) {
                unsigned char d = static_cast<unsigned char>(text[i + 1]);
                if (!bpe::is_ascii_space(d) && !bpe::is_ascii_digit(d)) {
                    fold = true;
                    ++i;
                    c = d;
                }
            }
            if (!fold) {
                pieces.emplace_back(text.substr(i, 1));
                ++i;
                continue;
            }
        } else if (bpe::is_ascii_space(c)) {
            // Non-space whitespace (tab/newline/...) emits as its own run.
            std::size_t j = i;
            while (j < text.size() &&
                   bpe::is_ascii_space(static_cast<unsigned char>(text[j])) &&
                   text[j] != ' ') ++j;
            pieces.emplace_back(text.substr(i, j - i));
            i = j;
            continue;
        }

        if (bpe::is_ascii_letter(c)) {
            std::size_t j = i;
            while (j < text.size() &&
                   bpe::is_ascii_letter(static_cast<unsigned char>(text[j]))) ++j;
            pieces.emplace_back(text.substr(start, j - start));
            i = j;
            continue;
        }

        if (bpe::is_ascii_digit(c)) {
            // Digit groups of up to three (cl100k \p{N}{1,3}). No folded space:
            // a space before a digit was emitted standalone above, so start==i.
            std::size_t j = i;
            int n = 0;
            while (j < text.size() &&
                   bpe::is_ascii_digit(static_cast<unsigned char>(text[j])) &&
                   n < 3) { ++j; ++n; }
            pieces.emplace_back(text.substr(start, j - start));
            i = j;
            continue;
        }

        // Punctuation run (non-space, non-letter, non-digit), breaking before a
        // contraction so "it" + "'s" splits the way the apostrophe rule expects.
        std::size_t j = i;
        while (j < text.size()) {
            unsigned char u = static_cast<unsigned char>(text[j]);
            if (bpe::is_ascii_space(u) || bpe::is_ascii_letter(u) ||
                bpe::is_ascii_digit(u)) break;
            if (u == '\'' && j != i) {
                std::string_view rest = text.substr(j);
                if (rest.substr(0, 3) == "'re" || rest.substr(0, 3) == "'ve" ||
                    rest.substr(0, 3) == "'ll" ||
                    rest.substr(0, 2) == "'s"  || rest.substr(0, 2) == "'t"  ||
                    rest.substr(0, 2) == "'m"  || rest.substr(0, 2) == "'d") {
                    break;
                }
            }
            ++j;
        }
        pieces.emplace_back(text.substr(start, j - start));
        i = j;
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

    // Special tokens occupy ids [0, num_special_tokens). Register the ones the
    // file declares, then pad the remaining slots with "<SPECIAL_n>" so every
    // id in the reserved range maps (mirrors mistral-common's filler scheme).
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
