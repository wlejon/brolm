#include "brolm/qwen_tokenizer.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace brolm::qwen {

// ─── GPT-2 byte↔unicode mapping ────────────────────────────────────────────
//
// GPT-2/Qwen BPE works in a "byte-level unicode" space: every byte 0..255
// maps to a unicode codepoint and the resulting UTF-8 string is what BPE
// merges operate on. Visible printable bytes map to themselves; the rest map
// to codepoints starting at U+0100 in the order they appear. A space (0x20)
// is not in the self-mapped range, so it maps to U+0120 ('Ġ').

static void encode_codepoint_utf8(uint32_t cp, std::string& out) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

// Build byte->unicode (UTF-8) and the inverse codepoint->byte map.
static void build_byte_unicode_maps(
    std::string out[256], std::unordered_map<uint32_t, unsigned char>& inverse) {
    bool self_map[256] = {false};
    auto mark_range = [&](int lo, int hi) {
        for (int b = lo; b <= hi; ++b) self_map[b] = true;
    };
    mark_range(33, 126);
    mark_range(161, 172);
    mark_range(174, 255);

    int next_cp = 256;
    inverse.clear();
    for (int b = 0; b < 256; ++b) {
        uint32_t cp = self_map[b] ? static_cast<uint32_t>(b)
                                  : static_cast<uint32_t>(next_cp++);
        out[b].clear();
        encode_codepoint_utf8(cp, out[b]);
        inverse[cp] = static_cast<unsigned char>(b);
    }
}

// Decode the first UTF-8 codepoint at s[i], returning the codepoint and
// advancing `i` past it.
static uint32_t next_codepoint(const std::string& s, std::size_t& i) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    uint32_t cp;
    std::size_t n;
    if ((c & 0x80) == 0x00) { cp = c;          n = 1; }
    else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; n = 2; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; n = 3; }
    else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; n = 4; }
    else { cp = c; n = 1; }
    for (std::size_t k = 1; k < n && i + k < s.size(); ++k) {
        cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
    }
    i += n;
    return cp;
}

// ─── Tiny JSON loader for vocab.json ───────────────────────────────────────
//
// vocab.json is a flat object: `{"token_string": id, ...}`.

namespace {

struct JsonReader {
    const char* p;
    const char* end;

    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error("vocab.json: " + msg);
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
                        encode_codepoint_utf8(cp, out);
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

    int32_t parse_int() {
        skip_ws();
        bool neg = false;
        if (p < end && *p == '-') { neg = true; ++p; }
        if (p >= end || *p < '0' || *p > '9') fail("expected integer");
        int64_t v = 0;
        while (p < end && *p >= '0' && *p <= '9') {
            const int digit = *p - '0';
            if (v > (INT64_MAX - digit) / 10) fail("integer out of range");
            v = v * 10 + digit;
            ++p;
        }
        v = neg ? -v : v;
        if (v < INT32_MIN || v > INT32_MAX) fail("integer out of int32 range");
        return static_cast<int32_t>(v);
    }
};

}  // namespace

static std::unordered_map<std::string, int32_t>
load_vocab_json(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("vocab.json: cannot open '" + path + "'");
    std::stringstream ss;
    ss << f.rdbuf();
    std::string text = ss.str();

    JsonReader r{text.data(), text.data() + text.size()};
    r.expect('{');
    std::unordered_map<std::string, int32_t> vocab;
    bool first = true;
    while (true) {
        r.skip_ws();
        if (r.eat('}')) break;
        if (!first) r.expect(',');
        first = false;
        std::string key = r.parse_string();
        r.expect(':');
        int32_t id = r.parse_int();
        vocab.emplace(std::move(key), id);
    }
    return vocab;
}

// ─── merges.txt loader ─────────────────────────────────────────────────────

static std::unordered_map<std::string, int32_t>
load_merges_txt(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("merges.txt: cannot open '" + path + "'");

    std::unordered_map<std::string, int32_t> ranks;
    std::string line;
    int32_t rank = 0;
    bool first = true;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        // The first non-empty line of a merges.txt is a "#version" comment.
        if (first) {
            first = false;
            if (line[0] == '#') continue;
        }
        auto sp = line.find(' ');
        if (sp == std::string::npos) {
            throw std::runtime_error("merges.txt: malformed line: '" + line + "'");
        }
        std::string a = line.substr(0, sp);
        std::string b = line.substr(sp + 1);
        std::string key = a + '\x01' + b;
        ranks.emplace(std::move(key), rank++);
    }
    return ranks;
}

// ─── BPE ───────────────────────────────────────────────────────────────────

namespace {

// Split a UTF-8 string into a list of unicode codepoints (each as its own
// UTF-8-encoded substring). Input is valid UTF-8 by construction.
std::vector<std::string> split_codepoints(const std::string& s) {
    std::vector<std::string> out;
    out.reserve(s.size());
    std::size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        std::size_t n = 1;
        if      ((c & 0x80) == 0x00) n = 1;
        else if ((c & 0xE0) == 0xC0) n = 2;
        else if ((c & 0xF0) == 0xE0) n = 3;
        else if ((c & 0xF8) == 0xF0) n = 4;
        out.emplace_back(s.substr(i, n));
        i += n;
    }
    return out;
}

}  // namespace

std::vector<std::string> Tokenizer::bpe_(const std::string& token) const {
    if (token.empty()) return {};

    // word = codepoints. Unlike CLIP, no </w> marker is glued onto the last.
    std::vector<std::string> word = split_codepoints(token);
    if (word.size() <= 1) return word;

    while (true) {
        // Find the lowest-rank adjacent pair.
        int best_rank = -1;
        std::size_t best_i = 0;
        for (std::size_t i = 0; i + 1 < word.size(); ++i) {
            std::string key = word[i];
            key += '\x01';
            key += word[i + 1];
            auto it = merge_ranks_.find(key);
            if (it != merge_ranks_.end()) {
                if (best_rank < 0 || it->second < best_rank) {
                    best_rank = it->second;
                    best_i = i;
                }
            }
        }
        if (best_rank < 0) break;

        // Merge every non-overlapping occurrence of the chosen pair.
        const std::string a = word[best_i];
        const std::string b = word[best_i + 1];
        std::vector<std::string> next;
        next.reserve(word.size());
        std::size_t i = 0;
        while (i < word.size()) {
            if (i + 1 < word.size() && word[i] == a && word[i + 1] == b) {
                next.emplace_back(a + b);
                i += 2;
            } else {
                next.emplace_back(word[i]);
                i += 1;
            }
        }
        word = std::move(next);
        if (word.size() == 1) break;
    }
    return word;
}

// ─── Pre-tokenization (ASCII-focused, GPT-2 style) ─────────────────────────
//
// GPT-2 regex (simplified):
//   's | 't | 're | 've | 'm | 'll | 'd
//   | ?letters+ | ?digits+ | ?punctuation+ | whitespace-runs
//
// where "?" is an optional single leading space folded into the token. We
// approximate this for ASCII: a single leading space is attached to the
// following run; non-ASCII bytes fall into the "punctuation run" category.
// Unlike CLIP, whitespace is preserved (folded in), not dropped.

namespace {

bool is_ascii_letter(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
bool is_ascii_digit(unsigned char c) { return c >= '0' && c <= '9'; }
bool is_ascii_space(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

bool starts_with(std::string_view s, std::size_t i, const char* lit) {
    std::size_t n = std::strlen(lit);
    if (i + n > s.size()) return false;
    return std::memcmp(s.data() + i, lit, n) == 0;
}

// GPT-2-style pre-tokenization. A single leading space is folded into the
// next pre-token (" word" stays as one piece). Other whitespace bytes are
// emitted as their own pre-tokens so they round-trip through byte encoding.
std::vector<std::string> pre_tokenize(std::string_view text) {
    std::vector<std::string> pieces;
    std::size_t i = 0;
    while (i < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[i]);

        // Contractions: ' followed by s|t|re|ve|m|ll|d.
        if (c == '\'') {
            if (starts_with(text, i, "'re")) { pieces.emplace_back("'re"); i += 3; continue; }
            if (starts_with(text, i, "'ve")) { pieces.emplace_back("'ve"); i += 3; continue; }
            if (starts_with(text, i, "'ll")) { pieces.emplace_back("'ll"); i += 3; continue; }
            if (starts_with(text, i, "'s"))  { pieces.emplace_back("'s");  i += 2; continue; }
            if (starts_with(text, i, "'t"))  { pieces.emplace_back("'t");  i += 2; continue; }
            if (starts_with(text, i, "'m"))  { pieces.emplace_back("'m");  i += 2; continue; }
            if (starts_with(text, i, "'d"))  { pieces.emplace_back("'d");  i += 2; continue; }
            // Bare apostrophe falls into the punct branch below.
        }

        // An optional single leading space folds into the following run.
        std::size_t start = i;
        bool have_lead_space = false;
        if (c == ' ') {
            // Look at what follows the space.
            if (i + 1 < text.size()) {
                unsigned char d = static_cast<unsigned char>(text[i + 1]);
                if (!is_ascii_space(d)) {
                    have_lead_space = true;
                    ++i;
                    c = d;
                }
            }
            if (!have_lead_space) {
                // Lone space (followed by EOF or more whitespace): emit it
                // alone so it byte-encodes to a standalone Ġ.
                pieces.emplace_back(text.substr(i, 1));
                ++i;
                continue;
            }
        } else if (is_ascii_space(c)) {
            // Non-space whitespace (tab/newline/...): emit as its own run.
            std::size_t j = i;
            while (j < text.size() &&
                   is_ascii_space(static_cast<unsigned char>(text[j])) &&
                   text[j] != ' ') ++j;
            pieces.emplace_back(text.substr(i, j - i));
            i = j;
            continue;
        }

        if (is_ascii_letter(c)) {
            std::size_t j = i;
            while (j < text.size() &&
                   is_ascii_letter(static_cast<unsigned char>(text[j]))) ++j;
            pieces.emplace_back(text.substr(start, j - start));
            i = j;
            continue;
        }

        if (is_ascii_digit(c)) {
            std::size_t j = i;
            while (j < text.size() &&
                   is_ascii_digit(static_cast<unsigned char>(text[j]))) ++j;
            pieces.emplace_back(text.substr(start, j - start));
            i = j;
            continue;
        }

        // Everything else: run of non-whitespace, non-letter, non-digit bytes.
        std::size_t j = i;
        while (j < text.size()) {
            unsigned char u = static_cast<unsigned char>(text[j]);
            if (is_ascii_space(u) || is_ascii_letter(u) || is_ascii_digit(u)) break;
            // Break on apostrophe-contraction lookahead.
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

// ─── Tokenizer driver ──────────────────────────────────────────────────────

Tokenizer Tokenizer::load(const std::string& vocab_json_path,
                          const std::string& merges_txt_path,
                          const std::vector<std::string>& extra_special_tokens) {
    Tokenizer t;
    t.vocab_ = load_vocab_json(vocab_json_path);
    t.merge_ranks_ = load_merges_txt(merges_txt_path);
    build_byte_unicode_maps(t.byte_to_unicode_, t.unicode_to_byte_);

    for (const auto& [tok, id] : t.vocab_) {
        t.id_to_token_.emplace(id, tok);
    }

    // Built-in Qwen3 special tokens plus any caller extensions. A special
    // token is only registered if it is actually present in vocab.json.
    std::vector<std::string> specials = {
        "<|endoftext|>", "<|im_start|>", "<|im_end|>"
    };
    specials.insert(specials.end(),
                    extra_special_tokens.begin(), extra_special_tokens.end());
    for (const auto& s : specials) {
        auto it = t.vocab_.find(s);
        if (it == t.vocab_.end()) continue;
        t.special_tokens_.emplace(s, it->second);
        t.special_ids_.emplace(it->second, s);
    }

    auto lookup = [&](const char* s) -> int {
        auto it = t.vocab_.find(s);
        return it != t.vocab_.end() ? it->second : -1;
    };
    t.endoftext_id_ = lookup("<|endoftext|>");
    t.im_start_id_  = lookup("<|im_start|>");
    t.im_end_id_    = lookup("<|im_end|>");
    return t;
}

void Tokenizer::register_special_token(const std::string& token, int32_t id) {
    // Drop any previous binding (forward and inverse) before re-inserting,
    // so re-registration is well-defined.
    auto prev = special_tokens_.find(token);
    if (prev != special_tokens_.end()) {
        special_ids_.erase(prev->second);
        special_tokens_.erase(prev);
    }
    special_tokens_.emplace(token, id);
    special_ids_.emplace(id, token);
    if (token == "<|endoftext|>") endoftext_id_ = id;
    else if (token == "<|im_start|>") im_start_id_ = id;
    else if (token == "<|im_end|>")   im_end_id_   = id;
}

void Tokenizer::encode_piece_(std::string_view piece,
                              std::vector<int32_t>& out) const {
    if (piece.empty()) return;

    // Byte-encode the piece into the GPT-2 unicode space.
    std::string encoded;
    encoded.reserve(piece.size() * 2);
    for (char c : piece) {
        encoded += byte_to_unicode_[static_cast<unsigned char>(c)];
    }

    auto units = bpe_(encoded);
    for (const auto& u : units) {
        auto it = vocab_.find(u);
        if (it != vocab_.end()) out.push_back(it->second);
        // Unknown unit: drop. With a complete vocab every single byte-encoded
        // character is itself in the vocab, so this cannot happen.
    }
}

std::vector<int32_t> Tokenizer::encode(std::string_view text,
                                       bool add_special) const {
    std::vector<int32_t> ids;
    ids.reserve(text.size());

    // Walk the input, splitting out verbatim special-token substrings and
    // BPE-encoding the spans between them.
    std::size_t i = 0;
    std::size_t span_start = 0;
    while (i < text.size()) {
        bool matched = false;
        // Try the longest special token that starts at i. (Specials are few,
        // a linear scan is fine.)
        std::size_t best_len = 0;
        int32_t best_id = -1;
        for (const auto& [s, id] : special_tokens_) {
            if (s.size() > best_len &&
                i + s.size() <= text.size() &&
                std::memcmp(text.data() + i, s.data(), s.size()) == 0) {
                best_len = s.size();
                best_id = id;
            }
        }
        if (best_len > 0) {
            // Flush the BPE span before this special token.
            if (i > span_start) {
                std::string_view span = text.substr(span_start, i - span_start);
                for (const auto& p : pre_tokenize(span)) encode_piece_(p, ids);
            }
            ids.push_back(best_id);
            i += best_len;
            span_start = i;
            matched = true;
        }
        if (!matched) ++i;
    }
    // Trailing span.
    if (text.size() > span_start) {
        std::string_view span = text.substr(span_start);
        for (const auto& p : pre_tokenize(span)) encode_piece_(p, ids);
    }

    // Qwen3 has no BOS. add_special only appends <|endoftext|> as an EOS hook.
    if (add_special && endoftext_id_ >= 0) {
        ids.push_back(endoftext_id_);
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    // Concatenate the byte-encoded forms of every non-special piece, then
    // byte-decode the whole thing back to raw bytes. Special-token ids emit
    // their literal string directly.
    std::string out;
    std::string encoded;  // pending run of byte-encoded pieces
    auto flush_encoded = [&]() {
        std::size_t i = 0;
        while (i < encoded.size()) {
            uint32_t cp = next_codepoint(encoded, i);
            auto it = unicode_to_byte_.find(cp);
            if (it != unicode_to_byte_.end()) {
                out += static_cast<char>(it->second);
            }
            // A codepoint not in the byte map is not a valid byte-level piece;
            // skip it (cannot happen for vocab produced by encode()).
        }
        encoded.clear();
    };

    for (int32_t id : ids) {
        auto sp = special_ids_.find(id);
        if (sp != special_ids_.end()) {
            flush_encoded();
            out += sp->second;
            continue;
        }
        auto it = id_to_token_.find(id);
        if (it != id_to_token_.end()) {
            encoded += it->second;
        }
        // Unknown id: skip.
    }
    flush_encoded();
    return out;
}

std::string Tokenizer::apply_chat_template(
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool add_generation_prompt) const {
    std::string out;
    for (const auto& [role, content] : messages) {
        out += "<|im_start|>";
        out += role;
        out += '\n';
        out += content;
        out += "<|im_end|>\n";
    }
    if (add_generation_prompt) {
        out += "<|im_start|>assistant\n";
    }
    return out;
}

}  // namespace brolm::qwen
