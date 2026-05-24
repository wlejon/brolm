#include "brolm/detail/byte_level_bpe.h"

#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace brolm::detail::bpe {

// ─── UTF-8 ─────────────────────────────────────────────────────────────────

void encode_codepoint_utf8(uint32_t cp, std::string& out) {
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

uint32_t next_codepoint(const std::string& s, std::size_t& i) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    uint32_t cp;
    std::size_t n;
    if      ((c & 0x80) == 0x00) { cp = c;          n = 1; }
    else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F;   n = 2; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F;   n = 3; }
    else if ((c & 0xF8) == 0xF0) { cp = c & 0x07;   n = 4; }
    else                         { cp = c;          n = 1; }
    for (std::size_t k = 1; k < n && i + k < s.size(); ++k) {
        cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
    }
    i += n;
    return cp;
}

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

// ─── Byte ↔ unicode ────────────────────────────────────────────────────────

namespace {

void build_self_map(bool self_map[256]) {
    for (int b = 0; b < 256; ++b) self_map[b] = false;
    auto mark_range = [&](int lo, int hi) {
        for (int b = lo; b <= hi; ++b) self_map[b] = true;
    };
    mark_range(33, 126);
    mark_range(161, 172);
    mark_range(174, 255);
}

}  // namespace

void build_byte_to_unicode(std::string out[256]) {
    bool self_map[256];
    build_self_map(self_map);
    int next_cp = 256;
    for (int b = 0; b < 256; ++b) {
        uint32_t cp = self_map[b] ? static_cast<uint32_t>(b)
                                  : static_cast<uint32_t>(next_cp++);
        out[b].clear();
        encode_codepoint_utf8(cp, out[b]);
    }
}

void build_byte_unicode_maps(
    std::string out[256],
    std::unordered_map<uint32_t, unsigned char>& inverse) {
    bool self_map[256];
    build_self_map(self_map);
    inverse.clear();
    int next_cp = 256;
    for (int b = 0; b < 256; ++b) {
        uint32_t cp = self_map[b] ? static_cast<uint32_t>(b)
                                  : static_cast<uint32_t>(next_cp++);
        out[b].clear();
        encode_codepoint_utf8(cp, out[b]);
        inverse[cp] = static_cast<unsigned char>(b);
    }
}

// ─── Vocab + merges loaders ────────────────────────────────────────────────
//
// Tiny inline JSON reader — vocab.json is a flat {"token": id, ...} object,
// so the general-purpose brolm::detail::json parser is overkill (and slower
// at million-entry vocab sizes). Strings handle the standard JSON escapes.

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

std::unordered_map<std::string, int32_t>
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

std::unordered_map<std::string, int32_t>
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

// ─── BPE merge loop ────────────────────────────────────────────────────────

std::vector<std::string> bpe_merge(
    const std::string& token,
    const std::unordered_map<std::string, int32_t>& merge_ranks,
    bool append_end_of_word) {
    if (token.empty()) return {};

    std::vector<std::string> word = split_codepoints(token);
    if (word.empty()) return {};
    if (append_end_of_word) word.back() += "</w>";
    if (word.size() == 1) return word;

    while (true) {
        // Find the lowest-rank adjacent pair.
        int best_rank = -1;
        std::size_t best_i = 0;
        for (std::size_t i = 0; i + 1 < word.size(); ++i) {
            std::string key = word[i];
            key += '\x01';
            key += word[i + 1];
            auto it = merge_ranks.find(key);
            if (it != merge_ranks.end()) {
                if (best_rank < 0 || it->second < best_rank) {
                    best_rank = it->second;
                    best_i = i;
                }
            }
        }
        if (best_rank < 0) break;

        // Merge every non-overlapping occurrence of the chosen pair in one
        // pass over the word — the canonical BPE behaviour.
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

void encode_piece(
    std::string_view piece,
    const std::string byte_to_unicode[256],
    const std::unordered_map<std::string, int32_t>& vocab,
    const std::unordered_map<std::string, int32_t>& merge_ranks,
    bool append_end_of_word,
    std::vector<int32_t>& out) {
    if (piece.empty()) return;

    std::string encoded;
    encoded.reserve(piece.size() * 2);
    for (char c : piece) {
        encoded += byte_to_unicode[static_cast<unsigned char>(c)];
    }

    auto units = bpe_merge(encoded, merge_ranks, append_end_of_word);
    for (const auto& u : units) {
        auto it = vocab.find(u);
        if (it != vocab.end()) out.push_back(it->second);
        // Unknown unit: drop. With a complete byte-level vocab every single
        // byte-encoded character is itself in vocab, so this never happens.
    }
}

// ─── ASCII helper ──────────────────────────────────────────────────────────

bool starts_with(std::string_view s, std::size_t i, const char* lit) {
    std::size_t n = std::strlen(lit);
    if (i + n > s.size()) return false;
    return std::memcmp(s.data() + i, lit, n) == 0;
}

}  // namespace brolm::detail::bpe
