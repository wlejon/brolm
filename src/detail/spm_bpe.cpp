#include "brolm/detail/spm_bpe.h"

#include <cstdio>
#include <cstring>
#include <limits>

namespace brolm::detail::spm {

namespace {

// The metaspace character U+2581 ("\xE2\x96\x81"), 3 UTF-8 bytes.
const char kMetaspace[] = "\xE2\x96\x81";

// Byte-length of the UTF-8 character starting at s[i]. Assumes valid UTF-8.
std::size_t utf8_char_len(std::string_view s, std::size_t i) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    if      ((c & 0x80) == 0x00) return 1;
    else if ((c & 0xE0) == 0xC0) return 2;
    else if ((c & 0xF0) == 0xE0) return 3;
    else if ((c & 0xF8) == 0xF0) return 4;
    return 1;  // invalid lead byte — treat as 1 to make progress
}

// The SentencePiece byte-fallback piece for raw byte `b`: "<0xNN>" (uppercase
// hex, matching HF tokenizers / SentencePiece).
std::string byte_piece(unsigned char b) {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
    return std::string(buf);
}

// Metaspace pre-tokenization: replace every ASCII space with U+2581, and
// optionally prepend one U+2581 (add_prefix_space / add_dummy_prefix).
std::string metaspace(std::string_view text, bool add_prefix_space) {
    std::string out;
    out.reserve(text.size() + text.size() / 4 + 3);
    if (add_prefix_space) out += kMetaspace;
    for (char c : text) {
        if (c == ' ') out += kMetaspace;
        else          out += c;
    }
    return out;
}

}  // namespace

void Bpe::ensure_id(int32_t id) {
    if (id >= 0 && static_cast<std::size_t>(id) >= id_to_piece_.size())
        id_to_piece_.resize(static_cast<std::size_t>(id) + 1);
}

void Bpe::add(const std::string& piece, int32_t id) {
    vocab_[piece] = id;
    ensure_id(id);
    if (id >= 0) id_to_piece_[static_cast<std::size_t>(id)] = piece;
}

void Bpe::add_decode_only(const std::string& piece, int32_t id) {
    ensure_id(id);
    if (id >= 0) id_to_piece_[static_cast<std::size_t>(id)] = piece;
}

void Bpe::add_merge(const std::string& left, const std::string& right,
                    int32_t rank) {
    std::string key = left;
    key += '\x01';
    key += right;
    merge_ranks_.emplace(std::move(key), rank);
}

void Bpe::finalize() {}

// ─── tokenize (greedy lowest-rank BPE with byte-fallback) ───────────────────

void Bpe::merge_word(const std::string& word, int unk_id,
                     std::vector<int32_t>& out) const {
    // Build the initial symbol sequence: each UTF-8 character is one symbol if
    // it is in the vocab; otherwise (byte_fallback) it decomposes into one
    // "<0xNN>" symbol per UTF-8 byte. Without byte_fallback the raw character
    // is kept as a symbol and will miss the vocab at the final lookup.
    std::vector<std::string> syms;
    syms.reserve(word.size());
    for (std::size_t i = 0; i < word.size();) {
        const std::size_t clen = utf8_char_len(word, i);
        std::string ch = word.substr(i, clen);
        i += clen;
        if (vocab_.find(ch) != vocab_.end()) {
            syms.push_back(std::move(ch));
        } else if (byte_fallback_) {
            for (char bc : ch)
                syms.push_back(byte_piece(static_cast<unsigned char>(bc)));
        } else {
            syms.push_back(std::move(ch));
        }
    }
    if (syms.empty()) return;

    // Greedy merge loop: repeatedly merge the lowest-rank adjacent pair.
    while (syms.size() > 1) {
        int best_rank = std::numeric_limits<int>::max();
        std::size_t best_pos = 0;
        bool found = false;
        for (std::size_t j = 0; j + 1 < syms.size(); ++j) {
            std::string key = syms[j];
            key += '\x01';
            key += syms[j + 1];
            auto it = merge_ranks_.find(key);
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos = j;
                found = true;
            }
        }
        if (!found) break;
        syms[best_pos] += syms[best_pos + 1];
        syms.erase(syms.begin() + static_cast<std::ptrdiff_t>(best_pos) + 1);
    }

    // Map symbols to ids; fold runs of misses into <unk> (fuse_unk) or one unk
    // per miss.
    for (const auto& s : syms) {
        auto it = vocab_.find(s);
        if (it != vocab_.end()) {
            out.push_back(it->second);
        } else if (!fuse_unk_ || out.empty() || out.back() != unk_id) {
            out.push_back(unk_id);
        }
    }
}

std::vector<int32_t> Bpe::tokenize(std::string_view text, int unk_id) const {
    const std::string s = metaspace(text, add_prefix_space_);
    std::vector<int32_t> out;
    if (s.empty()) return out;
    out.reserve(s.size() / 2 + 1);

    const std::size_t ms = sizeof(kMetaspace) - 1;  // 3 bytes

    // Split on metaspace boundaries: each "word" starts at a metaspace (or at
    // string start). SentencePiece pieces never carry a metaspace except at
    // their start, so no merge can fire across a boundary — splitting only
    // bounds the merge-loop window.
    std::size_t i = 0;
    while (i < s.size()) {
        std::size_t next = s.find(kMetaspace, i + ms);
        if (next == std::string::npos) next = s.size();
        merge_word(s.substr(i, next - i), unk_id, out);
        i = next;
    }
    return out;
}

// ─── decode ─────────────────────────────────────────────────────────────────

std::string Bpe::decode(const std::vector<int32_t>& ids,
                        const std::unordered_set<int32_t>* skip) const {
    // A piece "<0xNN>" is a SentencePiece byte-fallback token standing for the
    // single raw byte 0xNN. Returns -1 when `p` is not such a token.
    auto byte_fallback = [](const std::string& p) -> int {
        if (p.size() != 6 || p[0] != '<' || p[1] != '0' ||
            (p[2] != 'x' && p[2] != 'X') || p[5] != '>') {
            return -1;
        }
        auto hex = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            return -1;
        };
        const int hi = hex(p[3]), lo = hex(p[4]);
        if (hi < 0 || lo < 0) return -1;
        return (hi << 4) | lo;
    };

    // Concatenate the raw piece bytes (metaspace still embedded).
    std::string raw;
    for (int32_t id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= id_to_piece_.size())
            continue;
        if (skip && skip->count(id)) continue;
        const std::string& p = id_to_piece_[static_cast<std::size_t>(id)];
        const int b = byte_fallback(p);
        if (b >= 0) raw.push_back(static_cast<char>(b));
        else        raw += p;
    }

    // Replace every metaspace (U+2581) with an ASCII space.
    std::string out;
    out.reserve(raw.size());
    const std::size_t ms = sizeof(kMetaspace) - 1;  // 3 bytes
    for (std::size_t i = 0; i < raw.size();) {
        if (i + ms <= raw.size() &&
            std::memcmp(raw.data() + i, kMetaspace, ms) == 0) {
            out.push_back(' ');
            i += ms;
        } else {
            out.push_back(raw[i]);
            ++i;
        }
    }

    // Strip the single leading space introduced by add_prefix_space.
    if (add_prefix_space_ && !out.empty() && out.front() == ' ')
        out.erase(out.begin());
    return out;
}

}  // namespace brolm::detail::spm
