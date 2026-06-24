#include "brolm/detail/spm_unigram.h"

#include <algorithm>
#include <cmath>
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

// Metaspace pre-tokenization: replace every ASCII space with U+2581 and
// prepend one U+2581 (add_prefix_space).
std::string metaspace(std::string_view text) {
    std::string out;
    out.reserve(text.size() + text.size() / 4 + 3);
    out += kMetaspace;
    for (char c : text) {
        if (c == ' ') out += kMetaspace;
        else          out += c;
    }
    return out;
}

}  // namespace

void Unigram::ensure_id(int32_t id) {
    if (id >= 0 && static_cast<std::size_t>(id) >= id_to_piece_.size())
        id_to_piece_.resize(static_cast<std::size_t>(id) + 1);
}

void Unigram::add(const std::string& piece, int32_t id, double score) {
    vocab_[piece] = Entry{id, score};
    ensure_id(id);
    if (id >= 0) id_to_piece_[static_cast<std::size_t>(id)] = piece;
}

void Unigram::add_decode_only(const std::string& piece, int32_t id) {
    ensure_id(id);
    if (id >= 0) id_to_piece_[static_cast<std::size_t>(id)] = piece;
}

void Unigram::finalize() {
    max_piece_bytes_ = 1;
    min_score_ = std::numeric_limits<double>::infinity();
    for (const auto& kv : vocab_) {
        if (!kv.first.empty() && kv.first.size() > max_piece_bytes_)
            max_piece_bytes_ = kv.first.size();
        if (kv.second.score < min_score_) min_score_ = kv.second.score;
    }
    if (!std::isfinite(min_score_)) min_score_ = 0.0;
}

// ─── tokenize (Unigram Viterbi) ─────────────────────────────────────────────

std::vector<int32_t> Unigram::tokenize(std::string_view text, int unk_id) const {
    const std::string s = metaspace(text);
    const std::size_t n = s.size();
    if (n == 0) return {};

    const double NEG_INF = -std::numeric_limits<double>::infinity();
    const double unk_penalty = min_score_ - 10.0;

    // best_score[i] = best log-prob to reach byte position i.
    // back[i] = {start_index, token_id} of the piece ending at i.
    std::vector<double>  best_score(n + 1, NEG_INF);
    std::vector<std::size_t> back_start(n + 1, 0);
    std::vector<int32_t> back_id(n + 1, unk_id);
    best_score[0] = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        if (best_score[i] == NEG_INF) continue;

        // Scan piece candidates s[i:j] at UTF-8 character boundaries, up to the
        // longest vocab piece byte-length.
        const std::size_t scan_end = std::min(n, i + max_piece_bytes_);
        std::size_t j = i;
        while (j < scan_end) {
            j += utf8_char_len(s, j);
            if (j > scan_end) break;  // would overrun the scan window
            const std::string piece = s.substr(i, j - i);
            auto it = vocab_.find(piece);
            if (it != vocab_.end()) {
                const double cand = best_score[i] + it->second.score;
                if (cand > best_score[j]) {
                    best_score[j] = cand;
                    back_start[j] = i;
                    back_id[j]    = it->second.id;
                }
            }
        }

        // Unknown-char fallback: the single character s[i:i+charlen] always
        // gets a transition via the unk token, guaranteeing a complete path.
        const std::size_t one_char_end = i + utf8_char_len(s, i);
        if (one_char_end <= n) {
            const std::string one = s.substr(i, one_char_end - i);
            if (vocab_.find(one) == vocab_.end()) {
                const double cand = best_score[i] + unk_penalty;
                if (cand > best_score[one_char_end]) {
                    best_score[one_char_end] = cand;
                    back_start[one_char_end] = i;
                    back_id[one_char_end]    = unk_id;
                }
            }
        }
    }

    // Backtrack from the end to recover the token-id sequence.
    std::vector<int32_t> ids;
    std::size_t pos = n;
    while (pos > 0) {
        if (best_score[pos] == NEG_INF) break;  // guarded by unk fallback
        ids.push_back(back_id[pos]);
        pos = back_start[pos];
    }
    std::reverse(ids.begin(), ids.end());
    return ids;
}

// ─── decode ─────────────────────────────────────────────────────────────────

std::string Unigram::decode(const std::vector<int32_t>& ids,
                            const std::unordered_set<int32_t>* skip) const {
    // A piece "<0xNN>" is a SentencePiece byte-fallback token: it stands for
    // the single raw byte 0xNN. Returns -1 when `p` is not such a token.
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
    const std::size_t ms = sizeof(kMetaspace) - 1;   // 3 bytes
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

    // Strip the single leading space from add_prefix_space.
    if (!out.empty() && out.front() == ' ') out.erase(out.begin());
    return out;
}

}  // namespace brolm::detail::spm
