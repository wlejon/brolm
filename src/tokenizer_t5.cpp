#include "brolm/tokenizer_t5.h"

#include "brolm/detail/json.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace brolm::t5 {

namespace json = ::brolm::detail::json;

namespace {

// The metaspace character U+2581 ("▁"), UTF-8 0xE2 0x96 0x81.
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

// ─── load ──────────────────────────────────────────────────────────────────

Tokenizer Tokenizer::load(const std::string& tokenizer_json_path) {
    std::ifstream f(tokenizer_json_path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("t5::Tokenizer: cannot open '" +
                                 tokenizer_json_path + "'");
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();

    json::Value root = json::parse(text);  // throws on malformed input
    const json::Value* model = root.find("model");
    if (!model || !model->is_object()) {
        throw std::runtime_error("t5::Tokenizer: tokenizer.json has no "
                                 "'model' object");
    }
    const json::Value* vocab = model->find("vocab");
    if (!vocab || !(vocab->is_array() || vocab->is_object())) {
        throw std::runtime_error("t5::Tokenizer: model has no 'vocab' "
                                 "array or object");
    }

    Tokenizer t;
    t.unk_id_ = model->get_int("unk_id", 2);

    int eos_id = -1;
    int pad_id = -1;
    double min_score = std::numeric_limits<double>::infinity();
    std::size_t max_bytes = 1;

    // Register one (piece, id, score) entry into the vocab + id->piece map.
    auto add_piece = [&](const std::string& piece, int32_t id, double score) {
        t.vocab_.emplace(piece, Tokenizer::Entry{id, score});
        if (static_cast<std::size_t>(id) >= t.id_to_piece_.size())
            t.id_to_piece_.resize(static_cast<std::size_t>(id) + 1);
        t.id_to_piece_[static_cast<std::size_t>(id)] = piece;
        if (!piece.empty() && piece.size() > max_bytes) max_bytes = piece.size();
        if (score < min_score) min_score = score;
        if (piece == "</s>") eos_id = id;
        if (piece == "<pad>") pad_id = id;
    };

    if (vocab->is_array()) {
        // SentencePiece Unigram: vocab is [[piece, log-prob], ...], id = index.
        const auto& entries = vocab->as_array();
        if (entries.empty())
            throw std::runtime_error("t5::Tokenizer: empty vocab");
        t.id_to_piece_.assign(entries.size(), std::string());
        for (std::size_t i = 0; i < entries.size(); ++i) {
            const auto& pair = entries[i];
            if (!pair.is_array() || pair.as_array().size() < 2)
                throw std::runtime_error("t5::Tokenizer: malformed vocab entry");
            add_piece(pair.as_array()[0].as_string(),
                      static_cast<int32_t>(i), pair.as_array()[1].as_number());
        }
    } else {
        // HF BPE (SentencePiece exported as BPE, e.g. NeMo Parakeet): vocab is
        // a {piece: id} object with no per-piece scores. encode()/tokenize()'s
        // Unigram Viterbi is not meaningful here (scores are 0), but decode()
        // — the inverse id->piece metaspace join the ASR drivers need — is
        // exact, which is the supported operation for this vocab shape.
        const auto& members = vocab->as_object();
        if (members.empty())
            throw std::runtime_error("t5::Tokenizer: empty vocab");
        for (const auto& m : members)
            add_piece(m.first, static_cast<int32_t>(m.second.as_number()), 0.0);
    }

    t.eos_id_ = (eos_id >= 0) ? eos_id : 1;
    t.pad_id_ = (pad_id >= 0) ? pad_id : 0;
    t.max_piece_bytes_ = max_bytes;
    t.min_vocab_score_ = min_score;
    return t;
}

// ─── tokenize (Unigram Viterbi) ────────────────────────────────────────────

std::vector<int32_t> Tokenizer::tokenize(std::string_view text) const {
    const std::string s = metaspace(text);
    const std::size_t n = s.size();
    if (n == 0) return {};

    const double NEG_INF = -std::numeric_limits<double>::infinity();
    const double unk_penalty = min_vocab_score_ - 10.0;

    // best_score[i] = best log-prob to reach byte position i.
    // back[i] = {start_index, token_id} of the piece ending at i.
    std::vector<double>  best_score(n + 1, NEG_INF);
    std::vector<std::size_t> back_start(n + 1, 0);
    std::vector<int32_t> back_id(n + 1, unk_id_);
    best_score[0] = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        if (best_score[i] == NEG_INF) continue;

        // Scan piece candidates s[i:j] at UTF-8 character boundaries, up to
        // the longest vocab piece byte-length.
        const std::size_t scan_end =
            std::min(n, i + max_piece_bytes_);
        std::size_t j = i;
        bool first_char = true;
        std::size_t first_char_end = i;
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
            if (first_char) {
                first_char_end = j;
                first_char = false;
            }
        }

        // Unknown-char fallback: the single character s[i:i+charlen] always
        // gets a transition via the unk token, guaranteeing a complete path.
        const std::size_t one_char_end = i + utf8_char_len(s, i);
        if (one_char_end <= n) {
            const std::string one = s.substr(i, one_char_end - i);
            const bool known = vocab_.find(one) != vocab_.end();
            if (!known) {
                const double cand = best_score[i] + unk_penalty;
                if (cand > best_score[one_char_end]) {
                    best_score[one_char_end] = cand;
                    back_start[one_char_end] = i;
                    back_id[one_char_end]    = unk_id_;
                }
            }
        }
        (void)first_char_end;
    }

    // Backtrack from the end to recover the token-id sequence.
    std::vector<int32_t> ids;
    std::size_t pos = n;
    while (pos > 0) {
        if (best_score[pos] == NEG_INF) {
            // No path reached `pos` — shouldn't happen with the unk
            // fallback, but guard anyway by stepping back one character.
            break;
        }
        ids.push_back(back_id[pos]);
        pos = back_start[pos];
    }
    std::reverse(ids.begin(), ids.end());
    return ids;
}

// ─── decode ────────────────────────────────────────────────────────────────

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
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

// ─── encode ────────────────────────────────────────────────────────────────

std::vector<int32_t> Tokenizer::encode(std::string_view text,
                                       int max_length) const {
    std::vector<int32_t> ids = tokenize(text);

    std::vector<int32_t> out;
    if (max_length <= 0) return out;
    out.reserve(static_cast<std::size_t>(max_length));

    // Content fills up to max_length-1 slots; eos always takes the last used
    // slot. If content + eos overflows, truncate the content.
    const std::size_t content_cap =
        static_cast<std::size_t>(max_length) - 1;
    const std::size_t n =
        (ids.size() > content_cap) ? content_cap : ids.size();
    for (std::size_t i = 0; i < n; ++i) out.push_back(ids[i]);

    out.push_back(eos_id_);
    while (out.size() < static_cast<std::size_t>(max_length)) {
        out.push_back(pad_id_);
    }
    return out;
}

}  // namespace brolm::t5
