#pragma once

// Shared SentencePiece Unigram core for brolm's Unigram-family tokenizers
// (T5, NLLB-200 / M2M-100, and any future SPM-Unigram model).
//
// The model is a table of (piece, log-prob score) entries whose array index is
// the token id. Segmentation is a Unigram Viterbi over the metaspace-
// transformed UTF-8 string; decoding is the inverse id->piece join.
//
// Pre-tokenization is metaspace-based: every ASCII space becomes the metaspace
// character U+2581 ("\xE2\x96\x81"), and one metaspace is prepended to the
// whole string (add_prefix_space). Heavy Unicode normalization is skipped — for
// the common case metaspace is the part that matters; this mirrors what the T5
// tokenizer has always done.
//
// Each model family wraps this core with its own load() framing (which special
// ids and added tokens it cares about) and its own encode() sequence layout.

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brolm::detail::spm {

class Unigram {
public:
    struct Entry { int32_t id; double score; };

    // Register a Viterbi-matchable piece at `id` with `score`. The id also
    // becomes decodable (id -> piece). Call finalize() once after all adds.
    void add(const std::string& piece, int32_t id, double score);

    // Register a token that is decodable (id -> piece) but NOT Viterbi-
    // matchable — for special / added tokens (eos, pad, language codes) that
    // are only ever inserted explicitly, never segmented out of normal text.
    void add_decode_only(const std::string& piece, int32_t id);

    // Compute the Viterbi scan window and the unk-fallback penalty. Idempotent.
    void finalize();

    // Unigram Viterbi over the metaspace-transformed `text`. Pieces only — no
    // special tokens. Unknown characters route through `unk_id`.
    std::vector<int32_t> tokenize(std::string_view text, int unk_id) const;

    // id->piece join: metaspace U+2581 -> space, the single leading space
    // (add_prefix_space) stripped, byte-fallback pieces "<0xNN>" -> raw byte.
    // Ids in `skip` (if non-null) are dropped before joining; ids outside the
    // vocab are skipped.
    std::string decode(const std::vector<int32_t>& ids,
                       const std::unordered_set<int32_t>* skip = nullptr) const;

    bool has(const std::string& piece) const {
        return vocab_.find(piece) != vocab_.end();
    }
    // Number of distinct registered pieces (Viterbi + decode-only).
    std::size_t size() const { return vocab_.size(); }
    std::size_t id_count() const { return id_to_piece_.size(); }
    double min_score() const { return min_score_; }

private:
    void ensure_id(int32_t id);

    std::unordered_map<std::string, Entry> vocab_;  // Viterbi-matchable pieces
    std::vector<std::string> id_to_piece_;          // for decode()
    std::size_t max_piece_bytes_ = 1;
    double min_score_ = 0.0;
};

}  // namespace brolm::detail::spm
