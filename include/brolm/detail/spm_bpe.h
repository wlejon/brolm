#pragma once

// Shared SentencePiece-BPE core for brolm's SPM-BPE-family tokenizers
// (Gemma / Gemma-2, and any future Llama-family SPM-BPE model).
//
// Unlike the Unigram core (spm_unigram.h, used by T5 / NLLB), this model is a
// BPE: a table of pieces (piece -> id) plus an *ordered* merge list (a rank per
// adjacent piece-pair). Tokenization is the standard greedy lowest-rank merge
// loop run over the metaspace-transformed UTF-8 string, with SentencePiece
// **byte-fallback**: a character not present in the vocab as a single piece is
// decomposed into its UTF-8 bytes, each emitted as a "<0xNN>" byte-piece (the
// merge loop may then re-merge them). This is the HF `tokenizers` BPE backend's
// algorithm (byte_fallback at the initial-character level, before merges), which
// is what Gemma's tokenizer.json drives.
//
// Pre-tokenization is metaspace-based: every ASCII space becomes the metaspace
// character U+2581 ("\xE2\x96\x81"). Whether a single leading metaspace is
// prepended (SentencePiece add_dummy_prefix / "add_prefix_space") is a
// per-family switch: Gemma does NOT prepend (its fast tokenizer documents "no
// prefix space"); Llama-family SPM-BPE models do. Heavy Unicode normalization
// is skipped — for the common case the space->metaspace fold is what matters.
//
// The decode rules are identical to the Unigram core's: metaspace U+2581 ->
// space, the single leading space (when add_prefix_space) stripped, and
// "<0xNN>" byte-fallback pieces -> their raw byte.
//
// Each model family wraps this core with its own load() framing (which special
// ids it cares about) and its own encode() sequence layout. The core is kept
// container-agnostic so a future gguf-metadata loader (tokens + merges arrays)
// can feed add()/add_merge() without reworking it.

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brolm::detail::spm {

class Bpe {
public:
    // Register a mergeable / decodable vocab piece at `id`. The id also becomes
    // decodable (id -> piece). Call finalize() once after all add()/add_merge().
    void add(const std::string& piece, int32_t id);

    // Register a token that is decodable (id -> piece) but NOT produced by the
    // merge loop — for added / control tokens inserted only explicitly.
    void add_decode_only(const std::string& piece, int32_t id);

    // Register a merge rule "left right" with priority `rank` (lower rank =
    // higher priority = applied first), in the order they appear in the file.
    void add_merge(const std::string& left, const std::string& right,
                   int32_t rank);

    // Idempotent. Reserved for any precomputation; safe to call repeatedly.
    void finalize();

    // Per-family switches (set before tokenize()).
    void set_add_prefix_space(bool v) { add_prefix_space_ = v; }
    void set_byte_fallback(bool v)    { byte_fallback_ = v; }
    void set_fuse_unk(bool v)         { fuse_unk_ = v; }
    bool add_prefix_space() const { return add_prefix_space_; }

    // Greedy lowest-rank BPE over the metaspace-transformed `text`. Pieces only
    // — no special tokens. A character missing from the vocab decomposes to
    // "<0xNN>" byte-pieces when byte_fallback is on; otherwise (or if a needed
    // byte-piece is itself absent) it routes through `unk_id`.
    std::vector<int32_t> tokenize(std::string_view text, int unk_id) const;

    // id->piece join: metaspace U+2581 -> space, the single leading space
    // (when add_prefix_space) stripped, byte-fallback pieces "<0xNN>" -> raw
    // byte. Ids in `skip` (if non-null) are dropped; ids outside the vocab are
    // skipped.
    std::string decode(const std::vector<int32_t>& ids,
                       const std::unordered_set<int32_t>* skip = nullptr) const;

    bool has(const std::string& piece) const {
        return vocab_.find(piece) != vocab_.end();
    }
    // Number of distinct registered pieces.
    std::size_t size() const { return vocab_.size(); }
    std::size_t id_count() const { return id_to_piece_.size(); }
    std::size_t merge_count() const { return merge_ranks_.size(); }

private:
    void ensure_id(int32_t id);
    // Merge one already-metaspaced "word" (a run starting at a metaspace) into
    // ids, appending to `out`.
    void merge_word(const std::string& word, int unk_id,
                    std::vector<int32_t>& out) const;

    std::unordered_map<std::string, int32_t> vocab_;    // piece -> id
    std::vector<std::string> id_to_piece_;              // for decode()
    // Key: "left\x01right" (\x01 never appears in a piece) -> rank.
    std::unordered_map<std::string, int32_t> merge_ranks_;

    bool add_prefix_space_ = false;
    bool byte_fallback_    = true;
    bool fuse_unk_         = false;
};

}  // namespace brolm::detail::spm
