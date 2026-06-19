#pragma once

// SentencePiece-BPE tokenizer for Gemma / Gemma-2.
//
// Parsed from a HuggingFace `tokenizer.json` (the fast tokenizer's BPE model):
// `model.vocab` is a piece->id map, `model.merges` is the ranked merge list,
// and `model.byte_fallback` is true. Tokenization is SentencePiece-BPE over a
// metaspace pre-tokenization (every ASCII space -> U+2581, '▁'), with byte-
// fallback: a character missing from the vocab decomposes into "<0xNN>" byte-
// pieces. The shared `brolm::detail::spm::Bpe` core supplies the merge loop and
// the metaspace/byte-fallback decode join; this class adds the tokenizer.json
// parsing, the special-token table, and the sequence framing.
//
// Gemma's fast tokenizer documents "ByteFallback and no prefix space" — unlike
// T5 / NLLB it does NOT prepend a leading metaspace, so "Hello" tokenizes as
// the piece "Hello", not "▁Hello". (The core supports add_prefix_space for
// future Llama-family SPM-BPE models; the Gemma wrapper leaves it off.)
//
// Sequence framing: add_bos_token defaults to true (prepend <bos>, id 2) and
// add_eos_token defaults to false (<eos> is id 1). Default special ids:
// <pad>=0, <eos>=1, <bos>=2, <unk>=3 — read from the file's added_tokens when
// present. Added / control tokens (<bos>, <eos>, <start_of_turn>, ...) are
// matched verbatim in the input before BPE and emitted as their single id.

#include "brolm/detail/byte_level_bpe.h"  // bpe::SpecialTokens
#include "brolm/detail/spm_bpe.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace brolm::gemma {

class Tokenizer {
public:
    // Parse a HF tokenizer.json. Throws std::runtime_error on I/O / parse
    // error.
    static Tokenizer load(const std::string& tokenizer_json_path);

    // Encode `text` to int32 ids: optional <bos>, the BPE pieces (with verbatim
    // special-token matching), optional <eos>. Defaults match GemmaTokenizer:
    // add_bos=true, add_eos=false.
    std::vector<int32_t> encode(std::string_view text,
                                bool add_bos = true,
                                bool add_eos = false) const;

    // BPE pieces only — no <bos>/<eos>, but special tokens present in the text
    // are still matched verbatim.
    std::vector<int32_t> tokenize(std::string_view text) const;

    // Detokenize ids back to text (the inverse of tokenize): metaspace U+2581
    // -> space and "<0xNN>" byte-fallback pieces -> their raw byte. Special /
    // added tokens render as their literal piece string; ids outside the vocab
    // are skipped.
    std::string decode(const std::vector<int32_t>& ids) const;

    std::size_t vocab_count() const { return model_.size(); }
    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }
    int pad_id() const { return pad_id_; }
    int unk_id() const { return unk_id_; }

private:
    Tokenizer() = default;

    brolm::detail::spm::Bpe model_;
    brolm::detail::bpe::SpecialTokens specials_;

    int pad_id_ = 0;
    int eos_id_ = 1;
    int bos_id_ = 2;
    int unk_id_ = 3;
};

}  // namespace brolm::gemma
