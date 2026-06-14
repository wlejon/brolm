#pragma once

// SentencePiece Unigram tokenizer for NLLB-200 (M2M-100 architecture).
//
// Parsed from a HuggingFace `tokenizer.json`: the Unigram `model.vocab`
// supplies the base subword pieces (shared brolm::detail::spm::Unigram core),
// and the top-level `added_tokens` array supplies the special tokens
// (<s>, <pad>, </s>, <unk>, <mask>) and the 200+ FLORES-200 language codes
// (e.g. "eng_Latn", "fra_Latn"). Language codes and specials are decode-only:
// they are inserted explicitly, never segmented out of normal text.
//
// Sequence framing (the post-April-2023 NLLB default, which prefixes the
// SOURCE language for better zero-shot transfer):
//   encoder input : [src_lang]  piece...  </s>
//   decoder start : </s>  [tgt_lang]          (decoder_start_token_id then the
//                                              forced target-language BOS)
// Generation then produces target pieces and stops at </s>.

#include "brolm/detail/spm_unigram.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brolm::nllb {

class Tokenizer {
public:
    // Parse a HF tokenizer.json. Throws std::runtime_error on I/O / parse
    // error.
    static Tokenizer load(const std::string& tokenizer_json_path);

    // Encoder input ids for translating `text` FROM `src_lang`:
    //   [src_lang]  Unigram pieces...  </s>
    std::vector<int32_t> encode_source(std::string_view text,
                                       const std::string& src_lang) const;

    // Unigram pieces only — no language code, no eos. Useful for tests and for
    // callers that build their own framing.
    std::vector<int32_t> tokenize(std::string_view text) const;

    // Decoder prefill for generating INTO `tgt_lang`:
    //   </s>  [tgt_lang]
    // i.e. decoder_start_token_id (= </s>) followed by the forced target-
    // language BOS. Beam search seeds the decoder with this and decodes until
    // </s>.
    std::vector<int32_t> decoder_start(const std::string& tgt_lang) const;

    // Detokenize ids back to text. With `skip_special` (the default, matching
    // HF batch_decode(skip_special_tokens=True)), special tokens and language
    // codes are dropped.
    std::string decode(const std::vector<int32_t>& ids,
                       bool skip_special = true) const;

    // Token id for a FLORES-200 language code ("eng_Latn"). Throws
    // std::runtime_error if the code is unknown.
    int lang_id(const std::string& code) const;
    bool has_lang(const std::string& code) const {
        return lang_ids_.find(code) != lang_ids_.end();
    }
    std::size_t language_count() const { return lang_ids_.size(); }

    int bos_id() const { return bos_id_; }
    int pad_id() const { return pad_id_; }
    int eos_id() const { return eos_id_; }
    int unk_id() const { return unk_id_; }
    std::size_t vocab_count() const { return model_.size(); }

private:
    Tokenizer() = default;

    brolm::detail::spm::Unigram model_;
    // All added tokens (specials + language codes): content -> id.
    std::unordered_map<std::string, int32_t> added_;
    // Subset that are language codes ("eng_Latn", ...): content -> id.
    std::unordered_map<std::string, int32_t> lang_ids_;
    // Ids dropped by decode(skip_special=true): every added token.
    std::unordered_set<int32_t> special_ids_;

    int bos_id_ = 0;   // <s>
    int pad_id_ = 1;   // <pad>
    int eos_id_ = 2;   // </s>  (also decoder_start_token_id)
    int unk_id_ = 3;   // <unk>
};

}  // namespace brolm::nllb
