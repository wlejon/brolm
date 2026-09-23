#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace brolm::laya {

struct LayaQuestion;

struct SequenceResult {
    std::vector<int32_t> input_ids;
    std::vector<int32_t> marker_pos;
};

// Re-space compact JSON (JSON.stringify output) the way Python's json.dumps
// separates items and keys (", " and ": "), which is how the reference
// serialises a dict/list state before tokenizing. String contents are left
// untouched. Number formatting is not reconciled (JS has no int/float split).
std::string python_json_spacing(std::string_view json);

// The tokenizer of a Laya checkpoint, driven by its tokenizer.json (and the
// tokenizer_config.json beside it for the special-token roles), as HF
// `tokenizers` runs it with add_special_tokens=False:
//   1. added tokens are split out of the raw text first (leftmost-longest);
//   2. each remaining span is normalized (NFC, and/or Replace) and
//      pre-tokenized — ByteLevel (GPT-2 regex, byte-to-unicode map: the
//      ModernBERT checkpoints) or Metaspace (' ' -> U+2581, prepend scheme,
//      split before each metaspace: the Gemma-vocabulary mmBERT checkpoint);
//   3. each piece runs the BPE model (byte fallback / fused unk as declared).
// Anything else in the file (another normalizer, pre-tokenizer or model
// type) is rejected at load rather than tokenized approximately.
class LayaTokenizer {
public:
    enum class Kind { ByteLevel, Metaspace };

    LayaTokenizer();

    // `tokenizer_json_path` is <checkpoint>/tokenizer/tokenizer.json; the
    // special-token roles (cls / sep / pad / mask / unk) come from the
    // tokenizer_config.json beside it, falling back to ModernBERT's [CLS] /
    // [SEP] / [PAD] / [MASK] / [UNK] when that file is absent.
    static LayaTokenizer load(const std::string& tokenizer_json_path);

    std::vector<int32_t> encode(std::string_view text) const;

    // [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ...
    // [SEP] state [SEP], exactly as the reference build_sequence. The state
    // is cut from the right (keep the oldest tokens) unless truncate_left,
    // which keeps the newest — how the reference trained multi-turn
    // conversations. marker_pos holds only markers that landed inside
    // max_len; fewer markers than options means the options did not fit.
    SequenceResult build_sequence(const std::string& state_json_or_text,
                                  const LayaQuestion& q,
                                  int max_len = 512,
                                  int head_max_len = 192,
                                  bool truncate_left = false) const;

    // The same, over a state already run through encode_state() — lets one
    // predict() tokenize the state once for all of its questions.
    SequenceResult build_sequence_ids(const std::vector<int32_t>& state_ids,
                                      const LayaQuestion& q,
                                      int max_len = 512,
                                      int head_max_len = 192,
                                      bool truncate_left = false) const;

    // State text -> ids as build_sequence tokenizes it (the mask token's
    // text blanked out, as the reference does).
    std::vector<int32_t> encode_state(const std::string& state_json_or_text) const;

    static std::vector<std::string> render_options(const LayaQuestion& q);

    int32_t cls_token_id() const { return cls_id_; }
    int32_t sep_token_id() const { return sep_id_; }
    int32_t pad_token_id() const { return pad_id_; }
    int32_t mask_token_id() const { return mask_id_; }
    int32_t unk_token_id() const { return unk_id_; }
    // The mask token's text ("[MASK]", "<mask>"): blanked out of every
    // instruction, option and state before tokenizing.
    const std::string& mask_token() const { return mask_token_; }
    Kind kind() const { return kind_; }
    const char* kind_name() const { return kind_ == Kind::ByteLevel ? "byte-level-bpe" : "metaspace-bpe"; }
    std::size_t vocab_size() const;

private:
    struct Model;
    std::shared_ptr<const Model> model_;  // shared by copies (read-only after load)

    Kind kind_ = Kind::ByteLevel;
    int32_t cls_id_ = 50281, sep_id_ = 50282, pad_id_ = 50283, mask_id_ = 50284, unk_id_ = 50280;
    std::string mask_token_ = "[MASK]";

    void encode_span_(std::string_view span, bool at_start, std::vector<int32_t>& out) const;

    // Pre-token -> BPE ids memo (the merge loop dominates encode(); real text
    // repeats words heavily). Thread-safe; shared by copies of a tokenizer,
    // which share its vocabulary. Cleared wholesale when it reaches its cap.
    struct PieceCache;
    std::shared_ptr<PieceCache> cache_;
    void encode_piece_cached_(std::string_view piece, std::vector<int32_t>& out) const;
};

}  // namespace brolm::laya
