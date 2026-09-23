#pragma once

#include "brolm/detail/byte_level_bpe.h"

#include <cstdint>
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

class LayaTokenizer {
public:
    static constexpr int32_t kClsTokenId  = 50281;
    static constexpr int32_t kSepTokenId  = 50282;
    static constexpr int32_t kPadTokenId  = 50283;
    static constexpr int32_t kMaskTokenId = 50284;
    static constexpr int32_t kUnkTokenId  = 50280;

    LayaTokenizer() = default;

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

    static std::vector<std::string> render_options(const LayaQuestion& q);

    int32_t cls_token_id() const { return kClsTokenId; }
    int32_t sep_token_id() const { return kSepTokenId; }
    int32_t pad_token_id() const { return kPadTokenId; }
    int32_t mask_token_id() const { return kMaskTokenId; }

private:
    std::unordered_map<std::string, int32_t> vocab_;
    std::unordered_map<std::string, int32_t> merge_ranks_;
    std::string byte_to_unicode_[256];
    brolm::detail::bpe::SpecialTokens specials_;
    bool normalize_nfc_ = true;
};

}  // namespace brolm::laya
