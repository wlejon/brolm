#pragma once

#include "brolm/laya_config.h"
#include "brolm/laya_tokenizer.h"
#include "brolm/modernbert.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace brolm::laya {

struct LayaQuestion {
    std::string id;
    std::string type;  // "choice", "score", "noul"
    std::string instructions;

    // For choice: ordered list of (key, description)
    std::vector<std::pair<std::string, std::string>> criteria_choice;

    // For score: ordered list of criteria descriptions
    std::vector<std::string> criteria_score;

    // For noul: optional criteria descriptions for false and true
    std::string criteria_noul_false;
    std::string criteria_noul_true;

    int qtype_index() const {
        if (type == "choice") return 0;
        if (type == "score") return 1;
        return 2;
    }
};

struct LayaAnswer {
    std::string type;  // "choice", "score", "noul"
    std::string choice;
    float score = 0.0f;
    float noul = 0.0f;
    float confidence = 0.0f;
    std::vector<std::pair<std::string, float>> probabilities;
    float act_probability = 0.0f;
};

struct LayaResult {
    std::string model = "rl-agent";
    std::unordered_map<std::string, LayaAnswer> answers;
    int input_tokens = 0;
};

struct TransformerHeadLayer {
    brotensor::Tensor norm1_g;
    brotensor::Tensor norm1_b;
    brotensor::Tensor in_proj_W;
    brotensor::Tensor in_proj_b;
    brotensor::Tensor out_proj_W;
    brotensor::Tensor out_proj_b;
    brotensor::Tensor norm2_g;
    brotensor::Tensor norm2_b;
    brotensor::Tensor linear1_W;
    brotensor::Tensor linear1_b;
    brotensor::Tensor linear2_W;
    brotensor::Tensor linear2_b;
};

class DecisionModel {
public:
    DecisionModel();
    ~DecisionModel();

    DecisionModel(const DecisionModel&) = delete;
    DecisionModel& operator=(const DecisionModel&) = delete;
    DecisionModel(DecisionModel&&) noexcept = default;
    DecisionModel& operator=(DecisionModel&&) noexcept = default;

    void load_model(const std::string& model_dir);
    void load_safetensors(const std::string& safetensors_path);

    void init_synthetic(const modernbert::Config& enc_cfg = modernbert::Config{},
                        const Config& laya_cfg = Config{});

    LayaAnswer forward_question(const LayaQuestion& q,
                                const std::vector<int32_t>& input_ids,
                                const std::vector<int32_t>& marker_pos);

    LayaResult predict(const std::string& state_json_or_text,
                       const std::vector<LayaQuestion>& questions);

    const modernbert::ModernBertModel& encoder() const { return encoder_; }
    const Config& config() const { return cfg_; }
    const LayaTokenizer& tokenizer() const { return tokenizer_; }

private:
    Config cfg_;
    modernbert::ModernBertModel encoder_;
    LayaTokenizer tokenizer_;

    brotensor::Tensor type_emb_;  // (3, D)
    std::vector<TransformerHeadLayer> head_layers_;

    // Scorer
    brotensor::Tensor scorer_ln_g_;
    brotensor::Tensor scorer_ln_b_;
    brotensor::Tensor scorer_l1_W_;
    brotensor::Tensor scorer_l1_b_;
    brotensor::Tensor scorer_l2_W_;
    brotensor::Tensor scorer_l2_b_;

    // Act Head
    brotensor::Tensor act_l1_W_;
    brotensor::Tensor act_l1_b_;
    brotensor::Tensor act_l2_W_;
    brotensor::Tensor act_l2_b_;

    // Scratch tensors
    brotensor::Tensor h_;
    brotensor::Tensor h_norm_;
    brotensor::Tensor qkv_;
    brotensor::Tensor q_;
    brotensor::Tensor k_;
    brotensor::Tensor v_;
    brotensor::Tensor attn_out_;
    brotensor::Tensor proj_out_;
    brotensor::Tensor ffn1_;
    brotensor::Tensor ffn2_;
    brotensor::Tensor m_;
    brotensor::Tensor scorer_out0_;
    brotensor::Tensor scorer_out1_;
    brotensor::Tensor scorer_out2_;
    brotensor::Tensor scorer_out3_;
    brotensor::Tensor act_h1_;
    brotensor::Tensor act_h2_;
    brotensor::Tensor act_logits_;
};

}  // namespace brolm::laya
