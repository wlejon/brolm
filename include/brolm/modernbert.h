#pragma once

#include "brolm/modernbert_config.h"
#include "brolm/detail/weights.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brolm::modernbert {

struct LayerWeights {
    brotensor::Tensor attn_norm;  // empty for layer 0
    brotensor::Tensor Wqkv;       // (3 * hidden_size, hidden_size)
    brotensor::Tensor Wo;         // (hidden_size, hidden_size)
    brotensor::Tensor mlp_norm;   // (hidden_size, 1)
    brotensor::Tensor mlp_Wi;     // (2 * intermediate_size, hidden_size)
    brotensor::Tensor mlp_Wo;     // (hidden_size, intermediate_size)
};

class ModernBertModel {
public:
    explicit ModernBertModel(Config cfg = Config{});
    ~ModernBertModel();

    ModernBertModel(const ModernBertModel&) = delete;
    ModernBertModel& operator=(const ModernBertModel&) = delete;
    ModernBertModel(ModernBertModel&&) noexcept = default;
    ModernBertModel& operator=(ModernBertModel&&) noexcept = default;

    const Config& config() const { return cfg_; }

    void load_weights(const brolm::detail::weights::Source& src,
                      const std::string& prefix = "encoder.");

    void init_synthetic();

    // Forward pass over a sequence of token IDs:
    //   input_ids: array of length seq_len
    //   h_out: (seq_len, hidden_size) at compute_dtype()
    void forward(const int32_t* input_ids, int seq_len, brotensor::Tensor& h_out);

private:
    void build_rotary_tables_(int seq_len, float theta,
                              brotensor::Tensor& cos_out,
                              brotensor::Tensor& sin_out);

    Config cfg_;

    brotensor::Tensor tok_embeddings_;  // (vocab_size, hidden_size)
    brotensor::Tensor embed_norm_;      // (hidden_size, 1)
    std::vector<LayerWeights> layers_;
    brotensor::Tensor final_norm_;      // (hidden_size, 1)

    // Intermediate scratch buffers
    brotensor::Tensor ids_dev_;
    brotensor::Tensor embeds_;
    brotensor::Tensor h_;
    brotensor::Tensor attn_in_;
    brotensor::Tensor qkv_;
    brotensor::Tensor q_;
    brotensor::Tensor k_;
    brotensor::Tensor v_;
    brotensor::Tensor q_rope_;
    brotensor::Tensor k_rope_;
    brotensor::Tensor attn_out_;
    brotensor::Tensor proj_out_;
    brotensor::Tensor mlp_in_;
    brotensor::Tensor geglu_in_;
    brotensor::Tensor geglu_out_;
    brotensor::Tensor mlp_out_;
};

}  // namespace brolm::modernbert
