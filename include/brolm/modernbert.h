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

// Bidirectional multi-head self-attention over pre-projected (L, H*hd) Q/K/V:
// the fused WMMA flash kernel on FP16/BF16, the generic GQA kernel on FP32.
// Shared by the encoder's global layers and Laya's head layers.
void full_attention(const brotensor::Tensor& q, const brotensor::Tensor& k,
                    const brotensor::Tensor& v, int num_heads, brotensor::Tensor& out);

// Encoder wall time split by op family, accumulated over forward() calls
// while profiling is on (each op family is bracketed by a device sync).
struct EncoderTimings {
    double embed_ms = 0;      // id upload + embedding lookup + embedding norm
    double norm_ms = 0;       // attn_norm / mlp_norm / final_norm
    double qkv_ms = 0;        // Wqkv projection + q/k/v split
    double rope_ms = 0;
    double attn_full_ms = 0;
    double attn_local_ms = 0;
    double wo_ms = 0;
    double mlp_in_ms = 0;     // Wi projection
    double geglu_ms = 0;
    double mlp_out_ms = 0;    // Wo projection
    double residual_ms = 0;
    int forwards = 0;
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

    // Per-op-family timing (syncs between families; slows the forward).
    void set_profiling(bool on) { profiling_ = on; }
    EncoderTimings& timings() { return timings_; }

private:
    // Grow the cached cos/sin tables (full + sliding theta) to cover
    // `seq_len` positions. Built once on the host and uploaded; forward()
    // views their first seq_len rows, so a call uploads nothing.
    void ensure_rotary_tables_(int seq_len);

    Config cfg_;
    bool profiling_ = false;
    EncoderTimings timings_;

    int rope_rows_ = 0;
    brotensor::Tensor cos_full_, sin_full_, cos_slid_, sin_slid_;

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
