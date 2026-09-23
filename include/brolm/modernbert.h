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

// A packed batch: many sequences back to back in T rows, no padding between
// them. Every buffer is an INT32 tensor on the compute device.
//   ids:    (T, 1) token ids.
//   pos:    (T, 1) position of each row inside its own sequence (0-based).
//   bounds: (T, 2) [start, end) row range of each row's sequence.
// Rows only ever attend inside their own sequence, so the result for a
// sequence is the same as encoding it alone.
struct PackedInputs {
    const brotensor::Tensor* ids = nullptr;
    const brotensor::Tensor* pos = nullptr;
    const brotensor::Tensor* bounds = nullptr;
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
    // (Packs the one sequence and runs forward_packed.)
    void forward(const int32_t* input_ids, int seq_len, brotensor::Tensor& h_out);

    // Packed forward over in.ids->rows rows: h_out is (T, hidden_size) at
    // compute_dtype(). Every position must be below reserve_positions()'s
    // count. Issues device work only — no host transfer, no allocation once
    // reserve_rows() covers T — so it can be captured into a CUDA graph.
    void forward_packed(const PackedInputs& in, brotensor::Tensor& h_out);

    // Grow the rotary tables to cover positions [0, n). Returns true when the
    // tables were reallocated (device pointers changed).
    bool reserve_positions(int n);
    // Size every scratch buffer for T rows, so later forwards of up to T rows
    // neither allocate nor move a buffer.
    void reserve_rows(int T);

    // FP32 split-K scratch every linear here passes to
    // linear_forward_batched_ex; callers running their own linears in the same
    // graph may share it. Its device pointer is part of a captured graph:
    // reserve_rows() pre-sizes it so it does not move in practice, and a
    // caller holding graphs re-checks workspace_data() after an eager run.
    brotensor::Tensor& gemm_workspace() { return ws_; }

    // Let FP16 linears accumulate each k16 step in FP16 before folding into
    // FP32 (brotensor kLinearEpiFastAccum): ~2x tensor rate on consumer GPUs.
    void set_fast_accum(bool on) { fast_accum_ = on; }
    bool fast_accum() const { return fast_accum_; }
    const void* workspace_data() const { return ws_.data; }

    // Per-op-family timing (syncs between families; slows the forward).
    void set_profiling(bool on) { profiling_ = on; }
    EncoderTimings& timings() { return timings_; }

private:
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
    brotensor::Tensor single_idx_;  // forward(): ids | pos | bounds of one sequence
    brotensor::Tensor embeds_;
    brotensor::Tensor h_;
    brotensor::Tensor attn_in_;
    brotensor::Tensor qkv_;
    brotensor::Tensor attn_out_;
    brotensor::Tensor mlp_in_;
    brotensor::Tensor geglu_out_;
    brotensor::Tensor ws_;  // split-K partials (FP32)
    bool fast_accum_ = false;
    static constexpr int kWorkspaceFloats = 2 << 20;

    // Y = epilogue(X · Wᵀ) via linear_forward_batched_ex on ws_.
    void linear(const brotensor::Tensor& W, const brotensor::Tensor& X, int epilogue,
                brotensor::Tensor& Y);
};

}  // namespace brolm::modernbert
