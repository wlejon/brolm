#pragma once

#include "brolm/modernbert_config.h"
#include "brolm/detail/weights.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace brolm::laya {
class LayaGrad;
}

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
//   soft, soft_idx (optional, both or neither): soft-token input. soft is
//           (S, hidden_size) at the compute dtype; row s REPLACES the token
//           embedding of row soft_idx[s] (INT32 (S, 1)) before the embedding
//           norm, so a projected non-text signal (audio frames) enters the
//           encoder where a token would. The ids at those rows are ignored.
//           Duplicate soft_idx entries must carry identical rows.
struct PackedInputs {
    const brotensor::Tensor* ids = nullptr;
    const brotensor::Tensor* pos = nullptr;
    const brotensor::Tensor* bounds = nullptr;
    const brotensor::Tensor* soft = nullptr;
    const brotensor::Tensor* soft_idx = nullptr;
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

    // (vocab_size, hidden_size) token embedding table at the compute dtype —
    // the space soft-token inputs (PackedInputs::soft) live in.
    const brotensor::Tensor& token_embeddings() const { return tok_embeddings_; }

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

    // Every activation buffer a forward writes, plus the split-K workspace.
    // A captured CUDA graph holds these pointers, so a caller that keeps
    // graphs across a regrow gives the encoder a FRESH scratch set to grow
    // (set_scratch) and keeps the old one alive for the old graphs, instead
    // of reallocating the one they captured.
    struct Scratch {
        brotensor::Tensor single_idx;  // forward(): ids | pos | bounds of one sequence
        brotensor::Tensor embeds, h, attn_in, qkv, attn_out, mlp_in, geglu_out;
        brotensor::Tensor ws;  // split-K partials (FP32)
    };
    const std::shared_ptr<Scratch>& scratch() const { return s_; }
    void set_scratch(std::shared_ptr<Scratch> s) { s_ = s ? std::move(s) : std::make_shared<Scratch>(); }

    // FP32 split-K scratch every linear here passes to
    // linear_forward_batched_ex; callers running their own linears in the same
    // graph may share it. Its device pointer is part of a captured graph:
    // reserve_rows() pre-sizes it so it does not move in practice, and a
    // caller holding graphs re-checks workspace_data() after an eager run.
    brotensor::Tensor& gemm_workspace() { return s_->ws; }

    // Let FP16 linears accumulate each k16 step in FP16 before folding into
    // FP32 (brotensor kLinearEpiFastAccum): ~2x tensor rate on consumer GPUs.
    void set_fast_accum(bool on) { fast_accum_ = on; }
    bool fast_accum() const { return fast_accum_; }
    const void* workspace_data() const { return s_->ws.data; }

    // Per-op-family timing (syncs between families; slows the forward).
    void set_profiling(bool on) { profiling_ = on; }
    EncoderTimings& timings() { return timings_; }

private:
    // The training-side forward/backward (laya_grad.cpp) reads the weights
    // and rotary tables directly.
    friend class ::brolm::laya::LayaGrad;

    Config cfg_;
    bool profiling_ = false;
    EncoderTimings timings_;

    int rope_rows_ = 0;
    brotensor::Tensor cos_full_, sin_full_, cos_slid_, sin_slid_;

    brotensor::Tensor tok_embeddings_;  // (vocab_size, hidden_size)
    brotensor::Tensor embed_norm_;      // (hidden_size, 1)
    std::vector<LayerWeights> layers_;
    brotensor::Tensor final_norm_;      // (hidden_size, 1)

    std::shared_ptr<Scratch> s_ = std::make_shared<Scratch>();
    bool fast_accum_ = false;
    static constexpr int kWorkspaceFloats = 2 << 20;

    // Y = epilogue(X · Wᵀ) via linear_forward_batched_ex on ws_.
    void linear(const brotensor::Tensor& W, const brotensor::Tensor& X, int epilogue,
                brotensor::Tensor& Y);
};

}  // namespace brolm::modernbert
