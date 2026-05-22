#pragma once

// AlignmentAdapter — a small trainable projection + pooling head that
// retargets LLM / encoder hidden states into a diffusion denoiser's
// conditioning.
//
// This is brolm's first trainable module: it has a backward pass and an Adam
// optimizer. The forward path projects a sequence of LLM hidden states into
// two tensors that map directly onto a denoiser's conditioning inputs:
//
//   * text_embeddings (L, d_cond) — the per-token cross-attention context.
//     A diffusion U-Net consumes this as its cross-attention K/V source.
//   * pooled (1, d_cond) — the single pooled conditioning vector. SD-family
//     denoisers add this (after a time-embedding-style projection) to the
//     timestep embedding; SDXL/Flux carry it as the "pooled prompt" input.
//
// AlignmentAdapter deliberately does NOT name brodiffusion::Conditioning —
// brodiffusion depends on brolm, so referencing that type here would be a
// circular dependency. It outputs plain brotensor tensors; the app layer
// (diffusion-lab) assembles the actual Conditioning object from them.
//
// Architecture, given LLM hidden states H (L, d_in):
//   A_pre           = H @ W1^T + b1            (L, d_model)
//   A               = silu(A_pre)              (L, d_model)
//   text_embeddings = A @ W_text^T + b_text    (L, d_cond)
//   p               = mean_pool_rows(A)        (d_model, 1)
//   pooled          = p^T @ W_pool^T + b_pool  (1, d_cond)
//
// All parameters and activations are allocated at brolm::compute_dtype()
// (FP32 on the CPU backend, FP16 on a GPU backend).

#include "brotensor/tensor.h"

#include <cstdint>
#include <string>

namespace brotensor::safetensors { class File; }

namespace brolm {

struct AlignmentAdapterConfig {
    int d_in    = 1024;   // LLM / encoder hidden size (input)
    int d_model = 512;    // adapter inner width
    int d_cond  = 768;    // diffusion conditioning width (e.g. SD1.5 cross-attn dim)
};

class AlignmentAdapter {
public:
    // Construct with all six parameters initialised: weights via xavier-uniform
    // (deterministic from `init_seed` — two adapters built with the same seed
    // hold byte-identical weights), biases zero. Adam m/v state is zero-init
    // and the step counter starts at 0.
    explicit AlignmentAdapter(const AlignmentAdapterConfig& cfg,
                              uint64_t init_seed = 0);
    ~AlignmentAdapter();

    // Non-copyable; movable.
    AlignmentAdapter(const AlignmentAdapter&) = delete;
    AlignmentAdapter& operator=(const AlignmentAdapter&) = delete;
    AlignmentAdapter(AlignmentAdapter&&) noexcept;
    AlignmentAdapter& operator=(AlignmentAdapter&&) noexcept;

    // Forward pass over a length-L sequence of LLM hidden states.
    //   H:               (L, d_in) at the compute dtype.
    //   text_embeddings: (L, d_cond) output, resized as needed.
    //   pooled:          (1, d_cond) output, resized as needed.
    // Caches H, A_pre, A and p internally for the backward pass.
    // brotensor::init() must have run; the caller syncs before reading output.
    void forward(const brotensor::Tensor& H,
                 brotensor::Tensor& text_embeddings,
                 brotensor::Tensor& pooled);

    // Backward pass. Must follow a forward() with matching shapes.
    //   d_text_embeddings: (L, d_cond) upstream gradient w.r.t. text_embeddings.
    //   d_pooled:          (1, d_cond) upstream gradient w.r.t. pooled.
    //   dH_out:            optional (L, d_in) gradient w.r.t. the input H.
    //                      Pass null when the LLM is frozen (the usual case).
    // Parameter gradients ACCUMULATE into internal grad buffers — call
    // zero_grads() before the forward/backward of each training step.
    void backward(const brotensor::Tensor& d_text_embeddings,
                  const brotensor::Tensor& d_pooled,
                  brotensor::Tensor* dH_out = nullptr);

    // Zero every internal parameter-gradient buffer. Call once per step,
    // before forward().
    void zero_grads();

    // Apply one Adam optimiser step to all six parameters from the currently
    // accumulated gradients. Increments the internal 1-based step counter.
    void step(float lr = 1e-3f, float beta1 = 0.9f,
              float beta2 = 0.999f, float eps = 1e-8f);

    // Persistence. save_weights writes the six parameters to a safetensors
    // file (tensor names: W1, b1, W_text, b_text, W_pool, b_pool — optionally
    // prefixed). load_weights reads them back. Together they let tensor-lab
    // save a finetuned adapter and diffusion-lab load it.
    void save_weights(const std::string& path) const;
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "");

    const AlignmentAdapterConfig& config() const { return cfg_; }

private:
    AlignmentAdapterConfig cfg_;

    // ── Parameters (each at compute_dtype()) ──
    brotensor::Tensor W1_;       // (d_model, d_in)
    brotensor::Tensor b1_;       // (d_model, 1)
    brotensor::Tensor W_text_;   // (d_cond, d_model)
    brotensor::Tensor b_text_;   // (d_cond, 1)
    brotensor::Tensor W_pool_;   // (d_cond, d_model)
    brotensor::Tensor b_pool_;   // (d_cond, 1)

    // ── Gradient buffers (accumulate; cleared by zero_grads) ──
    brotensor::Tensor dW1_, db1_;
    brotensor::Tensor dW_text_, db_text_;
    brotensor::Tensor dW_pool_, db_pool_;

    // ── Adam state per parameter (m, v) + 1-based step counter ──
    brotensor::Tensor mW1_, vW1_, mb1_, vb1_;
    brotensor::Tensor mW_text_, vW_text_, mb_text_, vb_text_;
    brotensor::Tensor mW_pool_, vW_pool_, mb_pool_, vb_pool_;
    int adam_step_ = 0;

    // ── Forward caches (needed by backward) ──
    brotensor::Tensor H_;        // (L, d_in)      input
    brotensor::Tensor A_pre_;    // (L, d_model)   pre-activation
    brotensor::Tensor A_;        // (L, d_model)   silu(A_pre)
    brotensor::Tensor p_;        // (d_model, 1)   mean-pooled rows of A
    int seq_len_ = 0;

    // ── Per-call scratch (kept alive to avoid realloc) ──
    brotensor::Tensor dA_;         // (L, d_model)  summed dA contribution
    brotensor::Tensor dA_text_;    // (L, d_model)  text-head contribution to dA
    brotensor::Tensor dA_pool_;    // (L, d_model)  pool-head contribution to dA
    brotensor::Tensor dA_pre_;     // (L, d_model)  through silu
    brotensor::Tensor dp_;         // (d_model, 1)  grad w.r.t. pooled-row vector
    brotensor::Tensor pooled_in_;  // (1, d_model)  p^T staged as a 1-row matrix
    brotensor::Tensor d_pooled_in_;// (1, d_model)  grad of pooled_in_
    brotensor::Tensor dummy_dx_;   // scratch dX target when dH not requested
};

}  // namespace brolm
