#pragma once

// Qwen3-VL text decoder — inference-only, KV-cached.
//
// Faithful port of Hugging Face `Qwen3VLTextModel` (the text backbone
// consumed by `Qwen3VLForConditionalGeneration`). Forward-only. Runs on
// whichever brotensor backend is resolved at runtime — FP32 on CPU, FP16 on
// GPU.
//
// Unlike qwen35::TextModel, EVERY layer is the same plain dense block — no
// hybrid Gated DeltaNet linear-attention layers, no attn_output_gate. It is
// architecturally the same decoder as brolm::qwen::Qwen3Model /
// brolm::detail::DenseDecoder (GQA, per-head QK-norm, SwiGLU, RMSNorm), with
// one difference: full-rotary M-RoPE (three interleaved position streams
// t/h/w spanning the WHOLE head_dim — DenseDecoder only supports a single
// scalar-position 1-D RoPE stream) plus an optional DeepStack injection hook.
// Because of that RoPE mismatch this is its own implementation rather than a
// DenseDecoder specialization — the same call qwen35_text.cpp made for its
// (also M-RoPE) hybrid decoder.
//
// One decoder layer, residual stream `h` of width hidden_size:
//
//   residual = h
//   h = rms_norm(h, input_layernorm.weight, eps)         plain gain, no +1
//     q = q_proj(h)                                  (L, n_q  * head_dim), no bias
//     k = k_proj(h)                                  (L, n_kv * head_dim), no bias
//     v = v_proj(h)                                  (L, n_kv * head_dim), no bias
//     q = per_head_rmsnorm(q, q_norm.weight, head_dim)
//     k = per_head_rmsnorm(k, k_norm.weight, head_dim)
//     q = mrope_full(q, head_dim, t,h,w, mrope_section)   FULL rotation —
//     k = mrope_full(k, head_dim, t,h,w, mrope_section)   no pass-through tail
//     kv_cache_append(k, v, cache_len, K_cache, V_cache)   n_kv-width cache
//     attn = flash_attention_decode(q, K_cache, V_cache, cache_len + L, n_q, n_kv)
//     attn = o_proj(attn)                            (L, hidden), no bias
//   h = residual + attn
//   [DeepStack: if this is one of the first N decoder layers (N ==
//    deepstack feature count for this call), add the corresponding
//    per-image DeepStack feature into h at that image's token rows.]
//   residual = h
//   h = rms_norm(h, post_attention_layernorm.weight, eps)
//     h_mlp = down_proj( silu(gate_proj(h)) * up_proj(h) )       SwiGLU
//   h = residual + h_mlp
//
// After all layers: h = rms_norm(h, language_model.norm.weight, eps);
//                   logits = embed_tokens.weight @ h    (tied lm_head).
//
// RoPE convention: brotensor's rope_apply_mrope rotates ADJACENT pairs
// (interleaved / GPT-J convention). HF's apply_rotary_pos_emb uses
// rotate_half. As in qwen.cpp/qwen35_text.cpp, we permute the rotary-subrange
// rows of q_proj/k_proj (and q_norm/k_norm) at load time so projected q/k
// land in interleaved-pair order. Because Qwen3-VL rotates the WHOLE
// head_dim (no partial_rotary_factor), there is no pass-through tail to
// preserve — every row of every head is permuted.

#include "brolm/qwen3vl_config.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brotensor::safetensors { class File; struct TensorView; }

namespace brolm::qwen3vl {

// Per-layer KV cache at true KV width (n_kv * head_dim) — GQA mapping is
// handled inside flash_attention_decode.
struct LayerCache {
    brotensor::Tensor k, v;
    int len = 0;
};

// One image's DeepStack splice: the row range in the current forward's L-row
// batch where that image's expanded `<|image_pad|>` tokens live, plus one
// feature tensor per decoder layer that should receive it. `per_layer[li]`
// (shape (n_tokens, hidden_size), compute dtype) is added in place into `h`
// at rows [row_start, row_start + n_tokens) right after decoder layer `li`
// finishes — for li in [0, per_layer.size()). Layers beyond per_layer.size()
// are untouched. Only meaningful during prefill (the pass that actually
// contains image tokens); pass an empty vector for decode steps.
struct DeepstackSplice {
    int row_start = 0;
    std::vector<brotensor::Tensor> per_layer;
};

class TextModel {
public:
    explicit TextModel(const Qwen3VLConfig::Text& cfg);
    ~TextModel();

    TextModel(const TextModel&)            = delete;
    TextModel& operator=(const TextModel&) = delete;
    TextModel(TextModel&&) noexcept        = default;
    TextModel& operator=(TextModel&&) noexcept = default;

    // Load every weight. Tensor names follow the HF convention under
    // `prefix` (default "model.language_model."): `embed_tokens.weight`,
    // `norm.weight`, `layers.N.self_attn.{q,k,v,o}_proj.weight`,
    // `layers.N.self_attn.{q,k}_norm.weight`,
    // `layers.N.{input_layernorm,post_attention_layernorm}.weight`,
    // `layers.N.mlp.{gate,up,down}_proj.weight`. Throws std::runtime_error
    // on a missing name or shape mismatch. `tie_word_embeddings=false` (an
    // untied lm_head) is not supported — no released Qwen3-VL checkpoint
    // uses it.
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "model.language_model.");
    void load_weights(
        const std::vector<const brotensor::safetensors::File*>& shards,
        const std::string& prefix = "model.language_model.");

    // Allocate the per-layer K/V cache for sequences up to `max_seq` tokens.
    std::vector<LayerCache> make_cache(int max_seq) const;

    // Roll every layer's cache_len back to `len` (0 <= len <= current len).
    // Throws if out of range.
    void truncate_cache(std::vector<LayerCache>& cache, int len) const;

    // Embedding lookup only: (L, hidden_size) input embeddings for
    // `token_ids` at the compute dtype. Does NOT touch the KV cache.
    brotensor::Tensor embed_tokens(const std::vector<int>& token_ids) const;

    // Append L tokens at M-RoPE positions (mrope_t[i], mrope_h[i],
    // mrope_w[i]), run the decoder, and write `logits_out` := (L,
    // vocab_size). Advances every layer's cache_len by L.
    void forward(const std::vector<int>& token_ids,
                const std::vector<int64_t>& mrope_t,
                const std::vector<int64_t>& mrope_h,
                const std::vector<int64_t>& mrope_w,
                std::vector<LayerCache>& cache,
                brotensor::Tensor& logits_out);

    // Like forward(), but starts from precomputed input embeddings (e.g. the
    // output of embed_tokens with selected rows overwritten by spliced
    // vision-tower embeddings), and optionally injects DeepStack features
    // into the first few decoder layers at each image's token rows. Pass an
    // empty `deepstack` vector when there is nothing to inject (e.g. every
    // single-token decode step).
    void forward_embeds(const brotensor::Tensor& embeds,
                        const std::vector<int64_t>& mrope_t,
                        const std::vector<int64_t>& mrope_h,
                        const std::vector<int64_t>& mrope_w,
                        std::vector<LayerCache>& cache,
                        brotensor::Tensor& logits_out,
                        const std::vector<DeepstackSplice>& deepstack = {});

    const Qwen3VLConfig::Text& config() const { return cfg_; }

private:
    struct LayerSlot {
        brotensor::Tensor in_norm, post_attn_norm;   // (hidden,)
        brotensor::Tensor Wq, Wk, Wv, Wo;
        brotensor::Tensor q_norm, k_norm;             // (head_dim,)
        brotensor::Tensor gate_W, up_W, down_W;       // SwiGLU MLP
    };

    void load_weights_impl_(const std::vector<const brotensor::safetensors::File*>& shards,
                            const std::string& prefix);

    // Stage the M-RoPE position streams + per-axis sin/cos tables for this
    // forward call.
    void prepare_mrope_(const std::vector<int32_t>& pos_t,
                        const std::vector<int32_t>& pos_h,
                        const std::vector<int32_t>& pos_w, int L);
    // Apply full-rotation M-RoPE to a (L, num_heads*head_dim) q or k tensor
    // in place.
    void apply_mrope_(brotensor::Tensor& qk, int num_heads, int L);

    void mlp_block_(const LayerSlot& layer, int L);

    Qwen3VLConfig::Text cfg_;

    int d_t_ = 0, d_h_ = 0, d_w_ = 0;   // mrope_section, in pairs

    // Weights.
    brotensor::Tensor embed_;        // (vocab, hidden)
    std::vector<LayerSlot> layers_;
    brotensor::Tensor final_norm_;   // (hidden,)

    // M-RoPE per-axis tables, rebuilt (grown) lazily as max position grows.
    brotensor::Tensor mrope_cos_t_, mrope_sin_t_;
    brotensor::Tensor mrope_cos_h_, mrope_sin_h_;
    brotensor::Tensor mrope_cos_w_, mrope_sin_w_;
    int mrope_tbl_max_pos_ = -1;
    brotensor::Tensor pos_t_dev_, pos_h_dev_, pos_w_dev_;

    // Per-call scratch.
    brotensor::Tensor h_;                 // residual stream (L, hidden)
    brotensor::Tensor norm_;              // rms_norm output
    brotensor::Tensor q_, k_, v_;         // projected q/k/v
    brotensor::Tensor qn_, kn_;            // QK-normed q/k
    brotensor::Tensor q_rot_, k_rot_;      // rotary-subrange scratch (== full head_dim here)
    brotensor::Tensor attn_;              // flash-attention output
    brotensor::Tensor proj_;              // o_proj / down_proj output
    brotensor::Tensor mlp_gate_, mlp_up_;
};

}  // namespace brolm::qwen3vl
