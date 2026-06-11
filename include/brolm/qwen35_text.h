#pragma once

// Qwen3.5 hybrid text decoder — inference-only, KV-cached.
//
// Faithful port of Hugging Face `Qwen3_5TextModel` (the text backbone consumed
// by `Qwen3_5ForConditionalGeneration`). Forward-only. Runs on whichever
// brotensor backend is resolved at runtime — FP32 on CPU, FP16 on GPU.
//
// Hybrid layer schedule. The text backbone alternates two layer types per a
// fixed `layer_types` schedule (for the 0.8B release: [L,L,L,F] x 6):
//
//   * "full_attention" (F) — standard softmax GQA attention with an output
//                            gate (`attn_output_gate=true`). Implemented here.
//
//   * "linear_attention" (L) — Gated DeltaNet (Mamba-2-family) recurrence.
//                              Stubbed as identity passthrough in this chunk;
//                              the tensors are loaded into `LinearAttnLayer`
//                              but unused. Stage 3b wires the recurrence.
//
// Full-attention block (one F layer, residual stream `h` of width hidden_size):
//
//   residual = h
//   h = rms_norm(h, input_layernorm.weight, eps)
//     qg = q_proj(h)                                  (L, 2 * n_q * head_dim)
//         per head, qg is laid out [q (head_dim), gate (head_dim)] so we
//         split at load time into Wq (n_q*head_dim, hidden) and Wg of the
//         same shape, then run two batched linears.
//     q    = Wq @ h
//     gate = Wg @ h
//     k    = k_proj(h)                                (L, n_kv * head_dim)
//     v    = v_proj(h)                                (L, n_kv * head_dim)
//     q = per_head_rmsnorm(q, q_norm.weight, head_dim)
//     k = per_head_rmsnorm(k, k_norm.weight, head_dim)
//     q = mrope_partial(q, rotary_dim, t,h,w, sections)   partial M-RoPE
//     k = mrope_partial(k, rotary_dim, t,h,w, sections)
//     kv_cache_append(k, v, cache_len, K_cache, V_cache)   n_kv-width cache
//     attn = flash_attention_decode(q, K_cache, V_cache, cache_len + L, n_q, n_kv)
//     attn = attn * sigmoid(gate)            ← attn_output_gate (PRE-o_proj)
//     attn = o_proj(attn)                           (L, hidden)
//   h = residual + attn
//   residual = h
//   h = rms_norm(h, post_attention_layernorm.weight, eps)
//     h_mlp = down_proj( silu(gate_proj(h)) * up_proj(h) )       SwiGLU
//   h = residual + h_mlp
//
// After all layers: h = rms_norm(h, language_model.norm.weight, eps);
//                   logits = embed_tokens.weight @ h    (tied lm_head).
//
// RoPE convention. Qwen3.5 uses partial multi-axis RoPE: only the first
// `rotary_dim = round(head_dim * partial_rotary_factor) = 64` dims of each
// 256-dim head are rotated; the remaining 192 pass through. Within the rotary
// subrange, HF interleaves three per-axis position streams (t, h, w) per
// `mrope_section = [11, 11, 10]` (in PAIRS — 2 * sum = rotary_dim).
//
// brotensor's `rope_apply_mrope` rotates ADJACENT pairs (interleaved
// convention). HF's `apply_rotary_pos_emb` uses rotate_half. We follow the
// qwen.cpp trick: permute the rotated subrange of q_proj/k_proj rows and of
// q_norm/k_norm at load time so that after projection + QK-norm the head_dim
// already lands in interleaved-pair order in dims [0, rotary_dim). The
// pass-through tail (dims [rotary_dim, head_dim)) is left untouched.

#include "brolm/qwen35_config.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brotensor::safetensors { class File; struct TensorView; }

namespace brolm::qwen35 {

// Per-layer KV-cache slot (full_attention layers).
struct FullAttnKVCache {
    brotensor::Tensor k;   // (max_seq, n_q * head_dim) — KV expanded to q heads
    brotensor::Tensor v;
    int len = 0;
};

// Per-layer recurrent state slot (linear_attention layers).
//
//   recurrent : (num_heads, value_head_dim * key_head_dim) FP32
//               Row h holds S_h in row-major v,k order: state[h,v*d_k+k].
//               brotensor::gated_delta_rule_step reads + overwrites this in place.
//
//   conv_state: (1, qkv_channels * (kernel_dim - 1)) FP32
//               Rolling left-context shift register for the depthwise causal
//               conv1d that runs over the concatenated q||k||v stream
//               (qkv_channels == 3 * num_heads * head_dim). Read + overwritten
//               in place by brotensor::causal_conv1d_update.
//
// `initialized` is true once make_cache() has zeroed both buffers. Allocated
// only for linear-attention layer slots; full-attention slots leave it false.
struct LinearAttnState {
    brotensor::Tensor recurrent;
    brotensor::Tensor conv_state;
    bool initialized = false;
};

struct LayerCache {
    FullAttnKVCache full;
    LinearAttnState lin;
};

class TextModel {
public:
    explicit TextModel(const Qwen35Config::Text& cfg);
    ~TextModel();
    TextModel(const TextModel&) = delete;
    TextModel& operator=(const TextModel&) = delete;
    TextModel(TextModel&&) noexcept = default;
    TextModel& operator=(TextModel&&) noexcept = default;

    // Load weights from one or more safetensors shards. Tensor keys follow the
    // HF convention `model.language_model.<...>`. Pass the matching prefix.
    // For the real checkpoint use the default ("model.language_model."). For
    // a synthetic test fixture rooted at the same path, pass "".
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "model.language_model.");
    void load_weights(
        const std::vector<const brotensor::safetensors::File*>& shards,
        const std::string& prefix = "model.language_model.");

    // Allocate per-layer caches sized for `max_seq` tokens. Full-attn layers
    // get K/V; linear-attn slots are left empty in this chunk.
    std::vector<LayerCache> make_cache(int max_seq) const;

    // Run the decoder over `token_ids` (length T) at the per-token M-RoPE
    // positions (mrope_t/h/w each length T). For pure-text decoding pass the
    // same value on all three axes (positions 0..T-1, or cache_len..cache_len+T-1
    // for an offset prefill).
    //   cache: per-layer caches built via make_cache(), updated in place.
    //   logits_out: (T, vocab_size), resized; values at the model's compute
    //               dtype.
    void forward(const std::vector<int>& token_ids,
                 const std::vector<int64_t>& mrope_t,
                 const std::vector<int64_t>& mrope_h,
                 const std::vector<int64_t>& mrope_w,
                 std::vector<LayerCache>& cache,
                 brotensor::Tensor& logits_out);

    // Same forward, but starts from already-embedded inputs. `embeds` is a
    // (T, hidden) tensor at the compute dtype (typically produced by
    // `embed_tokens(...)` and then sliced/overwritten by the VLM glue to
    // splice vision-tower outputs into <|image_pad|> positions). The token-id
    // forward(...) above internally calls this after its own embedding lookup,
    // so `forward(ids,...) == forward_embeds(embed_tokens(ids),...)` exactly.
    void forward_embeds(const brotensor::Tensor& embeds,
                        const std::vector<int64_t>& mrope_t,
                        const std::vector<int64_t>& mrope_h,
                        const std::vector<int64_t>& mrope_w,
                        std::vector<LayerCache>& cache,
                        brotensor::Tensor& logits_out);

    // Embed a sequence of token ids via the tied embedding table, returning a
    // freshly-allocated (T, hidden) tensor at the compute dtype. Exposed so
    // the VLM glue can splice vision-tower embeddings into image-pad slots
    // before running forward_embeds.
    brotensor::Tensor embed_tokens(const std::vector<int>& token_ids) const;

    // Read-only handle to the embedding table (vocab, hidden) on device.
    const brotensor::Tensor& embed_table() const { return embed_; }

    const Qwen35Config::Text& config() const { return cfg_; }

private:
    // Per-layer weight bundle for a full_attention layer.
    struct FullAttnLayer {
        brotensor::Tensor Wq;       // (n_q * head_dim, hidden) — rotary-permuted
        brotensor::Tensor Wg;       // (n_q * head_dim, hidden) — gate half
        brotensor::Tensor Wk;       // (n_kv * head_dim, hidden) — rotary-permuted
        brotensor::Tensor Wv;       // (n_kv * head_dim, hidden)
        brotensor::Tensor Wo;       // (hidden, n_q * head_dim)
        brotensor::Tensor q_norm;   // (head_dim,) — rotary-permuted in [0, rotary_dim)
        brotensor::Tensor k_norm;   // (head_dim,)
    };

    // Per-layer weight bundle for a linear_attention layer. Tensors are loaded
    // so the safetensors check succeeds, but they are unused in Stage 3a.
    struct LinearAttnLayer {
        brotensor::Tensor A_log;
        brotensor::Tensor conv1d;
        brotensor::Tensor dt_bias;
        brotensor::Tensor in_proj_a;
        brotensor::Tensor in_proj_b;
        brotensor::Tensor in_proj_qkv;
        brotensor::Tensor in_proj_z;
        brotensor::Tensor norm;
        brotensor::Tensor out_proj;
    };

    struct MLP {
        brotensor::Tensor gate_W;   // (intermediate, hidden)
        brotensor::Tensor up_W;     // (intermediate, hidden)
        brotensor::Tensor down_W;   // (hidden, intermediate)
    };

    struct LayerSlot {
        LayerType type;
        brotensor::Tensor in_norm;          // (hidden,)
        brotensor::Tensor post_attn_norm;   // (hidden,)
        FullAttnLayer  full;
        LinearAttnLayer lin;
        MLP mlp;
    };

    void load_weights_impl_(
        const std::vector<const brotensor::safetensors::File*>& shards,
        const std::string& prefix);

    // Run the MLP sub-layer in place on h_.
    void mlp_block_(const MLP& mlp, int L);

    // Per-forward M-RoPE setup: upload the three position streams once and
    // make sure the cached per-axis cos/sin tables (members below) cover the
    // call's peak position. Tables only depend on the model's rope geometry,
    // so they are built to a bucketed capacity and reused across forwards —
    // the per-layer / per-token host rebuild they replace dominated decode.
    void prepare_mrope_(const std::vector<int32_t>& pos_t,
                        const std::vector<int32_t>& pos_h,
                        const std::vector<int32_t>& pos_w, int L);

    // Apply partial M-RoPE in place on `qk` viewed as (L, num_heads *
    // head_dim), using the tables + position streams prepare_mrope_ staged.
    // Only dims [0, rotary_dim) per head are rotated; pass-through dims are
    // untouched.
    void apply_partial_mrope_(brotensor::Tensor& qk, int num_heads, int L);

    Qwen35Config::Text cfg_;
    int rotary_dim_ = 0;
    int d_t_ = 0, d_h_ = 0, d_w_ = 0;     // mrope_section in PAIRS

    // Weights.
    brotensor::Tensor embed_;       // (vocab, hidden)
    brotensor::Tensor final_norm_;  // (hidden,)
    std::vector<LayerSlot> layers_;

    // Per-call scratch buffers, kept alive across forwards to avoid realloc.
    brotensor::Tensor ids_dev_;     // (L,1) INT32
    brotensor::Tensor h_;           // (L, hidden) residual stream
    brotensor::Tensor norm_;        // (L, hidden) post-RMSNorm
    brotensor::Tensor q_, k_, v_;   // projected q/k/v
    brotensor::Tensor gate_;        // attn output gate, (L, n_q*head_dim)
    brotensor::Tensor qn_, kn_;     // QK-normed q/k
    brotensor::Tensor q_rot_, k_rot_;     // rotary subrange (L, n_q*rotary_dim) / (L, n_kv*rotary_dim)
    brotensor::Tensor attn_;
    brotensor::Tensor gate_sig_;    // sigmoid(gate)
    brotensor::Tensor proj_;        // o_proj / down_proj output
    brotensor::Tensor mlp_gate_, mlp_up_;   // mlp_gate_ becomes the SwiGLU
                                            // activation in place

    // M-RoPE state staged by prepare_mrope_: per-axis cos/sin tables cached
    // to `mrope_tbl_max_pos_` (inclusive), and the call's device-resident
    // int32 position streams.
    brotensor::Tensor mrope_cos_t_, mrope_sin_t_;
    brotensor::Tensor mrope_cos_h_, mrope_sin_h_;
    brotensor::Tensor mrope_cos_w_, mrope_sin_w_;
    int mrope_tbl_max_pos_ = -1;
    brotensor::Tensor pos_t_dev_, pos_h_dev_, pos_w_dev_;

    // Linear-attention scratch (see qwen35_text.cpp linear_attn_block_).
    brotensor::Tensor lin_qkv_;        // (T, 3*num_heads*head_dim)
    brotensor::Tensor lin_qkv_conv_;   // (T, 3*num_heads*head_dim) after conv + silu
    brotensor::Tensor lin_q_, lin_k_, lin_v_;   // (T, num_heads*head_dim) each
    brotensor::Tensor lin_a_raw_;      // (T, num_heads) FP32
    brotensor::Tensor lin_beta_;       // (T, num_heads) FP32
    brotensor::Tensor lin_z_;          // (T, num_heads*value_head_dim)
    brotensor::Tensor lin_zsilu_;      // silu(z)
    brotensor::Tensor lin_O_;          // (T, num_heads*value_head_dim) recurrence out
    brotensor::Tensor lin_O_norm_;     // per-head RMSNormed O
    brotensor::Tensor lin_log_A_;      // (num_heads, 1) FP32 — cached per layer view
    brotensor::Tensor lin_x_fp32_;     // FP32 cast of norm_ for the linear-attn block
    brotensor::Tensor lin_proj_cast_;  // compute-dtype cast of FP32 out_proj output
};

}  // namespace brolm::qwen35
