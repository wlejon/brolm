#include "brolm/gemma2.h"

#include "brolm/detail/compute.h"
#include "brolm/detail/device.h"
#include "brolm/detail/weights.h"

#include "brotensor/gguf.h"
#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace brolm::gemma {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("gemma::Gemma2Model: " + msg);
}

// Build a device-resident INT32 buffer holding `n` token ids. brotensor has no
// from_host path for INT32, so stage on the host then migrate to the default
// device.
bt::Tensor make_idx_device(const int32_t* host, int n) {
    bt::Tensor cpu = bt::Tensor::empty_on(bt::Device::CPU, n, 1, bt::Dtype::INT32);
    std::memcpy(cpu.host_raw_mut(), host,
                static_cast<std::size_t>(n) * sizeof(int32_t));
    return cpu.to(bt::default_device());
}

bool starts_with(std::string_view s, std::string_view p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

}  // namespace

// ─── HF → ggml tensor-name map (Gemma-2) ───────────────────────────────────
//
// HF naming:  model.embed_tokens.weight,
//             model.layers.N.{self_attn.{q,k,v,o}_proj, input_layernorm,
//                             post_attention_layernorm, pre_feedforward_layernorm,
//                             post_feedforward_layernorm,
//                             mlp.{gate,up,down}_proj}.weight,
//             model.norm.weight, lm_head.weight.
// ggml naming (llama.cpp Gemma-2 convention, verified against
// gguf-py tensor_mapping.py / constants.py MODEL_ARCH.GEMMA2):
//             token_embd.weight,
//             blk.N.{attn_q, attn_k, attn_v, attn_output, attn_norm,
//                    post_attention_norm, ffn_norm, post_ffw_norm,
//                    ffn_gate, ffn_up, ffn_down}.weight,
//             output_norm.weight, output.weight (absent when tied).
//
// Note the two feed-forward norms: HF pre_feedforward_layernorm maps to ggml
// `ffn_norm` (FFN_PRE_NORM), and HF post_feedforward_layernorm maps to ggml
// `post_ffw_norm` (FFN_POST_NORM).

std::string gemma2_hf_to_ggml(std::string_view hf_name) {
    if (hf_name == "model.embed_tokens.weight") return "token_embd.weight";
    if (hf_name == "model.norm.weight")         return "output_norm.weight";
    if (hf_name == "lm_head.weight")            return "output.weight";

    constexpr std::string_view kLayersPrefix = "model.layers.";
    if (!starts_with(hf_name, kLayersPrefix)) return {};
    const auto dot = hf_name.find('.', kLayersPrefix.size());
    if (dot == std::string_view::npos) return {};
    const std::string_view idx = hf_name.substr(
        kLayersPrefix.size(), dot - kLayersPrefix.size());
    const std::string_view tail = hf_name.substr(dot + 1);

    auto blk = [&](std::string_view suffix) -> std::string {
        std::string out;
        out.reserve(4 + idx.size() + 1 + suffix.size());
        out.append("blk.");
        out.append(idx);
        out.push_back('.');
        out.append(suffix);
        return out;
    };

    if (tail == "self_attn.q_proj.weight")            return blk("attn_q.weight");
    if (tail == "self_attn.k_proj.weight")            return blk("attn_k.weight");
    if (tail == "self_attn.v_proj.weight")            return blk("attn_v.weight");
    if (tail == "self_attn.o_proj.weight")            return blk("attn_output.weight");
    if (tail == "input_layernorm.weight")             return blk("attn_norm.weight");
    if (tail == "post_attention_layernorm.weight")    return blk("post_attention_norm.weight");
    if (tail == "pre_feedforward_layernorm.weight")   return blk("ffn_norm.weight");
    if (tail == "post_feedforward_layernorm.weight")  return blk("post_ffw_norm.weight");
    if (tail == "mlp.gate_proj.weight")               return blk("ffn_gate.weight");
    if (tail == "mlp.up_proj.weight")                 return blk("ffn_up.weight");
    if (tail == "mlp.down_proj.weight")               return blk("ffn_down.weight");
    return {};
}

// ─── ctor ────────────────────────────────────────────────────────────────────

Gemma2Model::Gemma2Model(const Gemma2Config& cfg) : cfg_(cfg) {
    cfg_.validate();
    layers_.resize(static_cast<std::size_t>(cfg_.num_hidden_layers));
}

// ─── load_weights ──────────────────────────────────────────────────────────

void Gemma2Model::load_weights(const st::File& f, const std::string& prefix) {
    const std::vector<const st::File*> shards = {&f};
    load_weights(shards, prefix);
}

void Gemma2Model::load_weights(const std::vector<const st::File*>& shards,
                               const std::string& prefix) {
    if (shards.empty()) fail("load_weights: no safetensors shards");
    brolm::detail::weights::SafetensorsSource src(shards, prefix);
    // Safetensors stores raw norm weights; fold +1 at load.
    load_weights_(src, /*norms_prefolded=*/false);
}

void Gemma2Model::load_weights(const brotensor::gguf::File& f) {
    const std::vector<const brotensor::gguf::File*> shards = {&f};
    load_weights(shards);
}

void Gemma2Model::load_weights(
    const std::vector<const brotensor::gguf::File*>& shards) {
    if (shards.empty()) fail("load_weights: no gguf shards");
    brolm::detail::weights::GgufSource src(
        shards, [](std::string_view hf) { return gemma2_hf_to_ggml(hf); });
    // llama.cpp's Gemma-2 converter already baked the +1 into every norm.weight,
    // so the gguf path must NOT fold +1 again.
    load_weights_(src, /*norms_prefolded=*/true);
}

void Gemma2Model::load_weights_(const brolm::detail::weights::Source& src,
                                bool norms_prefolded) {
    const int V    = cfg_.vocab_size;
    const int H    = cfg_.hidden_size;
    const int F    = cfg_.intermediate_size;
    const int HD   = cfg_.head_dim;
    const int n_q  = cfg_.num_attention_heads;
    const int n_kv = cfg_.num_key_value_heads;
    const int q_dim  = n_q  * HD;
    const int kv_dim = n_kv * HD;

    // Gemma-2 RMSNorm computes `_norm(x) * (1 + weight)`, so the gain handed to
    // rms_norm_forward (which does a plain `x * gamma`) must be `1 + weight`.
    //   - safetensors (norms_prefolded == false): HF stores the raw `weight`.
    //     Read it to an FP32 host buffer (exact on FP32/CPU; the FP16/GPU path
    //     reads back the FP16 the kernel would use anyway), add 1.0f, re-upload.
    //   - gguf (norms_prefolded == true): llama.cpp's Gemma2 converter already
    //     baked the +1 in (modify_tensors: name.endswith("norm.weight") -> +1),
    //     so load AS-IS. Folding +1 again would double-shift every norm and
    //     silently corrupt all outputs.
    auto load_norm = [&](const std::string& name, bt::Tensor& dst,
                         const std::string& label) {
        if (norms_prefolded) {
            src.upload_compute_dequant(name, H, 1, dst, label);
            return;
        }
        bt::Tensor tmp;
        src.upload_compute_dequant(name, H, 1, tmp, label);
        std::vector<float> host =
            brolm::detail::weights::detail_::download_fp32(tmp);
        for (float& x : host) x += 1.0f;
        dst = brolm::detail::upload_host(host.data(), H, 1);
    };

    // HF `Gemma2ForCausalLM` checkpoints wrap the decoder under a "model."
    // prefix (model.embed_tokens.weight, model.layers.N..., model.norm.weight).
    // The bare `Gemma2Model` (e.g. Sana's Gemma-2 text encoder) saves the same
    // tensors WITHOUT that wrapper. Detect which convention this checkpoint uses
    // from the embedding name and prefix all decoder tensors accordingly. The
    // external `prefix` (SafetensorsSource prefix_) is still applied on top.
    const std::string mp =
        src.has("model.embed_tokens.weight") ? "model." : "";

    src.upload_compute_dequant(mp + "embed_tokens.weight",
                               V, H, embed_tokens_, "embed_tokens.weight");

    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        const std::string p = mp + "layers." + std::to_string(i) + ".";
        Layer& L = layers_[static_cast<std::size_t>(i)];

        load_norm(p + "input_layernorm.weight", L.input_ln,
                        "input_layernorm.weight");

        // RoPE permute on q/k exactly as the dense LLaMA-family decoder.
        src.upload_compute_rope_permuted(p + "self_attn.q_proj.weight",
                                         q_dim, H, n_q, HD, L.Wq, "q_proj.weight");
        src.upload_compute_rope_permuted(p + "self_attn.k_proj.weight",
                                         kv_dim, H, n_kv, HD, L.Wk, "k_proj.weight");
        src.upload_compute_checked(p + "self_attn.v_proj.weight",
                                   kv_dim, H, L.Wv, "v_proj.weight");
        src.upload_compute_checked(p + "self_attn.o_proj.weight",
                                   H, q_dim, L.Wo, "o_proj.weight");

        load_norm(p + "post_attention_layernorm.weight", L.post_attn_ln,
                        "post_attention_layernorm.weight");
        load_norm(p + "pre_feedforward_layernorm.weight", L.pre_ffn_ln,
                        "pre_feedforward_layernorm.weight");
        load_norm(p + "post_feedforward_layernorm.weight", L.post_ffn_ln,
                        "post_feedforward_layernorm.weight");

        src.upload_compute_checked(p + "mlp.gate_proj.weight",
                                   F, H, L.gate_W, "gate_proj.weight");
        src.upload_compute_checked(p + "mlp.up_proj.weight",
                                   F, H, L.up_W, "up_proj.weight");
        src.upload_compute_checked(p + "mlp.down_proj.weight",
                                   H, F, L.down_W, "down_proj.weight");
    }

    load_norm(mp + "norm.weight", final_norm_, "norm.weight");

    if (src.has("lm_head.weight")) {
        src.upload_compute_checked("lm_head.weight",
                                   V, H, lm_head_, "lm_head.weight");
    } else {
        if (!cfg_.tie_word_embeddings) {
            fail("load_weights: lm_head.weight missing and "
                 "tie_word_embeddings is false");
        }
        // Tied: lm_head shares the embedding matrix. Clone so the two tensors
        // are independent storage.
        lm_head_ = embed_tokens_.clone();
    }
}

// ─── cache management ────────────────────────────────────────────────────────

void Gemma2Model::allocate_cache(int max_seq_len) {
    if (max_seq_len <= 0) fail("allocate_cache: max_seq_len must be positive");
    const int n_kv = cfg_.num_key_value_heads;
    const int cache_cols = n_kv * cfg_.head_dim;
    const bt::Dtype dt = brolm::compute_dtype();
    const bt::Device dev = bt::default_device();
    for (Layer& L : layers_) {
        brolm::detail::resize_like(L.K_cache, max_seq_len, cache_cols, dt, dev);
        brolm::detail::resize_like(L.V_cache, max_seq_len, cache_cols, dt, dev);
    }
    max_seq_len_ = max_seq_len;
    cache_len_   = 0;
}

void Gemma2Model::truncate_cache(int len) {
    if (len < 0) fail("truncate_cache: len must be non-negative");
    if (len > cache_len_) fail("truncate_cache: len exceeds current cache_len");
    cache_len_ = len;
}

// ─── forward ─────────────────────────────────────────────────────────────────

void Gemma2Model::embed_tokens(const int32_t* ids, int L, bt::Tensor& out) {
    if (!ids) fail("embed_tokens: ids pointer is null");
    if (L <= 0) fail("embed_tokens: L must be positive");
    if (embed_tokens_.size() == 0) fail("embed_tokens: weights not loaded");

    bt::Tensor idx = make_idx_device(ids, L);
    bt::embedding_lookup_forward(
        embed_tokens_, static_cast<const int32_t*>(idx.data), L, out);
    out = out.clone();
}

void Gemma2Model::forward(const int32_t* ids, int L, bt::Tensor& logits_out,
                          bt::Tensor* hidden_out) {
    if (!ids) fail("forward: ids pointer is null");
    if (L <= 0) fail("forward: L must be positive");
    if (embed_tokens_.size() == 0) fail("forward: weights not loaded");
    if (max_seq_len_ == 0) fail("forward: cache not allocated");
    if (cache_len_ + L > max_seq_len_) {
        fail("forward: cache_len + L exceeds allocated cache capacity");
    }

    ids_dev_ = make_idx_device(ids, L);
    bt::embedding_lookup_forward(
        embed_tokens_, static_cast<const int32_t*>(ids_dev_.data), L, h_);
    h_ = h_.clone();

    run_layers_(L, logits_out, hidden_out);
}

void Gemma2Model::forward_last(const int32_t* ids, int L,
                               bt::Tensor& logits_out) {
    if (!ids) fail("forward_last: ids pointer is null");
    if (L <= 0) fail("forward_last: L must be positive");
    if (embed_tokens_.size() == 0) fail("forward_last: weights not loaded");
    if (max_seq_len_ == 0) fail("forward_last: cache not allocated");
    if (cache_len_ + L > max_seq_len_) {
        fail("forward_last: cache_len + L exceeds allocated cache capacity");
    }

    ids_dev_ = make_idx_device(ids, L);
    bt::embedding_lookup_forward(
        embed_tokens_, static_cast<const int32_t*>(ids_dev_.data), L, h_);
    h_ = h_.clone();

    run_layers_(L, logits_out, /*hidden_out=*/nullptr,
                /*logits_last_row_only=*/true);
}

void Gemma2Model::forward_embeds(const bt::Tensor& embeds, int L,
                                 bt::Tensor& logits_out,
                                 bt::Tensor* hidden_out) {
    if (L <= 0) fail("forward_embeds: L must be positive");
    if (embed_tokens_.size() == 0) fail("forward_embeds: weights not loaded");
    if (max_seq_len_ == 0) fail("forward_embeds: cache not allocated");
    if (cache_len_ + L > max_seq_len_) {
        fail("forward_embeds: cache_len + L exceeds allocated cache capacity");
    }
    if (embeds.rows != L || embeds.cols != cfg_.hidden_size) {
        fail("forward_embeds: embeds must be (L, hidden_size)");
    }

    h_ = embeds.clone();
    run_layers_(L, logits_out, hidden_out);
}

void Gemma2Model::run_layers_(int L, bt::Tensor& logits_out,
                              bt::Tensor* hidden_out,
                              bool logits_last_row_only) {
    namespace detail = brolm::detail;
    const int HD   = cfg_.head_dim;
    const int n_q  = cfg_.num_attention_heads;
    const int n_kv = cfg_.num_key_value_heads;
    const float eps = cfg_.rms_norm_eps;
    const float theta = cfg_.rope_theta;
    const float attn_softcap = cfg_.attn_logit_softcapping;
    // seq_offset for RoPE = absolute position of the first new token, which is
    // the cache length BEFORE this forward.
    const int seq_offset = cache_len_;

    // Embedding scale: multiply the residual stream by sqrt(hidden_size) at the
    // compute dtype before the first layer. Folded in here so forward(),
    // forward_last() and forward_embeds() all apply it consistently.
    bt::scale_inplace(h_, std::sqrt(static_cast<float>(cfg_.hidden_size)));

    for (int idx = 0; idx < cfg_.num_hidden_layers; ++idx) {
        Layer& layer = layers_[static_cast<std::size_t>(idx)];

        // ── self-attention sub-layer ──────────────────────────────────────
        bt::rms_norm_forward(h_, layer.input_ln, eps, norm_);

        detail::linear_batched(layer.Wq, /*bias=*/nullptr, norm_, q_);
        detail::linear_batched(layer.Wk, /*bias=*/nullptr, norm_, k_);
        detail::linear_batched(layer.Wv, /*bias=*/nullptr, norm_, v_);

        bt::rope_forward(q_, HD, n_q,  seq_offset, theta, q_);
        bt::rope_forward(k_, HD, n_kv, seq_offset, theta, k_);

        bt::kv_cache_append(k_, v_, cache_len_, layer.K_cache, layer.V_cache);

        // Even layers slide a `sliding_window`; odd layers are global
        // (window == 0). attn_softcap applies the tanh logit soft-cap inside
        // the kernel. The op's built-in 1/sqrt(head_dim) scale is correct
        // because query_pre_attn_scalar == head_dim (validated at config time).
        const int window = (idx % 2 == 0) ? cfg_.sliding_window : 0;
        bt::flash_attention_decode(q_, layer.K_cache, layer.V_cache,
                                   cache_len_ + L, n_q, n_kv, attn_,
                                   attn_softcap, window);

        detail::linear_batched(layer.Wo, /*bias=*/nullptr, attn_, proj_);

        // post_attention_layernorm applies to the SUBLAYER OUTPUT, then add.
        bt::rms_norm_forward(proj_, layer.post_attn_ln, eps, sub_);
        bt::add_inplace(h_, sub_);

        // ── MLP sub-layer (GeGLU, gelu-tanh) ──────────────────────────────
        bt::rms_norm_forward(h_, layer.pre_ffn_ln, eps, norm_);

        detail::linear_batched(layer.gate_W, /*bias=*/nullptr, norm_, gate_);
        detail::linear_batched(layer.up_W,   /*bias=*/nullptr, norm_, up_);
        bt::gelu_forward(gate_, gate_);    // tanh-approx == gelu_pytorch_tanh
        bt::mul_inplace(gate_, up_);
        detail::linear_batched(layer.down_W, /*bias=*/nullptr, gate_, proj_);

        // post_feedforward_layernorm applies to the SUBLAYER OUTPUT, then add.
        bt::rms_norm_forward(proj_, layer.post_ffn_ln, eps, sub_);
        bt::add_inplace(h_, sub_);
    }

    // Final norm + LM head.
    bt::rms_norm_forward(h_, final_norm_, eps, norm_);
    if (hidden_out) *hidden_out = norm_.clone();

    if (logits_last_row_only && L > 1) {
        bt::Tensor last = bt::Tensor::view(
            norm_.device,
            static_cast<char*>(norm_.data) +
                static_cast<std::size_t>(L - 1) *
                    static_cast<std::size_t>(norm_.cols) *
                    static_cast<std::size_t>(bt::dtype_size_bytes(norm_.dtype)),
            1, norm_.cols, norm_.dtype);
        detail::linear_batched(lm_head_, /*bias=*/nullptr, last, logits_out);
    } else {
        detail::linear_batched(lm_head_, /*bias=*/nullptr, norm_, logits_out);
    }

    // Final logit soft-cap: logits = c * tanh(logits / c).
    const float c = cfg_.final_logit_softcapping;
    if (c > 0.0f) {
        bt::scale_inplace(logits_out, 1.0f / c);
        bt::tanh_forward(logits_out, logits_out);
        bt::scale_inplace(logits_out, c);
    }

    cache_len_ += L;
}

}  // namespace brolm::gemma
