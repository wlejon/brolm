#include "brolm/qwen.h"

#include "brotensor/gguf.h"
#include "brotensor/safetensors.h"
#include "brolm/detail/compute.h"
#include "brolm/detail/device.h"
#include "brolm/detail/weights.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace brolm::qwen {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

// ─── helpers ───────────────────────────────────────────────────────────────

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("qwen::Qwen3Model: " + msg);
}

// Build a device-resident INT32 buffer holding `n` token ids. brotensor has
// no from_host path for INT32, so stage on the host then migrate to the
// default device.
bt::Tensor make_idx_device(const int32_t* host, int n) {
    bt::Tensor cpu = bt::Tensor::empty_on(bt::Device::CPU, n, 1, bt::Dtype::INT32);
    std::memcpy(cpu.host_raw_mut(), host,
                static_cast<std::size_t>(n) * sizeof(int32_t));
    return cpu.to(bt::default_device());
}

}  // namespace

// ─── Qwen3Config::from_gguf ────────────────────────────────────────────────

namespace {

namespace gg = ::brotensor::gguf;

const gg::Value& need_meta(const gg::File& f, const char* key) {
    const gg::Value* v = f.find_meta(key);
    if (!v) throw std::runtime_error(
        std::string("qwen::Qwen3Config::from_gguf: missing metadata '") + key + "'");
    return *v;
}

// Pull an integer-valued metadata into int. Accepts the common int-shaped
// gguf scalar types (u32/i32/u64/i64/u16/i16/u8/i8).
int meta_as_int(const gg::Value& v, const char* key) {
    switch (v.type) {
        case gg::ValueType::U8:  return static_cast<int>(v.scalar.u8);
        case gg::ValueType::I8:  return static_cast<int>(v.scalar.i8);
        case gg::ValueType::U16: return static_cast<int>(v.scalar.u16);
        case gg::ValueType::I16: return static_cast<int>(v.scalar.i16);
        case gg::ValueType::U32: return static_cast<int>(v.scalar.u32);
        case gg::ValueType::I32: return static_cast<int>(v.scalar.i32);
        case gg::ValueType::U64: return static_cast<int>(v.scalar.u64);
        case gg::ValueType::I64: return static_cast<int>(v.scalar.i64);
        default:
            throw std::runtime_error(
                std::string("qwen::Qwen3Config::from_gguf: metadata '") + key +
                "' is not an integer scalar");
    }
}

float meta_as_f32(const gg::Value& v, const char* key) {
    if (v.type == gg::ValueType::F32) return v.scalar.f32;
    if (v.type == gg::ValueType::F64) return static_cast<float>(v.scalar.f64);
    throw std::runtime_error(
        std::string("qwen::Qwen3Config::from_gguf: metadata '") + key +
        "' is not a float scalar");
}

}  // namespace

Qwen3Config Qwen3Config::from_gguf(const gg::File& f) {
    // Architecture check — guards against pointing at, say, a LLaMA gguf.
    if (const auto* arch = f.find_meta("general.architecture")) {
        if (arch->type == gg::ValueType::String && arch->str != "qwen3") {
            throw std::runtime_error(
                "qwen::Qwen3Config::from_gguf: general.architecture is '" +
                arch->str + "', expected 'qwen3'");
        }
    }

    Qwen3Config cfg;
    cfg.hidden_size = meta_as_int(
        need_meta(f, "qwen3.embedding_length"), "qwen3.embedding_length");
    cfg.intermediate_size = meta_as_int(
        need_meta(f, "qwen3.feed_forward_length"), "qwen3.feed_forward_length");
    cfg.num_hidden_layers = meta_as_int(
        need_meta(f, "qwen3.block_count"), "qwen3.block_count");
    cfg.num_attention_heads = meta_as_int(
        need_meta(f, "qwen3.attention.head_count"), "qwen3.attention.head_count");

    if (const auto* v = f.find_meta("qwen3.attention.head_count_kv")) {
        cfg.num_key_value_heads = meta_as_int(*v, "qwen3.attention.head_count_kv");
    } else {
        cfg.num_key_value_heads = cfg.num_attention_heads;
    }

    if (const auto* v = f.find_meta("qwen3.attention.key_length")) {
        cfg.head_dim = meta_as_int(*v, "qwen3.attention.key_length");
    } else {
        cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads;
    }

    if (const auto* v = f.find_meta("qwen3.attention.layer_norm_rms_epsilon")) {
        cfg.rms_norm_eps = meta_as_f32(*v, "qwen3.attention.layer_norm_rms_epsilon");
    }
    if (const auto* v = f.find_meta("qwen3.rope.freq_base")) {
        cfg.rope_theta = meta_as_f32(*v, "qwen3.rope.freq_base");
    }
    if (const auto* v = f.find_meta("qwen3.context_length")) {
        cfg.max_position_embeddings = meta_as_int(*v, "qwen3.context_length");
    }

    // vocab_size: prefer the tokenizer's actual token list length; fall back
    // to a model-level scalar if present.
    if (const auto* toks = f.find_meta("tokenizer.ggml.tokens");
        toks && toks->type == gg::ValueType::Array) {
        cfg.vocab_size = static_cast<int>(toks->array.size());
    } else if (const auto* vs = f.find_meta("qwen3.vocab_size")) {
        cfg.vocab_size = meta_as_int(*vs, "qwen3.vocab_size");
    }

    // Tied embeddings: if there's an explicit metadata flag use it; otherwise
    // infer from the presence of `output.weight`.
    if (const auto* v = f.find_meta("qwen3.tie_word_embeddings");
        v && v->type == gg::ValueType::Bool) {
        cfg.tie_word_embeddings = v->scalar.b;
    } else {
        cfg.tie_word_embeddings = (f.find_tensor("output.weight") == nullptr);
    }

    return cfg;
}

// ─── HF → ggml tensor-name map (Qwen3) ─────────────────────────────────────
//
// HF naming:  model.embed_tokens.weight,
//             model.layers.N.{self_attn.{q,k,v,o}_proj, self_attn.{q,k}_norm,
//                             input_layernorm, post_attention_layernorm,
//                             mlp.{gate,up,down}_proj}.weight,
//             model.norm.weight, lm_head.weight.
// ggml naming (llama.cpp Qwen3 convention):
//             token_embd.weight,
//             blk.N.{attn_q, attn_k, attn_v, attn_output, attn_q_norm,
//                    attn_k_norm, attn_norm, ffn_norm,
//                    ffn_gate, ffn_up, ffn_down}.weight,
//             output_norm.weight, output.weight.

namespace {

bool starts_with(std::string_view s, std::string_view p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

}  // namespace

std::string qwen3_hf_to_ggml(std::string_view hf_name) {
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

    if (tail == "self_attn.q_proj.weight")           return blk("attn_q.weight");
    if (tail == "self_attn.k_proj.weight")           return blk("attn_k.weight");
    if (tail == "self_attn.v_proj.weight")           return blk("attn_v.weight");
    if (tail == "self_attn.o_proj.weight")           return blk("attn_output.weight");
    if (tail == "self_attn.q_norm.weight")           return blk("attn_q_norm.weight");
    if (tail == "self_attn.k_norm.weight")           return blk("attn_k_norm.weight");
    if (tail == "input_layernorm.weight")            return blk("attn_norm.weight");
    if (tail == "post_attention_layernorm.weight")   return blk("ffn_norm.weight");
    if (tail == "mlp.gate_proj.weight")              return blk("ffn_gate.weight");
    if (tail == "mlp.up_proj.weight")                return blk("ffn_up.weight");
    if (tail == "mlp.down_proj.weight")              return blk("ffn_down.weight");
    return {};
}

// ─── RoPE weight permutation ───────────────────────────────────────────────
//
// brotensor's rope_forward rotates ADJACENT pairs (x_{2i}, x_{2i+1}) — the
// GPT-J / interleaved convention. Hugging Face Qwen3 is trained with the
// rotate_half convention, where head_dim index i pairs with i + head_dim/2.
// The two are NOT equivalent.
//
// To run brotensor's fast rope_forward on real HF weights we permute the
// per-head head_dim ordering of q_proj/k_proj (their OUTPUT rows) and of
// q_norm/k_norm (length head_dim) at load time, so that after projection +
// QK-norm the head_dim already lands in interleaved-pair order:
//
//   interleaved index 2i   <- HF head-dim index i
//   interleaved index 2i+1 <- HF head-dim index i + hd/2     for i in [0, hd/2)
//
// i.e. perm[2i] = i, perm[2i+1] = i + hd/2: row r of the permuted weight is
// row `perm[r]` of the HF weight, applied independently within each head's
// head_dim block.
//
// V is not roped, so v_proj is left alone. o_proj is left alone too: permuting
// q and k consistently leaves Q·Kᵀ invariant (the per-head dot product sums
// over head_dim and is permutation-invariant), and the attention output is the
// same consistent permutation of V which... is NOT permuted — so attention
// output stays in HF head order and o_proj sees HF-order activations. This is
// the standard HF→interleaved trick (llama.cpp does the same).
//
// The actual row permutation lives in brolm::detail::weights::Source and is
// applied uniformly across safetensors (FP32 host roundtrip) and gguf
// (byte-level row swap for quant carriers, FP32 roundtrip for dense).

// ─── ctor / dtor ───────────────────────────────────────────────────────────

Qwen3Model::Qwen3Model(const Qwen3Config& cfg) : cfg_(cfg) {
    if (cfg_.hidden_size <= 0 || cfg_.intermediate_size <= 0 ||
        cfg_.num_hidden_layers <= 0 || cfg_.vocab_size <= 0 ||
        cfg_.head_dim <= 0) {
        fail("config has non-positive dimension");
    }
    if (cfg_.num_attention_heads <= 0 || cfg_.num_key_value_heads <= 0) {
        fail("num_attention_heads / num_key_value_heads must be positive");
    }
    if (cfg_.num_attention_heads % cfg_.num_key_value_heads != 0) {
        fail("num_attention_heads must be a multiple of num_key_value_heads");
    }
    if (cfg_.head_dim % 2 != 0) {
        fail("head_dim must be even (RoPE rotates dimension pairs)");
    }
    layers_.resize(static_cast<std::size_t>(cfg_.num_hidden_layers));
}

Qwen3Model::~Qwen3Model() = default;

// ─── load_weights ──────────────────────────────────────────────────────────

void Qwen3Model::load_weights(const st::File& f, const std::string& prefix) {
    const std::vector<const st::File*> shards = {&f};
    load_weights(shards, prefix);
}

void Qwen3Model::load_weights(const std::vector<const st::File*>& shards,
                              const std::string& prefix) {
    if (shards.empty()) fail("load_weights: no safetensors shards");
    brolm::detail::weights::SafetensorsSource src(shards, prefix);
    load_weights_impl_(src);
}

void Qwen3Model::load_weights(const bt::gguf::File& f) {
    const std::vector<const bt::gguf::File*> shards = {&f};
    load_weights(shards);
}

void Qwen3Model::load_weights(
    const std::vector<const bt::gguf::File*>& shards) {
    if (shards.empty()) fail("load_weights: no gguf shards");
    brolm::detail::weights::GgufSource src(
        shards, [](std::string_view hf) { return qwen3_hf_to_ggml(hf); });
    load_weights_impl_(src);
}

void Qwen3Model::load_weights_impl_(
    const brolm::detail::weights::Source& src) {
    const int V    = cfg_.vocab_size;
    const int H    = cfg_.hidden_size;
    const int F    = cfg_.intermediate_size;
    const int HD   = cfg_.head_dim;
    const int n_q  = cfg_.num_attention_heads;
    const int n_kv = cfg_.num_key_value_heads;
    const int q_dim  = n_q  * HD;
    const int kv_dim = n_kv * HD;

    src.upload_compute_checked("model.embed_tokens.weight",
                               V, H, embed_tokens_, "embed_tokens.weight");

    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        Layer& L = layers_[static_cast<std::size_t>(i)];

        src.upload_compute_checked(p + "input_layernorm.weight",
                                   H, 1, L.input_ln, "input_layernorm.weight");

        src.upload_compute_rope_permuted(p + "self_attn.q_proj.weight",
                                         q_dim, H, n_q, HD, L.Wq, "q_proj.weight");
        src.upload_compute_rope_permuted(p + "self_attn.k_proj.weight",
                                         kv_dim, H, n_kv, HD, L.Wk, "k_proj.weight");
        src.upload_compute_checked(p + "self_attn.v_proj.weight",
                                   kv_dim, H, L.Wv, "v_proj.weight");
        src.upload_compute_checked(p + "self_attn.o_proj.weight",
                                   H, q_dim, L.Wo, "o_proj.weight");

        // q_norm / k_norm are length head_dim (cols == 1); permute as if a
        // single head over head_dim entries.
        src.upload_compute_rope_permuted(p + "self_attn.q_norm.weight",
                                         HD, 1, /*num_heads=*/1, HD,
                                         L.q_norm, "q_norm.weight");
        src.upload_compute_rope_permuted(p + "self_attn.k_norm.weight",
                                         HD, 1, /*num_heads=*/1, HD,
                                         L.k_norm, "k_norm.weight");

        src.upload_compute_checked(p + "post_attention_layernorm.weight",
                                   H, 1, L.post_attn_ln,
                                   "post_attention_layernorm.weight");

        src.upload_compute_checked(p + "mlp.gate_proj.weight",
                                   F, H, L.gate_W, "gate_proj.weight");
        src.upload_compute_checked(p + "mlp.up_proj.weight",
                                   F, H, L.up_W, "up_proj.weight");
        src.upload_compute_checked(p + "mlp.down_proj.weight",
                                   H, F, L.down_W, "down_proj.weight");
    }

    src.upload_compute_checked("model.norm.weight",
                               H, 1, final_norm_, "model.norm.weight");

    if (src.has("lm_head.weight")) {
        src.upload_compute_checked("lm_head.weight",
                                   V, H, lm_head_, "lm_head.weight");
    } else {
        if (!cfg_.tie_word_embeddings) {
            fail("load_weights: lm_head.weight missing and "
                 "tie_word_embeddings is false");
        }
        // Tied: lm_head shares the embedding matrix. Clone so the two tensors
        // are independent storage (cheap; avoids aliasing surprises).
        lm_head_ = embed_tokens_.clone();
    }
}

// ─── cache management ──────────────────────────────────────────────────────

void Qwen3Model::allocate_cache(int max_seq_len) {
    if (max_seq_len <= 0) fail("allocate_cache: max_seq_len must be positive");
    const int n_q = cfg_.num_attention_heads;
    const int cache_cols = n_q * cfg_.head_dim;
    const bt::Dtype dt = brolm::compute_dtype();
    const bt::Device dev = bt::default_device();
    for (Layer& L : layers_) {
        // KV cache stores K/V already expanded to n_q heads (see GQA notes).
        brolm::detail::resize_like(L.K_cache, max_seq_len, cache_cols, dt, dev);
        brolm::detail::resize_like(L.V_cache, max_seq_len, cache_cols, dt, dev);
    }
    max_seq_len_ = max_seq_len;
    cache_len_   = 0;
}

void Qwen3Model::reset_cache() { cache_len_ = 0; }

// ─── GQA head expansion ────────────────────────────────────────────────────
//
// flash_attention_decode takes a single num_heads and requires K/V cache cols
// to equal Q cols. Qwen3 has n_kv < n_q, so K and V must be widened from n_kv
// heads to n_q heads before they are appended to the cache: with
// group = n_q / n_kv, query head h reads KV head h / group.
//
// Implemented as a per-(row, query-head) copy_d2d. This is a non-optimal path
// — correctness over speed; a fused gather kernel would be the optimisation.

void Qwen3Model::expand_kv_heads_(const bt::Tensor& src, bt::Tensor& dst) {
    const int HD   = cfg_.head_dim;
    const int n_q  = cfg_.num_attention_heads;
    const int n_kv = cfg_.num_key_value_heads;
    const int group = n_q / n_kv;
    const int L = src.rows;             // src: (L, n_kv*HD)
    const int q_dim = n_q * HD;

    brolm::detail::resize_like(dst, L, q_dim, src.dtype, src.device);
    for (int r = 0; r < L; ++r) {
        for (int h = 0; h < n_q; ++h) {
            const int kv_h = h / group;
            const int src_off = r * (n_kv * HD) + kv_h * HD;
            const int dst_off = r * q_dim + h * HD;
            bt::copy_d2d(src, src_off, dst, dst_off, HD);
        }
    }
}

// ─── forward ───────────────────────────────────────────────────────────────

void Qwen3Model::forward(const int32_t* ids, int L, bt::Tensor& logits_out) {
    if (!ids) fail("forward: ids pointer is null");
    if (L <= 0) fail("forward: L must be positive");
    if (embed_tokens_.size() == 0) fail("forward: weights not loaded");
    if (max_seq_len_ == 0) fail("forward: cache not allocated");
    if (cache_len_ + L > max_seq_len_) {
        fail("forward: cache_len + L exceeds allocated cache capacity");
    }

    const int HD   = cfg_.head_dim;
    const int n_q  = cfg_.num_attention_heads;
    const int n_kv = cfg_.num_key_value_heads;
    const float eps = cfg_.rms_norm_eps;
    // seq_offset for RoPE = absolute position of the first new token, which is
    // the cache length BEFORE this forward.
    const int seq_offset = cache_len_;

    // Per-head RMSNorm: view (L, num_heads*HD) as (L*num_heads, HD), run
    // rms_norm with a (HD,1) gain shared across heads into `dst`, then reshape
    // `dst` back to (L, num_heads*HD). `src` and `dst` are distinct buffers.
    // Mirrors brodiffusion/src/dit/flux.cpp headnorm.
    auto headnorm = [&](const bt::Tensor& src, int num_heads,
                        const bt::Tensor& gain, bt::Tensor& dst) -> void {
        const int rows = src.rows;
        bt::Tensor src_v = bt::Tensor::view(
            src.device, src.data, rows * num_heads, HD, src.dtype);
        bt::rms_norm_forward(src_v, gain, eps, dst);
        // rms_norm_forward sized dst to (rows*num_heads, HD); the storage is
        // identical to (rows, num_heads*HD) row-major. Reshape in place.
        dst.rows = rows;
        dst.cols = num_heads * HD;
    };

    // Embedding lookup -> own a stable residual stream in h_.
    ids_dev_ = make_idx_device(ids, L);
    bt::embedding_lookup_forward(
        embed_tokens_, static_cast<const int32_t*>(ids_dev_.data), L, h_);
    h_ = h_.clone();

    for (Layer& layer : layers_) {
        // ── self-attention sub-layer ──────────────────────────────────────
        bt::rms_norm_forward(h_, layer.input_ln, eps, norm_);

        detail::linear_batched(layer.Wq, /*bias=*/nullptr, norm_, q_);
        detail::linear_batched(layer.Wk, /*bias=*/nullptr, norm_, k_);
        detail::linear_batched(layer.Wv, /*bias=*/nullptr, norm_, v_);

        // QK-norm: per-head RMSNorm over head_dim, into distinct buffers.
        headnorm(q_, n_q,  layer.q_norm, qn_);
        headnorm(k_, n_kv, layer.k_norm, kn_);

        // RoPE on the QK-normed q/k. Weights were permuted at load so this
        // interleaved-pair rotation reproduces HF's rotate_half.
        bt::rope_forward(qn_, HD, n_q,  seq_offset, cfg_.rope_theta, qn_);
        bt::rope_forward(kn_, HD, n_kv, seq_offset, cfg_.rope_theta, kn_);

        // GQA: widen k/v from n_kv heads to n_q heads, then append the
        // n_q-head-width K/V to the cache.
        expand_kv_heads_(kn_, k_exp_);
        expand_kv_heads_(v_, v_exp_);
        bt::kv_cache_append(k_exp_, v_exp_, cache_len_,
                            layer.K_cache, layer.V_cache);

        // Causal flash-attention against the populated cache. valid_len is the
        // total cache length after the append; the op applies causal masking
        // and the 1/sqrt(head_dim) scale internally.
        bt::flash_attention_decode(qn_, layer.K_cache, layer.V_cache,
                                   cache_len_ + L, n_q, attn_);

        detail::linear_batched(layer.Wo, /*bias=*/nullptr, attn_, proj_);
        bt::add_inplace(h_, proj_);

        // ── MLP sub-layer (SwiGLU) ────────────────────────────────────────
        bt::rms_norm_forward(h_, layer.post_attn_ln, eps, norm_);

        detail::linear_batched(layer.gate_W, /*bias=*/nullptr, norm_, gate_);
        detail::linear_batched(layer.up_W,   /*bias=*/nullptr, norm_, up_);

        // swiglu_forward expects (L, 2*F) = concat(gate, up) along columns and
        // computes silu(gate) * up.
        const int F = cfg_.intermediate_size;
        brolm::detail::resize_like(swiglu_in_, L, 2 * F, gate_.dtype,
                                   gate_.device);
        for (int r = 0; r < L; ++r) {
            bt::copy_d2d(gate_, r * F, swiglu_in_, r * (2 * F), F);
            bt::copy_d2d(up_,   r * F, swiglu_in_, r * (2 * F) + F, F);
        }
        bt::swiglu_forward(swiglu_in_, mlp_act_);
        detail::linear_batched(layer.down_W, /*bias=*/nullptr, mlp_act_, proj_);
        bt::add_inplace(h_, proj_);
    }

    // Final norm + LM head.
    bt::rms_norm_forward(h_, final_norm_, eps, norm_);
    detail::linear_batched(lm_head_, /*bias=*/nullptr, norm_, logits_out);

    cache_len_ += L;
}

}  // namespace brolm::qwen
