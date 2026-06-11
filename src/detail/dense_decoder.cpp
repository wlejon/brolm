#include "brolm/detail/dense_decoder.h"

#include "brolm/detail/compute.h"
#include "brolm/detail/device.h"
#include "brolm/detail/profile.h"
#include "brolm/detail/weights.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#ifdef BROTENSOR_HAS_CUDA
#include "brotensor/cuda_graph.h"
#include "brotensor/detail/dispatch.h"
#endif

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace brolm::detail {

namespace bt = ::brotensor;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("DenseDecoder: " + msg);
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

// ─── CUDA-graph decode session ───────────────────────────────────────────────
//
// One captured single-token decode step over persistent fixed-shape buffers,
// replayed per generated token (the brosoundml Qwen-TTS Talker treatment).
// The eager forward_last(ids, 1, ...) re-issues ~18 ops per layer per token,
// each paying a host launch; the captured step replays the whole stack as one
// cudaGraphLaunch. Three host scalars normally bake shapes into the step —
// they are replaced by device-resident state the staging code updates between
// replays:
//   - KV append row (kv_cache_append's cur_len)  -> scatter_rows + idx_row
//   - attention length (flash_attention_decode's valid_len)
//                                  -> flash_attention_decode_masked + key mask
//   - RoPE position (rope_forward's seq_offset)  -> rope_apply + a staged row
//     of a prebuilt table. The table is generated ON DEVICE by rope_forward
//     itself (over [1,0] pairs), so the staged cos/sin values are bit-identical
//     to what the eager kernel computes and the graph path reproduces the
//     eager decode exactly.
// The layer K/V caches themselves are reused as the fixed-cap buffers — they
// are already allocated at max_seq_len and never move between allocate_cache
// calls (which invalidate the session).

struct DecodeGraphSession {
    int  cap      = 0;    // cache capacity (== max_seq_len_ at build time)
    int  mask_len = -1;   // leading 1s in `mask`; -1 forces a rebuild
    bool captured = false;

    // Staged per-step inputs (written between replays, read by the graph).
    brotensor::Tensor idx_tok;            // (1,1) INT32 — token id (embedding)
    brotensor::Tensor idx_row;            // (1,1) INT32 — KV append row
    brotensor::Tensor mask;               // (cap,1) FP32 — valid-key mask
    brotensor::Tensor cos_step, sin_step; // (1, hd/2) FP32 — RoPE row at pos

    // Prebuilt device constants the staging copies read from.
    brotensor::Tensor tok_ramp;           // (vocab,1) INT32 0..vocab-1
    brotensor::Tensor row_ramp;           // (cap,1)  INT32 0..cap-1
    brotensor::Tensor ones;               // (1,1) FP32 = 1
    brotensor::Tensor cos_all, sin_all;   // (cap, hd/2) FP32 generation table

    // Step-resident activations (allocated by the warm-up run; the captured
    // replay reuses these exact buffers). Distinct from the eager scratch —
    // an eager prefill between replays must not resize captured storage.
    brotensor::Tensor h, norm, q, k, v, qn, kn, attn, proj, gate, up;
    brotensor::Tensor logits;             // (1, vocab)

#ifdef BROTENSOR_HAS_CUDA
    brotensor::CudaGraph graph;
#endif
};

// ─── ctor / dtor ─────────────────────────────────────────────────────────────

DenseDecoder::DenseDecoder(const DenseDecoderConfig& cfg) : cfg_(cfg) {
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

DenseDecoder::~DenseDecoder() = default;
DenseDecoder::DenseDecoder(DenseDecoder&&) noexcept = default;
DenseDecoder& DenseDecoder::operator=(DenseDecoder&&) noexcept = default;

// ─── load_weights ────────────────────────────────────────────────────────────

void DenseDecoder::load_weights(const brolm::detail::weights::Source& src) {
    // Fresh weight uploads replace tensor storage — any captured graph holds
    // dangling weight pointers.
    invalidate_graph_();
    const int V    = cfg_.vocab_size;
    const int H    = cfg_.hidden_size;
    const int F    = cfg_.intermediate_size;
    const int HD   = cfg_.head_dim;
    const int n_q  = cfg_.num_attention_heads;
    const int n_kv = cfg_.num_key_value_heads;
    const int q_dim  = n_q  * HD;
    const int kv_dim = n_kv * HD;

    src.upload_compute_dequant("model.embed_tokens.weight",
                               V, H, embed_tokens_, "embed_tokens.weight");

    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        Layer& L = layers_[static_cast<std::size_t>(i)];

        src.upload_compute_dequant(p + "input_layernorm.weight",
                                   H, 1, L.input_ln, "input_layernorm.weight");

        src.upload_compute_rope_permuted(p + "self_attn.q_proj.weight",
                                         q_dim, H, n_q, HD, L.Wq, "q_proj.weight");
        src.upload_compute_rope_permuted(p + "self_attn.k_proj.weight",
                                         kv_dim, H, n_kv, HD, L.Wk, "k_proj.weight");
        src.upload_compute_checked(p + "self_attn.v_proj.weight",
                                   kv_dim, H, L.Wv, "v_proj.weight");
        src.upload_compute_checked(p + "self_attn.o_proj.weight",
                                   H, q_dim, L.Wo, "o_proj.weight");

        if (cfg_.use_qk_norm) {
            // q_norm / k_norm are length head_dim (cols == 1); permute as if a
            // single head over head_dim entries so the rotary subrange matches
            // the rope-permuted q_proj / k_proj rows.
            src.upload_compute_rope_permuted(p + "self_attn.q_norm.weight",
                                             HD, 1, /*num_heads=*/1, HD,
                                             L.q_norm, "q_norm.weight");
            src.upload_compute_rope_permuted(p + "self_attn.k_norm.weight",
                                             HD, 1, /*num_heads=*/1, HD,
                                             L.k_norm, "k_norm.weight");
        }

        src.upload_compute_dequant(p + "post_attention_layernorm.weight",
                                   H, 1, L.post_attn_ln,
                                   "post_attention_layernorm.weight");

        src.upload_compute_checked(p + "mlp.gate_proj.weight",
                                   F, H, L.gate_W, "gate_proj.weight");
        src.upload_compute_checked(p + "mlp.up_proj.weight",
                                   F, H, L.up_W, "up_proj.weight");
        src.upload_compute_checked(p + "mlp.down_proj.weight",
                                   H, F, L.down_W, "down_proj.weight");
    }

    src.upload_compute_dequant("model.norm.weight",
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

// ─── cache management ────────────────────────────────────────────────────────

void DenseDecoder::allocate_cache(int max_seq_len) {
    if (max_seq_len <= 0) fail("allocate_cache: max_seq_len must be positive");
    // A same-capacity call reuses the cache storage (resize fast path), so the
    // captured decode graph — which bakes the K/V pointers in — stays valid.
    // A different capacity reallocates; drop the session.
    if (max_seq_len != max_seq_len_) invalidate_graph_();
    const int n_kv = cfg_.num_key_value_heads;
    const int cache_cols = n_kv * cfg_.head_dim;
    const bt::Dtype dt = brolm::compute_dtype();
    const bt::Device dev = bt::default_device();
    for (Layer& L : layers_) {
        // KV cache stores true n_kv-width K/V; flash_attention_decode does the
        // GQA head-mapping internally, so no per-head widening is needed.
        brolm::detail::resize_like(L.K_cache, max_seq_len, cache_cols, dt, dev);
        brolm::detail::resize_like(L.V_cache, max_seq_len, cache_cols, dt, dev);
    }
    max_seq_len_ = max_seq_len;
    cache_len_   = 0;
}

void DenseDecoder::reset_cache() { cache_len_ = 0; }

void DenseDecoder::truncate_cache(int len) {
    if (len < 0) fail("truncate_cache: len must be non-negative");
    if (len > cache_len_) {
        fail("truncate_cache: len exceeds current cache_len");
    }
    // The cached K/V rows past `len` are left in place; they are dead until the
    // next forward overwrites them at [len, len + L). RoPE positions derive from
    // cache_len_ at forward time, so they correctly restart from `len`.
    cache_len_ = len;
}

// ─── forward ─────────────────────────────────────────────────────────────────

void DenseDecoder::embed_tokens(const int32_t* ids, int L, bt::Tensor& out) {
    if (!ids) fail("embed_tokens: ids pointer is null");
    if (L <= 0) fail("embed_tokens: L must be positive");
    if (embed_tokens_.size() == 0) fail("embed_tokens: weights not loaded");

    bt::Tensor idx = make_idx_device(ids, L);
    bt::embedding_lookup_forward(
        embed_tokens_, static_cast<const int32_t*>(idx.data), L, out);
    out = out.clone();
}

void DenseDecoder::forward(const int32_t* ids, int L, bt::Tensor& logits_out,
                           bt::Tensor* hidden_out) {
    if (!ids) fail("forward: ids pointer is null");
    if (L <= 0) fail("forward: L must be positive");
    if (embed_tokens_.size() == 0) fail("forward: weights not loaded");
    if (max_seq_len_ == 0) fail("forward: cache not allocated");
    if (cache_len_ + L > max_seq_len_) {
        fail("forward: cache_len + L exceeds allocated cache capacity");
    }

    // Embedding lookup -> own a stable residual stream in h_.
    {
        profile::ScopedStage ps(profile::Stage::idx_upload);
        ids_dev_ = make_idx_device(ids, L);
    }
    {
        profile::ScopedStage ps(profile::Stage::embed);
        bt::embedding_lookup_forward(
            embed_tokens_, static_cast<const int32_t*>(ids_dev_.data), L, h_);
        h_ = h_.clone();
    }

    run_layers_(L, logits_out, hidden_out);
}

void DenseDecoder::forward_last(const int32_t* ids, int L,
                                bt::Tensor& logits_out) {
    if (!ids) fail("forward_last: ids pointer is null");
    if (L <= 0) fail("forward_last: L must be positive");
    if (embed_tokens_.size() == 0) fail("forward_last: weights not loaded");
    if (max_seq_len_ == 0) fail("forward_last: cache not allocated");
    if (cache_len_ + L > max_seq_len_) {
        fail("forward_last: cache_len + L exceeds allocated cache capacity");
    }

    // Single-token decode goes through the captured CUDA-graph step when one
    // is available; prefill (L > 1), CPU, and opt-outs fall through to eager.
    if (L == 1 && try_graph_step_(ids[0], logits_out)) return;

    {
        profile::ScopedStage ps(profile::Stage::idx_upload);
        ids_dev_ = make_idx_device(ids, L);
    }
    {
        profile::ScopedStage ps(profile::Stage::embed);
        bt::embedding_lookup_forward(
            embed_tokens_, static_cast<const int32_t*>(ids_dev_.data), L, h_);
        h_ = h_.clone();
    }

    run_layers_(L, logits_out, /*hidden_out=*/nullptr,
                /*logits_last_row_only=*/true);
}

void DenseDecoder::forward_embeds(const bt::Tensor& embeds, int L,
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

    // Own a stable residual stream in h_ (the layer loop mutates it in place).
    h_ = embeds.clone();

    run_layers_(L, logits_out, hidden_out);
}

// ─── graph decode step ───────────────────────────────────────────────────────

void DenseDecoder::invalidate_graph_() { graph_.reset(); }

bool DenseDecoder::try_graph_step_(int32_t token, bt::Tensor& logits_out) {
#ifndef BROTENSOR_HAS_CUDA
    (void)token;
    (void)logits_out;
    return false;
#else
    if (bt::default_device() != bt::Device::CUDA) return false;
    // BROLM_PROFILE brackets every op in sync pairs — capturing those scopes
    // is illegal and pointless (profiling wants per-stage attribution, which
    // a single graph launch hides). BROLM_NO_GRAPH is the explicit opt-out.
    static const bool disabled = [] {
        const char* v = std::getenv("BROLM_NO_GRAPH");
        return v != nullptr && *v != '\0' && !(v[0] == '0' && v[1] == '\0');
    }();
    if (disabled || profile::enabled()) return false;
    if (token < 0 || token >= cfg_.vocab_size) {
        fail("forward_last: token id out of range");
    }

    const bt::Device dev = bt::default_device();
    const int HD   = cfg_.head_dim;
    const int half = HD / 2;
    const int n_q  = cfg_.num_attention_heads;
    const int n_kv = cfg_.num_key_value_heads;
    const float eps = cfg_.rms_norm_eps;

    if (!graph_ || graph_->cap != max_seq_len_) {
        auto s   = std::make_unique<DecodeGraphSession>();
        const int cap = max_seq_len_;
        s->cap = cap;

        // Staging ramps + constants.
        {
            std::vector<int32_t> ramp(static_cast<std::size_t>(
                std::max(cap, cfg_.vocab_size)));
            for (std::size_t i = 0; i < ramp.size(); ++i) {
                ramp[i] = static_cast<int32_t>(i);
            }
            s->tok_ramp = make_idx_device(ramp.data(), cfg_.vocab_size);
            s->row_ramp = make_idx_device(ramp.data(), cap);
        }
        const float one = 1.0f;
        s->ones = bt::Tensor::from_host_on(dev, &one, 1, 1);
        s->mask = bt::Tensor::zeros_on(dev, cap, 1, bt::Dtype::FP32);
        s->mask_len = 0;
        s->idx_tok = bt::Tensor::empty_on(dev, 1, 1, bt::Dtype::INT32);
        s->idx_row = bt::Tensor::empty_on(dev, 1, 1, bt::Dtype::INT32);

        // Generation RoPE table, computed by the eager kernel's own math:
        // rope_forward over rows of [1, 0] pairs (one "head" of width HD)
        // yields y[2i] = cos(pos·θ_i), y[2i+1] = sin(pos·θ_i) with exactly
        // the fast-math cos/sin the eager decode path applies, so graph and
        // eager decode rotations are bit-identical.
        {
            std::vector<float> unit(static_cast<std::size_t>(cap) * HD, 0.0f);
            for (int p = 0; p < cap; ++p) {
                for (int i = 0; i < half; ++i) {
                    unit[static_cast<std::size_t>(p) * HD + 2 * i] = 1.0f;
                }
            }
            bt::Tensor unit_dev =
                bt::Tensor::from_host_on(dev, unit.data(), cap, HD);
            bt::Tensor rot;
            bt::rope_forward(unit_dev, HD, /*num_heads=*/1, /*seq_offset=*/0,
                             cfg_.rope_theta, rot);
            brolm::detail::resize_like(s->cos_all, cap, half, bt::Dtype::FP32,
                                       dev);
            brolm::detail::resize_like(s->sin_all, cap, half, bt::Dtype::FP32,
                                       dev);
            bt::copy_d2d_strided(rot, 0, 2, s->cos_all, 0, 1, 1, cap * half);
            bt::copy_d2d_strided(rot, 1, 2, s->sin_all, 0, 1, 1, cap * half);
        }
        brolm::detail::resize_like(s->cos_step, 1, half, bt::Dtype::FP32, dev);
        brolm::detail::resize_like(s->sin_step, 1, half, bt::Dtype::FP32, dev);

        graph_ = std::move(s);
    }
    DecodeGraphSession& s = *graph_;

    const int pos = cache_len_;   // capacity already checked by forward_last

    // Resync the valid-key mask when this isn't the next consecutive step
    // (fresh session, post-prefill, reset_cache, truncate_cache). One cap-row
    // host rebuild + upload; consecutive steps just light one element below.
    if (s.mask_len != pos) {
        std::vector<float> hmask(static_cast<std::size_t>(s.cap), 0.0f);
        std::fill(hmask.begin(), hmask.begin() + pos, 1.0f);
        bt::detail::alloc_for(dev).memcpy_h2d(
            s.mask.data, hmask.data(),
            static_cast<std::size_t>(s.cap) * sizeof(float));
        s.mask_len = pos;
    }

    // Stage this step's variables — all device-side, stream-ordered ahead of
    // the replay.
    bt::copy_d2d(s.tok_ramp, token, s.idx_tok, 0, 1);
    bt::copy_d2d(s.row_ramp, pos, s.idx_row, 0, 1);
    bt::copy_d2d(s.ones, 0, s.mask, pos, 1);          // mask[pos] = 1
    bt::copy_d2d(s.cos_all, pos * half, s.cos_step, 0, half);
    bt::copy_d2d(s.sin_all, pos * half, s.sin_step, 0, half);

    // The single-token layer stack over the session's persistent buffers —
    // run_layers_'s math at L == 1, with the three host-scalar ops swapped
    // for their device-state twins.
    auto step_body = [&]() {
        bt::embedding_lookup_forward(
            embed_tokens_, static_cast<const int32_t*>(s.idx_tok.data), 1,
            s.h);

        auto headnorm = [&](const bt::Tensor& src, int num_heads,
                            const bt::Tensor& gain, bt::Tensor& dst) {
            bt::Tensor src_v =
                bt::Tensor::view(src.device, src.data, num_heads, HD,
                                 src.dtype);
            bt::rms_norm_forward(src_v, gain, eps, dst);
            dst.rows = 1;
            dst.cols = num_heads * HD;
        };

        for (Layer& layer : layers_) {
            bt::rms_norm_forward(s.h, layer.input_ln, eps, s.norm);

            detail::linear_batched(layer.Wq, /*bias=*/nullptr, s.norm, s.q);
            detail::linear_batched(layer.Wk, /*bias=*/nullptr, s.norm, s.k);
            detail::linear_batched(layer.Wv, /*bias=*/nullptr, s.norm, s.v);

            bt::Tensor* q_attn = &s.q;
            bt::Tensor* k_attn = &s.k;
            if (cfg_.use_qk_norm) {
                headnorm(s.q, n_q,  layer.q_norm, s.qn);
                headnorm(s.k, n_kv, layer.k_norm, s.kn);
                q_attn = &s.qn;
                k_attn = &s.kn;
            }

            bt::rope_apply(*q_attn, s.cos_step, s.sin_step, HD, n_q, *q_attn);
            bt::rope_apply(*k_attn, s.cos_step, s.sin_step, HD, n_kv, *k_attn);

            bt::scatter_rows(*k_attn, s.idx_row, layer.K_cache);
            bt::scatter_rows(s.v,     s.idx_row, layer.V_cache);

            bt::flash_attention_decode_masked(
                *q_attn, layer.K_cache, layer.V_cache,
                static_cast<const float*>(s.mask.data), n_q, n_kv, s.attn);

            detail::linear_batched(layer.Wo, /*bias=*/nullptr, s.attn, s.proj);
            bt::add_inplace(s.h, s.proj);

            bt::rms_norm_forward(s.h, layer.post_attn_ln, eps, s.norm);
            detail::linear_batched(layer.gate_W, /*bias=*/nullptr, s.norm,
                                   s.gate);
            detail::linear_batched(layer.up_W, /*bias=*/nullptr, s.norm, s.up);
            bt::silu_forward(s.gate, s.gate);
            bt::mul_inplace(s.gate, s.up);
            detail::linear_batched(layer.down_W, /*bias=*/nullptr, s.gate,
                                   s.proj);
            bt::add_inplace(s.h, s.proj);
        }

        bt::rms_norm_forward(s.h, final_norm_, eps, s.norm);
        detail::linear_batched(lm_head_, /*bias=*/nullptr, s.norm, s.logits);
    };

    if (!s.captured) {
        // Warm-up sizes every step buffer (capture must not allocate), and
        // already computes this token's correct step — capture records the
        // identical op sequence without executing it.
        step_body();
        bt::sync_all();
        {
            bt::CudaGraphCapture cap;
            step_body();
            s.graph = cap.finish();
        }
        s.captured = true;
    } else {
        s.graph.launch();
    }

    // Hand the caller stable (1, vocab) logits; s.logits is rewritten by the
    // next replay. The download in the sampler synchronises before reading.
    brolm::detail::resize_like(logits_out, 1, cfg_.vocab_size, s.logits.dtype,
                               dev);
    bt::copy_d2d(s.logits, 0, logits_out, 0, cfg_.vocab_size);

    cache_len_ += 1;
    s.mask_len = pos + 1;
    return true;
#endif
}

void DenseDecoder::run_layers_(int L, bt::Tensor& logits_out,
                               bt::Tensor* hidden_out,
                               bool logits_last_row_only) {
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

    namespace prof = profile;
    using PStage   = profile::Stage;

    for (Layer& layer : layers_) {
        // ── self-attention sub-layer ──────────────────────────────────────
        {
            prof::ScopedStage ps(PStage::rms_norm);
            bt::rms_norm_forward(h_, layer.input_ln, eps, norm_);
        }

        {
            prof::ScopedStage ps(PStage::qkv_proj);
            detail::linear_batched(layer.Wq, /*bias=*/nullptr, norm_, q_);
            detail::linear_batched(layer.Wk, /*bias=*/nullptr, norm_, k_);
            detail::linear_batched(layer.Wv, /*bias=*/nullptr, norm_, v_);
        }

        // QK-norm (Qwen3): per-head RMSNorm over head_dim into distinct
        // buffers. Mistral has none — q_/k_ feed RoPE directly. `q_attn`/
        // `k_attn` select the active query/key tensors for the rest of the
        // attention path.
        bt::Tensor* q_attn = &q_;
        bt::Tensor* k_attn = &k_;
        if (cfg_.use_qk_norm) {
            prof::ScopedStage ps(PStage::qk_norm);
            headnorm(q_, n_q,  layer.q_norm, qn_);
            headnorm(k_, n_kv, layer.k_norm, kn_);
            q_attn = &qn_;
            k_attn = &kn_;
        }

        // RoPE on q/k. Weights were permuted at load so this interleaved-pair
        // rotation reproduces HF's rotate_half.
        {
            prof::ScopedStage ps(PStage::rope);
            bt::rope_forward(*q_attn, HD, n_q,  seq_offset, cfg_.rope_theta,
                             *q_attn);
            bt::rope_forward(*k_attn, HD, n_kv, seq_offset, cfg_.rope_theta,
                             *k_attn);
        }

        // GQA: append the n_kv-width K/V straight to the cache. No widening —
        // flash_attention_decode maps query head h to KV head h/(n_q/n_kv).
        {
            prof::ScopedStage ps(PStage::kv_append);
            bt::kv_cache_append(*k_attn, v_, cache_len_,
                                layer.K_cache, layer.V_cache);
        }

        // Causal flash-attention against the populated cache. valid_len is the
        // total cache length after the append; the op applies causal masking
        // and the 1/sqrt(head_dim) scale internally.
        {
            prof::ScopedStage ps(PStage::attention);
            bt::flash_attention_decode(*q_attn, layer.K_cache, layer.V_cache,
                                       cache_len_ + L, n_q, n_kv, attn_);
        }

        {
            prof::ScopedStage ps(PStage::o_proj);
            detail::linear_batched(layer.Wo, /*bias=*/nullptr, attn_, proj_);
        }
        {
            prof::ScopedStage ps(PStage::residual_add);
            bt::add_inplace(h_, proj_);
        }

        // ── MLP sub-layer (SwiGLU) ────────────────────────────────────────
        {
            prof::ScopedStage ps(PStage::rms_norm);
            bt::rms_norm_forward(h_, layer.post_attn_ln, eps, norm_);
        }

        {
            prof::ScopedStage ps(PStage::mlp_proj);
            detail::linear_batched(layer.gate_W, /*bias=*/nullptr, norm_, gate_);
            detail::linear_batched(layer.up_W,   /*bias=*/nullptr, norm_, up_);
        }

        // SwiGLU without the concat staging: gate <- silu(gate) * up, in
        // place (silu_forward allows aliasing), then down-project.
        {
            prof::ScopedStage ps(PStage::swiglu);
            bt::silu_forward(gate_, gate_);
            bt::mul_inplace(gate_, up_);
        }
        {
            prof::ScopedStage ps(PStage::mlp_proj);
            detail::linear_batched(layer.down_W, /*bias=*/nullptr, gate_,
                                   proj_);
        }
        {
            prof::ScopedStage ps(PStage::residual_add);
            bt::add_inplace(h_, proj_);
        }
    }

    // Final norm + LM head.
    {
        prof::ScopedStage ps(PStage::final_norm);
        bt::rms_norm_forward(h_, final_norm_, eps, norm_);
        if (hidden_out) {
            // norm_ is reused scratch across calls; hand the caller stable
            // storage.
            *hidden_out = norm_.clone();
        }
    }
    {
        prof::ScopedStage ps(PStage::lm_head);
        if (logits_last_row_only && L > 1) {
            // lm_head over the final row only — a (1, hidden) view into the
            // contiguous row-major norm_ buffer.
            bt::Tensor last = bt::Tensor::view(
                norm_.device,
                static_cast<char*>(norm_.data) +
                    static_cast<std::size_t>(L - 1) *
                        static_cast<std::size_t>(norm_.cols) *
                        static_cast<std::size_t>(
                            bt::dtype_size_bytes(norm_.dtype)),
                1, norm_.cols, norm_.dtype);
            detail::linear_batched(lm_head_, /*bias=*/nullptr, last,
                                   logits_out);
        } else {
            detail::linear_batched(lm_head_, /*bias=*/nullptr, norm_,
                                   logits_out);
        }
    }

    cache_len_ += L;
}

}  // namespace brolm::detail
