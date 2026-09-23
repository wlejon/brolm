// LayaGrad, encoder half: the ModernBERT forward over the packed rows (soft
// rows scattered over their token embeddings), checkpointing each layer's
// input, and the backward that recomputes one layer at a time from its
// checkpoint and carries dL/dh down to the embeddings.
//
// Per layer (pre-LN; layer 0 has no attention norm):
//   a     = LN(h; attn_norm)                    bias-free LayerNorm
//   qkv   = a Wqkv^T, RoPE on q/k at each row's position
//   h'    = h + attn(qkv) Wo^T                  packed, windowed on sliding layers
//   m     = LN(h'; mlp_norm)
//   pre   = m Wi^T                              (T, 2F), GeGLU halves interleaved
//   h_out = h' + geglu(pre) Wo_mlp^T
// Weights are frozen, so the backward needs only input gradients:
// dX = dY W for a linear, the LayerNorm / GeGLU / attention adjoints, and
// RoPE's inverse rotation (the forward kernel with -sin).

#include "laya_grad_state.h"

#include "brolm/detail/compute.h"
#include "brolm/detail/device.h"

#include "brotensor/runtime.h"

#include <algorithm>
#include <stdexcept>

namespace brolm::laya {

namespace {

using brolm::detail::resize_like;

void linear_fwd(const bt::Tensor& W, const bt::Tensor* b, const bt::Tensor& X, int act, int epi, bt::Tensor& ws,
                bt::Tensor& Y) {
    bt::linear_forward_batched_ex(W, b, X, act, epi, &ws, Y);
}

// Y = LN(X) with caches for the backward (beta may be the zero vector).
void ln_train(const bt::Tensor& X, const bt::Tensor& g, const bt::Tensor& b, bt::Tensor& Y, bt::Tensor& xhat,
              bt::Tensor& mean, bt::Tensor& rstd, float eps) {
    bt::layernorm_forward_batched_with_caches(X, g, b, Y, xhat, mean, rstd, eps);
}

// dX = LN backward (gamma/beta gradients go to throwaway accumulators).
void ln_back(const bt::Tensor& dY, const bt::Tensor& xhat, const bt::Tensor& g, const bt::Tensor& rstd,
             bt::Tensor& dgamma, bt::Tensor& dbeta, bt::Tensor& dX) {
    dgamma.zero();
    dbeta.zero();
    bt::layernorm_backward_batched_with_caches(dY, xhat, g, rstd, dX, dgamma, dbeta);
}

}  // namespace

void LayaGrad::encoder_forward_() {
    State& s = *s_;
    modernbert::ModernBertModel& E = m_.encoder_;
    const modernbert::Config& c = E.config();
    const int T = s.T, D = c.hidden_size, H = c.num_attention_heads, hd = c.head_dim();
    const int L = c.num_hidden_layers;
    const float eps = c.norm_eps;
    const bt::Device dev = bt::default_device();
    const bt::Dtype dt = brolm::compute_dtype();

    if (s.zero_beta.rows != D || s.zero_beta.dtype != dt) {
        s.zero_beta = bt::Tensor::zeros_on(dev, D, 1, dt);
        s.dgamma = bt::Tensor::zeros_on(dev, D, 1, dt);
        s.dbeta = bt::Tensor::zeros_on(dev, D, 1, dt);
    }
    // RoPE inverse tables follow the encoder's (rebuilt when it regrew them).
    if (s.neg_sin_src != E.sin_full_.data) {
        s.neg_sin_full = E.sin_full_.clone();
        bt::scale_inplace(s.neg_sin_full, -1.0f);
        s.neg_sin_slid = E.sin_slid_.clone();
        bt::scale_inplace(s.neg_sin_slid, -1.0f);
        s.neg_sin_src = E.sin_full_.data;
    }

    bt::embedding_lookup_forward(E.tok_embeddings_, static_cast<const int32_t*>(s.v_ids.data), T, s.embeds);
    if (s.S > 0) bt::scatter_rows(s.soft_seq_rows, s.v_soft_seq, s.embeds);

    s.h_in.resize(static_cast<std::size_t>(L + 1));
    ln_train(s.embeds, E.embed_norm_, s.zero_beta, s.h_in[0], s.xhat1, s.mean1, s.rstd1, eps);

    for (int i = 0; i < L; ++i) {
        const modernbert::LayerWeights& W = E.layers_[static_cast<std::size_t>(i)];
        const bool slid = c.is_sliding(i);
        bt::Tensor& h = s.h_in[static_cast<std::size_t>(i + 1)];
        resize_like(h, T, D, dt, dev);
        bt::copy_d2d(s.h_in[static_cast<std::size_t>(i)], 0, h, 0, T * D);
        const bt::Tensor* a = &h;
        if (i > 0) {
            ln_train(h, W.attn_norm, s.zero_beta, s.a, s.xhat1, s.mean1, s.rstd1, eps);
            a = &s.a;
        }
        linear_fwd(W.Wqkv, nullptr, *a, bt::kLinearActNone, bt::kLinearEpiStore, s.ws, s.qkv);
        bt::rope_qkv_packed_inplace(s.qkv, slid ? E.cos_slid_ : E.cos_full_, slid ? E.sin_slid_ : E.sin_full_,
                                    s.v_pos, H, hd);
        bt::flash_attention_packed_qkv_forward(s.qkv, s.v_bounds, H, slid ? c.local_attention : 0, s.attn);
        linear_fwd(W.Wo, nullptr, s.attn, bt::kLinearActNone, bt::kLinearEpiAccumulate, s.ws, h);
        ln_train(h, W.mlp_norm, s.zero_beta, s.m, s.xhat2, s.mean2, s.rstd2, eps);
        linear_fwd(W.mlp_Wi, nullptr, s.m, bt::kLinearActNone, bt::kLinearEpiGeglu, s.ws, s.geglu);
        linear_fwd(W.mlp_Wo, nullptr, s.geglu, bt::kLinearActNone, bt::kLinearEpiAccumulate, s.ws, h);
    }
    ln_train(s.h_in[static_cast<std::size_t>(L)], E.final_norm_, s.zero_beta, s.enc_out, s.xhat1, s.mean1,
             s.rstd1, eps);
}

void LayaGrad::encoder_backward_(bt::Tensor& d_h) {
    State& s = *s_;
    modernbert::ModernBertModel& E = m_.encoder_;
    const modernbert::Config& c = E.config();
    const int T = s.T, D = c.hidden_size, F = c.intermediate_size, H = c.num_attention_heads, hd = c.head_dim();
    const int L = c.num_hidden_layers;
    const float eps = c.norm_eps;
    const bt::Device dev = bt::default_device();
    const bt::Dtype dt = brolm::compute_dtype();

    // Final norm.
    ln_train(s.h_in[static_cast<std::size_t>(L)], E.final_norm_, s.zero_beta, s.a, s.xhat1, s.mean1, s.rstd1, eps);
    ln_back(d_h, s.xhat1, E.final_norm_, s.rstd1, s.dgamma, s.dbeta, s.g_a);
    std::swap(d_h, s.g_a);  // d_h = dL/dh_out of the last layer

    for (int i = L - 1; i >= 0; --i) {
        const modernbert::LayerWeights& W = E.layers_[static_cast<std::size_t>(i)];
        const bool slid = c.is_sliding(i);
        const int window = slid ? c.local_attention : 0;

        // ── recompute layer i from its checkpoint ──
        const bt::Tensor& h_in = s.h_in[static_cast<std::size_t>(i)];
        const bt::Tensor* a = &h_in;
        if (i > 0) {
            ln_train(h_in, W.attn_norm, s.zero_beta, s.a, s.xhat1, s.mean1, s.rstd1, eps);
            a = &s.a;
        }
        linear_fwd(W.Wqkv, nullptr, *a, bt::kLinearActNone, bt::kLinearEpiStore, s.ws, s.qkv);
        bt::rope_qkv_packed_inplace(s.qkv, slid ? E.cos_slid_ : E.cos_full_, slid ? E.sin_slid_ : E.sin_full_,
                                    s.v_pos, H, hd);
        bt::flash_attention_packed_qkv_forward(s.qkv, s.v_bounds, H, window, s.attn);
        resize_like(s.h, T, D, dt, dev);
        bt::copy_d2d(h_in, 0, s.h, 0, T * D);
        linear_fwd(W.Wo, nullptr, s.attn, bt::kLinearActNone, bt::kLinearEpiAccumulate, s.ws, s.h);  // h'
        ln_train(s.h, W.mlp_norm, s.zero_beta, s.m, s.xhat2, s.mean2, s.rstd2, eps);
        linear_fwd(W.mlp_Wi, nullptr, s.m, bt::kLinearActNone, bt::kLinearEpiStore, s.ws, s.pre);

        // ── MLP ──
        bt::matmul(d_h, W.mlp_Wo, s.g_b);  // dgeglu (T, F)
        {
            // GeGLU over the interleaved (T, 2F) pre-activation viewed as
            // (T*F, 2) pairs [gate, input]: out = gate * gelu(input).
            const bt::Tensor pre2 = bt::Tensor::view(dev, s.pre.data, T * F, 2, dt);
            const bt::Tensor dg1 = bt::Tensor::view(dev, s.g_b.data, T * F, 1, dt);
            resize_like(s.g_c, T, 2 * F, dt, dev);
            bt::Tensor dpre2 = bt::Tensor::view(dev, s.g_c.data, T * F, 2, dt);
            bt::geglu_exact_backward(pre2, dg1, dpre2);
        }
        bt::matmul(s.g_c, W.mlp_Wi, s.g_b);  // dm (T, D)
        ln_back(s.g_b, s.xhat2, W.mlp_norm, s.rstd2, s.dgamma, s.dbeta, s.g_a);
        bt::add_inplace(d_h, s.g_a);  // d_h = dL/dh'

        // ── attention ──
        bt::matmul(d_h, W.Wo, s.g_b);  // dattn (T, D)
        bt::flash_attention_packed_qkv_backward(s.qkv, s.g_b, s.v_bounds, H, window, s.g_qkv);
        bt::rope_qkv_packed_inplace(s.g_qkv, slid ? E.cos_slid_ : E.cos_full_,
                                    slid ? s.neg_sin_slid : s.neg_sin_full, s.v_pos, H, hd);
        bt::matmul(s.g_qkv, W.Wqkv, s.g_b);  // da (T, D)
        if (i > 0) {
            ln_back(s.g_b, s.xhat1, W.attn_norm, s.rstd1, s.dgamma, s.dbeta, s.g_a);
            bt::add_inplace(d_h, s.g_a);
        } else {
            bt::add_inplace(d_h, s.g_b);
        }
    }

    // Embedding norm: d_h -> dL/d(embeds).
    ln_train(s.embeds, E.embed_norm_, s.zero_beta, s.a, s.xhat1, s.mean1, s.rstd1, eps);
    ln_back(d_h, s.xhat1, E.embed_norm_, s.rstd1, s.dgamma, s.dbeta, s.g_a);
    std::swap(d_h, s.g_a);
}

}  // namespace brolm::laya
