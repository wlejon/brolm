// LayaGrad: the training-side forward/backward of Laya's packed forward, for
// gradients with respect to soft-token inputs (see laya_grad.h). This file:
// packing, the question-type embedding, the two head layers, the scorer, and
// the soft-row gradient; laya_grad_encoder.cpp holds the ModernBERT half.
//
// Head layer (nn.TransformerEncoderLayer, norm_first, ReLU, biases):
//   x_mid = x + out_proj(attn(in_proj(LN1(x))))
//   x_out = x_mid + linear2(relu(linear1(LN2(x_mid))))
// Scorer on each marker row: LN -> Linear -> GELU(exact) -> Linear(1).

#include "laya_grad_state.h"

#include "brolm/detail/compute.h"
#include "brolm/detail/device.h"

#include "brotensor/runtime.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace brolm::laya {

namespace {

using brolm::detail::resize_like;
using Clock = std::chrono::steady_clock;

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("laya::LayaGrad: " + msg); }

void ln_back(const bt::Tensor& dY, const bt::Tensor& xhat, const bt::Tensor& g, const bt::Tensor& rstd,
             bt::Tensor& dgamma, bt::Tensor& dbeta, bt::Tensor& dX) {
    dgamma.zero();
    dbeta.zero();
    bt::layernorm_backward_batched_with_caches(dY, xhat, g, rstd, dX, dgamma, dbeta);
}

std::vector<float> to_host_f32(const bt::Tensor& t) {
    std::vector<float> out(static_cast<std::size_t>(t.rows) * t.cols);
    if (t.dtype == bt::Dtype::FP32) {
        t.copy_to_host_raw(out.data(), out.size() * sizeof(float));
        return out;
    }
    std::vector<uint16_t> bits(out.size());
    t.copy_to_host_raw(bits.data(), bits.size() * sizeof(uint16_t));
    const bool bf = t.dtype == bt::Dtype::BF16;
    for (std::size_t i = 0; i < bits.size(); ++i) out[i] = bf ? bt::bf16_bits_to_fp32(bits[i]) : bt::fp16_bits_to_fp32(bits[i]);
    return out;
}

// Upload host FP32 at `dt` (FP16/BF16/FP32) onto the default device.
bt::Tensor upload_at(const std::vector<float>& v, int rows, int cols, bt::Dtype dt) {
    const bt::Device dev = bt::default_device();
    if (dt == bt::Dtype::FP32) return bt::Tensor::from_host_on(dev, v.data(), rows, cols);
    std::vector<uint16_t> bits(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        bits[i] = dt == bt::Dtype::BF16 ? bt::fp32_to_bf16_bits(v[i]) : bt::fp32_to_fp16_bits(v[i]);
    }
    return bt::Tensor::from_raw_bytes_on(dev, bits.data(), rows, cols, dt, bits.size() * sizeof(uint16_t));
}

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

}  // namespace

LayaGrad::LayaGrad(DecisionModel& model) : m_(model), s_(std::make_unique<State>()) {}
LayaGrad::~LayaGrad() = default;

std::vector<float> LayaGrad::forward(const std::vector<LayaItem>& items, const bt::Tensor* soft) {
    const auto t0 = Clock::now();
    State& s = *s_;
    modernbert::ModernBertModel& E = m_.encoder_;
    const int D = E.config().hidden_size;
    const int32_t vocab = E.config().vocab_size;
    const bt::Device dev = bt::default_device();
    const bt::Dtype dt = brolm::compute_dtype();
    if (items.empty()) fail("forward: no items");

    // ── pack ──
    int T = 0, K = 0, S = 0, max_len = 0;
    for (const LayaItem& it : items) {
        if (it.num_ids <= 0 || !it.input_ids || it.num_markers <= 0 || !it.marker_pos) fail("forward: empty item");
        for (int i = 0; i < it.num_ids; ++i) {
            if (static_cast<uint32_t>(it.input_ids[i]) >= static_cast<uint32_t>(vocab)) fail("forward: token id out of range");
        }
        for (int k = 0; k < it.num_markers; ++k) {
            if (it.marker_pos[k] < 0 || it.marker_pos[k] >= it.num_ids) fail("forward: marker outside its item");
        }
        if (it.soft_count > 0) {
            if (!soft || soft->cols != D || soft->dtype != dt) fail("forward: soft table missing or mis-shaped");
            if (it.soft_pos < 0 || it.soft_pos + it.soft_count > it.num_ids || it.soft_row < 0 ||
                it.soft_row + it.soft_count > soft->rows) {
                fail("forward: soft rows outside the item or table");
            }
            S += it.soft_count;
        }
        T += it.num_ids;
        K += it.num_markers;
        max_len = std::max(max_len, it.num_ids);
    }
    s.T = T;
    s.K = K;
    s.S = S;
    s.N = static_cast<int>(items.size());
    s.soft_table_rows = soft ? soft->rows : 0;
    s.markers_per_item.clear();

    const int o_pos = T, o_bounds = 2 * T, o_type = 4 * T, o_markers = 5 * T, o_sseq = 5 * T + K, o_stab = o_sseq + S;
    std::vector<int32_t> hi(static_cast<std::size_t>(o_stab + S), 0);
    {
        int row = 0, mk = 0, sr = 0;
        for (const LayaItem& it : items) {
            std::memcpy(hi.data() + row, it.input_ids, static_cast<std::size_t>(it.num_ids) * sizeof(int32_t));
            for (int i = 0; i < it.num_ids; ++i) {
                hi[static_cast<std::size_t>(o_pos + row + i)] = i;
                hi[static_cast<std::size_t>(o_bounds + 2 * (row + i))] = row;
                hi[static_cast<std::size_t>(o_bounds + 2 * (row + i) + 1)] = row + it.num_ids;
                hi[static_cast<std::size_t>(o_type + row + i)] = it.qtype;
            }
            for (int k = 0; k < it.num_markers; ++k) hi[static_cast<std::size_t>(o_markers + mk++)] = row + it.marker_pos[k];
            for (int j = 0; j < it.soft_count; ++j) {
                hi[static_cast<std::size_t>(o_sseq + sr)] = row + it.soft_pos + j;
                hi[static_cast<std::size_t>(o_stab + sr)] = it.soft_row + j;
                ++sr;
            }
            s.markers_per_item.push_back(it.num_markers);
            row += it.num_ids;
        }
    }
    s.idx = bt::Tensor::from_raw_bytes_on(dev, hi.data(), static_cast<int>(hi.size()), 1, bt::Dtype::INT32,
                                          hi.size() * sizeof(int32_t));
    int32_t* base = static_cast<int32_t*>(s.idx.data);
    s.v_ids = bt::Tensor::view(dev, base, T, 1, bt::Dtype::INT32);
    s.v_pos = bt::Tensor::view(dev, base + o_pos, T, 1, bt::Dtype::INT32);
    s.v_bounds = bt::Tensor::view(dev, base + o_bounds, T, 2, bt::Dtype::INT32);
    s.v_type = bt::Tensor::view(dev, base + o_type, T, 1, bt::Dtype::INT32);
    s.v_markers = bt::Tensor::view(dev, base + o_markers, K, 1, bt::Dtype::INT32);
    if (S > 0) {
        s.v_soft_seq = bt::Tensor::view(dev, base + o_sseq, S, 1, bt::Dtype::INT32);
        s.v_soft_tab = bt::Tensor::view(dev, base + o_stab, S, 1, bt::Dtype::INT32);
        bt::gather_rows(*soft, s.v_soft_tab, s.soft_seq_rows);
    }
    if (s.ws.rows * s.ws.cols < (2 << 20)) s.ws = bt::Tensor::zeros_on(dev, 1, 2 << 20, bt::Dtype::FP32);
    E.reserve_positions(max_len);

    encoder_forward_();
    head_forward_();

    // ── scorer ──
    const float eps = 1e-5f;
    bt::gather_rows(s.x_final, s.v_markers, s.m_rows);
    bt::layernorm_forward_batched_with_caches(s.m_rows, m_.scorer_ln_g_, m_.scorer_ln_b_, s.s_y, s.s_xhat, s.s_mean,
                                              s.s_rstd, eps);
    bt::linear_forward_batched_ex(m_.scorer_l1_W_, &m_.scorer_l1_b_, s.s_y, bt::kLinearActNone, bt::kLinearEpiStore,
                                  &s.ws, s.pre1);
    bt::gelu_exact_forward(s.pre1, s.g1);
    bt::linear_forward_batched_ex(m_.scorer_l2_W_, &m_.scorer_l2_b_, s.g1, bt::kLinearActNone, bt::kLinearEpiStore,
                                  &s.ws, s.wide);
    const std::vector<float> wide = to_host_f32(s.wide);  // (K, kOutPad); column 0 is the logit
    std::vector<float> logits(static_cast<std::size_t>(K));
    for (int k = 0; k < K; ++k) logits[static_cast<std::size_t>(k)] = wide[static_cast<std::size_t>(k) * s.wide.cols];
    fwd_ms_ = ms_since(t0);
    return logits;
}

void LayaGrad::head_forward_() {
    State& s = *s_;
    const int T = s.T, D = m_.encoder_.config().hidden_size, H = std::max(1, D / 64);
    const bt::Device dev = bt::default_device();
    const bt::Dtype dt = brolm::compute_dtype();
    const float eps = 1e-5f;

    s.head.resize(m_.head_layers_.size());
    bt::embedding_lookup_forward(m_.type_emb_, static_cast<const int32_t*>(s.v_type.data), T, s.type_rows);
    resize_like(s.head[0].x_in, T, D, dt, dev);
    bt::copy_d2d(s.enc_out, 0, s.head[0].x_in, 0, T * D);
    bt::add_inplace(s.head[0].x_in, s.type_rows);
    for (std::size_t l = 0; l < m_.head_layers_.size(); ++l) {
        const TransformerHeadLayer& W = m_.head_layers_[l];
        State::HeadSave& h = s.head[l];
        bt::layernorm_forward_batched_with_caches(h.x_in, W.norm1_g, W.norm1_b, h.ln1_y, h.ln1_xhat, h.ln1_mean,
                                                  h.ln1_rstd, eps);
        bt::linear_forward_batched_ex(W.in_proj_W, &W.in_proj_b, h.ln1_y, bt::kLinearActNone, bt::kLinearEpiStore,
                                      &s.ws, h.qkv);
        bt::flash_attention_packed_qkv_forward(h.qkv, s.v_bounds, H, 0, h.attn);
        resize_like(h.x_mid, T, D, dt, dev);
        bt::copy_d2d(h.x_in, 0, h.x_mid, 0, T * D);
        bt::linear_forward_batched_ex(W.out_proj_W, &W.out_proj_b, h.attn, bt::kLinearActNone,
                                      bt::kLinearEpiAccumulate, &s.ws, h.x_mid);
        bt::layernorm_forward_batched_with_caches(h.x_mid, W.norm2_g, W.norm2_b, h.ln2_y, h.ln2_xhat, h.ln2_mean,
                                                  h.ln2_rstd, eps);
        bt::linear_forward_batched_ex(W.linear1_W, &W.linear1_b, h.ln2_y, bt::kLinearActRelu, bt::kLinearEpiStore,
                                      &s.ws, h.act);
        bt::Tensor& out = l + 1 < s.head.size() ? s.head[l + 1].x_in : s.x_final;
        resize_like(out, T, D, dt, dev);
        bt::copy_d2d(h.x_mid, 0, out, 0, T * D);
        bt::linear_forward_batched_ex(W.linear2_W, &W.linear2_b, h.act, bt::kLinearActNone, bt::kLinearEpiAccumulate,
                                      &s.ws, out);
    }
}

void LayaGrad::head_backward_(bt::Tensor& d_x) {
    State& s = *s_;
    const int D = m_.encoder_.config().hidden_size, H = std::max(1, D / 64);
    for (std::size_t l = m_.head_layers_.size(); l-- > 0;) {
        const TransformerHeadLayer& W = m_.head_layers_[l];
        State::HeadSave& h = s.head[l];
        // FFN: d_x is dL/dx_out.
        bt::matmul(d_x, W.linear2_W, s.g_b);                   // d act (T, 4D)
        bt::relu_backward(h.act, s.g_b, s.g_c);                // d pre-relu (act > 0 <=> pre > 0)
        bt::matmul(s.g_c, W.linear1_W, s.g_b);                 // d ln2_y (T, D)
        ln_back(s.g_b, h.ln2_xhat, W.norm2_g, h.ln2_rstd, s.dgamma, s.dbeta, s.g_a);
        bt::add_inplace(d_x, s.g_a);                           // dL/dx_mid
        // Attention.
        bt::matmul(d_x, W.out_proj_W, s.g_b);                  // d attn (T, D)
        bt::flash_attention_packed_qkv_backward(h.qkv, s.g_b, s.v_bounds, H, 0, s.g_qkv);
        bt::matmul(s.g_qkv, W.in_proj_W, s.g_b);               // d ln1_y (T, D)
        ln_back(s.g_b, h.ln1_xhat, W.norm1_g, h.ln1_rstd, s.dgamma, s.dbeta, s.g_a);
        bt::add_inplace(d_x, s.g_a);                           // dL/dx_in
    }
    // The type embedding is an additive constant: dL/d(encoder out) = d_x.
}

bool LayaGrad::backward(const std::vector<float>& dlogits, bt::Tensor& d_soft, float loss_scale) {
    const auto t0 = Clock::now();
    State& s = *s_;
    const int T = s.T, K = s.K, D = m_.encoder_.config().hidden_size;
    const bt::Device dev = bt::default_device();
    const bt::Dtype dt = brolm::compute_dtype();
    if (static_cast<int>(dlogits.size()) != K) fail("backward: dlogits must hold one value per marker");

    // ── scorer ── d wide (K, kOutPad): column 0 carries the scaled dL/dlogit.
    const int P = s.wide.cols;
    std::vector<float> dw(static_cast<std::size_t>(K) * P, 0.0f);
    for (int k = 0; k < K; ++k) dw[static_cast<std::size_t>(k) * P] = dlogits[static_cast<std::size_t>(k)] * loss_scale;
    s.g_wide = upload_at(dw, K, P, dt);
    bt::matmul(s.g_wide, m_.scorer_l2_W_, s.g_b);  // d g1 (K, D)
    bt::gelu_exact_backward(s.pre1, s.g_b, s.g_c);
    bt::matmul(s.g_c, m_.scorer_l1_W_, s.g_b);     // d s_y (K, D)
    ln_back(s.g_b, s.s_xhat, m_.scorer_ln_g_, s.s_rstd, s.dgamma, s.dbeta, s.g_a);  // d m_rows (K, D)

    bt::Tensor d_x = bt::Tensor::zeros_on(dev, T, D, dt);
    bt::scatter_rows(s.g_a, s.v_markers, d_x);  // marker rows are unique

    head_backward_(d_x);
    encoder_backward_(d_x);  // d_x: dL/d(embeds)

    // ── soft rows ──
    resize_like(d_soft, std::max(1, s.soft_table_rows), D, bt::Dtype::FP32, dev);
    d_soft.zero();
    if (s.S == 0) {
        bwd_ms_ = ms_since(t0);
        return true;
    }
    bt::gather_rows(d_x, s.v_soft_seq, s.g_rows);  // (S, D) at dt
    bt::cast(s.g_rows, s.g_rows32, bt::Dtype::FP32);
    bt::scale_inplace(s.g_rows32, 1.0f / loss_scale);
    bt::scatter_rows_add(s.g_rows32, s.v_soft_tab, s.soft_table_rows, d_soft);
    const std::vector<float> host = to_host_f32(d_soft);
    bool finite = true;
    for (float v : host) {
        if (!std::isfinite(v)) {
            finite = false;
            break;
        }
    }
    bwd_ms_ = ms_since(t0);
    return finite;
}

}  // namespace brolm::laya
