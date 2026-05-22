#include "brolm/alignment_adapter.h"

#include "brolm/detail/compute.h"
#include "brolm/detail/device.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace brolm {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("AlignmentAdapter: " + msg);
}

// Build a parameter tensor at the compute dtype, deterministically
// xavier-uniform initialised. xavier_init is CPU/FP32-only (it runs before
// training, on the host), so init on a CPU FP32 staging tensor, then upload
// host-side to the compute dtype + default device. This yields byte-identical
// weights for a given seed on either backend.
bt::Tensor make_xavier(int rows, int cols, uint64_t& rng) {
    bt::Tensor stage = bt::Tensor::zeros_on(bt::Device::CPU, rows, cols,
                                            bt::Dtype::FP32);
    bt::xavier_init(stage, rng);
    return detail::upload_host(stage.host_f32(), rows, cols);
}

// Zero-filled parameter-shaped tensor at the compute dtype on the default
// device — used for biases, grad buffers, and Adam m/v state.
bt::Tensor make_zeros(int rows, int cols) {
    return bt::Tensor::zeros_on(bt::default_device(), rows, cols,
                                compute_dtype());
}

// Download a tensor to a host FP32 vector regardless of compute dtype.
std::vector<float> to_f32(const bt::Tensor& t) {
    bt::sync_all();
    if (t.dtype == bt::Dtype::FP16) {
        std::vector<uint16_t> bits = t.to_host_vector_fp16();
        std::vector<float> out(bits.size());
        for (std::size_t i = 0; i < bits.size(); ++i) {
            out[i] = bt::fp16_bits_to_fp32(bits[i]);
        }
        return out;
    }
    return t.to_host_vector();
}

}  // namespace

// ─── ctor / dtor / move ────────────────────────────────────────────────────

AlignmentAdapter::AlignmentAdapter(const AlignmentAdapterConfig& cfg,
                                   uint64_t init_seed)
    : cfg_(cfg) {
    if (cfg_.d_in <= 0 || cfg_.d_model <= 0 || cfg_.d_cond <= 0) {
        fail("config has a non-positive dimension");
    }

    const int din = cfg_.d_in;
    const int dm  = cfg_.d_model;
    const int dc  = cfg_.d_cond;

    // Weights: xavier-uniform, deterministic from init_seed advanced in place
    // so the three weight matrices draw from one stream. Biases: zero.
    uint64_t rng = init_seed;
    W1_     = make_xavier(dm, din, rng);
    W_text_ = make_xavier(dc, dm,  rng);
    W_pool_ = make_xavier(dc, dm,  rng);
    b1_     = make_zeros(dm, 1);
    b_text_ = make_zeros(dc, 1);
    b_pool_ = make_zeros(dc, 1);

    // Gradient buffers.
    dW1_     = make_zeros(dm, din);
    db1_     = make_zeros(dm, 1);
    dW_text_ = make_zeros(dc, dm);
    db_text_ = make_zeros(dc, 1);
    dW_pool_ = make_zeros(dc, dm);
    db_pool_ = make_zeros(dc, 1);

    // Adam m / v state, zero-initialised.
    mW1_ = make_zeros(dm, din);  vW1_ = make_zeros(dm, din);
    mb1_ = make_zeros(dm, 1);    vb1_ = make_zeros(dm, 1);
    mW_text_ = make_zeros(dc, dm);  vW_text_ = make_zeros(dc, dm);
    mb_text_ = make_zeros(dc, 1);   vb_text_ = make_zeros(dc, 1);
    mW_pool_ = make_zeros(dc, dm);  vW_pool_ = make_zeros(dc, dm);
    mb_pool_ = make_zeros(dc, 1);   vb_pool_ = make_zeros(dc, 1);
}

AlignmentAdapter::~AlignmentAdapter() = default;
AlignmentAdapter::AlignmentAdapter(AlignmentAdapter&&) noexcept = default;
AlignmentAdapter& AlignmentAdapter::operator=(AlignmentAdapter&&) noexcept =
    default;

// ─── forward ───────────────────────────────────────────────────────────────

void AlignmentAdapter::forward(const bt::Tensor& H,
                               bt::Tensor& text_embeddings,
                               bt::Tensor& pooled) {
    if (H.cols != cfg_.d_in) {
        fail("forward: H.cols (" + std::to_string(H.cols) +
             ") != config d_in (" + std::to_string(cfg_.d_in) + ")");
    }
    if (H.rows <= 0) fail("forward: H has no rows");

    const int L  = H.rows;
    const int dm = cfg_.d_model;
    seq_len_ = L;

    // Cache the input for the backward pass.
    H_ = H.clone();

    // A_pre = H @ W1^T + b1   (batched linear: Y[l] = W1 @ H[l] + b1).
    detail::linear_batched(W1_, &b1_, H_, A_pre_);

    // A = silu(A_pre).
    bt::silu_forward(A_pre_, A_);

    // text_embeddings = A @ W_text^T + b_text.
    detail::linear_batched(W_text_, &b_text_, A_, text_embeddings);

    // p = mean_pool_rows(A)  -> (d_model, 1).
    bt::masked_mean_pool_forward(A_, /*d_mask=*/nullptr, p_);

    // pooled = p^T @ W_pool^T + b_pool. Stage p as a 1-row (1, d_model)
    // matrix so the batched linear (which wants (B, in_dim)) runs with B = 1.
    detail::resize_like(pooled_in_, 1, dm, compute_dtype(), A_.device);
    bt::copy_d2d(p_, /*src_off=*/0, pooled_in_, /*dst_off=*/0, /*count=*/dm);
    detail::linear_batched(W_pool_, &b_pool_, pooled_in_, pooled);
}

// ─── backward ──────────────────────────────────────────────────────────────

void AlignmentAdapter::backward(const bt::Tensor& d_text_embeddings,
                                const bt::Tensor& d_pooled,
                                bt::Tensor* dH_out) {
    if (seq_len_ == 0 || A_.empty()) {
        fail("backward: called before a matching forward");
    }
    const int L   = seq_len_;
    const int din = cfg_.d_in;
    const int dm  = cfg_.d_model;
    const int dc  = cfg_.d_cond;

    if (d_text_embeddings.rows != L || d_text_embeddings.cols != dc) {
        fail("backward: d_text_embeddings shape mismatch");
    }
    if (d_pooled.rows != 1 || d_pooled.cols != dc) {
        fail("backward: d_pooled must be (1, d_cond)");
    }

    // ── pooled head: pooled = pooled_in @ W_pool^T + b_pool ──────────────
    // linear_backward_batched accumulates dW_pool / db_pool, overwrites the
    // input grad d_pooled_in (a (1, d_model) row).
    detail::resize_like(d_pooled_in_, 1, dm, compute_dtype(), A_.device);
    bt::linear_backward_batched(W_pool_, pooled_in_, d_pooled,
                                d_pooled_in_, dW_pool_, db_pool_);

    // dp = grad w.r.t. the pooled-row vector p, reshaped (1,d_model)->(d_model,1).
    detail::resize_like(dp_, dm, 1, compute_dtype(), A_.device);
    bt::copy_d2d(d_pooled_in_, /*src_off=*/0, dp_, /*dst_off=*/0, /*count=*/dm);

    // Backprop the mean-pool: dA_pool (L, d_model), overwritten by the op.
    bt::masked_mean_pool_backward(dp_, /*d_mask=*/nullptr, L, dA_pool_);

    // ── text head: text_embeddings = A @ W_text^T + b_text ───────────────
    // Accumulates dW_text / db_text, overwrites dA_text (L, d_model).
    bt::linear_backward_batched(W_text_, A_, d_text_embeddings,
                                dA_text_, dW_text_, db_text_);

    // ── sum the two contributions to dA ──────────────────────────────────
    dA_ = dA_text_.clone();          // (L, d_model)
    bt::add_inplace(dA_, dA_pool_);

    // ── through silu: dA_pre = silu'(A_pre) * dA ─────────────────────────
    bt::silu_backward(A_pre_, dA_, dA_pre_);

    // ── first linear: A_pre = H @ W1^T + b1 ──────────────────────────────
    // Accumulates dW1 / db1. dX target is dH_out when requested, else a
    // throwaway scratch (the op always overwrites its dX argument).
    bt::Tensor* dx = dH_out;
    if (dx == nullptr) {
        detail::resize_like(dummy_dx_, L, din, compute_dtype(), A_.device);
        dx = &dummy_dx_;
    }
    bt::linear_backward_batched(W1_, H_, dA_pre_, *dx, dW1_, db1_);
}

// ─── zero_grads / step ─────────────────────────────────────────────────────

void AlignmentAdapter::zero_grads() {
    dW1_.zero();     db1_.zero();
    dW_text_.zero(); db_text_.zero();
    dW_pool_.zero(); db_pool_.zero();
}

void AlignmentAdapter::step(float lr, float beta1, float beta2, float eps) {
    ++adam_step_;
    const int s = adam_step_;
    bt::adam_step(W1_,     dW1_,     mW1_,     vW1_,     lr, beta1, beta2, eps, s);
    bt::adam_step(b1_,     db1_,     mb1_,     vb1_,     lr, beta1, beta2, eps, s);
    bt::adam_step(W_text_, dW_text_, mW_text_, vW_text_, lr, beta1, beta2, eps, s);
    bt::adam_step(b_text_, db_text_, mb_text_, vb_text_, lr, beta1, beta2, eps, s);
    bt::adam_step(W_pool_, dW_pool_, mW_pool_, vW_pool_, lr, beta1, beta2, eps, s);
    bt::adam_step(b_pool_, db_pool_, mb_pool_, vb_pool_, lr, beta1, beta2, eps, s);
}

// ─── persistence ───────────────────────────────────────────────────────────

namespace {

// Append a parameter to a safetensors write plan as F32. Downloads to host
// FP32 (compute dtype may be FP16) and stows the bytes in `storage` so they
// outlive the write_file call.
void stage_param(const bt::Tensor& t, const std::string& name,
                 std::vector<std::vector<float>>& storage,
                 std::vector<st::WriteEntry>& entries) {
    storage.push_back(to_f32(t));
    const std::vector<float>& buf = storage.back();
    st::WriteEntry e;
    e.name      = name;
    e.dtype     = st::Dtype::F32;
    e.shape     = {t.rows, t.cols};
    e.host_data = buf.data();
    e.bytes     = buf.size() * sizeof(float);
    entries.push_back(e);
}

const st::TensorView& need(const st::File& f, const std::string& key) {
    const auto* v = f.find(key);
    if (!v) throw std::runtime_error("AlignmentAdapter: missing tensor '" +
                                     key + "'");
    return *v;
}

}  // namespace

void AlignmentAdapter::save_weights(const std::string& path) const {
    std::vector<std::vector<float>> storage;
    storage.reserve(6);
    std::vector<st::WriteEntry> entries;
    entries.reserve(6);

    stage_param(W1_,     "W1",     storage, entries);
    stage_param(b1_,     "b1",     storage, entries);
    stage_param(W_text_, "W_text", storage, entries);
    stage_param(b_text_, "b_text", storage, entries);
    stage_param(W_pool_, "W_pool", storage, entries);
    stage_param(b_pool_, "b_pool", storage, entries);

    st::write_file(path, entries);
}

void AlignmentAdapter::load_weights(const st::File& f,
                                    const std::string& prefix) {
    const int din = cfg_.d_in;
    const int dm  = cfg_.d_model;
    const int dc  = cfg_.d_cond;

    st::upload_compute_checked(need(f, prefix + "W1"),     dm, din, W1_,     "W1");
    st::upload_compute_checked(need(f, prefix + "b1"),     dm, 1,   b1_,     "b1");
    st::upload_compute_checked(need(f, prefix + "W_text"), dc, dm,  W_text_, "W_text");
    st::upload_compute_checked(need(f, prefix + "b_text"), dc, 1,   b_text_, "b_text");
    st::upload_compute_checked(need(f, prefix + "W_pool"), dc, dm,  W_pool_, "W_pool");
    st::upload_compute_checked(need(f, prefix + "b_pool"), dc, 1,   b_pool_, "b_pool");
}

}  // namespace brolm
