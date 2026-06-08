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

// Build an FP32 parameter tensor, deterministically xavier-uniform
// initialised. The adapter is a trainable module, so its master weights,
// gradients, and Adam state are kept in FP32 regardless of the compute dtype
// (FP16 master weights + FP16 Adam underflow/diverge to NaN). xavier_init is
// CPU/FP32-only; init on a CPU FP32 staging tensor, then migrate to the default
// device as FP32. This yields byte-identical weights for a given seed.
bt::Tensor make_xavier(int rows, int cols, uint64_t& rng) {
    bt::Tensor stage = bt::Tensor::zeros_on(bt::Device::CPU, rows, cols,
                                            bt::Dtype::FP32);
    bt::xavier_init(stage, rng);
    return bt::Tensor::from_host(stage.host_f32(), rows, cols);
}

// Zero-filled FP32 parameter-shaped tensor on the default device — used for
// biases, grad buffers, and Adam m/v state (all FP32, see make_xavier).
bt::Tensor make_zeros(int rows, int cols) {
    return bt::Tensor::zeros_on(bt::default_device(), rows, cols,
                                bt::Dtype::FP32);
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

    // Master weights are FP32; cast the compute-dtype input up so the whole
    // forward runs in FP32, and cast the two outputs back to the compute dtype
    // at the end (the public I/O contract — diffusion consumers read them at
    // compute_dtype). H_ is the FP32 input cache for backward.
    bt::cast(H, H_, bt::Dtype::FP32);

    // A_pre = H @ W1^T + b1   (batched linear: Y[l] = W1 @ H[l] + b1).
    detail::linear_batched(W1_, &b1_, H_, A_pre_);

    // A = silu(A_pre).
    bt::silu_forward(A_pre_, A_);

    // text_embeddings = A @ W_text^T + b_text   (computed in FP32, cast below).
    detail::linear_batched(W_text_, &b_text_, A_, text_f32_);

    // p = mean_pool_rows(A)  -> (d_model, 1).
    bt::masked_mean_pool_forward(A_, /*d_mask=*/nullptr, p_);

    // pooled = p^T @ W_pool^T + b_pool. Stage p as a 1-row (1, d_model)
    // matrix so the batched linear (which wants (B, in_dim)) runs with B = 1.
    detail::resize_like(pooled_in_, 1, dm, bt::Dtype::FP32, A_.device);
    bt::copy_d2d(p_, /*src_off=*/0, pooled_in_, /*dst_off=*/0, /*count=*/dm);
    detail::linear_batched(W_pool_, &b_pool_, pooled_in_, pooled_f32_);

    // Cast the FP32 results to the compute dtype for the caller.
    bt::cast(text_f32_,   text_embeddings, compute_dtype());
    bt::cast(pooled_f32_, pooled,          compute_dtype());
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

    // Master weights are FP32; the upstream grads arrive at the compute dtype,
    // so cast them up to FP32 for the FP32 backward.
    bt::cast(d_pooled,          dpool_f32_, bt::Dtype::FP32);
    bt::cast(d_text_embeddings, dtext_f32_, bt::Dtype::FP32);

    // ── pooled head: pooled = pooled_in @ W_pool^T + b_pool ──────────────
    // linear_backward_batched accumulates dW_pool / db_pool, overwrites the
    // input grad d_pooled_in (a (1, d_model) row).
    detail::resize_like(d_pooled_in_, 1, dm, bt::Dtype::FP32, A_.device);
    bt::linear_backward_batched(W_pool_, pooled_in_, dpool_f32_,
                                d_pooled_in_, dW_pool_, db_pool_);

    // dp = grad w.r.t. the pooled-row vector p, reshaped (1,d_model)->(d_model,1).
    detail::resize_like(dp_, dm, 1, bt::Dtype::FP32, A_.device);
    bt::copy_d2d(d_pooled_in_, /*src_off=*/0, dp_, /*dst_off=*/0, /*count=*/dm);

    // Backprop the mean-pool: dA_pool (L, d_model), overwritten by the op.
    bt::masked_mean_pool_backward(dp_, /*d_mask=*/nullptr, L, dA_pool_);

    // ── text head: text_embeddings = A @ W_text^T + b_text ───────────────
    // Accumulates dW_text / db_text, overwrites dA_text (L, d_model).
    bt::linear_backward_batched(W_text_, A_, dtext_f32_,
                                dA_text_, dW_text_, db_text_);

    // ── sum the two contributions to dA ──────────────────────────────────
    dA_ = dA_text_.clone();          // (L, d_model)
    bt::add_inplace(dA_, dA_pool_);

    // ── through silu: dA_pre = silu'(A_pre) * dA ─────────────────────────
    bt::silu_backward(A_pre_, dA_, dA_pre_);

    // ── first linear: A_pre = H @ W1^T + b1 ──────────────────────────────
    // Accumulates dW1 / db1. The input-grad dX is computed in FP32; when the
    // caller wants it (dH_out), cast the FP32 result back to the compute dtype,
    // else write to a throwaway FP32 scratch.
    bt::Tensor* dx = (dH_out != nullptr) ? &dH_f32_ : &dummy_dx_;
    detail::resize_like(*dx, L, din, bt::Dtype::FP32, A_.device);
    bt::linear_backward_batched(W1_, H_, dA_pre_, *dx, dW1_, db1_);
    if (dH_out != nullptr) bt::cast(dH_f32_, *dH_out, compute_dtype());
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

    // Master weights are FP32 (saved as F32); upload preserves the F32 dtype.
    st::upload(need(f, prefix + "W1"),     dm, din, W1_);
    st::upload(need(f, prefix + "b1"),     dm, 1,   b1_);
    st::upload(need(f, prefix + "W_text"), dc, dm,  W_text_);
    st::upload(need(f, prefix + "b_text"), dc, 1,   b_text_);
    st::upload(need(f, prefix + "W_pool"), dc, dm,  W_pool_);
    st::upload(need(f, prefix + "b_pool"), dc, 1,   b_pool_);
}

}  // namespace brolm
