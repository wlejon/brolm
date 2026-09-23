#include "brolm/modernbert.h"

#include "brolm/detail/compute.h"
#include "brolm/detail/device.h"
#include "brolm/detail/weights.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace brolm::modernbert {

namespace bt = ::brotensor;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("modernbert::ModernBertModel: " + msg);
}

inline void layernorm_bias_free(const bt::Tensor& X, const bt::Tensor& gamma,
                                bt::Tensor& Y, float eps) {
    if (X.dtype == bt::Dtype::FP16) {
        bt::layernorm_forward_inference_batched_fp16(X, gamma, Y, eps);
    } else {
        bt::layernorm_forward_inference_batched(X, gamma, Y, eps);
    }
}

}  // namespace

// Bidirectional multi-head attention. On FP16/BF16 this is brotensor's fused
// FlashAttention-2 WMMA kernel (flash_attention_forward); the GQA entry point
// runs a scalar one-block-per-(query, head) kernel ~50x slower at L = 512.
// FP32 (the CPU backend) keeps the GQA path, which is the one with an FP32
// implementation.
void full_attention(const bt::Tensor& q, const bt::Tensor& k, const bt::Tensor& v,
                    int num_heads, bt::Tensor& out) {
    if (q.dtype == bt::Dtype::FP16 || q.dtype == bt::Dtype::BF16) {
        bt::flash_attention_forward(q, k, v, nullptr, num_heads, /*causal=*/false, out);
    } else {
        bt::flash_attention_gqa_forward(q, k, v, nullptr, num_heads, num_heads,
                                        /*causal=*/false, out);
    }
}

namespace {

// Upload host FP16 bits at the compute dtype (FP16 on GPU, FP32 on CPU).
bt::Tensor upload_fp16_bits(const std::vector<std::uint16_t>& bits, int rows, int cols) {
    if (brolm::compute_dtype() == bt::Dtype::FP16) {
        return bt::Tensor::from_host_fp16(bits.data(), rows, cols);
    }
    std::vector<float> f(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) f[i] = bt::fp16_bits_to_fp32(bits[i]);
    return bt::Tensor::from_host(f.data(), rows, cols);
}

// Brackets one op family with device syncs while profiling is on.
class FamilyTimer {
public:
    FamilyTimer(bool on, double& acc) : on_(on), acc_(acc) {
        if (on_) {
            bt::sync_all();
            t0_ = std::chrono::steady_clock::now();
        }
    }
    ~FamilyTimer() {
        if (on_) {
            bt::sync_all();
            acc_ += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0_).count();
        }
    }
    FamilyTimer(const FamilyTimer&) = delete;
    FamilyTimer& operator=(const FamilyTimer&) = delete;

private:
    bool on_;
    double& acc_;
    std::chrono::steady_clock::time_point t0_;
};

}  // namespace

ModernBertModel::ModernBertModel(Config cfg) : cfg_(std::move(cfg)) {
    layers_.resize(static_cast<std::size_t>(cfg_.num_hidden_layers));
}

ModernBertModel::~ModernBertModel() = default;

void ModernBertModel::load_weights(const brolm::detail::weights::Source& src,
                                  const std::string& prefix) {
    const int V = cfg_.vocab_size;
    const int D = cfg_.hidden_size;
    const int F = cfg_.intermediate_size;
    const int H = cfg_.num_attention_heads;
    const int head_dim = cfg_.head_dim();

    src.upload_compute_checked(prefix + "embeddings.tok_embeddings.weight",
                               V, D, tok_embeddings_, "tok_embeddings");
    src.upload_compute_checked(prefix + "embeddings.norm.weight",
                               D, 1, embed_norm_, "embed_norm");

    std::vector<std::uint16_t> raw, fixed;
    layers_.resize(static_cast<std::size_t>(cfg_.num_hidden_layers));
    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        const std::string p = prefix + "layers." + std::to_string(i) + ".";
        LayerWeights& L = layers_[static_cast<std::size_t>(i)];

        if (i > 0) {
            src.upload_compute_checked(p + "attn_norm.weight", D, 1,
                                       L.attn_norm, "attn_norm");
        }

        // Wqkv: permute the Q and K row blocks from HF rotate_half order into
        // brotensor's interleaved-pair RoPE order; V rows pass through. Done
        // on the host FP16 bits straight from the mapped file, one upload.
        src.download_host_fp16(p + "attn.Wqkv.weight", 3 * D, D, raw, "attn.Wqkv");
        fixed = raw;
        const std::size_t row_bytes = static_cast<std::size_t>(D) * sizeof(std::uint16_t);
        const std::size_t block = static_cast<std::size_t>(D) * static_cast<std::size_t>(D);
        for (int part = 0; part < 2; ++part) {
            brolm::detail::weights::detail_::permute_rope_bytes(
                reinterpret_cast<const uint8_t*>(raw.data() + part * block),
                reinterpret_cast<uint8_t*>(fixed.data() + part * block),
                H, head_dim, row_bytes);
        }
        L.Wqkv = upload_fp16_bits(fixed, 3 * D, D);

        src.upload_compute_checked(p + "attn.Wo.weight", D, D,
                                   L.Wo, "attn.Wo");
        src.upload_compute_checked(p + "mlp_norm.weight", D, 1,
                                   L.mlp_norm, "mlp_norm");

        // mlp.Wi: HF rows are [input (F) ; gate (F)], out = gelu(input) * gate.
        // Interleave them pairwise — row 2j = gate j, row 2j+1 = input j — the
        // layout linear_forward_batched_ex's fused GeGLU epilogue reads.
        src.download_host_fp16(p + "mlp.Wi.weight", 2 * F, D, raw, "mlp.Wi");
        fixed.resize(raw.size());
        const std::size_t drow = static_cast<std::size_t>(D);
        for (std::size_t j = 0; j < static_cast<std::size_t>(F); ++j) {
            std::memcpy(fixed.data() + (2 * j) * drow, raw.data() + (F + j) * drow, row_bytes);
            std::memcpy(fixed.data() + (2 * j + 1) * drow, raw.data() + j * drow, row_bytes);
        }
        L.mlp_Wi = upload_fp16_bits(fixed, 2 * F, D);

        src.upload_compute_checked(p + "mlp.Wo.weight", D, F,
                                   L.mlp_Wo, "mlp.Wo");
    }

    src.upload_compute_checked(prefix + "final_norm.weight", D, 1,
                               final_norm_, "final_norm");
}

void ModernBertModel::init_synthetic() {
    const int V = cfg_.vocab_size;
    const int D = cfg_.hidden_size;
    const int F = cfg_.intermediate_size;

    std::vector<float> ones_D(D, 1.0f);
    std::vector<float> small_VD(static_cast<std::size_t>(V) * D, 0.01f);
    std::vector<float> small_3DD(static_cast<std::size_t>(3 * D) * D, 0.01f);
    std::vector<float> small_DD(static_cast<std::size_t>(D) * D, 0.01f);
    std::vector<float> small_2FD(static_cast<std::size_t>(2 * F) * D, 0.01f);
    std::vector<float> small_DF(static_cast<std::size_t>(D) * F, 0.01f);

    tok_embeddings_ = brolm::detail::upload_host(small_VD.data(), V, D);
    embed_norm_     = brolm::detail::upload_host(ones_D.data(), D, 1);

    layers_.resize(static_cast<std::size_t>(cfg_.num_hidden_layers));
    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        LayerWeights& L = layers_[static_cast<std::size_t>(i)];
        if (i > 0) {
            L.attn_norm = brolm::detail::upload_host(ones_D.data(), D, 1);
        }
        L.Wqkv     = brolm::detail::upload_host(small_3DD.data(), 3 * D, D);
        L.Wo       = brolm::detail::upload_host(small_DD.data(), D, D);
        L.mlp_norm = brolm::detail::upload_host(ones_D.data(), D, 1);
        L.mlp_Wi   = brolm::detail::upload_host(small_2FD.data(), 2 * F, D);
        L.mlp_Wo   = brolm::detail::upload_host(small_DF.data(), D, F);
    }
    final_norm_ = brolm::detail::upload_host(ones_D.data(), D, 1);
}

bool ModernBertModel::reserve_positions(int seq_len) {
    if (seq_len <= rope_rows_) return false;
    // Grow in 512-row steps so a run of slightly longer inputs does not
    // rebuild every call.
    const int rows = ((seq_len + 511) / 512) * 512;
    const int head_dim = cfg_.head_dim();
    const int half = head_dim / 2;
    const bt::Device dev = bt::default_device();
    auto build = [&](float theta, bt::Tensor& cos_out, bt::Tensor& sin_out) {
        // HF: inv_freq = 1 / theta ** (arange(0, dim, 2) / dim), angles in FP32.
        std::vector<float> inv_freq(static_cast<std::size_t>(half));
        for (int i = 0; i < half; ++i) {
            const float p = static_cast<float>(2 * i) / static_cast<float>(head_dim);
            inv_freq[static_cast<std::size_t>(i)] = 1.0f / std::pow(theta, p);
        }
        std::vector<float> cos_h(static_cast<std::size_t>(rows) * half);
        std::vector<float> sin_h(static_cast<std::size_t>(rows) * half);
        for (int pos = 0; pos < rows; ++pos) {
            for (int i = 0; i < half; ++i) {
                const float ang = static_cast<float>(pos) * inv_freq[static_cast<std::size_t>(i)];
                cos_h[static_cast<std::size_t>(pos) * half + i] = std::cos(ang);
                sin_h[static_cast<std::size_t>(pos) * half + i] = std::sin(ang);
            }
        }
        cos_out = bt::Tensor::from_host_on(dev, cos_h.data(), rows, half);
        sin_out = bt::Tensor::from_host_on(dev, sin_h.data(), rows, half);
    };
    build(cfg_.rope_theta_full, cos_full_, sin_full_);
    build(cfg_.rope_theta_sliding, cos_slid_, sin_slid_);
    rope_rows_ = rows;
    return true;
}

void ModernBertModel::reserve_rows(int T) {
    if (T <= 0) return;
    const int D = cfg_.hidden_size;
    const int F = cfg_.intermediate_size;
    const bt::Device dev = bt::default_device();
    const bt::Dtype dt = brolm::compute_dtype();
    Scratch& S = *s_;
    brolm::detail::resize_like(S.embeds, T, D, dt, dev);
    brolm::detail::resize_like(S.h, T, D, dt, dev);
    brolm::detail::resize_like(S.attn_in, T, D, dt, dev);
    brolm::detail::resize_like(S.qkv, T, 3 * D, dt, dev);
    brolm::detail::resize_like(S.attn_out, T, D, dt, dev);
    brolm::detail::resize_like(S.mlp_in, T, D, dt, dev);
    brolm::detail::resize_like(S.geglu_out, T, F, dt, dev);
    // Split-K only engages while a product's output tiles underfill the GPU,
    // which caps its partials near 2 * SMs * 64 * 64 * splits-per-tile; 2M
    // floats (8 MB) covers a 128-SM part, and the op grows it if ever short.
    if (S.ws.rows * S.ws.cols < kWorkspaceFloats) {
        S.ws = bt::Tensor::zeros_on(dev, 1, kWorkspaceFloats, bt::Dtype::FP32);
    }
}

void ModernBertModel::linear(const bt::Tensor& W, const bt::Tensor& X, int epilogue, bt::Tensor& Y) {
    bt::linear_forward_batched_ex(W, nullptr, X, bt::kLinearActNone, epilogue | (fast_accum_ ? bt::kLinearEpiFastAccum : 0),
                                  &s_->ws, Y);
}

void ModernBertModel::forward(const int32_t* input_ids, int seq_len, bt::Tensor& h_out) {
    if (seq_len <= 0) fail("forward: seq_len must be positive");
    std::vector<int32_t> idx(static_cast<std::size_t>(seq_len) * 4);
    for (int r = 0; r < seq_len; ++r) {
        idx[static_cast<std::size_t>(r)] = input_ids[r];
        idx[static_cast<std::size_t>(seq_len + r)] = r;
        idx[static_cast<std::size_t>(2 * seq_len + 2 * r)] = 0;
        idx[static_cast<std::size_t>(2 * seq_len + 2 * r + 1)] = seq_len;
    }
    const bt::Device dev = bt::default_device();
    s_->single_idx = bt::Tensor::from_raw_bytes_on(dev, idx.data(), seq_len * 4, 1, bt::Dtype::INT32,
                                                   idx.size() * sizeof(int32_t));
    int32_t* base = static_cast<int32_t*>(s_->single_idx.data);
    const bt::Tensor ids = bt::Tensor::view(dev, base, seq_len, 1, bt::Dtype::INT32);
    const bt::Tensor pos = bt::Tensor::view(dev, base + seq_len, seq_len, 1, bt::Dtype::INT32);
    const bt::Tensor bounds = bt::Tensor::view(dev, base + 2 * seq_len, seq_len, 2, bt::Dtype::INT32);
    reserve_positions(seq_len);
    forward_packed(PackedInputs{&ids, &pos, &bounds}, h_out);
}

void ModernBertModel::forward_packed(const PackedInputs& in, bt::Tensor& h_out) {
    if (!in.ids || !in.pos || !in.bounds) fail("forward_packed: missing input buffer");
    const int seq_len = in.ids->rows;
    if (seq_len <= 0) fail("forward_packed: no rows");
    if (tok_embeddings_.size() == 0) fail("forward_packed: weights not loaded");
    if (rope_rows_ <= 0) fail("forward_packed: reserve_positions() first");

    const int H = cfg_.num_attention_heads;
    const int head_dim = cfg_.head_dim();
    const float eps = cfg_.norm_eps;
    const bool prof = profiling_;
    EncoderTimings& T = timings_;
    if (prof) ++T.forwards;

    Scratch& S = *s_;
    {
        FamilyTimer t(prof, T.embed_ms);
        bt::embedding_lookup_forward(tok_embeddings_, static_cast<const int32_t*>(in.ids->data),
                                     seq_len, S.embeds);
        if (in.soft && in.soft_idx && in.soft_idx->rows > 0) bt::scatter_rows(*in.soft, *in.soft_idx, S.embeds);
        layernorm_bias_free(S.embeds, embed_norm_, S.h, eps);
    }

    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        LayerWeights& L = layers_[static_cast<std::size_t>(i)];

        // ── Attention block ── (layer 0 has no attn_norm: attend over h)
        const bt::Tensor* attn_in = &S.h;
        if (i > 0) {
            FamilyTimer t(prof, T.norm_ms);
            layernorm_bias_free(S.h, L.attn_norm, S.attn_in, eps);
            attn_in = &S.attn_in;
        }

        {
            FamilyTimer t(prof, T.qkv_ms);
            linear(L.Wqkv, *attn_in, bt::kLinearEpiStore, S.qkv);
        }

        // RoPE in place on the Q/K sections, each row at its own position.
        const bool is_slid = cfg_.is_sliding(i);
        {
            FamilyTimer t(prof, T.rope_ms);
            bt::rope_qkv_packed_inplace(S.qkv, is_slid ? cos_slid_ : cos_full_,
                                        is_slid ? sin_slid_ : sin_full_, *in.pos, H, head_dim);
        }

        // Attention straight off the fused QKV. HF sliding layers attend
        // |q - k| <= local_attention / 2 — brotensor's bidirectional `window`.
        {
            FamilyTimer t(prof, is_slid ? T.attn_local_ms : T.attn_full_ms);
            bt::flash_attention_packed_qkv_forward(S.qkv, *in.bounds, H,
                                                   is_slid ? cfg_.local_attention : 0, S.attn_out);
        }

        {
            // Output projection with the residual add fused into its store.
            FamilyTimer t(prof, T.wo_ms);
            linear(L.Wo, S.attn_out, bt::kLinearEpiAccumulate, S.h);
        }

        // ── MLP block ──
        {
            FamilyTimer t(prof, T.norm_ms);
            layernorm_bias_free(S.h, L.mlp_norm, S.mlp_in, eps);
        }
        {
            // Wi with GeGLU fused into its store (rows interleaved at load).
            FamilyTimer t(prof, T.mlp_in_ms);
            linear(L.mlp_Wi, S.mlp_in, bt::kLinearEpiGeglu, S.geglu_out);
        }
        {
            FamilyTimer t(prof, T.mlp_out_ms);
            linear(L.mlp_Wo, S.geglu_out, bt::kLinearEpiAccumulate, S.h);
        }
    }

    FamilyTimer t(prof, T.norm_ms);
    layernorm_bias_free(S.h, final_norm_, h_out, eps);
}

}  // namespace brolm::modernbert
