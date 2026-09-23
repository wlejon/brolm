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

        // mlp.Wi: swap the two F-row halves so brotensor's geglu_exact_forward
        // (A * gelu(B)) computes ModernBERT's gelu(input) * gate.
        src.download_host_fp16(p + "mlp.Wi.weight", 2 * F, D, raw, "mlp.Wi");
        fixed.resize(raw.size());
        const std::size_t half = static_cast<std::size_t>(F) * static_cast<std::size_t>(D);
        std::memcpy(fixed.data(), raw.data() + half, half * sizeof(std::uint16_t));
        std::memcpy(fixed.data() + half, raw.data(), half * sizeof(std::uint16_t));
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

void ModernBertModel::ensure_rotary_tables_(int seq_len) {
    if (seq_len <= rope_rows_) return;
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
}

void ModernBertModel::forward(const int32_t* input_ids, int seq_len, bt::Tensor& h_out) {
    if (seq_len <= 0) fail("forward: seq_len must be positive");
    if (tok_embeddings_.size() == 0) fail("forward: weights not loaded");

    const int D = cfg_.hidden_size;
    const int H = cfg_.num_attention_heads;
    const int head_dim = cfg_.head_dim();
    const float eps = cfg_.norm_eps;
    const bt::Device dev = bt::default_device();
    const bt::Dtype dt = brolm::compute_dtype();
    const bool prof = profiling_;
    EncoderTimings& T = timings_;
    if (prof) ++T.forwards;

    ensure_rotary_tables_(seq_len);
    const int half = head_dim / 2;
    const bt::Tensor cos_full = bt::Tensor::view(dev, cos_full_.data, seq_len, half);
    const bt::Tensor sin_full = bt::Tensor::view(dev, sin_full_.data, seq_len, half);
    const bt::Tensor cos_slid = bt::Tensor::view(dev, cos_slid_.data, seq_len, half);
    const bt::Tensor sin_slid = bt::Tensor::view(dev, sin_slid_.data, seq_len, half);

    {
        FamilyTimer t(prof, T.embed_ms);
        ids_dev_ =bt::Tensor::from_raw_bytes_on(dev, input_ids, seq_len, 1, bt::Dtype::INT32,
                                                 static_cast<std::size_t>(seq_len) * sizeof(int32_t));
        bt::embedding_lookup_forward(tok_embeddings_,
                                     static_cast<const int32_t*>(ids_dev_.data),
                                     seq_len, embeds_);
        layernorm_bias_free(embeds_, embed_norm_, h_, eps);
    }

    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        LayerWeights& L = layers_[static_cast<std::size_t>(i)];

        // ── Attention block ── (layer 0 has no attn_norm: attend over h_)
        const bt::Tensor* attn_in = &h_;
        if (i > 0) {
            FamilyTimer t(prof, T.norm_ms);
            layernorm_bias_free(h_, L.attn_norm, attn_in_, eps);
            attn_in = &attn_in_;
        }

        {
            FamilyTimer t(prof, T.qkv_ms);
            brolm::detail::linear_batched(L.Wqkv, nullptr, *attn_in, qkv_);
            brolm::detail::resize_like(q_, seq_len, D, dt, dev);
            brolm::detail::resize_like(k_, seq_len, D, dt, dev);
            brolm::detail::resize_like(v_, seq_len, D, dt, dev);
            bt::copy_d2d_strided(qkv_, 0,     3 * D, q_, 0, D, D, seq_len);
            bt::copy_d2d_strided(qkv_, D,     3 * D, k_, 0, D, D, seq_len);
            bt::copy_d2d_strided(qkv_, 2 * D, 3 * D, v_, 0, D, D, seq_len);
        }

        const bool is_slid = cfg_.is_sliding(i);
        {
            FamilyTimer t(prof, T.rope_ms);
            const bt::Tensor& cos_tbl = is_slid ? cos_slid : cos_full;
            const bt::Tensor& sin_tbl = is_slid ? sin_slid : sin_full;
            bt::rope_apply(q_, cos_tbl, sin_tbl, head_dim, H, q_rope_);
            bt::rope_apply(k_, cos_tbl, sin_tbl, head_dim, H, k_rope_);
        }

        if (is_slid) {
            // HF: |q - k| <= local_attention / 2, which is exactly the
            // bidirectional window brotensor takes as `window`.
            FamilyTimer t(prof, T.attn_local_ms);
            bt::flash_attention_windowed_forward(q_rope_, k_rope_, v_, nullptr,
                                                 H, cfg_.local_attention, attn_out_,
                                                 /*causal=*/false);
        } else {
            FamilyTimer t(prof, T.attn_full_ms);
            full_attention(q_rope_, k_rope_, v_, H, attn_out_);
        }

        {
            FamilyTimer t(prof, T.wo_ms);
            brolm::detail::linear_batched(L.Wo, nullptr, attn_out_, proj_out_);
        }
        {
            FamilyTimer t(prof, T.residual_ms);
            bt::add_inplace(h_, proj_out_);
        }

        // ── MLP block ──
        {
            FamilyTimer t(prof, T.norm_ms);
            layernorm_bias_free(h_, L.mlp_norm, mlp_in_, eps);
        }
        {
            FamilyTimer t(prof, T.mlp_in_ms);
            brolm::detail::linear_batched(L.mlp_Wi, nullptr, mlp_in_, geglu_in_);
        }
        {
            FamilyTimer t(prof, T.geglu_ms);
            bt::geglu_exact_forward(geglu_in_, geglu_out_);
        }
        {
            FamilyTimer t(prof, T.mlp_out_ms);
            brolm::detail::linear_batched(L.mlp_Wo, nullptr, geglu_out_, mlp_out_);
        }
        {
            FamilyTimer t(prof, T.residual_ms);
            bt::add_inplace(h_, mlp_out_);
        }
    }

    FamilyTimer t(prof, T.norm_ms);
    layernorm_bias_free(h_, final_norm_, h_out, eps);
}

}  // namespace brolm::modernbert
