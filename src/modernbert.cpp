#include "brolm/modernbert.h"

#include "brolm/detail/compute.h"
#include "brolm/detail/device.h"
#include "brolm/detail/weights.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"

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

bt::Tensor make_idx_device(const int32_t* host, int n) {
    bt::Tensor cpu = bt::Tensor::empty_on(bt::Device::CPU, n, 1, bt::Dtype::INT32);
    std::memcpy(cpu.host_raw_mut(), host,
                static_cast<std::size_t>(n) * sizeof(int32_t));
    return cpu.to(bt::default_device());
}

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

    layers_.resize(static_cast<std::size_t>(cfg_.num_hidden_layers));
    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        const std::string p = prefix + "layers." + std::to_string(i) + ".";
        LayerWeights& L = layers_[static_cast<std::size_t>(i)];

        if (i > 0) {
            src.upload_compute_checked(p + "attn_norm.weight", D, 1,
                                       L.attn_norm, "attn_norm");
        }

        src.upload_compute_checked(p + "attn.Wqkv.weight", 3 * D, D,
                                   L.Wqkv, "attn.Wqkv");

        // Permute Q and K rows from HF rotate_half layout into brotensor interleaved layout
        std::vector<float> host = brolm::detail::weights::detail_::download_fp32(L.Wqkv);
        const std::size_t block = static_cast<std::size_t>(D) * static_cast<std::size_t>(D);
        for (int part = 0; part < 2; ++part) {
            const auto begin = host.begin() + static_cast<std::ptrdiff_t>(part * block);
            std::vector<float> sub(begin, begin + static_cast<std::ptrdiff_t>(block));
            std::vector<float> perm =
                brolm::detail::weights::detail_::permute_rope_fp32(sub, H, head_dim, D);
            std::copy(perm.begin(), perm.end(), begin);
        }
        L.Wqkv = brolm::detail::upload_host(host.data(), 3 * D, D);

        src.upload_compute_checked(p + "attn.Wo.weight", D, D,
                                   L.Wo, "attn.Wo");
        src.upload_compute_checked(p + "mlp_norm.weight", D, 1,
                                   L.mlp_norm, "mlp_norm");
        src.upload_compute_checked(p + "mlp.Wi.weight", 2 * F, D,
                                   L.mlp_Wi, "mlp.Wi");

        // Swap top and bottom F rows of mlp_Wi so that brotensor geglu_exact_forward
        // (which does A * gelu(B)) matches ModernBERT's gelu(input) * gate.
        {
            std::vector<float> wi_host = brolm::detail::weights::detail_::download_fp32(L.mlp_Wi);
            std::vector<float> wi_swapped(wi_host.size());
            const std::size_t half_elems = static_cast<std::size_t>(F) * static_cast<std::size_t>(D);
            std::copy(wi_host.begin() + half_elems, wi_host.end(), wi_swapped.begin());
            std::copy(wi_host.begin(), wi_host.begin() + half_elems, wi_swapped.begin() + half_elems);
            L.mlp_Wi = brolm::detail::upload_host(wi_swapped.data(), 2 * F, D);
        }

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

void ModernBertModel::build_rotary_tables_(int seq_len, float theta,
                                          bt::Tensor& cos_out,
                                          bt::Tensor& sin_out) {
    const int head_dim = cfg_.head_dim();
    const int half = head_dim / 2;
    std::vector<float> inv_freq(static_cast<std::size_t>(half));
    for (int i = 0; i < half; ++i) {
        const float p = static_cast<float>(2 * i) / static_cast<float>(head_dim);
        inv_freq[static_cast<std::size_t>(i)] = 1.0f / std::pow(theta, p);
    }
    std::vector<float> cos_h(static_cast<std::size_t>(seq_len) * half);
    std::vector<float> sin_h(static_cast<std::size_t>(seq_len) * half);
    for (int pos = 0; pos < seq_len; ++pos) {
        for (int i = 0; i < half; ++i) {
            const float ang = static_cast<float>(pos) * inv_freq[static_cast<std::size_t>(i)];
            cos_h[static_cast<std::size_t>(pos) * half + i] = std::cos(ang);
            sin_h[static_cast<std::size_t>(pos) * half + i] = std::sin(ang);
        }
    }
    const bt::Device dev = bt::default_device();
    cos_out = bt::Tensor::from_host(cos_h.data(), seq_len, half).to(dev);
    sin_out = bt::Tensor::from_host(sin_h.data(), seq_len, half).to(dev);
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

    ids_dev_ = make_idx_device(input_ids, seq_len);
    bt::embedding_lookup_forward(tok_embeddings_,
                                 static_cast<const int32_t*>(ids_dev_.data),
                                 seq_len, embeds_);
    layernorm_bias_free(embeds_, embed_norm_, h_, eps);

    bt::Tensor cos_full, sin_full;
    bt::Tensor cos_sliding, sin_sliding;
    build_rotary_tables_(seq_len, cfg_.rope_theta_full, cos_full, sin_full);
    build_rotary_tables_(seq_len, cfg_.rope_theta_sliding, cos_sliding, sin_sliding);

    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        LayerWeights& L = layers_[static_cast<std::size_t>(i)];

        // ── Attention block ──
        if (i == 0) {
            attn_in_ = h_;
        } else {
            layernorm_bias_free(h_, L.attn_norm, attn_in_, eps);
        }

        brolm::detail::linear_batched(L.Wqkv, nullptr, attn_in_, qkv_);

        brolm::detail::resize_like(q_, seq_len, D, dt, dev);
        brolm::detail::resize_like(k_, seq_len, D, dt, dev);
        brolm::detail::resize_like(v_, seq_len, D, dt, dev);
        bt::copy_d2d_strided(qkv_, 0,     3 * D, q_, 0, D, D, seq_len);
        bt::copy_d2d_strided(qkv_, D,     3 * D, k_, 0, D, D, seq_len);
        bt::copy_d2d_strided(qkv_, 2 * D, 3 * D, v_, 0, D, D, seq_len);

        const bool is_slid = cfg_.is_sliding(i);
        const bt::Tensor& cos_tbl = is_slid ? cos_sliding : cos_full;
        const bt::Tensor& sin_tbl = is_slid ? sin_sliding : sin_full;

        bt::rope_apply(q_, cos_tbl, sin_tbl, head_dim, H, q_rope_);
        bt::rope_apply(k_, cos_tbl, sin_tbl, head_dim, H, k_rope_);

        if (is_slid) {
            bt::flash_attention_windowed_forward(q_rope_, k_rope_, v_, nullptr,
                                                 H, cfg_.local_attention, attn_out_,
                                                 /*causal=*/false);
        } else {
            bt::flash_attention_gqa_forward(q_rope_, k_rope_, v_, nullptr,
                                             H, H, /*causal=*/false, attn_out_);
        }

        brolm::detail::linear_batched(L.Wo, nullptr, attn_out_, proj_out_);
        bt::add_inplace(h_, proj_out_);

        // ── MLP block ──
        layernorm_bias_free(h_, L.mlp_norm, mlp_in_, eps);
        brolm::detail::linear_batched(L.mlp_Wi, nullptr, mlp_in_, geglu_in_);
        bt::geglu_exact_forward(geglu_in_, geglu_out_);
        brolm::detail::linear_batched(L.mlp_Wo, nullptr, geglu_out_, mlp_out_);
        bt::add_inplace(h_, mlp_out_);
    }

    layernorm_bias_free(h_, final_norm_, h_out, eps);
}

}  // namespace brolm::modernbert
