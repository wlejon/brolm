#include "brolm/nllb.h"

#include "brolm/detail/compute.h"
#include "brolm/detail/nllb_ops.h"
#include "brolm/detail/weights.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace brolm::nllb {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

namespace {
[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("nllb::Encoder: " + msg);
}
}  // namespace

Encoder::Encoder(const NllbConfig& cfg) : cfg_(cfg) {
    cfg_.validate();
    layers_.resize(static_cast<std::size_t>(cfg_.encoder_layers));
}

Encoder::~Encoder() = default;

// ─── load ────────────────────────────────────────────────────────────────────

void Encoder::load_weights(const st::File& f, const std::string& prefix) {
    const std::vector<const st::File*> shards = {&f};
    load_weights(shards, prefix);
}

void Encoder::load_weights(const std::vector<const st::File*>& shards,
                           const std::string& prefix) {
    if (shards.empty()) fail("load_weights: no safetensors shards");
    brolm::detail::weights::SafetensorsSource src(shards, prefix);
    load_weights_impl_(src);
}

void Encoder::load_weights_impl_(const brolm::detail::weights::Source& src) {
    const int V  = cfg_.vocab_size;
    const int D  = cfg_.d_model;
    const int FF = cfg_.encoder_ffn_dim;

    // Shared token embedding — "model.shared.weight", with the standalone
    // encoder export's "model.encoder.embed_tokens.weight" as a fallback.
    if (src.has("model.shared.weight")) {
        src.upload_compute_checked("model.shared.weight", V, D, token_embed_,
                                   "model.shared.weight");
    } else if (src.has("model.encoder.embed_tokens.weight")) {
        src.upload_compute_checked("model.encoder.embed_tokens.weight", V, D,
                                   token_embed_, "model.shared.weight");
    } else {
        fail("missing token embedding 'model.shared.weight'");
    }

    // mha_forward wants FP32 projection biases regardless of compute dtype.
    auto load_bias_fp32 = [&](const std::string& name, int n, bt::Tensor& dst) {
        bt::Tensor tmp;
        src.upload_compute_checked(name, n, 1, tmp, name);
        if (brolm::compute_dtype() == bt::Dtype::FP16) {
            bt::cast(tmp, dst, bt::Dtype::FP32);
        } else {
            dst = tmp;
        }
    };

    for (int i = 0; i < cfg_.encoder_layers; ++i) {
        const std::string p =
            "model.encoder.layers." + std::to_string(i) + ".";
        Layer& Lr = layers_[static_cast<std::size_t>(i)];

        const std::string sa = p + "self_attn.";
        src.upload_compute_checked(sa + "q_proj.weight", D, D, Lr.Wq, sa + "q");
        src.upload_compute_checked(sa + "k_proj.weight", D, D, Lr.Wk, sa + "k");
        src.upload_compute_checked(sa + "v_proj.weight", D, D, Lr.Wv, sa + "v");
        src.upload_compute_checked(sa + "out_proj.weight", D, D, Lr.Wo, sa + "o");
        load_bias_fp32(sa + "q_proj.bias", D, Lr.bq);
        load_bias_fp32(sa + "k_proj.bias", D, Lr.bk);
        load_bias_fp32(sa + "v_proj.bias", D, Lr.bv);
        load_bias_fp32(sa + "out_proj.bias", D, Lr.bo);

        src.upload_compute_checked(p + "self_attn_layer_norm.weight", D, 1,
                                   Lr.sa_ln_w, "sa_ln.w");
        src.upload_compute_checked(p + "self_attn_layer_norm.bias", D, 1,
                                   Lr.sa_ln_b, "sa_ln.b");

        src.upload_compute_checked(p + "fc1.weight", FF, D, Lr.fc1_w, "fc1.w");
        src.upload_compute_checked(p + "fc1.bias", FF, 1, Lr.fc1_b, "fc1.b");
        src.upload_compute_checked(p + "fc2.weight", D, FF, Lr.fc2_w, "fc2.w");
        src.upload_compute_checked(p + "fc2.bias", D, 1, Lr.fc2_b, "fc2.b");

        src.upload_compute_checked(p + "final_layer_norm.weight", D, 1,
                                   Lr.ff_ln_w, "ff_ln.w");
        src.upload_compute_checked(p + "final_layer_norm.bias", D, 1,
                                   Lr.ff_ln_b, "ff_ln.b");
    }

    src.upload_compute_checked("model.encoder.layer_norm.weight", D, 1,
                               enc_ln_w_, "enc_ln.w");
    src.upload_compute_checked("model.encoder.layer_norm.bias", D, 1,
                               enc_ln_b_, "enc_ln.b");
}

// ─── forward ─────────────────────────────────────────────────────────────────

void Encoder::forward(const std::int32_t* ids, int L, bt::Tensor& enc_out) {
    if (!ids) fail("forward: ids pointer is null");
    if (L <= 0) fail("forward: L must be positive");
    if (token_embed_.size() == 0) fail("forward: weights not loaded");

    const int D = cfg_.d_model;
    const int H = cfg_.encoder_attention_heads;
    const float eps = cfg_.layer_norm_eps;

    // Embedding * sqrt(d_model) + sinusoidal positions.
    ids_dev_ = detail::upload_ids(ids, L);
    bt::embedding_lookup_forward(
        token_embed_, static_cast<const std::int32_t*>(ids_dev_.data), L, x_);
    x_ = x_.clone();   // own the residual stream
    if (cfg_.scale_embedding)
        bt::scale_inplace(x_, std::sqrt(static_cast<float>(D)));
    {
        bt::Tensor pos =
            detail::sinusoidal_positions(L, D, cfg_.pad_token_id, /*past_len=*/0);
        bt::add_inplace(x_, pos);
    }

    for (auto& Lr : layers_) {
        // self-attention (bidirectional, no mask for a single unpadded source)
        detail::layer_norm(x_, Lr.sa_ln_w, Lr.sa_ln_b, n_, eps);
        bt::mha_forward(n_, Lr.Wq, Lr.Wk, Lr.Wv, Lr.Wo,
                        &Lr.bq, &Lr.bk, &Lr.bv, &Lr.bo,
                        /*d_mask=*/nullptr, H,
                        Qh_, Kh_, Vh_, Attnh_, Yconcat_, attn_);
        bt::add_inplace(x_, attn_);

        // feed-forward (ReLU)
        detail::layer_norm(x_, Lr.ff_ln_w, Lr.ff_ln_b, n_, eps);
        detail::linear(Lr.fc1_w, Lr.fc1_b, n_, h1_);
        bt::relu_forward_batched(h1_, h1_);
        detail::linear(Lr.fc2_w, Lr.fc2_b, h1_, ff_out_);
        bt::add_inplace(x_, ff_out_);
    }

    detail::layer_norm(x_, enc_ln_w_, enc_ln_b_, enc_out, eps);
}

}  // namespace brolm::nllb
