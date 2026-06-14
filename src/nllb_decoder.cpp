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
    throw std::runtime_error("nllb::Decoder: " + msg);
}
std::size_t dtype_bytes(bt::Dtype dt) {
    return (dt == bt::Dtype::FP16) ? 2u : 4u;
}
}  // namespace

Decoder::Decoder(const NllbConfig& cfg) : cfg_(cfg) {
    cfg_.validate();
    layers_.resize(static_cast<std::size_t>(cfg_.decoder_layers));
}

Decoder::~Decoder() = default;

// ─── load ────────────────────────────────────────────────────────────────────

void Decoder::load_weights(const st::File& f, const std::string& prefix) {
    const std::vector<const st::File*> shards = {&f};
    load_weights(shards, prefix);
}

void Decoder::load_weights(const std::vector<const st::File*>& shards,
                           const std::string& prefix) {
    if (shards.empty()) fail("load_weights: no safetensors shards");
    brolm::detail::weights::SafetensorsSource src(shards, prefix);
    load_weights_impl_(src);
}

void Decoder::load_weights_impl_(const brolm::detail::weights::Source& src) {
    const int V  = cfg_.vocab_size;
    const int D  = cfg_.d_model;
    const int FF = cfg_.decoder_ffn_dim;

    if (src.has("model.shared.weight")) {
        src.upload_compute_checked("model.shared.weight", V, D, token_embed_,
                                   "model.shared.weight");
    } else if (src.has("model.decoder.embed_tokens.weight")) {
        src.upload_compute_checked("model.decoder.embed_tokens.weight", V, D,
                                   token_embed_, "model.shared.weight");
    } else {
        fail("missing token embedding 'model.shared.weight'");
    }

    // M2M-100 adds a (1, vocab) final_logits_bias buffer to the lm_head output;
    // it is zeros in the released NLLB checkpoints but load it when present so
    // any fine-tuned variant stays correct.
    if (src.has("final_logits_bias")) {
        src.upload_compute_checked("final_logits_bias", 1, V,
                                   final_logits_bias_, "final_logits_bias");
    }

    auto attn = [&](const std::string& base, const std::string& which,
                    bt::Tensor& Wq, bt::Tensor& bq, bt::Tensor& Wk,
                    bt::Tensor& bk, bt::Tensor& Wv, bt::Tensor& bv,
                    bt::Tensor& Wo, bt::Tensor& bo) {
        const std::string a = base + which + ".";
        src.upload_compute_checked(a + "q_proj.weight", D, D, Wq, a + "q");
        src.upload_compute_checked(a + "q_proj.bias",   D, 1, bq, a + "qb");
        src.upload_compute_checked(a + "k_proj.weight", D, D, Wk, a + "k");
        src.upload_compute_checked(a + "k_proj.bias",   D, 1, bk, a + "kb");
        src.upload_compute_checked(a + "v_proj.weight", D, D, Wv, a + "v");
        src.upload_compute_checked(a + "v_proj.bias",   D, 1, bv, a + "vb");
        src.upload_compute_checked(a + "out_proj.weight", D, D, Wo, a + "o");
        src.upload_compute_checked(a + "out_proj.bias",   D, 1, bo, a + "ob");
    };

    for (int i = 0; i < cfg_.decoder_layers; ++i) {
        const std::string p =
            "model.decoder.layers." + std::to_string(i) + ".";
        Layer& Lr = layers_[static_cast<std::size_t>(i)];

        attn(p, "self_attn", Lr.sWq, Lr.sbq, Lr.sWk, Lr.sbk,
             Lr.sWv, Lr.sbv, Lr.sWo, Lr.sbo);
        src.upload_compute_checked(p + "self_attn_layer_norm.weight", D, 1,
                                   Lr.sa_ln_w, "sa_ln.w");
        src.upload_compute_checked(p + "self_attn_layer_norm.bias", D, 1,
                                   Lr.sa_ln_b, "sa_ln.b");

        attn(p, "encoder_attn", Lr.cWq, Lr.cbq, Lr.cWk, Lr.cbk,
             Lr.cWv, Lr.cbv, Lr.cWo, Lr.cbo);
        src.upload_compute_checked(p + "encoder_attn_layer_norm.weight", D, 1,
                                   Lr.ca_ln_w, "ca_ln.w");
        src.upload_compute_checked(p + "encoder_attn_layer_norm.bias", D, 1,
                                   Lr.ca_ln_b, "ca_ln.b");

        src.upload_compute_checked(p + "fc1.weight", FF, D, Lr.fc1_w, "fc1.w");
        src.upload_compute_checked(p + "fc1.bias", FF, 1, Lr.fc1_b, "fc1.b");
        src.upload_compute_checked(p + "fc2.weight", D, FF, Lr.fc2_w, "fc2.w");
        src.upload_compute_checked(p + "fc2.bias", D, 1, Lr.fc2_b, "fc2.b");
        src.upload_compute_checked(p + "final_layer_norm.weight", D, 1,
                                   Lr.ff_ln_w, "ff_ln.w");
        src.upload_compute_checked(p + "final_layer_norm.bias", D, 1,
                                   Lr.ff_ln_b, "ff_ln.b");
    }

    src.upload_compute_checked("model.decoder.layer_norm.weight", D, 1,
                               dec_ln_w_, "dec_ln.w");
    src.upload_compute_checked("model.decoder.layer_norm.bias", D, 1,
                               dec_ln_b_, "dec_ln.b");
}

// ─── cross-attention memory ─────────────────────────────────────────────────

void Decoder::set_encoder_memory(const bt::Tensor& enc_out) {
    if (enc_out.cols != cfg_.d_model)
        fail("set_encoder_memory: enc_out width != d_model");
    enc_len_ = enc_out.rows;
    for (auto& Lr : layers_) {
        brolm::detail::linear_batched(Lr.cWk, &Lr.cbk, enc_out, Lr.K_enc);
        brolm::detail::linear_batched(Lr.cWv, &Lr.cbv, enc_out, Lr.V_enc);
    }
}

// ─── forward ─────────────────────────────────────────────────────────────────

void Decoder::forward_logits(const std::int32_t* dec_ids, int T,
                             bt::Tensor& logits) {
    if (!dec_ids) fail("forward_logits: dec_ids is null");
    if (T <= 0) fail("forward_logits: T must be positive");
    if (token_embed_.size() == 0) fail("forward_logits: weights not loaded");
    if (enc_len_ == 0) fail("forward_logits: set_encoder_memory not called");

    const int D = cfg_.d_model;
    const int H = cfg_.decoder_attention_heads;
    const float eps = cfg_.layer_norm_eps;

    // Embedding * sqrt(d_model) + sinusoidal positions (decode offset 0: the
    // whole prefix is recomputed each step).
    ids_dev_ = detail::upload_ids(dec_ids, T);
    bt::embedding_lookup_forward(
        token_embed_, static_cast<const std::int32_t*>(ids_dev_.data), T, x_);
    x_ = x_.clone();
    if (cfg_.scale_embedding)
        bt::scale_inplace(x_, std::sqrt(static_cast<float>(D)));
    {
        bt::Tensor pos =
            detail::sinusoidal_positions(T, D, cfg_.pad_token_id, /*past_len=*/0);
        bt::add_inplace(x_, pos);
    }

    for (auto& Lr : layers_) {
        // causal self-attention
        brolm::detail::layernorm_batched(x_, Lr.sa_ln_w, Lr.sa_ln_b, xln_, eps);
        brolm::detail::linear_batched(Lr.sWq, &Lr.sbq, xln_, Q_);
        brolm::detail::linear_batched(Lr.sWk, &Lr.sbk, xln_, K_);
        brolm::detail::linear_batched(Lr.sWv, &Lr.sbv, xln_, V_);
        bt::flash_attention_forward(Q_, K_, V_, /*d_mask=*/nullptr, H,
                                    /*causal=*/true, attn_);
        brolm::detail::linear_batched(Lr.sWo, &Lr.sbo, attn_, proj_);
        bt::add_inplace(x_, proj_);

        // cross-attention over the cached encoder K/V (non-causal)
        brolm::detail::layernorm_batched(x_, Lr.ca_ln_w, Lr.ca_ln_b, xln_, eps);
        brolm::detail::linear_batched(Lr.cWq, &Lr.cbq, xln_, Q_);
        bt::flash_attention_forward(Q_, Lr.K_enc, Lr.V_enc, /*d_mask=*/nullptr,
                                    H, /*causal=*/false, attn_);
        brolm::detail::linear_batched(Lr.cWo, &Lr.cbo, attn_, proj_);
        bt::add_inplace(x_, proj_);

        // feed-forward (ReLU)
        brolm::detail::layernorm_batched(x_, Lr.ff_ln_w, Lr.ff_ln_b, xln_, eps);
        brolm::detail::linear_batched(Lr.fc1_w, &Lr.fc1_b, xln_, h1_);
        bt::relu_forward_batched(h1_, h1_);
        brolm::detail::linear_batched(Lr.fc2_w, &Lr.fc2_b, h1_, proj_);
        bt::add_inplace(x_, proj_);
    }

    brolm::detail::layernorm_batched(x_, dec_ln_w_, dec_ln_b_, xn_, eps);

    // Tied lm_head over the LAST position only: view its (1, D) row, project
    // through the shared embedding to (1, vocab).
    const std::size_t es = dtype_bytes(xn_.dtype);
    void* last_row = static_cast<char*>(xn_.data) +
                     static_cast<std::size_t>(T - 1) *
                         static_cast<std::size_t>(D) * es;
    bt::Tensor x_last = bt::Tensor::view(xn_.device, last_row, 1, D, xn_.dtype);
    brolm::detail::linear_batched(token_embed_, /*bias=*/nullptr, x_last, logits);
    if (final_logits_bias_.size() > 0)
        bt::add_inplace(logits, final_logits_bias_);
}

}  // namespace brolm::nllb
