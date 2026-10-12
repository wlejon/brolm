#include "brolm/clip_image.h"
#include "brotensor/safetensors.h"
#include "brolm/detail/device.h"
#include "brolm/detail/compute.h"
#include "brolm/detail/weights.h"

#include "brotensor/ops.h"
#include "brotensor/tensor.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace brolm::clip_image {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("clip_image::ImageEncoder: " + msg);
}

}  // namespace

ImageEncoder::ImageEncoder(const ImageEncoderConfig& cfg) : cfg_(cfg) {
    if (cfg_.hidden_dim <= 0 || cfg_.num_heads <= 0 ||
        cfg_.hidden_dim % cfg_.num_heads != 0) {
        fail("hidden_dim must be a positive multiple of num_heads");
    }
    if (cfg_.num_layers <= 0 || cfg_.patch_size <= 0 ||
        cfg_.image_size <= 0 ||
        cfg_.image_size % cfg_.patch_size != 0) {
        fail("image_size must be a positive multiple of patch_size");
    }
    layers_.resize(static_cast<std::size_t>(cfg_.num_layers));
}

ImageEncoder::~ImageEncoder() = default;

// ─── load_weights ──────────────────────────────────────────────────────────

void ImageEncoder::load_weights(const st::File& f,
                                const std::string& prefix,
                                bool pre_layrnorm_alt) {
    const int D = cfg_.hidden_dim;
    const int F = cfg_.intermediate_dim;
    const int C = cfg_.in_channels;
    const int P = cfg_.patch_size;
    const int T = num_tokens(cfg_);

    brolm::detail::weights::SafetensorsSource src({&f}, prefix);

    // patch_embed conv: (D, C*P*P). HF stores (D, C, P, P) which is the
    // same C-contiguous layout brotensor's conv2d expects.
    src.upload_compute_checked("embeddings.patch_embedding.weight",
                               D, C * P * P, patch_W_, "patch_embedding.weight");

    // class_embedding: (D,) stored 1-D in HF. Load as (1, D) row to ease
    // the copy into seq_ row 0.
    src.upload_compute_checked("embeddings.class_embedding",
                               1, D, class_embed_, "class_embedding");

    src.upload_compute_checked("embeddings.position_embedding.weight",
                               T, D, position_embed_, "position_embedding");

    // pre_layrnorm is the upstream HF typo. Accept either spelling.
    const bool has_typo = src.has("pre_layrnorm.weight");
    const bool has_alt  = pre_layrnorm_alt && src.has("pre_layernorm.weight");
    const std::string pre = has_typo ? "pre_layrnorm." :
                            (has_alt ? "pre_layernorm." : std::string());
    if (pre.empty()) {
        fail("missing pre_layrnorm/pre_layernorm weight/bias");
    }
    src.upload_compute_checked(pre + "weight", D, 1, pre_g_, "pre_ln.weight");
    src.upload_compute_checked(pre + "bias",   D, 1, pre_b_, "pre_ln.bias");

    for (int i = 0; i < cfg_.num_layers; ++i) {
        const std::string p = "encoder.layers." + std::to_string(i) + ".";
        Layer& L = layers_[static_cast<std::size_t>(i)];

        src.upload_compute_checked(p + "layer_norm1.weight", D, 1, L.ln1_g, "ln1.weight");
        src.upload_compute_checked(p + "layer_norm1.bias",   D, 1, L.ln1_b, "ln1.bias");

        src.upload_compute_checked(p + "self_attn.q_proj.weight", D, D, L.Wq, "q_proj.W");
        src.upload_compute_checked(p + "self_attn.q_proj.bias",   D, 1, L.bq, "q_proj.b");
        src.upload_compute_checked(p + "self_attn.k_proj.weight", D, D, L.Wk, "k_proj.W");
        src.upload_compute_checked(p + "self_attn.k_proj.bias",   D, 1, L.bk, "k_proj.b");
        src.upload_compute_checked(p + "self_attn.v_proj.weight", D, D, L.Wv, "v_proj.W");
        src.upload_compute_checked(p + "self_attn.v_proj.bias",   D, 1, L.bv, "v_proj.b");
        src.upload_compute_checked(p + "self_attn.out_proj.weight", D, D, L.Wo, "out_proj.W");
        src.upload_compute_checked(p + "self_attn.out_proj.bias",   D, 1, L.bo, "out_proj.b");

        src.upload_compute_checked(p + "layer_norm2.weight", D, 1, L.ln2_g, "ln2.weight");
        src.upload_compute_checked(p + "layer_norm2.bias",   D, 1, L.ln2_b, "ln2.bias");

        src.upload_compute_checked(p + "mlp.fc1.weight", F, D, L.fc1_W, "fc1.W");
        src.upload_compute_checked(p + "mlp.fc1.bias",   F, 1, L.fc1_b, "fc1.b");
        src.upload_compute_checked(p + "mlp.fc2.weight", D, F, L.fc2_W, "fc2.W");
        src.upload_compute_checked(p + "mlp.fc2.bias",   D, 1, L.fc2_b, "fc2.b");
    }

    src.upload_compute_checked("post_layernorm.weight",
                               D, 1, post_g_, "post_ln.weight");
    src.upload_compute_checked("post_layernorm.bias",
                               D, 1, post_b_, "post_ln.bias");
}

// ─── forward ───────────────────────────────────────────────────────────────

namespace {

// Rows [row, row + rows) of a (R, cols) tensor as a non-owning view.
bt::Tensor rows_view(const bt::Tensor& t, int row, int rows) {
    const std::size_t off = static_cast<std::size_t>(row) *
                            static_cast<std::size_t>(t.cols) *
                            static_cast<std::size_t>(bt::dtype_size_bytes(t.dtype));
    return bt::Tensor::view(t.device, static_cast<char*>(t.data) + off,
                            rows, t.cols, t.dtype);
}

}  // namespace

void ImageEncoder::forward(const bt::Tensor& pixels, bt::Tensor& cls_out) {
    if (patch_W_.size() == 0) fail("forward: weights not loaded");

    const int D = cfg_.hidden_dim;
    const int H = cfg_.num_heads;
    const int P = cfg_.patch_size;
    const int C = cfg_.in_channels;
    const int S = cfg_.image_size;
    const int G = S / P;                   // grid side (16 for 224/14)
    const int T = 1 + G * G;               // 257 tokens
    const float eps = cfg_.layer_norm_eps;
    const int N = pixels.rows;             // images in the batch
    if (N <= 0 || pixels.cols != C * S * S) {
        fail("forward: pixels must be (N, " + std::to_string(C * S * S) + "), N >= 1");
    }

    // Every op below either works row by row over the (N*T, D) residual
    // stream — layer norms, the activation, the residual adds, the linear
    // layers (a row's dot products do not depend on the other rows) — or is
    // run image by image (patch conv, attention core), so image n of a batch
    // takes the arithmetic it takes alone.

    // ── patch embed: (1, 3*S*S) -> (1, D*G*G) via stride-P conv2d, per image ─
    // No bias on patch_embedding in HF CLIP.
    detail::resize_like(patch_nchw_, N, D * G * G, compute_dtype(), pixels.device);
    for (int n = 0; n < N; ++n) {
        bt::conv2d_forward(rows_view(pixels, n, 1), patch_W_, /*bias=*/nullptr,
                               /*N=*/1, C, S, S, D, P, P,
                               /*stride_h=*/P, /*stride_w=*/P,
                               /*pad_h=*/0, /*pad_w=*/0,
                               /*dil_h=*/1, /*dil_w=*/1,
                               patch_one_);
        bt::copy_d2d(patch_one_, 0, patch_nchw_, n * D * G * G, D * G * G);
    }

    // ── reshape NCHW -> sequence: (N, D*G*G) -> (N*G*G, D) ─────────────────
    // Reuse ln_out_ as the holding buffer for the patch sequences — they are
    // copied into seq_ in the next step.
    bt::nchw_to_sequence(patch_nchw_, N, /*C=*/D, /*H=*/G, /*W=*/G, ln_out_);

    // ── build (N*T, D): per image [CLS; patches] + position_embed ──────────
    detail::resize_like(seq_, N * T, D, compute_dtype(), pixels.device);
    for (int n = 0; n < N; ++n) {
        // Row n*T = class_embedding; rows n*T+1 .. n*T+T-1 = its patches.
        bt::copy_d2d(class_embed_, 0, seq_, n * T * D, D);
        bt::copy_d2d(ln_out_, n * (T - 1) * D, seq_, (n * T + 1) * D, (T - 1) * D);
        bt::Tensor img = rows_view(seq_, n * T, T);
        bt::add_inplace(img, position_embed_);
    }

    // ── pre-LN over the full sequence ──────────────────────────────────────
    detail::layernorm_batched(
        seq_, pre_g_, pre_b_, ln_out_, eps);
    seq_ = ln_out_.clone();

    // ── transformer layers (non-causal, biased Q/K/V/O, cfg_.activation MLP) ─
    // Attention: Q/K/V and the output projection run over all N*T rows; the
    // softmax core runs per image, so no image attends to another's tokens.
    // (flash_attention_qkvo_forward is exactly these steps for one image.)
    detail::resize_like(attn_, N * T, D, compute_dtype(), pixels.device);
    for (auto& L : layers_) {
        detail::layernorm_batched(
            seq_, L.ln1_g, L.ln1_b, ln_out_, eps);

        detail::linear_batched(L.Wq, &L.bq, ln_out_, q_);
        detail::linear_batched(L.Wk, &L.bk, ln_out_, k_);
        detail::linear_batched(L.Wv, &L.bv, ln_out_, v_);
        for (int n = 0; n < N; ++n) {
            bt::Tensor o = rows_view(attn_, n * T, T);
            bt::flash_attention_forward(rows_view(q_, n * T, T),
                                        rows_view(k_, n * T, T),
                                        rows_view(v_, n * T, T),
                                        /*d_mask=*/nullptr, H, /*causal=*/false, o);
        }
        detail::linear_batched(L.Wo, &L.bo, attn_, proj_out_);
        bt::add_inplace(seq_, proj_out_);

        detail::layernorm_batched(
            seq_, L.ln2_g, L.ln2_b, ln_out_, eps);
        detail::linear_batched(L.fc1_W, &L.fc1_b, ln_out_, ffn_mid_);
        clip::activation_forward(cfg_.activation, ffn_mid_, ffn_act_);
        detail::linear_batched(L.fc2_W, &L.fc2_b, ffn_act_, ffn_out_);
        bt::add_inplace(seq_, ffn_out_);
    }

    // ── post-LN of each image's CLS row (row n*T) ──────────────────────────
    // HF CLIPVisionTransformer applies post_layernorm to the *pooled* CLS
    // output only. Copy the CLS rows out, LN those N×D rows, emit.
    detail::resize_like(cls_out, N, D, compute_dtype(), seq_.device);
    for (int n = 0; n < N; ++n) {
        bt::copy_d2d(seq_, n * T * D, cls_out, n * D, D);
    }
    // In-place LN on cls_out by writing through ln_out_.
    detail::layernorm_batched(
        cls_out, post_g_, post_b_, ln_out_, eps);
    cls_out = ln_out_.clone();
}

}  // namespace brolm::clip_image
