#pragma once

// Qwen3.5-VL vision tower — port of HuggingFace `Qwen3VLVisionModel`.
//
// Consumes the flat patch tensor + grid_thw triple produced by
// `brolm::qwen35::preprocess_image` (see qwen35_preprocessor.h) and emits a
// dense `(num_image_tokens, text_hidden_size)` token block ready to be spliced
// in for the `<|image_pad|>` placeholders by the top-level VLM glue.
//
// Architecture (config matches Qwen/Qwen3.5-0.8B):
//
//   patch_embed.proj : nn.Conv3d(C=3, D=768,
//                                k=(tps=2, P=16, P=16),
//                                s=(tps, P, P), bias=True).
//                      Identical math to a flat Linear of shape
//                      (D, C*tps*P*P)=(768, 1536) over the preprocessor's
//                      already-patchified input — that is exactly how the
//                      weight is consumed here (verified against
//                      model.safetensors header: shape [768,3,2,16,16]).
//   pos_embed        : learnable (2304, 768) table laid out as a 48×48 grid.
//                      Bilinear-interpolated to (grid_h, grid_w) per image and
//                      broadcast across grid_t (HF
//                      `get_vision_bilinear_indices_and_weights`).
//   12 transformer blocks (pre-norm residual):
//     x = x + attn(LN(x), rotary_pos_emb, cu_seqlens)
//     x = x + mlp (LN(x))
//   attn   : qkv = Linear(D -> 3D, bias),
//            split into heads (12, head_dim=64);
//            2-D rotary (HF rotate_half form) applied to Q,K with
//            h_pos and w_pos streams over disjoint halves of head_dim;
//            full attention across all patches of the image
//            (cu_seqlens = [0, num_patches] for a single static image);
//            out = Linear(D -> D, bias).
//   mlp    : Linear(D -> 4D, bias) → gelu_pytorch_tanh →
//            Linear(4D -> D, bias).   (Plain, NOT gated SwiGLU — confirmed by
//            linear_fc1.weight shape [3072,768] in the index.)
//   merger : LayerNorm on D=768 (PRE-spatial-shuffle norm — confirmed by
//            merger.norm.weight shape [768] vs the post-shuffle variant which
//            would be [3072]);
//            reshape each merge×merge=2×2 patch window to a single token of
//            dim D*merge²=3072;
//            Linear(3072 -> 3072, bias) → GELU(exact) → Linear(3072 -> 1024,
//            bias) → final token at text_hidden_size.
//
// Inference-only, batch size = 1 image at a time (the preprocessor already
// produces one image per call). Runs at the pipeline compute dtype — FP32 on
// CPU, FP16 on a GPU backend. The full-attention path materialises an (N,N)
// score matrix per head; that is fine at the patch counts typical Qwen3.5-VL
// runs at (e.g. a 224² image gives N=14² × tps_collapse = 196 patches before
// merge, 49 tokens after).

#include "brolm/qwen35_config.h"
#include "brotensor/tensor.h"

#include <string>
#include <vector>

namespace brotensor::safetensors { class File; }

namespace brolm::qwen35 {

class VisionTower {
public:
    // Construct an unloaded tower from `vision_config` plus the text backbone's
    // hidden size (used to verify cfg.out_hidden_size == text_hidden_size, the
    // post-merger output width). Throws std::runtime_error on shape misconfig.
    VisionTower(const Qwen35Config::Vision& cfg, int text_hidden_size);
    ~VisionTower();

    VisionTower(const VisionTower&)            = delete;
    VisionTower& operator=(const VisionTower&) = delete;
    VisionTower(VisionTower&&) noexcept        = default;
    VisionTower& operator=(VisionTower&&) noexcept = default;

    // Load every `model.visual.*` tensor for the configured depth. Throws
    // std::runtime_error with "qwen35::VisionTower: " prefix on missing keys or
    // shape/dtype mismatch.
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "model.visual.");

    // Forward pass for a single image.
    //
    //   patches   : (num_patches, C * temporal_patch_size * P²) at the pipeline
    //               compute dtype, as emitted by
    //               brolm::qwen35::preprocess_image and uploaded by the caller.
    //               num_patches == grid_t * grid_h * grid_w.
    //   grid_t/h/w: shape of the patch grid for this image (grid_t == 1 for
    //               static images).
    //   tokens_out: (num_image_tokens, text_hidden_size) post-merger embeddings
    //               at the compute dtype; resized as needed.
    //               num_image_tokens == grid_t * (grid_h/m) * (grid_w/m),
    //               which is exactly the count of `<|image_pad|>` slots
    //               PreprocessedImage::num_image_tokens() reports for this
    //               image.
    void forward(const brotensor::Tensor& patches,
                 int grid_t, int grid_h, int grid_w,
                 brotensor::Tensor& tokens_out);

    const Qwen35Config::Vision& config() const { return cfg_; }
    int text_hidden_size() const { return text_hidden_; }

private:
    struct Block {
        // Pre-norm + LN-bias.
        brotensor::Tensor norm1_g, norm1_b;
        brotensor::Tensor norm2_g, norm2_b;
        // attn: combined qkv (3D, D), proj (D, D), all biased.
        brotensor::Tensor Wqkv, bqkv;
        brotensor::Tensor Wproj, bproj;
        // MLP: fc1 (4D, D), fc2 (D, 4D), all biased.
        brotensor::Tensor fc1_W, fc1_b;
        brotensor::Tensor fc2_W, fc2_b;
    };

    Qwen35Config::Vision cfg_;
    int text_hidden_;

    // Patch embed (Conv3d collapsed to flat Linear) and learnable pos table.
    brotensor::Tensor patch_W_;          // (D, C*tps*P*P)
    brotensor::Tensor patch_b_;          // (D, 1)
    brotensor::Tensor pos_table_;        // (num_pos=2304, D), host-pinned

    std::vector<Block> blocks_;

    // Patch merger.
    brotensor::Tensor merger_norm_g_, merger_norm_b_;  // (D,)
    brotensor::Tensor merger_fc1_W_, merger_fc1_b_;    // (D*m², D*m²)
    brotensor::Tensor merger_fc2_W_, merger_fc2_b_;    // (out, D*m²)

    // Scratch buffers reused across forward calls.
    brotensor::Tensor x_;                // (N, D) residual stream
    brotensor::Tensor xn_;               // (N, D) post-LN
    brotensor::Tensor qkv_;              // (N, 3D)
    brotensor::Tensor q_, k_, v_;        // (N, D) each
    brotensor::Tensor q_rope_, k_rope_;  // (N, D)
    brotensor::Tensor attn_out_;         // (N, D)
    brotensor::Tensor proj_out_;         // (N, D)
    brotensor::Tensor fc_mid_;           // (N, 4D)
    brotensor::Tensor fc_act_;           // (N, 4D)
    brotensor::Tensor fc_out_;           // (N, D)
    brotensor::Tensor pos_embed_;        // (N, D) per-image bilinear-interp
    brotensor::Tensor merged_;           // (N/m², D*m²)
    brotensor::Tensor merger_mid_;       // (N/m², D*m²)
    brotensor::Tensor merger_act_;       // (N/m², D*m²)

    // Host-side staging for cos/sin tables; on GPU we upload to device-side
    // buffers but the attention path runs scalar-friendly FP32 here.
    std::vector<float> cos_host_, sin_host_;  // (N, head_dim) each

    // Dense per-head attention helper, applied AFTER RoPE on Q/K.
    // Implements: O = softmax((Q · K^T) / sqrt(head_dim)) · V, per head, then
    // concatenates heads back to (N, D). Internally FP32; reads/writes the
    // configured compute dtype via brief host round-trip — simple and correct
    // on every backend brolm supports today (CPU + the eventual GPU port can
    // swap in flash_attention_varlen_forward).
    void dense_attention_(const brotensor::Tensor& q_in,
                          const brotensor::Tensor& k_in,
                          const brotensor::Tensor& v_in,
                          brotensor::Tensor& out);

    // Apply vision rotary embedding (HF rotate_half form, h+w concatenated
    // along head_dim) to one Q or K tensor in place via xn_-style buffer.
    //   in : (N, D), out : (N, D), both compute dtype.
    //   cos/sin tables prebuilt by build_rotary_tables_().
    void apply_rope_(const brotensor::Tensor& in, brotensor::Tensor& out);

    // Build the (N, head_dim) cos/sin tables for this image's (grid_h, grid_w)
    // grid following HF `Qwen3VLVisionModel.forward`. Result lives in
    // cos_host_/sin_host_ as length N*head_dim FP32 vectors.
    void build_rotary_tables_(int grid_t, int grid_h, int grid_w);

    // Bilinear-interpolate pos_table_ from the 48×48 reference grid to the
    // per-image (grid_h, grid_w) grid, replicated `grid_t` times, written into
    // pos_embed_ at the compute dtype. Matches HF
    // `get_vision_bilinear_indices_and_weights`.
    void build_pos_embed_(int grid_t, int grid_h, int grid_w);
};

}  // namespace brolm::qwen35
