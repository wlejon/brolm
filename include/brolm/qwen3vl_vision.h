#pragma once

// Qwen3-VL vision tower — port of HuggingFace `Qwen3VLVisionModel`.
//
// Consumes the flat patch tensor + grid_thw triple produced by
// `brolm::qwen3vl::preprocess_image` (re-exported from qwen35_preprocessor.h
// — see qwen3vl_preprocessor.h) and emits:
//   1. a dense `(num_image_tokens, text_hidden_size)` token block, ready to
//      be spliced in for the `<|image_pad|>` placeholders (identical role to
//      qwen35::VisionTower::forward's output), and
//   2. a DeepStack feature list — one `(num_image_tokens, text_hidden_size)`
//      tensor per entry of `cfg.deepstack_visual_indexes`, meant to be added
//      additively into the first N text-decoder layers (N ==
//      deepstack_visual_indexes.size()) at the same token positions. This is
//      the one architectural addition over qwen35::VisionTower.
//
// Architecture (config matches Qwen/Qwen3-VL-4B-Instruct):
//
//   patch_embed.proj : nn.Conv3d(C=3, D=1024,
//                                k=(tps=2, P=16, P=16),
//                                s=(tps, P, P), bias=True).
//                      Identical math to a flat Linear of shape
//                      (D, C*tps*P*P)=(1024, 1536) over the preprocessor's
//                      already-patchified input — same collapse trick
//                      qwen35::VisionTower uses (verified against the
//                      checkpoint: model.visual.patch_embed.proj.weight has
//                      shape [1024,3,2,16,16]).
//   pos_embed        : learnable (2304, 1024) table laid out as a 48×48 grid.
//                      Bilinear-interpolated to (grid_h, grid_w) per image and
//                      broadcast across grid_t — identical to qwen35.
//   `depth` transformer blocks (pre-norm residual, depth=24 for the 4B):
//     x = x + attn(LN(x), rotary_pos_emb, cu_seqlens)
//     x = x + mlp (LN(x))
//     [if this block's 0-based index is in cfg.deepstack_visual_indexes:
//        deepstack_features.push_back(deepstack_merger[k](x))   -- see below]
//   attn   : qkv = Linear(D -> 3D, bias),
//            split into heads (16, head_dim=64);
//            2-D rotary (HF rotate_half form) applied to Q,K with
//            h_pos and w_pos streams over disjoint halves of head_dim;
//            full attention across all patches of the image
//            (cu_seqlens = [0, num_patches] for a single static image);
//            out = Linear(D -> D, bias).
//   mlp    : Linear(D -> 4D, bias) → gelu_pytorch_tanh →
//            Linear(4D -> D, bias).   (Plain, NOT gated SwiGLU.)
//   merger (main, pre-shuffle norm — `model.visual.merger.*`):
//            LayerNorm on D=hidden_size (PRE-spatial-shuffle;
//            merger.norm.weight has shape [D]);
//            reshape each merge×merge=2×2 patch window to a single token of
//            dim D*merge²;
//            Linear(D*merge² -> D*merge², bias) → GELU(exact) →
//            Linear(D*merge² -> out_hidden_size, bias).
//   DeepStack mergers (`model.visual.deepstack_merger_list.{k}.*`, one per
//            entry of deepstack_visual_indexes): SAME shapes as the main
//            merger, but `use_postshuffle_norm=true` — the LayerNorm runs
//            AFTER the spatial-shuffle reshape, on width D*merge² rather than
//            D. Everything else (fc1/GELU/fc2) is identical in shape to the
//            main merger.
//
// Inference-only, batch size = 1 image at a time. Runs at the pipeline
// compute dtype — FP32 on CPU, FP16 on a GPU backend. The full-attention path
// materialises an (N,N) score matrix per head, same as qwen35::VisionTower —
// fine at the patch counts Qwen3-VL runs at.

#include "brolm/qwen3vl_config.h"
#include "brotensor/tensor.h"

#include <string>
#include <vector>

namespace brotensor::safetensors { class File; }

namespace brolm::qwen3vl {

class VisionTower {
public:
    // Construct an unloaded tower from `vision_config` plus the text
    // backbone's hidden size (used to verify cfg.out_hidden_size ==
    // text_hidden_size). Throws std::runtime_error on shape misconfig.
    VisionTower(const Qwen3VLConfig::Vision& cfg, int text_hidden_size);
    ~VisionTower();

    VisionTower(const VisionTower&)            = delete;
    VisionTower& operator=(const VisionTower&) = delete;
    VisionTower(VisionTower&&) noexcept        = default;
    VisionTower& operator=(VisionTower&&) noexcept = default;

    // Load every `model.visual.*` tensor for the configured depth, including
    // the DeepStack merger list. Throws std::runtime_error with
    // "qwen3vl::VisionTower: " prefix on missing keys or shape/dtype
    // mismatch.
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "model.visual.");

    // Sharded overload: a tensor is resolved by scanning the shards in
    // order, first match wins. Unlike the single-shard Qwen3.5-0.8B
    // checkpoint, larger Qwen3-VL releases may split `model.visual.*` across
    // more than one shard file.
    void load_weights(
        const std::vector<const brotensor::safetensors::File*>& shards,
        const std::string& prefix = "model.visual.");

    // Forward pass for a single image.
    //
    //   patches            : (num_patches, C * temporal_patch_size * P²) at
    //                         the pipeline compute dtype, as emitted by
    //                         brolm::qwen3vl::preprocess_image.
    //                         num_patches == grid_t * grid_h * grid_w.
    //   grid_t/h/w         : shape of the patch grid for this image
    //                         (grid_t == 1 for static images).
    //   tokens_out         : (num_image_tokens, text_hidden_size) post-main-
    //                         merger embeddings, resized as needed.
    //   deepstack_tokens_out: resized to deepstack_visual_indexes.size()
    //                         entries, each (num_image_tokens,
    //                         text_hidden_size) — in the same order as
    //                         cfg.deepstack_visual_indexes (ascending block
    //                         index). Cleared/resized even when the config
    //                         carries no DeepStack indexes (empty vector).
    //   num_image_tokens() == grid_t * (grid_h/m) * (grid_w/m) for both
    //   tokens_out and every deepstack_tokens_out entry.
    void forward(const brotensor::Tensor& patches,
                 int grid_t, int grid_h, int grid_w,
                 brotensor::Tensor& tokens_out,
                 std::vector<brotensor::Tensor>& deepstack_tokens_out);

    const Qwen3VLConfig::Vision& config() const { return cfg_; }
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

    // A patch-merger head: LayerNorm + fc1 + GELU(exact) + fc2. Shared shape
    // for both the main merger (pre-shuffle norm) and every DeepStack merger
    // (post-shuffle norm) — only the norm's application point differs, which
    // `run_merger_` takes as a bool.
    struct Merger {
        brotensor::Tensor norm_g, norm_b;   // (D,) pre-shuffle or (D*m²,) post-shuffle
        brotensor::Tensor fc1_W, fc1_b;     // (D*m², D*m²)
        brotensor::Tensor fc2_W, fc2_b;     // (out_hidden, D*m²)
    };

    Qwen3VLConfig::Vision cfg_;
    int text_hidden_;

    // Patch embed (Conv3d collapsed to flat Linear) and learnable pos table.
    brotensor::Tensor patch_W_;          // (D, C*tps*P*P)
    brotensor::Tensor patch_b_;          // (D, 1)
    brotensor::Tensor pos_table_;        // (num_pos=2304, D), host-pinned

    std::vector<Block> blocks_;

    Merger main_merger_;                       // pre-shuffle norm
    std::vector<Merger> deepstack_mergers_;     // post-shuffle norm, one per index

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
    brotensor::Tensor merged_;           // (N/m², D*m²) shuffled, pre-merger-norm
    brotensor::Tensor merger_norm_out_;  // (N/m², D*m²) or (N, D) depending on placement
    brotensor::Tensor merger_mid_;       // (N/m², D*m²)
    brotensor::Tensor merger_act_;       // (N/m², D*m²)

    // Host-side staging for cos/sin tables; on GPU we upload to device-side
    // buffers but the attention path runs scalar-friendly FP32 here.
    std::vector<float> cos_host_, sin_host_;  // (N, head_dim) each

    // Dense per-head attention helper, applied AFTER RoPE on Q/K.
    void dense_attention_(const brotensor::Tensor& q_in,
                          const brotensor::Tensor& k_in,
                          const brotensor::Tensor& v_in,
                          brotensor::Tensor& out);

    // Apply vision rotary embedding (HF rotate_half form, h+w concatenated
    // along head_dim) to one Q or K tensor in place via xn_-style buffer.
    void apply_rope_(const brotensor::Tensor& in, brotensor::Tensor& out);

    // Build the (N, head_dim) cos/sin tables for this image's (grid_h, grid_w)
    // grid. Result lives in cos_host_/sin_host_.
    void build_rotary_tables_(int grid_t, int grid_h, int grid_w);

    // Bilinear-interpolate pos_table_ from the reference grid to the
    // per-image (grid_h, grid_w) grid, replicated `grid_t` times, written
    // into pos_embed_ at the compute dtype.
    void build_pos_embed_(int grid_t, int grid_h, int grid_w);

    // Run one merger (main or DeepStack) over the CURRENT residual stream
    // `x_` (N rows, D cols), writing an (N/m², out_hidden_size) result into
    // `out`. `postshuffle_norm` selects where the LayerNorm runs:
    //   false (main merger)      : norm on x_ at width D, THEN spatial-shuffle
    //                              to width D*m², then fc1/GELU/fc2.
    //   true  (DeepStack mergers): spatial-shuffle x_ to width D*m² FIRST,
    //                              THEN norm at width D*m², then fc1/GELU/fc2.
    void run_merger_(const Merger& m, bool postshuffle_norm,
                     int N, int grid_t, int grid_h, int grid_w,
                     brotensor::Tensor& out);
};

}  // namespace brolm::qwen3vl
