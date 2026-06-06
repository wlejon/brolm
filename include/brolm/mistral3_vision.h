#pragma once

// Mistral 3.1 Pixtral vision tower — port of HuggingFace `PixtralVisionModel`
// (the `vision_tower` of `Mistral3ForConditionalGeneration`).
//
// Consumes the flat patch tensor for ONE image (num_patches, C*P*P) and the
// patch grid (grid_h, grid_w), and emits the per-patch hidden states
// (num_patches, hidden_size) at the vision width (1024 for Mistral 3.1). The
// 2-D spatial 2x2 merge and the projection up to the text hidden size are NOT
// done here — they live in the patch-merger / multimodal projector that runs
// on this tower's output (HF `Mistral3MultiModalProjector`), so the tower
// matches HF's `PixtralVisionModel` boundary exactly.
//
// Architecture (config matches Mistral-Small-3.1's vision_config):
//
//   patch_conv : nn.Conv2d(C=3, D=1024, k=(P,P)=(14,14), s=(P,P), bias=False).
//                Identical math to a flat Linear (D, C*P*P)=(1024, 588) over an
//                already-patchified input — that is how the weight is consumed
//                here. Patch input layout is channel-major then spatial:
//                [c0 row-major P*P pixels, c1 ..., c2 ...] per patch.
//   ln_pre     : PixtralRMSNorm(D) applied once after the patch embed.
//   N transformer blocks (pre-norm residual), full (bidirectional) attention
//   across every patch of the image:
//     x = x + attn(rmsnorm(x, attention_norm), rope2d)
//     x = x + mlp (rmsnorm(x, ffn_norm))
//   attn : separate q/k/v/o = Linear(D->D, bias=False), split into heads
//          (16, head_dim=64); 2-D rotary (HF rotate_half form) over disjoint
//          quarters of head_dim for the h- and w- position streams
//          (PixtralRotaryEmbedding, rope_theta=1e4); full attention.
//   mlp  : SiLU-gated SwiGLU — gate_proj/up_proj (D->F), down_proj (F->D),
//          all bias-free: down(silu(gate(x)) * up(x)).
//
// Unlike the Qwen3.5-VL tower there is NO learned position-embedding table
// (positions are pure 2-D RoPE), NO temporal patch axis, and NO in-tower
// merger. Inference-only, one image per call, at the pipeline compute dtype
// (FP32 on CPU, FP16 on a GPU backend). The full-attention path materialises an
// (N,N) score matrix per head; fine at Pixtral patch counts.

#include "brolm/mistral3_config.h"
#include "brotensor/tensor.h"

#include <string>
#include <string_view>
#include <vector>

namespace brotensor::safetensors { class File; }
namespace brotensor::gguf { class File; }
namespace brolm::detail::weights { class Source; }

namespace brolm::mistral3 {

// Translate a relative Pixtral vision tensor name (as used internally, e.g.
// "transformer.layers.3.attention.q_proj.weight") into the ggml name in a
// llama.cpp mmproj/clip gguf ("v.blk.3.attn_q.weight"). Returns "" if the name
// matches no known vision-tower weight. Exposed for callers building their own
// gguf pipelines.
std::string mistral3_vision_hf_to_ggml(std::string_view name);

class VisionTower {
public:
    // Construct an unloaded tower from the Pixtral vision config. Throws
    // std::runtime_error on a shape misconfig (head split, even-quarter
    // head_dim for the 2-D RoPE, positive dims).
    explicit VisionTower(const Mistral3Config::Vision& cfg);
    ~VisionTower();

    VisionTower(const VisionTower&)            = delete;
    VisionTower& operator=(const VisionTower&) = delete;
    VisionTower(VisionTower&&) noexcept            = default;
    VisionTower& operator=(VisionTower&&) noexcept = default;

    // Load every `vision_tower.*` tensor for the configured depth from an HF
    // safetensors file. All Pixtral weights are bias-free; RMSNorms carry only
    // a weight. Throws std::runtime_error ("mistral3::VisionTower: ...") on a
    // missing key or shape/dtype mismatch.
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "vision_tower.");

    // Load from a llama.cpp mmproj/clip gguf (the file Mistral 3.1 ships beside
    // the text gguf). Tensor names follow the ggml `v.*` convention; see
    // mistral3_vision_hf_to_ggml. F16/F32 dense weights land at the compute
    // dtype; an all-F16 mmproj therefore loads (and the tower runs) on CPU too.
    void load_weights(const brotensor::gguf::File& f);

    // Forward pass for a single image.
    //
    //   patches : (num_patches, C * patch_size²) at the compute dtype, in the
    //             channel-major patch layout described above. num_patches must
    //             equal grid_h * grid_w.
    //   grid_h  : patch rows  (image_height  / patch_size).
    //   grid_w  : patch cols  (image_width   / patch_size).
    //   out     : (num_patches, hidden_size) per-patch hidden states at the
    //             compute dtype; resized as needed. Patches stay in row-major
    //             (h, w) order — the projector consumes that ordering.
    void forward(const brotensor::Tensor& patches, int grid_h, int grid_w,
                 brotensor::Tensor& out);

    const Mistral3Config::Vision& config() const { return cfg_; }

private:
    struct Block {
        brotensor::Tensor attn_norm_g;          // (D,) RMSNorm weight
        brotensor::Tensor q_W, k_W, v_W, o_W;   // (D, D) each, bias-free
        brotensor::Tensor ffn_norm_g;           // (D,) RMSNorm weight
        brotensor::Tensor gate_W, up_W;         // (F, D) each, bias-free
        brotensor::Tensor down_W;               // (D, F), bias-free
    };

    Mistral3Config::Vision cfg_;

    brotensor::Tensor patch_W_;   // (D, C*P*P) — Conv2d collapsed to Linear
    brotensor::Tensor ln_pre_g_;  // (D,) RMSNorm weight
    std::vector<Block> blocks_;

    // Scratch buffers reused across forward calls.
    brotensor::Tensor x_;                 // (N, D) residual stream
    brotensor::Tensor xn_;                // (N, D) post-norm
    brotensor::Tensor q_, k_, v_;         // (N, D) each
    brotensor::Tensor q_rope_, k_rope_;   // (N, D)
    brotensor::Tensor attn_out_;          // (N, D)
    brotensor::Tensor proj_out_;          // (N, D)
    brotensor::Tensor gate_, up_;         // (N, F) each
    brotensor::Tensor swiglu_in_;         // (N, 2F) concat(gate, up)
    brotensor::Tensor mlp_act_;           // (N, F) silu(gate)*up
    brotensor::Tensor mlp_out_;           // (N, D)

    // Host-side cos/sin tables for the 2-D rotary, (N, head_dim) each.
    std::vector<float> cos_host_, sin_host_;

    // Shared loader over the container-agnostic weights Source (safetensors or
    // mmproj gguf). Names are relative (no `vision_tower.` prefix); the Source
    // applies the prefix / ggml mapping.
    void load_from_(const brolm::detail::weights::Source& src);

    // Build the per-patch (N, head_dim) cos/sin tables for this image's grid
    // following PixtralRotaryEmbedding: the h-position drives the first quarter
    // of head_dim/2 frequencies and the w-position the second quarter, then the
    // half-table is duplicated for the rotate_half convention.
    void build_rotary_tables_(int grid_h, int grid_w);

    // Apply the 2-D rotary (HF rotate_half form) to one (N, D) Q or K tensor.
    void apply_rope_(const brotensor::Tensor& in, brotensor::Tensor& out);

    // Dense per-head full attention after RoPE: O = softmax(QKᵀ/√d) V per head,
    // concatenated back to (N, D). Internally FP32 on the host — single image,
    // small matrices, correct on every backend.
    void dense_attention_(const brotensor::Tensor& q_in,
                          const brotensor::Tensor& k_in,
                          const brotensor::Tensor& v_in,
                          brotensor::Tensor& out);
};

}  // namespace brolm::mistral3
