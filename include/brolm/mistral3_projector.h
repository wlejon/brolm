#pragma once

// Mistral 3.1 multimodal projector — port of HF `Mistral3MultiModalProjector`
// (+ its `Mistral3PatchMerger`).
//
// Bridges the Pixtral vision tower to the text decoder: it consumes the tower's
// per-patch hidden states (num_patches, vision_hidden=1024) for one image plus
// the patch grid (grid_h, grid_w), and emits the image-token embeddings
// (num_image_tokens, text_hidden=5120) that splice in for the `[IMG]`
// placeholders. num_image_tokens == (grid_h / merge) * (grid_w / merge).
//
// Pipeline (HF `Mistral3MultiModalProjector.forward`):
//
//   norm          : Mistral3RMSNorm over vision_hidden, eps = text rms_norm_eps,
//                   applied to every patch.
//   patch_merger  : reshape the patches to a (grid_h, grid_w) grid and gather
//                   each non-overlapping merge×merge window into one token of
//                   width vision_hidden*merge² (torch `unfold`, channel-major
//                   then window row-major), then merging_layer:
//                   Linear(vision_hidden*merge² -> vision_hidden, bias=False).
//   linear_1      : Linear(vision_hidden -> text_hidden, bias=projector_bias).
//   act           : exact GELU (projector_hidden_act = "gelu").
//   linear_2      : Linear(text_hidden -> text_hidden, bias=projector_bias).
//
// For Mistral 3.1 `multimodal_projector_bias` is false, so the Linears are
// bias-free; the bias path is supported for completeness. Inference-only, one
// image per call, at the pipeline compute dtype.

#include "brolm/mistral3_config.h"
#include "brotensor/tensor.h"

#include <string>
#include <string_view>

namespace brotensor::safetensors { class File; }
namespace brotensor::gguf { class File; }
namespace brolm::detail::weights { class Source; }

namespace brolm::mistral3 {

// Translate a relative projector tensor name ("linear_1.weight",
// "patch_merger.merging_layer.weight", ...) into the ggml name in a llama.cpp
// mmproj/clip gguf ("mm.1.weight", "mm.patch_merger.weight", ...). Returns ""
// for an unknown name.
std::string mistral3_projector_hf_to_ggml(std::string_view name);

class MultiModalProjector {
public:
    // Construct an unloaded projector from the full Mistral 3.1 config (it needs
    // the vision hidden, text hidden, spatial_merge_size, the projector-bias
    // flag, and the text rms_norm_eps for the pre-merge norm). Throws on a shape
    // misconfig.
    explicit MultiModalProjector(const Mistral3Config& cfg);
    ~MultiModalProjector();

    MultiModalProjector(const MultiModalProjector&)            = delete;
    MultiModalProjector& operator=(const MultiModalProjector&) = delete;
    MultiModalProjector(MultiModalProjector&&) noexcept            = default;
    MultiModalProjector& operator=(MultiModalProjector&&) noexcept = default;

    // Load every `multi_modal_projector.*` tensor from an HF safetensors file.
    // The merging_layer is always bias-free; linear_1/linear_2 carry a bias only
    // when the config's multimodal_projector_bias is set. Throws
    // std::runtime_error ("mistral3::MultiModalProjector: ...") on a missing key
    // or shape/dtype mismatch.
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "multi_modal_projector.");

    // Load from a llama.cpp mmproj/clip gguf (the `mm.*` tensors). See
    // mistral3_projector_hf_to_ggml. The mmproj is bias-free.
    void load_weights(const brotensor::gguf::File& f);

    // Project one image's patch features.
    //   features : (num_patches, vision_hidden) at the compute dtype, in the
    //              tower's row-major (h, w) patch order. num_patches must equal
    //              grid_h * grid_w, with grid_h and grid_w multiples of merge.
    //   out      : (num_image_tokens, text_hidden) image-token embeddings at the
    //              compute dtype; resized as needed.
    void forward(const brotensor::Tensor& features, int grid_h, int grid_w,
                 brotensor::Tensor& out);

private:
    // Shared loader over the weights Source (safetensors or mmproj gguf).
    void load_from_(const brolm::detail::weights::Source& src);

    int vision_hidden_;
    int text_hidden_;
    int merge_;
    bool has_bias_;
    float norm_eps_;

    brotensor::Tensor norm_g_;        // (vision_hidden,) RMSNorm weight
    brotensor::Tensor merge_W_;       // (vision_hidden, vision_hidden*merge²)
    brotensor::Tensor lin1_W_, lin1_b_;  // (text_hidden, vision_hidden) [+bias]
    brotensor::Tensor lin2_W_, lin2_b_;  // (text_hidden, text_hidden)   [+bias]

    // Scratch buffers reused across calls.
    brotensor::Tensor normed_;        // (N, vision_hidden)
    brotensor::Tensor merged_in_;     // (L, vision_hidden*merge²)
    brotensor::Tensor merged_;        // (L, vision_hidden)
    brotensor::Tensor lin1_out_;      // (L, text_hidden)
    brotensor::Tensor act_;           // (L, text_hidden)
};

}  // namespace brolm::mistral3
