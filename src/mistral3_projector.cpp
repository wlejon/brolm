#include "brolm/mistral3_projector.h"

#include "brolm/detail/compute.h"
#include "brolm/detail/device.h"
#include "brolm/detail/weights.h"
#include "brotensor/gguf.h"
#include "brotensor/ops.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace brolm::mistral3 {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;
namespace wt = ::brolm::detail::weights;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("mistral3::MultiModalProjector: " + msg);
}

// Download a compute-dtype tensor to host FP32. (Same helper as the vision
// tower; the patch-merge gather runs on the host — one image, small matrices.)
std::vector<float> to_host_f32(const bt::Tensor& t) {
    const int n = t.size();
    std::vector<float> out(static_cast<std::size_t>(n));
    if (t.device == bt::Device::CPU) {
        if (t.dtype == bt::Dtype::FP32) {
            const float* p = t.host_f32();
            std::copy(p, p + n, out.begin());
            return out;
        }
        if (t.dtype == bt::Dtype::FP16) {
            const std::uint16_t* p = t.host_fp16();
            for (int i = 0; i < n; ++i) out[static_cast<std::size_t>(i)] = bt::fp16_bits_to_fp32(p[i]);
            return out;
        }
        if (t.dtype == bt::Dtype::BF16) {
            const std::uint16_t* p = t.host_bf16();
            for (int i = 0; i < n; ++i) out[static_cast<std::size_t>(i)] = bt::bf16_bits_to_fp32(p[i]);
            return out;
        }
        fail("to_host_f32: unsupported dtype");
    }
    if (t.dtype == bt::Dtype::FP32) return t.to_host_vector();
    if (t.dtype == bt::Dtype::FP16) {
        const auto bits = t.to_host_vector_fp16();
        for (int i = 0; i < n; ++i) out[static_cast<std::size_t>(i)] = bt::fp16_bits_to_fp32(bits[static_cast<std::size_t>(i)]);
        return out;
    }
    if (t.dtype == bt::Dtype::BF16) {
        const auto bits = t.to_host_vector_bf16();
        for (int i = 0; i < n; ++i) out[static_cast<std::size_t>(i)] = bt::bf16_bits_to_fp32(bits[static_cast<std::size_t>(i)]);
        return out;
    }
    fail("to_host_f32: unsupported dtype");
}

void from_host_f32(const std::vector<float>& src, int rows, int cols,
                   bt::Device dev, bt::Tensor& dst) {
    const int expected = rows * cols;
    if (static_cast<int>(src.size()) != expected) fail("from_host_f32: size mismatch");
    const bt::Dtype dt = compute_dtype();
    detail::resize_like(dst, rows, cols, dt, dev);
    if (dev == bt::Device::CPU) {
        if (dt == bt::Dtype::FP32) { std::copy(src.begin(), src.end(), dst.host_f32_mut()); return; }
        if (dt == bt::Dtype::FP16) {
            std::uint16_t* p = dst.host_fp16_mut();
            for (int i = 0; i < expected; ++i) p[i] = bt::fp32_to_fp16_bits(src[static_cast<std::size_t>(i)]);
            return;
        }
        fail("from_host_f32: unsupported CPU compute dtype");
    }
    if (dt == bt::Dtype::FP16) {
        std::vector<std::uint16_t> bits(static_cast<std::size_t>(expected));
        for (int i = 0; i < expected; ++i) bits[static_cast<std::size_t>(i)] = bt::fp32_to_fp16_bits(src[static_cast<std::size_t>(i)]);
        dst = bt::Tensor::from_host_fp16(bits.data(), rows, cols);
        return;
    }
    if (dt == bt::Dtype::FP32) { dst = bt::Tensor::from_host(src.data(), rows, cols); return; }
    fail("from_host_f32: unsupported GPU compute dtype");
}

}  // namespace

// ─── ctor / dtor ───────────────────────────────────────────────────────────

MultiModalProjector::MultiModalProjector(const Mistral3Config& cfg)
    : vision_hidden_(cfg.vision.hidden_size),
      text_hidden_(cfg.text.hidden_size),
      merge_(cfg.spatial_merge_size),
      has_bias_(cfg.multimodal_projector_bias),
      norm_eps_(cfg.text.rms_norm_eps) {
    if (vision_hidden_ <= 0 || text_hidden_ <= 0) fail("vision/text hidden_size must be positive");
    if (merge_ <= 0) fail("spatial_merge_size must be positive");
    if (norm_eps_ <= 0.0f) fail("rms_norm_eps must be positive");
}

MultiModalProjector::~MultiModalProjector() = default;

// ─── HF → ggml (mmproj/clip) tensor-name map ────────────────────────────────

std::string mistral3_projector_hf_to_ggml(std::string_view name) {
    if (name == "norm.weight")                          return "mm.input_norm.weight";
    if (name == "patch_merger.merging_layer.weight")    return "mm.patch_merger.weight";
    if (name == "linear_1.weight")                      return "mm.1.weight";
    if (name == "linear_2.weight")                      return "mm.2.weight";
    return {};
}

// ─── load_weights ──────────────────────────────────────────────────────────

void MultiModalProjector::load_from_(const wt::Source& src) {
    const int d  = vision_hidden_;
    const int T  = text_hidden_;
    const int dm = d * merge_ * merge_;

    // norm RMSNorm gamma is a dense-only op operand.
    src.upload_compute_dequant("norm.weight", d, 1, norm_g_, "norm.weight");
    src.upload_compute_checked("patch_merger.merging_layer.weight", d, dm, merge_W_, "patch_merger.merging_layer.weight");
    src.upload_compute_checked("linear_1.weight", T, d, lin1_W_, "linear_1.weight");
    src.upload_compute_checked("linear_2.weight", T, T, lin2_W_, "linear_2.weight");
    if (has_bias_) {
        src.upload_compute_dequant("linear_1.bias", T, 1, lin1_b_, "linear_1.bias");
        src.upload_compute_dequant("linear_2.bias", T, 1, lin2_b_, "linear_2.bias");
    }
}

void MultiModalProjector::load_weights(const st::File& f, const std::string& prefix) {
    wt::SafetensorsSource src({&f}, prefix);
    load_from_(src);
}

void MultiModalProjector::load_weights(const bt::gguf::File& f) {
    wt::GgufSource src({&f}, [](std::string_view n) { return mistral3_projector_hf_to_ggml(n); });
    load_from_(src);
}

// ─── forward ───────────────────────────────────────────────────────────────

void MultiModalProjector::forward(const bt::Tensor& features, int grid_h,
                                  int grid_w, bt::Tensor& out) {
    if (merge_W_.size() == 0) fail("forward: weights not loaded");
    const int d = vision_hidden_;
    const int m = merge_;
    const int N = grid_h * grid_w;
    if (grid_h <= 0 || grid_w <= 0) fail("forward: grid_h and grid_w must be positive");
    if (grid_h % m != 0 || grid_w % m != 0) fail("forward: grid_h and grid_w must be multiples of spatial_merge_size");
    if (features.rows != N) fail("forward: features.rows must equal grid_h*grid_w");
    if (features.cols != d) fail("forward: features.cols must equal vision hidden_size");

    const bt::Device dev = bt::default_device();

    // 1. Pre-merge RMSNorm over each patch.
    bt::rms_norm_forward(features, norm_g_, norm_eps_, normed_);

    // 2. Patch merge — gather every merge×merge window into one (d*merge²) token
    //    matching torch.nn.functional.unfold's channel-major / window-row-major
    //    layout, output blocks in row-major (oh, ow) order:
    //      feature index f = c*merge² + ki*merge + kj
    //      source patch    = (oh*merge + ki) * grid_w + (ow*merge + kj)
    const int Lh = grid_h / m;
    const int Lw = grid_w / m;
    const int L  = Lh * Lw;
    const int dm = d * m * m;
    std::vector<float> x = to_host_f32(normed_);   // (N, d)
    std::vector<float> g(static_cast<std::size_t>(L) * dm);
    for (int oh = 0; oh < Lh; ++oh) {
        for (int ow = 0; ow < Lw; ++ow) {
            const int l = oh * Lw + ow;
            float* dst = &g[static_cast<std::size_t>(l) * dm];
            for (int c = 0; c < d; ++c) {
                for (int ki = 0; ki < m; ++ki) {
                    for (int kj = 0; kj < m; ++kj) {
                        const int f   = c * (m * m) + ki * m + kj;
                        const int src = (oh * m + ki) * grid_w + (ow * m + kj);
                        dst[f] = x[static_cast<std::size_t>(src) * d + static_cast<std::size_t>(c)];
                    }
                }
            }
        }
    }
    from_host_f32(g, L, dm, dev, merged_in_);

    // merging_layer: (d*merge² -> d), bias-free.
    detail::linear_batched(merge_W_, /*bias=*/nullptr, merged_in_, merged_);

    // 3. linear_1 -> exact GELU -> linear_2.
    detail::linear_batched(lin1_W_, has_bias_ ? &lin1_b_ : nullptr, merged_, lin1_out_);
    bt::gelu_exact_forward(lin1_out_, act_);
    detail::linear_batched(lin2_W_, has_bias_ ? &lin2_b_ : nullptr, act_, out);
}

}  // namespace brolm::mistral3
