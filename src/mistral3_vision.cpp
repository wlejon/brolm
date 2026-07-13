#include "brolm/mistral3_vision.h"

#include "brolm/detail/compute.h"
#include "brolm/detail/device.h"
#include "brolm/detail/weights.h"
#include "brotensor/gguf.h"
#include "brotensor/ops.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace brolm::mistral3 {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;
namespace wt = ::brolm::detail::weights;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("mistral3::VisionTower: " + msg);
}

// Download a compute-dtype tensor to a host FP32 vector. The tower's RoPE and
// attention run directly in FP32 on the host for clarity/portability — small
// matrices, single image. (Mirrors qwen35_vision.cpp.)
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

// Upload an FP32 host buffer to a compute-dtype tensor on `dev`. Resizes `dst`.
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

VisionTower::VisionTower(const Mistral3Config::Vision& cfg) : cfg_(cfg) {
    if (cfg_.hidden_size <= 0 || cfg_.num_attention_heads <= 0 || cfg_.head_dim <= 0) {
        fail("vision hidden_size / num_attention_heads / head_dim must be positive");
    }
    if (cfg_.head_dim * cfg_.num_attention_heads != cfg_.hidden_size) {
        fail("vision head_dim * num_attention_heads must equal hidden_size");
    }
    if (cfg_.head_dim % 4 != 0) {
        // head_dim splits into (h-quarter, w-quarter) for the 2-D RoPE; each
        // half must hold pair-rotation dims, so head_dim must be a multiple of 4.
        fail("vision head_dim must be a multiple of 4");
    }
    if (cfg_.num_hidden_layers <= 0) fail("vision num_hidden_layers must be positive");
    if (cfg_.intermediate_size <= 0) fail("vision intermediate_size must be positive");
    if (cfg_.patch_size <= 0 || cfg_.num_channels <= 0) {
        fail("vision patch_size / num_channels must be positive");
    }
    blocks_.resize(static_cast<std::size_t>(cfg_.num_hidden_layers));
}

VisionTower::~VisionTower() = default;

namespace {
bool starts_with(std::string_view s, std::string_view p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}
}  // namespace

// ─── HF → ggml (mmproj/clip) tensor-name map ────────────────────────────────

std::string mistral3_vision_hf_to_ggml(std::string_view name) {
    if (name == "patch_conv.weight") return "v.patch_embd.weight";
    if (name == "ln_pre.weight")     return "v.pre_ln.weight";

    constexpr std::string_view kPrefix = "transformer.layers.";
    if (!starts_with(name, kPrefix)) return {};
    const auto dot = name.find('.', kPrefix.size());
    if (dot == std::string_view::npos) return {};
    const std::string_view idx = name.substr(kPrefix.size(), dot - kPrefix.size());
    const std::string_view tail = name.substr(dot + 1);

    auto blk = [&](std::string_view suffix) -> std::string {
        std::string out = "v.blk.";
        out.append(idx);
        out.push_back('.');
        out.append(suffix);
        return out;
    };

    if (tail == "attention.q_proj.weight")        return blk("attn_q.weight");
    if (tail == "attention.k_proj.weight")        return blk("attn_k.weight");
    if (tail == "attention.v_proj.weight")        return blk("attn_v.weight");
    if (tail == "attention.o_proj.weight")        return blk("attn_out.weight");
    if (tail == "attention_norm.weight")          return blk("ln1.weight");
    if (tail == "ffn_norm.weight")                return blk("ln2.weight");
    if (tail == "feed_forward.gate_proj.weight")  return blk("ffn_gate.weight");
    if (tail == "feed_forward.up_proj.weight")    return blk("ffn_up.weight");
    if (tail == "feed_forward.down_proj.weight")  return blk("ffn_down.weight");
    return {};
}

// ─── load_weights ──────────────────────────────────────────────────────────

void VisionTower::load_from_(const wt::Source& src) {
    const int D = cfg_.hidden_size;
    const int F = cfg_.intermediate_size;
    const int C = cfg_.num_channels;
    const int P = cfg_.patch_size;

    // patch_conv : Conv2d(C, D, P, P) flattened to (D, C*P*P), bias-free.
    src.upload_compute_checked("patch_conv.weight", D, C * P * P, patch_W_, "patch_conv.weight");
    // ln_pre RMSNorm gamma is a dense-only op operand (no quant kernel).
    src.upload_compute_dequant("ln_pre.weight", D, 1, ln_pre_g_, "ln_pre.weight");

    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        const std::string p = "transformer.layers." + std::to_string(i) + ".";
        Block& B = blocks_[static_cast<std::size_t>(i)];

        src.upload_compute_dequant(p + "attention_norm.weight", D, 1, B.attn_norm_g, "attention_norm.weight");
        // Q/K rows are permuted from HF's rotate_half pairing into brotensor's
        // adjacent-pair pairing, so rope_apply rotates the pairs HF rotates.
        // V is left alone: the permutation shuffles Q's and K's feature axis
        // identically, and a dot product does not care — scores are unchanged
        // and attention output stays in V's basis, so o_proj needs no change.
        src.upload_compute_rope_permuted(p + "attention.q_proj.weight", D, D,
                                         cfg_.num_attention_heads, cfg_.head_dim,
                                         B.q_W, "attention.q_proj.weight");
        src.upload_compute_rope_permuted(p + "attention.k_proj.weight", D, D,
                                         cfg_.num_attention_heads, cfg_.head_dim,
                                         B.k_W, "attention.k_proj.weight");
        src.upload_compute_checked(p + "attention.v_proj.weight", D, D, B.v_W, "attention.v_proj.weight");
        src.upload_compute_checked(p + "attention.o_proj.weight", D, D, B.o_W, "attention.o_proj.weight");

        src.upload_compute_dequant(p + "ffn_norm.weight", D, 1, B.ffn_norm_g, "ffn_norm.weight");
        src.upload_compute_checked(p + "feed_forward.gate_proj.weight", F, D, B.gate_W, "feed_forward.gate_proj.weight");
        src.upload_compute_checked(p + "feed_forward.up_proj.weight",   F, D, B.up_W,   "feed_forward.up_proj.weight");
        src.upload_compute_checked(p + "feed_forward.down_proj.weight", D, F, B.down_W, "feed_forward.down_proj.weight");
    }
}

void VisionTower::load_weights(const st::File& f, const std::string& prefix) {
    wt::SafetensorsSource src({&f}, prefix);
    load_from_(src);
}

void VisionTower::load_weights(const bt::gguf::File& f) {
    wt::GgufSource src({&f}, [](std::string_view n) { return mistral3_vision_hf_to_ggml(n); });
    load_from_(src);
}

// ─── rotary tables (PixtralRotaryEmbedding) ─────────────────────────────────

void VisionTower::build_rotary_tables_(int grid_h, int grid_w) {
    const int head_dim = cfg_.head_dim;
    const int half     = head_dim / 2;
    const int quarter  = head_dim / 4;
    const float theta  = cfg_.rope_theta;
    const int N        = grid_h * grid_w;

    // PixtralRotaryEmbedding builds freqs = theta^(-arange(0,dim,2)/dim) (dim/2
    // values); the h-axis takes the even-indexed freqs and the w-axis the odd.
    //   inv_h[k] = theta^(-(4k)  /dim),   inv_w[k] = theta^(-(4k+2)/dim).
    std::vector<float> inv_h(static_cast<std::size_t>(quarter));
    std::vector<float> inv_w(static_cast<std::size_t>(quarter));
    for (int k = 0; k < quarter; ++k) {
        const float eh = static_cast<float>(4 * k)     / static_cast<float>(head_dim);
        const float ew = static_cast<float>(4 * k + 2) / static_cast<float>(head_dim);
        inv_h[static_cast<std::size_t>(k)] = 1.0f / std::pow(theta, eh);
        inv_w[static_cast<std::size_t>(k)] = 1.0f / std::pow(theta, ew);
    }

    // The angle vector per patch has length half = [h-quarter | w-quarter], and
    // HF duplicates it over head_dim (emb = cat(freqs, freqs)) so the
    // [half, head_dim) columns mirror [0, half). Once the pairing is explicit
    // that mirror is redundant: rope_apply takes one angle per rotated pair,
    // (N, head_dim/2) shared across heads. Patches are row-major (h, w): patch
    // n sits at (ph = n / grid_w, pw = n % grid_w).
    cos_host_.assign(static_cast<std::size_t>(N) * half, 1.0f);
    sin_host_.assign(static_cast<std::size_t>(N) * half, 0.0f);
    for (int n = 0; n < N; ++n) {
        const int ph = n / grid_w;
        const int pw = n % grid_w;
        const std::size_t row = static_cast<std::size_t>(n) * half;
        for (int k = 0; k < quarter; ++k) {
            const float ah = static_cast<float>(ph) * inv_h[static_cast<std::size_t>(k)];
            const float aw = static_cast<float>(pw) * inv_w[static_cast<std::size_t>(k)];
            // h-quarter at column k; w-quarter at column quarter+k.
            cos_host_[row + static_cast<std::size_t>(k)]            = std::cos(ah);
            sin_host_[row + static_cast<std::size_t>(k)]            = std::sin(ah);
            cos_host_[row + static_cast<std::size_t>(quarter + k)]  = std::cos(aw);
            sin_host_[row + static_cast<std::size_t>(quarter + k)]  = std::sin(aw);
        }
    }
    cos_dev_ = bt::Tensor::from_host(cos_host_.data(), N, half).to(ln_pre_g_.device);
    sin_dev_ = bt::Tensor::from_host(sin_host_.data(), N, half).to(ln_pre_g_.device);
}


// ─── dense per-head full attention ──────────────────────────────────────────

void VisionTower::dense_attention_(const bt::Tensor& q_in,
                                   const bt::Tensor& k_in,
                                   const bt::Tensor& v_in,
                                   bt::Tensor& out) {
    const int D        = cfg_.hidden_size;
    const int H        = cfg_.num_attention_heads;
    const int head_dim = cfg_.head_dim;
    const int N        = q_in.rows;
    const float scale  = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> Q = to_host_f32(q_in);
    std::vector<float> K = to_host_f32(k_in);
    std::vector<float> V = to_host_f32(v_in);
    std::vector<float> O(static_cast<std::size_t>(N) * D, 0.0f);

    std::vector<float> scores(static_cast<std::size_t>(N));
    for (int h = 0; h < H; ++h) {
        for (int n = 0; n < N; ++n) {
            float max_s = -std::numeric_limits<float>::infinity();
            const std::size_t qoff = static_cast<std::size_t>(n) * D + static_cast<std::size_t>(h) * head_dim;
            for (int kk = 0; kk < N; ++kk) {
                float s = 0.0f;
                const std::size_t koff = static_cast<std::size_t>(kk) * D + static_cast<std::size_t>(h) * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    s += Q[qoff + static_cast<std::size_t>(d)] * K[koff + static_cast<std::size_t>(d)];
                }
                s *= scale;
                scores[static_cast<std::size_t>(kk)] = s;
                if (s > max_s) max_s = s;
            }
            float sum = 0.0f;
            for (int kk = 0; kk < N; ++kk) {
                scores[static_cast<std::size_t>(kk)] = std::exp(scores[static_cast<std::size_t>(kk)] - max_s);
                sum += scores[static_cast<std::size_t>(kk)];
            }
            const float inv = 1.0f / sum;
            const std::size_t ooff = static_cast<std::size_t>(n) * D + static_cast<std::size_t>(h) * head_dim;
            for (int kk = 0; kk < N; ++kk) {
                const float w = scores[static_cast<std::size_t>(kk)] * inv;
                const std::size_t voff = static_cast<std::size_t>(kk) * D + static_cast<std::size_t>(h) * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    O[ooff + static_cast<std::size_t>(d)] += w * V[voff + static_cast<std::size_t>(d)];
                }
            }
        }
    }
    from_host_f32(O, N, D, q_in.device, out);
}

// ─── forward ───────────────────────────────────────────────────────────────

void VisionTower::forward(const bt::Tensor& patches, int grid_h, int grid_w,
                          bt::Tensor& out) {
    if (patch_W_.size() == 0) fail("forward: weights not loaded");
    const int D = cfg_.hidden_size;
    const int F = cfg_.intermediate_size;
    const int N = grid_h * grid_w;
    if (grid_h <= 0 || grid_w <= 0) fail("forward: grid_h and grid_w must be positive");
    if (patches.rows != N) fail("forward: patches.rows must equal grid_h*grid_w");

    const bt::Dtype dt = compute_dtype();
    const bt::Device dev = bt::default_device();

    // patch embed (Conv2d collapsed to a bias-free Linear) → (N, D).
    detail::linear_batched(patch_W_, /*bias=*/nullptr, patches, x_);

    // ln_pre RMSNorm on the patch embeddings, in place via a swap.
    bt::rms_norm_forward(x_, ln_pre_g_, cfg_.rms_norm_eps, xn_);
    std::swap(x_, xn_);

    // Rotary tables for this image's grid (shared across all blocks).
    build_rotary_tables_(grid_h, grid_w);

    for (auto& B : blocks_) {
        // ── attention sub-block ──────────────────────────────────────────
        bt::rms_norm_forward(x_, B.attn_norm_g, cfg_.rms_norm_eps, xn_);
        detail::linear_batched(B.q_W, /*bias=*/nullptr, xn_, q_);
        detail::linear_batched(B.k_W, /*bias=*/nullptr, xn_, k_);
        detail::linear_batched(B.v_W, /*bias=*/nullptr, xn_, v_);

        // Q/K rows were permuted at load, so this is HF's rotate_half rotation.
        const bt::Dtype dt = compute_dtype();
        bt::rope_apply(q_, cos_dev_, sin_dev_, cfg_.head_dim,
                       cfg_.num_attention_heads, q_rope_);
        bt::rope_apply(k_, cos_dev_, sin_dev_, cfg_.head_dim,
                       cfg_.num_attention_heads, k_rope_);
        if (dt == bt::Dtype::FP16) {
            bt::flash_attention_forward(q_rope_, k_rope_, v_, /*d_mask=*/nullptr,
                                        cfg_.num_attention_heads, /*causal=*/false,
                                        attn_out_);
        } else {
            // CPU backend (FP32): brotensor's fused attention is FP16-only.
            dense_attention_(q_rope_, k_rope_, v_, attn_out_);
        }

        detail::linear_batched(B.o_W, /*bias=*/nullptr, attn_out_, proj_out_);
        bt::add_inplace(x_, proj_out_);

        // ── SiLU-gated MLP sub-block ─────────────────────────────────────
        bt::rms_norm_forward(x_, B.ffn_norm_g, cfg_.rms_norm_eps, xn_);
        detail::linear_batched(B.gate_W, /*bias=*/nullptr, xn_, gate_);
        detail::linear_batched(B.up_W,   /*bias=*/nullptr, xn_, up_);

        // swiglu_forward expects (N, 2F) = concat(gate, up) and computes
        // silu(gate) * up; assemble the concat by row-wise device copies.
        detail::resize_like(swiglu_in_, N, 2 * F, dt, dev);
        for (int r = 0; r < N; ++r) {
            bt::copy_d2d(gate_, r * F, swiglu_in_, r * (2 * F), F);
            bt::copy_d2d(up_,   r * F, swiglu_in_, r * (2 * F) + F, F);
        }
        bt::swiglu_forward(swiglu_in_, mlp_act_);
        detail::linear_batched(B.down_W, /*bias=*/nullptr, mlp_act_, mlp_out_);
        bt::add_inplace(x_, mlp_out_);
    }

    // Pixtral's vision tower returns the raw transformer output (no post-norm;
    // the projector applies its own norm).
    detail::resize_like(out, N, D, dt, dev);
    bt::copy_d2d(x_, 0, out, 0, N * D);
}

}  // namespace brolm::mistral3
