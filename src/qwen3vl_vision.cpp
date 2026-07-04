#include "brolm/qwen3vl_vision.h"

#include "brolm/detail/compute.h"
#include "brolm/detail/device.h"
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
#include <vector>

namespace brolm::qwen3vl {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

using st::upload_compute_checked;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("qwen3vl::VisionTower: " + msg);
}

const st::TensorView& need(const std::vector<const st::File*>& shards,
                           const std::string& key) {
    for (const st::File* f : shards) {
        if (const auto* v = f->find(key)) return *v;
    }
    throw std::runtime_error("qwen3vl::VisionTower: missing tensor '" + key + "'");
}

// Download a compute-dtype tensor to a host FP32 vector. Same rationale as
// qwen35::VisionTower: a few tight inner loops (RoPE, attention, merger
// shuffle) are implemented directly in FP32 on the host — small matrices,
// single image.
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
    if (t.dtype == bt::Dtype::FP32) {
        return t.to_host_vector();
    }
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

// Upload an FP32 host buffer to a compute-dtype tensor on the caller-supplied
// device. Resizes `dst` to (rows, cols, compute_dtype) before writing.
void from_host_f32(const std::vector<float>& src, int rows, int cols,
                   bt::Device dev, bt::Tensor& dst) {
    const int expected = rows * cols;
    if (static_cast<int>(src.size()) != expected) {
        fail("from_host_f32: size mismatch");
    }
    const bt::Dtype dt = compute_dtype();
    detail::resize_like(dst, rows, cols, dt, dev);
    if (dev == bt::Device::CPU) {
        if (dt == bt::Dtype::FP32) {
            std::copy(src.begin(), src.end(), dst.host_f32_mut());
            return;
        }
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
    if (dt == bt::Dtype::FP32) {
        dst = bt::Tensor::from_host(src.data(), rows, cols);
        return;
    }
    fail("from_host_f32: unsupported GPU compute dtype");
}

// Build per-token (h_pos, w_pos) streams over the patch grid AFTER the
// merge-aware permutation HF applies. Identical to qwen35's
// build_vision_position_ids.
void build_vision_position_ids(int grid_t, int grid_h, int grid_w, int merge,
                               std::vector<int>& h_out,
                               std::vector<int>& w_out) {
    if (grid_h % merge != 0 || grid_w % merge != 0) {
        fail("grid_h and grid_w must be multiples of merge_size");
    }
    const int N_per_t = grid_h * grid_w;
    const int N = grid_t * N_per_t;
    h_out.resize(static_cast<std::size_t>(N));
    w_out.resize(static_cast<std::size_t>(N));

    const int Hb = grid_h / merge;
    const int Wb = grid_w / merge;
    for (int t = 0; t < grid_t; ++t) {
        int idx = t * N_per_t;
        for (int hb = 0; hb < Hb; ++hb) {
            for (int wb = 0; wb < Wb; ++wb) {
                for (int mh = 0; mh < merge; ++mh) {
                    for (int mw = 0; mw < merge; ++mw) {
                        const int h = hb * merge + mh;
                        const int w = wb * merge + mw;
                        h_out[static_cast<std::size_t>(idx)] = h;
                        w_out[static_cast<std::size_t>(idx)] = w;
                        ++idx;
                    }
                }
            }
        }
    }
}

}  // namespace

// ─── ctor / dtor ───────────────────────────────────────────────────────────

VisionTower::VisionTower(const Qwen3VLConfig::Vision& cfg, int text_hidden_size)
    : cfg_(cfg), text_hidden_(text_hidden_size) {
    if (cfg_.depth <= 0) fail("vision.depth must be positive");
    if (cfg_.hidden_size <= 0 || cfg_.num_heads <= 0 ||
        cfg_.hidden_size % cfg_.num_heads != 0) {
        fail("vision.hidden_size must be a positive multiple of num_heads");
    }
    const int head_dim = cfg_.hidden_size / cfg_.num_heads;
    if (head_dim % 4 != 0) {
        // head_dim halves into (rot_h, rot_w); each half must hold pair-rotation
        // dims, so head_dim/2 must itself be even.
        fail("vision head_dim must be a multiple of 4");
    }
    if (cfg_.intermediate_size <= 0) fail("vision.intermediate_size must be positive");
    if (cfg_.spatial_merge_size <= 0) fail("vision.spatial_merge_size must be positive");
    if (cfg_.patch_size <= 0 || cfg_.temporal_patch_size <= 0 || cfg_.in_channels <= 0) {
        fail("vision patch/channel dims must be positive");
    }
    if (cfg_.out_hidden_size != text_hidden_size) {
        fail("vision.out_hidden_size must equal text.hidden_size");
    }
    const int side = static_cast<int>(std::round(std::sqrt(
        static_cast<double>(cfg_.num_position_embeddings))));
    if (side * side != cfg_.num_position_embeddings) {
        fail("vision.num_position_embeddings must be a perfect square");
    }
    for (int idx : cfg_.deepstack_visual_indexes) {
        if (idx < 0 || idx >= cfg_.depth) {
            fail("deepstack_visual_indexes entry out of range [0, depth)");
        }
    }
    blocks_.resize(static_cast<std::size_t>(cfg_.depth));
    deepstack_mergers_.resize(cfg_.deepstack_visual_indexes.size());
}

VisionTower::~VisionTower() = default;

// ─── load_weights ──────────────────────────────────────────────────────────

void VisionTower::load_weights(const st::File& f, const std::string& prefix) {
    load_weights({&f}, prefix);
}

void VisionTower::load_weights(const std::vector<const st::File*>& shards,
                               const std::string& prefix) {
    if (shards.empty()) fail("load_weights: no safetensors shards");
    const int D     = cfg_.hidden_size;
    const int F     = cfg_.intermediate_size;
    const int C     = cfg_.in_channels;
    const int P     = cfg_.patch_size;
    const int Tps   = cfg_.temporal_patch_size;
    const int m     = cfg_.spatial_merge_size;
    const int Npos  = cfg_.num_position_embeddings;
    const int Hmrg  = D * m * m;
    const int Hout  = cfg_.out_hidden_size;
    const int qkv   = 3 * D;

    // patch_embed.proj : shape on disk is (D, C, Tps, P, P); flatten to
    // (D, C*Tps*P*P).
    upload_compute_checked(need(shards, prefix + "patch_embed.proj.weight"),
                           D, C * Tps * P * P, patch_W_,
                           "patch_embed.proj.weight");
    upload_compute_checked(need(shards, prefix + "patch_embed.proj.bias"),
                           D, 1, patch_b_, "patch_embed.proj.bias");

    // pos_embed table (num_pos, D).
    upload_compute_checked(need(shards, prefix + "pos_embed.weight"),
                           Npos, D, pos_table_, "pos_embed.weight");

    // `depth` blocks.
    for (int i = 0; i < cfg_.depth; ++i) {
        const std::string p = prefix + "blocks." + std::to_string(i) + ".";
        Block& B = blocks_[static_cast<std::size_t>(i)];

        upload_compute_checked(need(shards, p + "norm1.weight"), D, 1, B.norm1_g, "norm1.weight");
        upload_compute_checked(need(shards, p + "norm1.bias"),   D, 1, B.norm1_b, "norm1.bias");
        upload_compute_checked(need(shards, p + "norm2.weight"), D, 1, B.norm2_g, "norm2.weight");
        upload_compute_checked(need(shards, p + "norm2.bias"),   D, 1, B.norm2_b, "norm2.bias");

        upload_compute_checked(need(shards, p + "attn.qkv.weight"), qkv, D, B.Wqkv, "attn.qkv.weight");
        upload_compute_checked(need(shards, p + "attn.qkv.bias"),   qkv, 1, B.bqkv, "attn.qkv.bias");

        upload_compute_checked(need(shards, p + "attn.proj.weight"), D, D, B.Wproj, "attn.proj.weight");
        upload_compute_checked(need(shards, p + "attn.proj.bias"),   D, 1, B.bproj, "attn.proj.bias");

        upload_compute_checked(need(shards, p + "mlp.linear_fc1.weight"), F, D, B.fc1_W, "mlp.linear_fc1.weight");
        upload_compute_checked(need(shards, p + "mlp.linear_fc1.bias"),   F, 1, B.fc1_b, "mlp.linear_fc1.bias");
        upload_compute_checked(need(shards, p + "mlp.linear_fc2.weight"), D, F, B.fc2_W, "mlp.linear_fc2.weight");
        upload_compute_checked(need(shards, p + "mlp.linear_fc2.bias"),   D, 1, B.fc2_b, "mlp.linear_fc2.bias");
    }

    // Main merger. Pre-shuffle LayerNorm on the un-merged D=hidden_size.
    upload_compute_checked(need(shards, prefix + "merger.norm.weight"), D, 1,
                           main_merger_.norm_g, "merger.norm.weight");
    upload_compute_checked(need(shards, prefix + "merger.norm.bias"), D, 1,
                           main_merger_.norm_b, "merger.norm.bias");
    upload_compute_checked(need(shards, prefix + "merger.linear_fc1.weight"), Hmrg, Hmrg,
                           main_merger_.fc1_W, "merger.linear_fc1.weight");
    upload_compute_checked(need(shards, prefix + "merger.linear_fc1.bias"), Hmrg, 1,
                           main_merger_.fc1_b, "merger.linear_fc1.bias");
    upload_compute_checked(need(shards, prefix + "merger.linear_fc2.weight"), Hout, Hmrg,
                           main_merger_.fc2_W, "merger.linear_fc2.weight");
    upload_compute_checked(need(shards, prefix + "merger.linear_fc2.bias"), Hout, 1,
                           main_merger_.fc2_b, "merger.linear_fc2.bias");

    // DeepStack mergers. Post-shuffle LayerNorm — the on-disk norm shape is
    // still just the width it normalises, D*m² here (verified against the
    // checkpoint: deepstack_merger_list.k.norm.weight has shape [D*m²], vs
    // the main merger's [D]).
    for (std::size_t k = 0; k < deepstack_mergers_.size(); ++k) {
        const std::string p =
            prefix + "deepstack_merger_list." + std::to_string(k) + ".";
        Merger& M = deepstack_mergers_[k];
        upload_compute_checked(need(shards, p + "norm.weight"), Hmrg, 1, M.norm_g, "deepstack norm.weight");
        upload_compute_checked(need(shards, p + "norm.bias"),   Hmrg, 1, M.norm_b, "deepstack norm.bias");
        upload_compute_checked(need(shards, p + "linear_fc1.weight"), Hmrg, Hmrg, M.fc1_W, "deepstack linear_fc1.weight");
        upload_compute_checked(need(shards, p + "linear_fc1.bias"),   Hmrg, 1, M.fc1_b, "deepstack linear_fc1.bias");
        upload_compute_checked(need(shards, p + "linear_fc2.weight"), Hout, Hmrg, M.fc2_W, "deepstack linear_fc2.weight");
        upload_compute_checked(need(shards, p + "linear_fc2.bias"),   Hout, 1, M.fc2_b, "deepstack linear_fc2.bias");
    }
}

// ─── helpers: rotary tables, pos embed ─────────────────────────────────────

void VisionTower::build_rotary_tables_(int grid_t, int grid_h, int grid_w) {
    const int D        = cfg_.hidden_size;
    const int H        = cfg_.num_heads;
    const int head_dim = D / H;
    const int rot_full = head_dim / 2;
    const int half      = rot_full / 2;

    std::vector<int> hpos, wpos;
    build_vision_position_ids(grid_t, grid_h, grid_w, cfg_.spatial_merge_size,
                              hpos, wpos);
    const int N = static_cast<int>(hpos.size());

    std::vector<float> inv_freq(static_cast<std::size_t>(half));
    const float theta = 10000.0f;
    for (int i = 0; i < half; ++i) {
        const float p = static_cast<float>(2 * i) / static_cast<float>(rot_full);
        inv_freq[static_cast<std::size_t>(i)] = 1.0f / std::pow(theta, p);
    }

    cos_host_.assign(static_cast<std::size_t>(N) * head_dim, 1.0f);
    sin_host_.assign(static_cast<std::size_t>(N) * head_dim, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < rot_full; ++c) {
            const int axis = (c / half) % 2;
            const int k    = c % half;
            const int pos  = (axis == 0) ? hpos[static_cast<std::size_t>(n)]
                                         : wpos[static_cast<std::size_t>(n)];
            const float ang = static_cast<float>(pos) * inv_freq[static_cast<std::size_t>(k)];
            const float cv = std::cos(ang);
            const float sv = std::sin(ang);
            cos_host_[static_cast<std::size_t>(n) * head_dim + c] = cv;
            sin_host_[static_cast<std::size_t>(n) * head_dim + c] = sv;
            cos_host_[static_cast<std::size_t>(n) * head_dim + rot_full + c] = cv;
            sin_host_[static_cast<std::size_t>(n) * head_dim + rot_full + c] = sv;
        }
    }
}

void VisionTower::apply_rope_(const bt::Tensor& in, bt::Tensor& out) {
    const int D        = cfg_.hidden_size;
    const int H        = cfg_.num_heads;
    const int head_dim = D / H;
    const int half     = head_dim / 2;
    const int N        = in.rows;

    std::vector<float> X = to_host_f32(in);
    std::vector<float> Y(X.size());
    for (int n = 0; n < N; ++n) {
        const float* cosrow = &cos_host_[static_cast<std::size_t>(n) * head_dim];
        const float* sinrow = &sin_host_[static_cast<std::size_t>(n) * head_dim];
        for (int h = 0; h < H; ++h) {
            const std::size_t base = static_cast<std::size_t>(n) * D + static_cast<std::size_t>(h) * head_dim;
            for (int c = 0; c < half; ++c) {
                const float x0 = X[base + static_cast<std::size_t>(c)];
                const float x1 = X[base + static_cast<std::size_t>(c + half)];
                Y[base + static_cast<std::size_t>(c)]        = x0 * cosrow[c]        + (-x1) * sinrow[c];
                Y[base + static_cast<std::size_t>(c + half)] = x1 * cosrow[c + half] +   x0  * sinrow[c + half];
            }
        }
    }

    from_host_f32(Y, N, D, in.device, out);
}

void VisionTower::build_pos_embed_(int grid_t, int grid_h, int grid_w) {
    const int D       = cfg_.hidden_size;
    const int m       = cfg_.spatial_merge_size;
    const int side    = static_cast<int>(std::round(std::sqrt(
        static_cast<double>(cfg_.num_position_embeddings))));
    const int N_per_t = grid_h * grid_w;
    const int N       = grid_t * N_per_t;

    auto linspace = [&](int n, std::vector<float>& out) {
        out.resize(static_cast<std::size_t>(n));
        if (n == 1) { out[0] = 0.0f; return; }
        const float step = static_cast<float>(side - 1) / static_cast<float>(n - 1);
        for (int i = 0; i < n; ++i) out[static_cast<std::size_t>(i)] = static_cast<float>(i) * step;
    };
    std::vector<float> h_grid, w_grid;
    linspace(grid_h, h_grid);
    linspace(grid_w, w_grid);

    std::vector<float> table = to_host_f32(pos_table_);

    std::vector<float> bilin(static_cast<std::size_t>(grid_h) * grid_w * D, 0.0f);
    for (int i = 0; i < grid_h; ++i) {
        const float hf  = h_grid[static_cast<std::size_t>(i)];
        const int   h0  = static_cast<int>(std::floor(hf));
        const int   h1  = std::min(h0 + 1, side - 1);
        const float dh  = hf - static_cast<float>(h0);
        for (int j = 0; j < grid_w; ++j) {
            const float wf = w_grid[static_cast<std::size_t>(j)];
            const int   w0 = static_cast<int>(std::floor(wf));
            const int   w1 = std::min(w0 + 1, side - 1);
            const float dw = wf - static_cast<float>(w0);

            const float w00 = (1.0f - dh) * (1.0f - dw);
            const float w01 = (1.0f - dh) * dw;
            const float w10 = dh * (1.0f - dw);
            const float w11 = dh * dw;

            const float* p00 = &table[(static_cast<std::size_t>(h0) * side + w0) * D];
            const float* p01 = &table[(static_cast<std::size_t>(h0) * side + w1) * D];
            const float* p10 = &table[(static_cast<std::size_t>(h1) * side + w0) * D];
            const float* p11 = &table[(static_cast<std::size_t>(h1) * side + w1) * D];
            float* dst = &bilin[(static_cast<std::size_t>(i) * grid_w + j) * D];
            for (int d = 0; d < D; ++d) {
                dst[d] = w00 * p00[d] + w01 * p01[d] + w10 * p10[d] + w11 * p11[d];
            }
        }
    }

    const int Hb = grid_h / m;
    const int Wb = grid_w / m;
    std::vector<float> reord(static_cast<std::size_t>(N_per_t) * D);
    {
        int idx = 0;
        for (int hb = 0; hb < Hb; ++hb) {
            for (int wb = 0; wb < Wb; ++wb) {
                for (int mh = 0; mh < m; ++mh) {
                    for (int mw = 0; mw < m; ++mw) {
                        const int h = hb * m + mh;
                        const int w = wb * m + mw;
                        const float* src = &bilin[(static_cast<std::size_t>(h) * grid_w + w) * D];
                        std::copy(src, src + D, reord.begin() + static_cast<std::ptrdiff_t>(idx) * D);
                        ++idx;
                    }
                }
            }
        }
    }

    std::vector<float> full(static_cast<std::size_t>(N) * D);
    for (int t = 0; t < grid_t; ++t) {
        std::copy(reord.begin(), reord.end(),
                  full.begin() + static_cast<std::ptrdiff_t>(t) * N_per_t * D);
    }

    from_host_f32(full, N, D, patch_W_.device, pos_embed_);
}

// ─── dense per-head attention ──────────────────────────────────────────────

void VisionTower::dense_attention_(const bt::Tensor& q_in,
                                   const bt::Tensor& k_in,
                                   const bt::Tensor& v_in,
                                   bt::Tensor& out) {
    const int D        = cfg_.hidden_size;
    const int H        = cfg_.num_heads;
    const int head_dim = D / H;
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
            for (int kk = 0; kk < N; ++kk) {
                float s = 0.0f;
                const std::size_t qoff = static_cast<std::size_t>(n) * D + static_cast<std::size_t>(h) * head_dim;
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

// ─── merger ─────────────────────────────────────────────────────────────

void VisionTower::run_merger_(const Merger& M, bool postshuffle_norm,
                              int N, int grid_t, int grid_h, int grid_w,
                              bt::Tensor& out) {
    (void)grid_t;
    const int D    = cfg_.hidden_size;
    const int m    = cfg_.spatial_merge_size;
    const int Hmrg = D * m * m;
    const float eps = 1.0e-6f;

    const int Nm = N / (m * m);

    auto shuffle = [&](const std::vector<float>& src_host) {
        std::vector<float> merged(static_cast<std::size_t>(Nm) * Hmrg);
        for (int g = 0; g < Nm; ++g) {
            for (int k = 0; k < m * m; ++k) {
                const float* src = &src_host[(static_cast<std::size_t>(g) * (m * m) + k) * D];
                float* dst = &merged[static_cast<std::size_t>(g) * Hmrg + static_cast<std::size_t>(k) * D];
                std::copy(src, src + D, dst);
            }
        }
        return merged;
    };

    bt::Tensor pre_fc1;   // (Nm, Hmrg) input to fc1, at compute dtype
    if (postshuffle_norm) {
        // Shuffle x_ (un-normed) first, then LayerNorm at width Hmrg.
        std::vector<float> x_host = to_host_f32(x_);
        std::vector<float> merged = shuffle(x_host);
        bt::Tensor merged_dev;
        from_host_f32(merged, Nm, Hmrg, x_.device, merged_dev);
        detail::layernorm_batched(merged_dev, M.norm_g, M.norm_b, pre_fc1, eps);
    } else {
        // LayerNorm at width D first, then shuffle.
        bt::Tensor normed;
        detail::layernorm_batched(x_, M.norm_g, M.norm_b, normed, eps);
        std::vector<float> normed_host = to_host_f32(normed);
        std::vector<float> merged = shuffle(normed_host);
        from_host_f32(merged, Nm, Hmrg, x_.device, pre_fc1);
    }

    bt::Tensor mid, act;
    detail::linear_batched(M.fc1_W, &M.fc1_b, pre_fc1, mid);
    bt::gelu_exact_forward(mid, act);
    detail::linear_batched(M.fc2_W, &M.fc2_b, act, out);
    (void)grid_h; (void)grid_w;
}

// ─── forward ───────────────────────────────────────────────────────────────

void VisionTower::forward(const bt::Tensor& patches,
                          int grid_t, int grid_h, int grid_w,
                          bt::Tensor& tokens_out,
                          std::vector<bt::Tensor>& deepstack_tokens_out) {
    if (patch_W_.size() == 0) fail("forward: weights not loaded");
    const float eps = 1.0e-6f;

    const int N = grid_t * grid_h * grid_w;
    if (patches.rows != N) fail("forward: patches.rows does not match grid_t*grid_h*grid_w");

    const int D = cfg_.hidden_size;

    // ── patch embed: (N, C*tps*P²) -> (N, D) via linear (=Conv3d collapsed) ──
    detail::linear_batched(patch_W_, &patch_b_, patches, x_);

    // ── add positional embedding (bilinear-interpolated -> grid_h×grid_w) ──
    build_pos_embed_(grid_t, grid_h, grid_w);
    bt::add_inplace(x_, pos_embed_);

    // ── rotary tables for this image ─────────────────────────────────────
    build_rotary_tables_(grid_t, grid_h, grid_w);

    deepstack_tokens_out.clear();
    deepstack_tokens_out.resize(deepstack_mergers_.size());
    std::size_t next_deepstack = 0;

    // ── `depth` transformer blocks (full attention across the image) ─────
    for (int li = 0; li < cfg_.depth; ++li) {
        Block& B = blocks_[static_cast<std::size_t>(li)];

        detail::layernorm_batched(x_, B.norm1_g, B.norm1_b, xn_, eps);
        detail::linear_batched(B.Wqkv, &B.bqkv, xn_, qkv_);

        {
            std::vector<float> qkv_host = to_host_f32(qkv_);
            std::vector<float> qh(static_cast<std::size_t>(N) * D);
            std::vector<float> kh(static_cast<std::size_t>(N) * D);
            std::vector<float> vh(static_cast<std::size_t>(N) * D);
            for (int n = 0; n < N; ++n) {
                const float* src = &qkv_host[static_cast<std::size_t>(n) * 3 * D];
                std::copy(src,           src + D,     qh.begin() + static_cast<std::ptrdiff_t>(n) * D);
                std::copy(src + D,       src + 2 * D, kh.begin() + static_cast<std::ptrdiff_t>(n) * D);
                std::copy(src + 2 * D,   src + 3 * D, vh.begin() + static_cast<std::ptrdiff_t>(n) * D);
            }
            from_host_f32(qh, N, D, x_.device, q_);
            from_host_f32(kh, N, D, x_.device, k_);
            from_host_f32(vh, N, D, x_.device, v_);
        }

        apply_rope_(q_, q_rope_);
        apply_rope_(k_, k_rope_);

        dense_attention_(q_rope_, k_rope_, v_, attn_out_);

        detail::linear_batched(B.Wproj, &B.bproj, attn_out_, proj_out_);
        bt::add_inplace(x_, proj_out_);

        detail::layernorm_batched(x_, B.norm2_g, B.norm2_b, xn_, eps);
        detail::linear_batched(B.fc1_W, &B.fc1_b, xn_, fc_mid_);
        bt::gelu_forward(fc_mid_, fc_act_);
        detail::linear_batched(B.fc2_W, &B.fc2_b, fc_act_, fc_out_);
        bt::add_inplace(x_, fc_out_);

        // DeepStack extraction: this block's 0-based index feeds a dedicated
        // post-shuffle-norm merger on the CURRENT residual stream.
        if (next_deepstack < cfg_.deepstack_visual_indexes.size() &&
            li == cfg_.deepstack_visual_indexes[next_deepstack]) {
            run_merger_(deepstack_mergers_[next_deepstack],
                       /*postshuffle_norm=*/true,
                       N, grid_t, grid_h, grid_w,
                       deepstack_tokens_out[next_deepstack]);
            ++next_deepstack;
        }
    }
    if (next_deepstack != deepstack_mergers_.size()) {
        fail("forward: not every DeepStack index was reached (depth too small?)");
    }

    // ── main merger (pre-shuffle norm) ────────────────────────────────────
    run_merger_(main_merger_, /*postshuffle_norm=*/false,
               N, grid_t, grid_h, grid_w, tokens_out);
}

}  // namespace brolm::qwen3vl
