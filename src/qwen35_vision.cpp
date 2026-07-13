#include "brolm/qwen35_vision.h"

#include "brolm/detail/compute.h"
#include "brolm/detail/device.h"
#include "brolm/detail/weights.h"
#include "brotensor/ops.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brolm::qwen35 {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

using st::upload_compute_checked;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("qwen35::VisionTower: " + msg);
}

const st::TensorView& need(const st::File& f, const std::string& key) {
    const auto* v = f.find(key);
    if (!v) throw std::runtime_error("qwen35::VisionTower: missing tensor '" + key + "'");
    return *v;
}

// Download a compute-dtype tensor to a host FP32 vector. The vision tower has
// a few tight inner loops (RoPE, attention) that we implement directly in
// FP32 on the host for clarity / portability — small matrices, single image.
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
    // GPU resident — round-trip through device→host via to_host_vector*.
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
    // GPU path: stage via host tensor + clone-to-device.
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

}  // namespace

// ─── ctor / dtor ───────────────────────────────────────────────────────────

VisionTower::VisionTower(const Qwen35Config::Vision& cfg, int text_hidden_size)
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
    // num_position_embeddings must be a perfect square — HF assumes a square
    // reference grid for bilinear interpolation.
    const int side = static_cast<int>(std::round(std::sqrt(
        static_cast<double>(cfg_.num_position_embeddings))));
    if (side * side != cfg_.num_position_embeddings) {
        fail("vision.num_position_embeddings must be a perfect square");
    }
    blocks_.resize(static_cast<std::size_t>(cfg_.depth));
}

VisionTower::~VisionTower() = default;

// ─── load_weights ──────────────────────────────────────────────────────────

void VisionTower::load_weights(const st::File& f, const std::string& prefix) {
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
    // (D, C*Tps*P*P). brotensor is 2D-only so the safetensors loader requires
    // the caller to pass the flattened (rows, cols) which it then validates
    // against the on-disk byte count regardless of original rank.
    upload_compute_checked(need(f, prefix + "patch_embed.proj.weight"),
                           D, C * Tps * P * P, patch_W_,
                           "patch_embed.proj.weight");
    upload_compute_checked(need(f, prefix + "patch_embed.proj.bias"),
                           D, 1, patch_b_, "patch_embed.proj.bias");

    // pos_embed table (num_pos, D).
    upload_compute_checked(need(f, prefix + "pos_embed.weight"),
                           Npos, D, pos_table_, "pos_embed.weight");

    // 12 blocks.
    for (int i = 0; i < cfg_.depth; ++i) {
        const std::string p = prefix + "blocks." + std::to_string(i) + ".";
        Block& B = blocks_[static_cast<std::size_t>(i)];

        upload_compute_checked(need(f, p + "norm1.weight"), D, 1, B.norm1_g, "norm1.weight");
        upload_compute_checked(need(f, p + "norm1.bias"),   D, 1, B.norm1_b, "norm1.bias");
        upload_compute_checked(need(f, p + "norm2.weight"), D, 1, B.norm2_g, "norm2.weight");
        upload_compute_checked(need(f, p + "norm2.bias"),   D, 1, B.norm2_b, "norm2.bias");

        // qkv: combined (3D, D), biased (3D,).
        upload_compute_checked(need(f, p + "attn.qkv.weight"), qkv, D, B.Wqkv, "attn.qkv.weight");
        upload_compute_checked(need(f, p + "attn.qkv.bias"),   qkv, 1, B.bqkv, "attn.qkv.bias");
        permute_qk_rope_rows_(B.Wqkv, D);
        permute_qk_rope_rows_(B.bqkv, 1);

        upload_compute_checked(need(f, p + "attn.proj.weight"), D, D, B.Wproj, "attn.proj.weight");
        upload_compute_checked(need(f, p + "attn.proj.bias"),   D, 1, B.bproj, "attn.proj.bias");

        upload_compute_checked(need(f, p + "mlp.linear_fc1.weight"), F, D, B.fc1_W, "mlp.linear_fc1.weight");
        upload_compute_checked(need(f, p + "mlp.linear_fc1.bias"),   F, 1, B.fc1_b, "mlp.linear_fc1.bias");
        upload_compute_checked(need(f, p + "mlp.linear_fc2.weight"), D, F, B.fc2_W, "mlp.linear_fc2.weight");
        upload_compute_checked(need(f, p + "mlp.linear_fc2.bias"),   D, 1, B.fc2_b, "mlp.linear_fc2.bias");
    }

    // Patch merger. Pre-shuffle LayerNorm on the un-merged D=hidden_size
    // (verified against checkpoint: merger.norm.weight has shape [D]).
    upload_compute_checked(need(f, prefix + "merger.norm.weight"), D, 1, merger_norm_g_, "merger.norm.weight");
    upload_compute_checked(need(f, prefix + "merger.norm.bias"),   D, 1, merger_norm_b_, "merger.norm.bias");

    upload_compute_checked(need(f, prefix + "merger.linear_fc1.weight"), Hmrg, Hmrg, merger_fc1_W_, "merger.linear_fc1.weight");
    upload_compute_checked(need(f, prefix + "merger.linear_fc1.bias"),   Hmrg, 1, merger_fc1_b_, "merger.linear_fc1.bias");
    upload_compute_checked(need(f, prefix + "merger.linear_fc2.weight"), Hout, Hmrg, merger_fc2_W_, "merger.linear_fc2.weight");
    upload_compute_checked(need(f, prefix + "merger.linear_fc2.bias"),   Hout, 1, merger_fc2_b_, "merger.linear_fc2.bias");
}

// ─── helpers: rotary tables, pos embed ─────────────────────────────────────

// Build per-token (h_pos, w_pos) streams over the patch grid AFTER the
// merge-aware permutation HF applies. See vision_utils.get_vision_position_ids
// — output is a (num_patches, 2) layout flattened to two parallel int vectors.
namespace {

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

    // hpos[h, w] = h ; wpos[h, w] = w ;
    // then reshape (h/m, m, w/m, m) and transpose(1,2) => (h/m, w/m, m, m).
    // Flatten and repeat `grid_t` times.
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

void VisionTower::build_rotary_tables_(int grid_t, int grid_h, int grid_w) {
    const int D        = cfg_.hidden_size;
    const int H        = cfg_.num_heads;
    const int head_dim = D / H;
    // Vision rotary dim is head_dim/2 — HF: Qwen3VLVisionRotaryEmbedding(head_dim//2).
    const int rot_full = head_dim / 2;        // dim arg to VisionRotaryEmbedding
    // half = rot_full/2 frequencies per axis (h, w).
    const int half     = rot_full / 2;        // number of pair frequencies per axis

    std::vector<int> hpos, wpos;
    build_vision_position_ids(grid_t, grid_h, grid_w, cfg_.spatial_merge_size,
                              hpos, wpos);
    const int N = static_cast<int>(hpos.size());

    // inv_freq[i] = 1 / theta^(2i/rot_full), i in [0..half).
    std::vector<float> inv_freq(static_cast<std::size_t>(half));
    const float theta = 10000.0f;
    for (int i = 0; i < half; ++i) {
        const float p = static_cast<float>(2 * i) / static_cast<float>(rot_full);
        inv_freq[static_cast<std::size_t>(i)] = 1.0f / std::pow(theta, p);
    }

    // HF builds rotary_pos_emb of shape (N, rot_full) by:
    //   freqs_per_axis = position_ids[:, :, None] * inv_freq         (N, 2, half)
    //   rotary_pos_emb = freqs_per_axis.flatten(1)                   (N, rot_full)
    //   emb = cat((rotary_pos_emb, rotary_pos_emb), dim=-1)          (N, head_dim)
    //   cos, sin = emb.cos(), emb.sin()
    // So cos/sin in column c for c in [0..rot_full):
    //   axis = (c / half) % 2     (0 => h, 1 => w)
    //   k    = c % half           (frequency index)
    //   pos  = (axis == 0) ? hpos[n] : wpos[n]
    //   angle = pos * inv_freq[k]
    // And columns [rot_full..head_dim) repeat columns [0..rot_full).
    // (N, rot_full): one angle per rotated pair, shared across heads — the
    // layout rope_apply takes. HF's duplicated [rot_full..head_dim) half is
    // redundant once the pairing is explicit.
    cos_host_.assign(static_cast<std::size_t>(N) * rot_full, 1.0f);
    sin_host_.assign(static_cast<std::size_t>(N) * rot_full, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < rot_full; ++c) {
            const int axis = (c / half) % 2;
            const int k    = c % half;
            const int pos  = (axis == 0) ? hpos[static_cast<std::size_t>(n)]
                                         : wpos[static_cast<std::size_t>(n)];
            const float ang = static_cast<float>(pos) * inv_freq[static_cast<std::size_t>(k)];
            cos_host_[static_cast<std::size_t>(n) * rot_full + c] = std::cos(ang);
            sin_host_[static_cast<std::size_t>(n) * rot_full + c] = std::sin(ang);
        }
    }
    cos_dev_ = bt::Tensor::from_host(cos_host_.data(), N, rot_full).to(patch_W_.device);
    sin_dev_ = bt::Tensor::from_host(sin_host_.data(), N, rot_full).to(patch_W_.device);
}

// HF stores each head's rotary dims in rotate_half order — dim c pairs with
// dim c + head_dim/2 — while brotensor's rope_apply rotates ADJACENT pairs
// (2i, 2i+1). Permuting the Q and K output rows of the fused qkv projection at
// load reconciles the two once, for free, instead of per image.
//
// V is deliberately left alone: the permutation is a per-head shuffle of Q's
// and K's feature axis, and a dot product is invariant to shuffling both sides
// the same way. Scores are unchanged, attention output stays in V's basis, and
// attn.proj needs no change.
void VisionTower::permute_qk_rope_rows_(bt::Tensor& W, int cols) {
    const int D        = cfg_.hidden_size;
    const int H        = cfg_.num_heads;
    const int head_dim = D / H;

    std::vector<float> host = to_host_f32(W);                 // (3D, cols)
    const std::size_t block = static_cast<std::size_t>(D) * static_cast<std::size_t>(cols);
    for (int part = 0; part < 2; ++part) {                    // Q, then K
        const auto begin = host.begin() + static_cast<std::ptrdiff_t>(part * block);
        std::vector<float> sub(begin, begin + static_cast<std::ptrdiff_t>(block));
        std::vector<float> perm =
            brolm::detail::weights::detail_::permute_rope_fp32(sub, H, head_dim, cols);
        std::copy(perm.begin(), perm.end(), begin);
    }
    from_host_f32(host, W.rows, cols, W.device, W);
}

void VisionTower::build_pos_embed_(int grid_t, int grid_h, int grid_w) {
    const int D       = cfg_.hidden_size;
    const int m       = cfg_.spatial_merge_size;
    const int side    = static_cast<int>(std::round(std::sqrt(
        static_cast<double>(cfg_.num_position_embeddings))));   // 48 for 0.8B
    const int N_per_t = grid_h * grid_w;
    const int N       = grid_t * N_per_t;

    // 1. Bilinear: for each output (i in [0..grid_h)) on the side-grid sample
    //    point u = i * (side - 1) / (grid_h - 1) if grid_h > 1 else 0 (HF uses
    //    torch.linspace(0, side-1, grid_h)). Same for w.
    auto linspace = [&](int n, std::vector<float>& out) {
        out.resize(static_cast<std::size_t>(n));
        if (n == 1) { out[0] = 0.0f; return; }
        const float step = static_cast<float>(side - 1) / static_cast<float>(n - 1);
        for (int i = 0; i < n; ++i) out[static_cast<std::size_t>(i)] = static_cast<float>(i) * step;
    };
    std::vector<float> h_grid, w_grid;
    linspace(grid_h, h_grid);
    linspace(grid_w, w_grid);

    // Download pos_table_ to host FP32 once.
    std::vector<float> table = to_host_f32(pos_table_);   // size side*side*D

    // 2. Per (h,w), gather 4 corners and weight by bilinear fractions.
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

    // 3. Apply the HF merge-reorder permutation: same one used in
    //    get_vision_bilinear_indices_and_weights via the `reorder` buffer.
    //    Output ordering matches build_vision_position_ids above.
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

    // 4. Replicate grid_t times for the temporal axis.
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
        // O_h[n] = sum_k softmax(Q_h[n]·K_h[k]/sqrt(d)) * V_h[k]
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

// ─── forward ───────────────────────────────────────────────────────────────

void VisionTower::forward(const bt::Tensor& patches,
                          int grid_t, int grid_h, int grid_w,
                          bt::Tensor& tokens_out) {
    if (patch_W_.size() == 0) fail("forward: weights not loaded");
    const int D        = cfg_.hidden_size;
    const int H        = cfg_.num_heads;
    const int head_dim = D / H;
    const int F     = cfg_.intermediate_size;
    const int m     = cfg_.spatial_merge_size;
    const int Hmrg  = D * m * m;
    const int Hout  = cfg_.out_hidden_size;
    const float eps = 1.0e-6f;   // HF LayerNorm default; merger norm also 1e-6

    const int N = grid_t * grid_h * grid_w;
    if (patches.rows != N) fail("forward: patches.rows does not match grid_t*grid_h*grid_w");

    // ── patch embed: (N, C*tps*P²) -> (N, D) via linear (=Conv3d collapsed) ──
    detail::linear_batched(patch_W_, &patch_b_, patches, x_);

    // ── add positional embedding (bilinear-interpolated 48×48 -> grid_h×grid_w)
    build_pos_embed_(grid_t, grid_h, grid_w);
    bt::add_inplace(x_, pos_embed_);

    // ── rotary tables for this image ─────────────────────────────────────
    build_rotary_tables_(grid_t, grid_h, grid_w);

    // ── 12 transformer blocks (full attention across the image) ──────────
    for (auto& B : blocks_) {
        // Pre-attn LayerNorm.
        detail::layernorm_batched(x_, B.norm1_g, B.norm1_b, xn_, eps);

        // qkv projection.
        detail::linear_batched(B.Wqkv, &B.bqkv, xn_, qkv_);

        // Split the fused (N, 3D) projection into Q/K/V without leaving the
        // device — three strided row copies.
        const bt::Dtype dt = compute_dtype();
        detail::resize_like(q_, N, D, dt, x_.device);
        detail::resize_like(k_, N, D, dt, x_.device);
        detail::resize_like(v_, N, D, dt, x_.device);
        bt::copy_d2d_strided(qkv_, 0,     3 * D, q_, 0, D, D, N);
        bt::copy_d2d_strided(qkv_, D,     3 * D, k_, 0, D, D, N);
        bt::copy_d2d_strided(qkv_, 2 * D, 3 * D, v_, 0, D, D, N);

        // Q/K rows were permuted into adjacent-pair order at load, so this is
        // HF's rotate_half rotation. V is untouched.
        bt::rope_apply(q_, cos_dev_, sin_dev_, head_dim, H, q_rope_);
        bt::rope_apply(k_, cos_dev_, sin_dev_, head_dim, H, k_rope_);

        // Full attention across all N patches (single image — the
        // windowed/varlen path is unnecessary).
        if (dt == bt::Dtype::FP16) {
            bt::flash_attention_forward(q_rope_, k_rope_, v_, /*d_mask=*/nullptr,
                                        H, /*causal=*/false, attn_out_);
        } else {
            // CPU backend (FP32): brotensor's fused attention is FP16-only.
            dense_attention_(q_rope_, k_rope_, v_, attn_out_);
        }

        // out projection.
        detail::linear_batched(B.Wproj, &B.bproj, attn_out_, proj_out_);
        bt::add_inplace(x_, proj_out_);

        // Pre-MLP LayerNorm.
        detail::layernorm_batched(x_, B.norm2_g, B.norm2_b, xn_, eps);
        // fc1.
        detail::linear_batched(B.fc1_W, &B.fc1_b, xn_, fc_mid_);
        // gelu_pytorch_tanh — brotensor's gelu_forward is exactly this (tanh
        // approximation; see ops.h comment "tanh-approx gelu_forward").
        bt::gelu_forward(fc_mid_, fc_act_);
        detail::linear_batched(B.fc2_W, &B.fc2_b, fc_act_, fc_out_);
        bt::add_inplace(x_, fc_out_);
    }

    // ── patch merger ─────────────────────────────────────────────────────
    // 1. LayerNorm on each of the N tokens at the un-merged width D.
    detail::layernorm_batched(x_, merger_norm_g_, merger_norm_b_, xn_, eps);

    // 2. Spatial-shuffle reshape: group every 2×2 patch window into one token
    //    of dim D*m². With our token ordering already (hb, wb, mh, mw, t)
    //    [see build_vision_position_ids], the m×m block is a contiguous run
    //    of m*m=4 tokens — concatenating those rows is the shuffle.
    const int Nm = N / (m * m);
    std::vector<float> xn_host = to_host_f32(xn_);
    std::vector<float> merged(static_cast<std::size_t>(Nm) * Hmrg);
    for (int g = 0; g < Nm; ++g) {
        for (int k = 0; k < m * m; ++k) {
            const float* src = &xn_host[(static_cast<std::size_t>(g) * (m * m) + k) * D];
            float* dst = &merged[static_cast<std::size_t>(g) * Hmrg + static_cast<std::size_t>(k) * D];
            std::copy(src, src + D, dst);
        }
    }
    from_host_f32(merged, Nm, Hmrg, x_.device, merged_);

    // 3. fc1 (Hmrg, Hmrg) -> GELU (exact, nn.GELU default) -> fc2 (Hout, Hmrg).
    detail::linear_batched(merger_fc1_W_, &merger_fc1_b_, merged_, merger_mid_);
    bt::gelu_exact_forward(merger_mid_, merger_act_);
    detail::linear_batched(merger_fc2_W_, &merger_fc2_b_, merger_act_, tokens_out);

    (void)F; (void)Hout;  // silence unused-in-Release warnings if any
}

}  // namespace brolm::qwen35
