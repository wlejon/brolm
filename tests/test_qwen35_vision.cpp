// Tests for brolm::qwen35::VisionTower.
//
// Two paths:
//   1. Synthetic — build a tiny vision tower (depth 2, hidden 32, num_heads 4,
//      merge 2, patch 16, tps 2, 16×16 reference pos grid) by writing a one-
//      shot in-memory safetensors file to a temp path with random-ish weights,
//      then run forward on a fake patch grid. Asserts shape, finiteness, and
//      a non-trivial output norm.
//   2. Real-checkpoint — gated on BROLM_QWEN35_DIR. Opens the real
//      0.8B safetensors, loads `model.visual.*`, runs a 224×224 ones-image
//      through the Stage 2a preprocessor + vision tower, and asserts the
//      output shape matches num_image_tokens × text.hidden_size with a
//      finite mean-abs in a reasonable range.

#include "brolm/qwen35_config.h"
#include "brolm/qwen35_preprocessor.h"
#include "brolm/qwen35_vision.h"
#include "brolm/detail/compute.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

namespace bt = brotensor;
namespace st = brotensor::safetensors;
namespace q  = brolm::qwen35;

// Build a tiny Qwen35Config::Vision suitable for synthetic testing.
q::Qwen35Config::Vision make_tiny_cfg(int& text_hidden) {
    q::Qwen35Config::Vision v;
    v.depth                   = 2;
    v.hidden_size             = 32;
    v.num_heads               = 4;          // head_dim = 8
    v.intermediate_size       = 64;
    v.in_channels             = 3;
    v.patch_size              = 16;
    v.temporal_patch_size     = 2;
    v.spatial_merge_size      = 2;
    v.out_hidden_size         = 48;         // arbitrary "text_hidden"
    v.num_position_embeddings = 256;        // 16×16 reference grid
    text_hidden               = v.out_hidden_size;
    return v;
}

// Helper: append a fp16-encoded random tensor entry to the writer entries.
struct OwnedEntry {
    std::string                name;
    st::Dtype                  dtype;
    std::vector<int64_t>       shape;
    std::vector<std::uint16_t> data_fp16;   // owns storage
};

void push_random(std::vector<OwnedEntry>& entries,
                 std::vector<st::WriteEntry>& view_out,
                 const std::string& name,
                 std::initializer_list<int64_t> shape,
                 std::mt19937& rng, float scale = 0.02f) {
    OwnedEntry e;
    e.name  = name;
    e.dtype = st::Dtype::F16;
    e.shape.assign(shape.begin(), shape.end());
    std::size_t n = 1;
    for (auto s : shape) n *= static_cast<std::size_t>(s);
    e.data_fp16.resize(n);
    std::normal_distribution<float> dist(0.0f, scale);
    for (std::size_t i = 0; i < n; ++i) {
        e.data_fp16[i] = bt::fp32_to_fp16_bits(dist(rng));
    }
    entries.push_back(std::move(e));
    (void)view_out;  // we'll build view_out after entries finalises (stable addrs).
}

void push_ones_fp16(std::vector<OwnedEntry>& entries,
                    const std::string& name,
                    std::initializer_list<int64_t> shape) {
    OwnedEntry e;
    e.name  = name;
    e.dtype = st::Dtype::F16;
    e.shape.assign(shape.begin(), shape.end());
    std::size_t n = 1;
    for (auto s : shape) n *= static_cast<std::size_t>(s);
    e.data_fp16.assign(n, bt::fp32_to_fp16_bits(1.0f));
    entries.push_back(std::move(e));
}

void push_zeros_fp16(std::vector<OwnedEntry>& entries,
                     const std::string& name,
                     std::initializer_list<int64_t> shape) {
    OwnedEntry e;
    e.name  = name;
    e.dtype = st::Dtype::F16;
    e.shape.assign(shape.begin(), shape.end());
    std::size_t n = 1;
    for (auto s : shape) n *= static_cast<std::size_t>(s);
    e.data_fp16.assign(n, bt::fp32_to_fp16_bits(0.0f));
    entries.push_back(std::move(e));
}

void build_tiny_safetensors(const std::string& path,
                            const q::Qwen35Config::Vision& v) {
    std::mt19937 rng(0xC0FFEEu);
    std::vector<OwnedEntry> entries;
    std::vector<st::WriteEntry> dummy;

    const int D    = v.hidden_size;
    const int F    = v.intermediate_size;
    const int C    = v.in_channels;
    const int P    = v.patch_size;
    const int Tps  = v.temporal_patch_size;
    const int m    = v.spatial_merge_size;
    const int Npos = v.num_position_embeddings;
    const int Hmrg = D * m * m;
    const int Hout = v.out_hidden_size;
    const int qkv  = 3 * D;

    push_random(entries, dummy, "model.visual.patch_embed.proj.weight",
                {D, C, Tps, P, P}, rng, 0.02f);
    push_zeros_fp16(entries, "model.visual.patch_embed.proj.bias", {D});
    push_random(entries, dummy, "model.visual.pos_embed.weight", {Npos, D}, rng, 0.02f);

    for (int i = 0; i < v.depth; ++i) {
        const std::string p = "model.visual.blocks." + std::to_string(i) + ".";
        push_ones_fp16 (entries, p + "norm1.weight", {D});
        push_zeros_fp16(entries, p + "norm1.bias",   {D});
        push_ones_fp16 (entries, p + "norm2.weight", {D});
        push_zeros_fp16(entries, p + "norm2.bias",   {D});
        push_random(entries, dummy, p + "attn.qkv.weight",  {qkv, D}, rng, 0.04f);
        push_zeros_fp16(entries, p + "attn.qkv.bias",  {qkv});
        push_random(entries, dummy, p + "attn.proj.weight", {D, D}, rng, 0.04f);
        push_zeros_fp16(entries, p + "attn.proj.bias", {D});
        push_random(entries, dummy, p + "mlp.linear_fc1.weight", {F, D}, rng, 0.04f);
        push_zeros_fp16(entries, p + "mlp.linear_fc1.bias", {F});
        push_random(entries, dummy, p + "mlp.linear_fc2.weight", {D, F}, rng, 0.04f);
        push_zeros_fp16(entries, p + "mlp.linear_fc2.bias", {D});
    }
    push_ones_fp16 (entries, "model.visual.merger.norm.weight", {D});
    push_zeros_fp16(entries, "model.visual.merger.norm.bias",   {D});
    push_random(entries, dummy, "model.visual.merger.linear_fc1.weight", {Hmrg, Hmrg}, rng, 0.02f);
    push_zeros_fp16(entries, "model.visual.merger.linear_fc1.bias", {Hmrg});
    push_random(entries, dummy, "model.visual.merger.linear_fc2.weight", {Hout, Hmrg}, rng, 0.02f);
    push_zeros_fp16(entries, "model.visual.merger.linear_fc2.bias", {Hout});

    // Finalise pointer view now that `entries` storage is stable.
    std::vector<st::WriteEntry> view;
    view.reserve(entries.size());
    for (auto& e : entries) {
        st::WriteEntry we;
        we.name      = e.name;
        we.dtype     = e.dtype;
        we.shape     = e.shape;
        we.host_data = e.data_fp16.data();
        we.bytes     = e.data_fp16.size() * sizeof(std::uint16_t);
        view.push_back(std::move(we));
    }
    st::write_file(path, view);
}

void test_synthetic() {
    int text_hidden = 0;
    const auto cfg = make_tiny_cfg(text_hidden);

    // Build a temp safetensors file in the binary dir (writable).
    const std::string path = "brolm_qwen35_vision_synth.safetensors";
    build_tiny_safetensors(path, cfg);

    q::VisionTower tower(cfg, text_hidden);
    auto f = st::File::open(path);
    tower.load_weights(f);

    // Build fake patches: grid_t=1, grid_h=4, grid_w=4, so N=16 patches and
    // post-merger tokens = 16 / (2*2) = 4.
    const int grid_t = 1, grid_h = 4, grid_w = 4;
    const int N      = grid_t * grid_h * grid_w;
    const int patch_cols = cfg.in_channels * cfg.temporal_patch_size *
                           cfg.patch_size * cfg.patch_size;
    std::vector<float> patch_data(static_cast<std::size_t>(N) * patch_cols);
    std::mt19937 rng(0xBEEFu);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : patch_data) v = dist(rng);

    // Upload at the compute dtype, on the current default device.
    bt::Tensor patches;
    if (brolm::compute_dtype() == bt::Dtype::FP16) {
        std::vector<std::uint16_t> bits(patch_data.size());
        for (std::size_t i = 0; i < patch_data.size(); ++i)
            bits[i] = bt::fp32_to_fp16_bits(patch_data[i]);
        patches = bt::Tensor::from_host_fp16(bits.data(), N, patch_cols);
    } else {
        patches = bt::Tensor::from_host(patch_data.data(), N, patch_cols);
    }

    bt::Tensor out;
    tower.forward(patches, grid_t, grid_h, grid_w, out);

    const int expected_tokens = (grid_h / cfg.spatial_merge_size) *
                                (grid_w / cfg.spatial_merge_size) * grid_t;
    assert(out.rows == expected_tokens);
    assert(out.cols == text_hidden);
    (void)expected_tokens; (void)text_hidden;

    // Pull to FP32 for finiteness + norm checks.
    std::vector<float> hostv;
    if (out.dtype == bt::Dtype::FP32) {
        hostv = out.to_host_vector();
    } else if (out.dtype == bt::Dtype::FP16) {
        auto bits = out.to_host_vector_fp16();
        hostv.resize(bits.size());
        for (std::size_t i = 0; i < bits.size(); ++i) hostv[i] = bt::fp16_bits_to_fp32(bits[i]);
    }
    double sum_abs = 0.0;
    for (float v : hostv) {
        assert(std::isfinite(v));
        sum_abs += std::fabs(static_cast<double>(v));
    }
    const double mean_abs = sum_abs / static_cast<double>(hostv.size());
    assert(mean_abs > 1.0e-6);
    assert(mean_abs < 1.0e3);
    std::printf("[ok] synthetic VisionTower forward: out=(%d,%d), mean_abs=%.5f\n",
                out.rows, out.cols, mean_abs);

    std::remove(path.c_str());
}

void test_real_checkpoint_if_present() {
    const char* dir = std::getenv("BROLM_QWEN35_DIR");
    if (!dir) {
        std::printf("[skip] BROLM_QWEN35_DIR not set (real-checkpoint VisionTower)\n");
        return;
    }
    const std::string cfg_path = std::string(dir) + "/config.json";
    q::Qwen35Config cfg = q::Qwen35Config::load(cfg_path);

    // Open the merged safetensors shard.
    const std::string st_path = std::string(dir) +
        "/model.safetensors-00001-of-00001.safetensors";
    auto f = st::File::open(st_path);

    q::VisionTower tower(cfg.vision, cfg.text.hidden_size);
    tower.load_weights(f);
    std::printf("[ok] loaded model.visual.* from real 0.8B checkpoint\n");

    // Build a 224×224 ones-image (CHW, [0,1]) and run through preprocessor.
    const int H = 224, W = 224, C = 3;
    std::vector<float> img(static_cast<std::size_t>(C) * H * W, 1.0f);

    q::PreprocessConfig pcfg;
    pcfg.patch_size          = cfg.vision.patch_size;
    pcfg.temporal_patch_size = cfg.vision.temporal_patch_size;
    pcfg.merge_size          = cfg.vision.spatial_merge_size;
    auto pre = q::preprocess_image(img.data(), H, W, pcfg);
    std::printf("[ok] preprocess: grid=(1,%d,%d), num_image_tokens=%d\n",
                pre.grid_h, pre.grid_w, pre.num_image_tokens());

    // Upload patches at compute dtype.
    const float* patch_src = pre.patches.host_f32();
    const int N        = pre.patches.rows;
    const int patch_cols = pre.patches.cols;
    bt::Tensor patches_dev;
    if (brolm::compute_dtype() == bt::Dtype::FP16) {
        std::vector<std::uint16_t> bits(static_cast<std::size_t>(N) * patch_cols);
        for (std::size_t i = 0; i < bits.size(); ++i)
            bits[i] = bt::fp32_to_fp16_bits(patch_src[i]);
        patches_dev = bt::Tensor::from_host_fp16(bits.data(), N, patch_cols);
    } else {
        patches_dev = bt::Tensor::from_host(patch_src, N, patch_cols);
    }

    bt::Tensor out;
    tower.forward(patches_dev, pre.grid_t, pre.grid_h, pre.grid_w, out);
    assert(out.rows == pre.num_image_tokens());
    assert(out.cols == cfg.text.hidden_size);

    std::vector<float> hostv;
    if (out.dtype == bt::Dtype::FP32) {
        hostv = out.to_host_vector();
    } else {
        auto bits = out.to_host_vector_fp16();
        hostv.resize(bits.size());
        for (std::size_t i = 0; i < bits.size(); ++i) hostv[i] = bt::fp16_bits_to_fp32(bits[i]);
    }
    double sum_abs = 0.0;
    for (float v : hostv) {
        assert(std::isfinite(v));
        sum_abs += std::fabs(static_cast<double>(v));
    }
    const double mean_abs = sum_abs / static_cast<double>(hostv.size());
    assert(mean_abs > 1.0e-4);
    assert(mean_abs < 1.0e2);
    std::printf("[ok] real VisionTower forward: out=(%d,%d), mean_abs=%.5f\n",
                out.rows, out.cols, mean_abs);
}

}  // namespace

int main() {
    test_synthetic();
    test_real_checkpoint_if_present();
    return 0;
}
