// Tests for brolm::qwen3vl::VisionTower.
//
// Two paths:
//   1. Synthetic — build a tiny vision tower (depth 6, hidden 32, num_heads 4,
//      merge 2, patch 16, tps 2, deepstack indexes [1,3]) by writing a one-
//      shot in-memory safetensors file with random-ish weights, then run
//      forward on a fake patch grid. Asserts main + DeepStack output shapes,
//      finiteness, and non-trivial output norms.
//   2. Real-checkpoint — gated on BROLM_QWEN3VL_DIR.

#include "brolm/qwen3vl_config.h"
#include "brolm/qwen3vl_preprocessor.h"
#include "brolm/qwen3vl_vision.h"
#include "brolm/detail/compute.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace {

namespace bt = brotensor;
namespace st = brotensor::safetensors;
namespace q  = brolm::qwen3vl;

q::Qwen3VLConfig::Vision make_tiny_cfg(int& text_hidden) {
    q::Qwen3VLConfig::Vision v;
    v.depth                   = 6;
    v.hidden_size             = 32;
    v.num_heads               = 4;          // head_dim = 8
    v.intermediate_size       = 64;
    v.in_channels             = 3;
    v.patch_size              = 16;
    v.temporal_patch_size     = 2;
    v.spatial_merge_size      = 2;
    v.out_hidden_size         = 48;         // arbitrary "text_hidden"
    v.num_position_embeddings = 256;        // 16×16 reference grid
    v.deepstack_visual_indexes = {1, 3};
    text_hidden               = v.out_hidden_size;
    return v;
}

struct OwnedEntry {
    std::string                name;
    st::Dtype                  dtype;
    std::vector<int64_t>       shape;
    std::vector<std::uint16_t> data_fp16;
};

void push_random(std::vector<OwnedEntry>& entries,
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
                            const q::Qwen3VLConfig::Vision& v) {
    std::mt19937 rng(0xC0FFEEu);
    std::vector<OwnedEntry> entries;

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

    push_random(entries, "model.visual.patch_embed.proj.weight",
               {D, C, Tps, P, P}, rng, 0.02f);
    push_zeros_fp16(entries, "model.visual.patch_embed.proj.bias", {D});
    push_random(entries, "model.visual.pos_embed.weight", {Npos, D}, rng, 0.02f);

    for (int i = 0; i < v.depth; ++i) {
        const std::string p = "model.visual.blocks." + std::to_string(i) + ".";
        push_ones_fp16 (entries, p + "norm1.weight", {D});
        push_zeros_fp16(entries, p + "norm1.bias",   {D});
        push_ones_fp16 (entries, p + "norm2.weight", {D});
        push_zeros_fp16(entries, p + "norm2.bias",   {D});
        push_random(entries, p + "attn.qkv.weight",  {qkv, D}, rng, 0.04f);
        push_zeros_fp16(entries, p + "attn.qkv.bias",  {qkv});
        push_random(entries, p + "attn.proj.weight", {D, D}, rng, 0.04f);
        push_zeros_fp16(entries, p + "attn.proj.bias", {D});
        push_random(entries, p + "mlp.linear_fc1.weight", {F, D}, rng, 0.04f);
        push_zeros_fp16(entries, p + "mlp.linear_fc1.bias", {F});
        push_random(entries, p + "mlp.linear_fc2.weight", {D, F}, rng, 0.04f);
        push_zeros_fp16(entries, p + "mlp.linear_fc2.bias", {D});
    }
    // Main merger (pre-shuffle norm, width D).
    push_ones_fp16 (entries, "model.visual.merger.norm.weight", {D});
    push_zeros_fp16(entries, "model.visual.merger.norm.bias",   {D});
    push_random(entries, "model.visual.merger.linear_fc1.weight", {Hmrg, Hmrg}, rng, 0.02f);
    push_zeros_fp16(entries, "model.visual.merger.linear_fc1.bias", {Hmrg});
    push_random(entries, "model.visual.merger.linear_fc2.weight", {Hout, Hmrg}, rng, 0.02f);
    push_zeros_fp16(entries, "model.visual.merger.linear_fc2.bias", {Hout});

    // DeepStack mergers (post-shuffle norm, width Hmrg).
    for (std::size_t k = 0; k < v.deepstack_visual_indexes.size(); ++k) {
        const std::string p =
            "model.visual.deepstack_merger_list." + std::to_string(k) + ".";
        push_ones_fp16 (entries, p + "norm.weight", {Hmrg});
        push_zeros_fp16(entries, p + "norm.bias",   {Hmrg});
        push_random(entries, p + "linear_fc1.weight", {Hmrg, Hmrg}, rng, 0.02f);
        push_zeros_fp16(entries, p + "linear_fc1.bias", {Hmrg});
        push_random(entries, p + "linear_fc2.weight", {Hout, Hmrg}, rng, 0.02f);
        push_zeros_fp16(entries, p + "linear_fc2.bias", {Hout});
    }

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

void check_finite_nontrivial(const bt::Tensor& out, int expected_rows, int expected_cols,
                             const char* label) {
    assert(out.rows == expected_rows);
    assert(out.cols == expected_cols);
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
    assert(mean_abs > 1.0e-6);
    assert(mean_abs < 1.0e3);
    std::printf("[ok] %s: out=(%d,%d), mean_abs=%.5f\n",
                label, out.rows, out.cols, mean_abs);
}

void test_synthetic() {
    int text_hidden = 0;
    const auto cfg = make_tiny_cfg(text_hidden);

    const std::string path = "brolm_qwen3vl_vision_synth.safetensors";
    build_tiny_safetensors(path, cfg);

    q::VisionTower tower(cfg, text_hidden);
    auto f = st::File::open(path);
    tower.load_weights(f);

    const int grid_t = 1, grid_h = 4, grid_w = 4;
    const int N      = grid_t * grid_h * grid_w;
    const int patch_cols = cfg.in_channels * cfg.temporal_patch_size *
                           cfg.patch_size * cfg.patch_size;
    std::vector<float> patch_data(static_cast<std::size_t>(N) * patch_cols);
    std::mt19937 rng(0xBEEFu);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : patch_data) v = dist(rng);

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
    std::vector<bt::Tensor> deepstack_out;
    tower.forward(patches, grid_t, grid_h, grid_w, out, deepstack_out);

    const int expected_tokens = (grid_h / cfg.spatial_merge_size) *
                                (grid_w / cfg.spatial_merge_size) * grid_t;
    check_finite_nontrivial(out, expected_tokens, text_hidden, "main merger");

    assert(deepstack_out.size() == cfg.deepstack_visual_indexes.size());
    for (std::size_t k = 0; k < deepstack_out.size(); ++k) {
        check_finite_nontrivial(deepstack_out[k], expected_tokens, text_hidden,
                                "deepstack merger");
    }

    std::remove(path.c_str());
}

void test_real_checkpoint_if_present() {
    namespace fs = std::filesystem;
    const char* dir = std::getenv("BROLM_QWEN3VL_DIR");
    if (!dir) {
        std::printf("[skip] BROLM_QWEN3VL_DIR not set (real-checkpoint VisionTower)\n");
        return;
    }
    try {
        const std::string cfg_path = std::string(dir) + "/config.json";
        q::Qwen3VLConfig cfg = q::Qwen3VLConfig::load(cfg_path);

        // Discover every model*.safetensors shard (HF's sharded checkpoints
        // split model.visual.* across more than one file for larger models).
        std::vector<fs::path> shard_paths;
        for (const auto& entry : fs::directory_iterator(dir)) {
            const std::string name = entry.path().filename().string();
            if (name.rfind("model", 0) == 0 && entry.path().extension() == ".safetensors")
                shard_paths.push_back(entry.path());
        }
        std::sort(shard_paths.begin(), shard_paths.end());
        if (shard_paths.empty()) {
            std::printf("[skip] no model*.safetensors shard found in %s\n", dir);
            return;
        }
        std::vector<st::File> shards;
        shards.reserve(shard_paths.size());
        for (const auto& p : shard_paths) shards.emplace_back(st::File::open(p.string()));
        std::vector<const st::File*> shard_ptrs;
        shard_ptrs.reserve(shards.size());
        for (const auto& f : shards) shard_ptrs.push_back(&f);

        q::VisionTower tower(cfg.vision, cfg.text.hidden_size);
        tower.load_weights(shard_ptrs);
        std::printf("[ok] loaded model.visual.* (incl. DeepStack mergers) from real checkpoint "
                    "(%zu shard(s))\n", shard_ptrs.size());

        const int H = 224, W = 224, C = 3;
        std::vector<float> img(static_cast<std::size_t>(C) * H * W, 1.0f);

        q::PreprocessConfig pcfg;
        pcfg.patch_size          = cfg.vision.patch_size;
        pcfg.temporal_patch_size = cfg.vision.temporal_patch_size;
        pcfg.merge_size          = cfg.vision.spatial_merge_size;
        auto pre = q::preprocess_image(img.data(), H, W, pcfg);
        std::printf("[ok] preprocess: grid=(1,%d,%d), num_image_tokens=%d\n",
                    pre.grid_h, pre.grid_w, pre.num_image_tokens());

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
        std::vector<bt::Tensor> deepstack_out;
        tower.forward(patches_dev, pre.grid_t, pre.grid_h, pre.grid_w, out, deepstack_out);
        check_finite_nontrivial(out, pre.num_image_tokens(), cfg.text.hidden_size,
                               "real main merger");
        assert(deepstack_out.size() == cfg.vision.deepstack_visual_indexes.size());
        for (auto& d : deepstack_out) {
            check_finite_nontrivial(d, pre.num_image_tokens(), cfg.text.hidden_size,
                                    "real deepstack merger");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "qwen3vl_vision real-checkpoint threw: %s\n", e.what());
        std::abort();
    }
}

}  // namespace

int main() {
    test_synthetic();
    test_real_checkpoint_if_present();
    return 0;
}
