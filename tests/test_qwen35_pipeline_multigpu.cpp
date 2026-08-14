#include "brolm/qwen35_config.h"
#include "brolm/qwen35_text.h"
#include "brolm/detail/compute.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace q35 = brolm::qwen35;
namespace st  = brotensor::safetensors;
namespace bt  = brotensor;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

namespace {

struct Builder {
    std::string entries;
    std::vector<uint8_t> payload;
    bool first = true;

    void add(const std::string& name, std::vector<int> shape,
             const std::vector<uint16_t>& fp16_bits) {
        std::size_t expected = 1;
        for (int d : shape) expected *= static_cast<std::size_t>(d);
        if (expected != fp16_bits.size()) {
            std::fprintf(stderr, "fixture: shape/data mismatch for %s\n",
                         name.c_str());
            std::abort();
        }
        std::uint64_t start = payload.size();
        const std::uint8_t* bytes =
            reinterpret_cast<const std::uint8_t*>(fp16_bits.data());
        payload.insert(payload.end(), bytes, bytes + fp16_bits.size() * 2);
        std::uint64_t end = payload.size();
        if (!first) entries += ",";
        first = false;
        entries += "\"" + name + "\":{\"dtype\":\"F16\",\"shape\":[";
        for (std::size_t i = 0; i < shape.size(); ++i) {
            if (i) entries += ",";
            entries += std::to_string(shape[i]);
        }
        entries += "],\"data_offsets\":[" + std::to_string(start) + "," +
                   std::to_string(end) + "]}";
    }

    void write(const std::filesystem::path& path) const {
        std::string header = "{" + entries + "}";
        std::uint64_t hdr_size = header.size();
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) std::abort();
        f.write(reinterpret_cast<const char*>(&hdr_size), 8);
        f.write(header.data(), header.size());
        f.write(reinterpret_cast<const char*>(payload.data()),
                static_cast<std::streamsize>(payload.size()));
    }
};

std::vector<uint16_t> fp16_rand(std::size_t n, uint32_t seed) {
    std::vector<uint16_t> out(n);
    uint32_t s = seed * 2654435761u + 1u;
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        float v = (static_cast<float>(s >> 8) / 16777216.0f - 0.5f) * 0.2f;
        out[i] = bt::fp32_to_fp16_bits(v);
    }
    return out;
}

std::vector<uint16_t> fp16_ones(std::size_t n) {
    return std::vector<uint16_t>(n, bt::fp32_to_fp16_bits(1.0f));
}

}  // namespace

int main() {
    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }

    int count = bt::cuda_device_count();
    std::cout << "[test_qwen35_pipeline_multigpu] Detected " << count << " CUDA device(s)" << std::endl;
    if (count < 2) {
        std::cout << "[test_qwen35_pipeline_multigpu] Multi-GPU test requires at least 2 GPUs. Skipping." << std::endl;
        return 0;
    }

    q35::Qwen35Config::Text cfg_single;
    cfg_single.vocab_size          = 64;
    cfg_single.hidden_size         = 64;
    cfg_single.intermediate_size   = 128;
    cfg_single.num_hidden_layers   = 4;
    cfg_single.num_attention_heads = 2;
    cfg_single.num_key_value_heads = 1;
    cfg_single.head_dim            = 32;
    cfg_single.attention_bias      = false;
    cfg_single.attn_output_gate    = true;
    cfg_single.rms_norm_eps        = 1e-6f;
    cfg_single.tie_word_embeddings = true;
    cfg_single.layer_types = {
        q35::LayerType::Linear, q35::LayerType::Linear,
        q35::LayerType::Linear, q35::LayerType::Full,
    };
    cfg_single.full_attention_interval = 4;
    cfg_single.linear_num_key_heads   = 2;
    cfg_single.linear_num_value_heads = 2;
    cfg_single.linear_key_head_dim    = 16;
    cfg_single.linear_value_head_dim  = 16;
    cfg_single.linear_conv_kernel_dim = 4;
    cfg_single.rope.rope_theta            = 10000000.0f;
    cfg_single.rope.partial_rotary_factor = 0.5f;
    cfg_single.rope.mrope_section         = {3, 3, 2};

    q35::Qwen35Config::Text cfg_pipe = cfg_single;
    cfg_pipe.pipeline_devices = { bt::Device::cuda(0), bt::Device::cuda(1) };

    const int V    = cfg_single.vocab_size;
    const int H    = cfg_single.hidden_size;
    const int Fm   = cfg_single.intermediate_size;
    const int HD   = cfg_single.head_dim;
    const int n_q  = cfg_single.num_attention_heads;
    const int n_kv = cfg_single.num_key_value_heads;
    const int q_d  = n_q  * HD;
    const int kv_d = n_kv * HD;

    const int H_lin   = cfg_single.linear_num_value_heads;
    const int D_k     = cfg_single.linear_key_head_dim;
    const int D_v     = cfg_single.linear_value_head_dim;
    const int lin_kv  = H_lin * D_k;
    const int lin_vv  = H_lin * D_v;
    const int lin_qkv = 3 * lin_kv;
    const int conv_ch = lin_qkv;
    const int conv_kd = cfg_single.linear_conv_kernel_dim;

    Builder b;
    uint32_t seed = 123;
    auto R = [&](std::size_t n) { return fp16_rand(n, seed++); };

    b.add("model.embed_tokens.weight", {V, H}, R(static_cast<std::size_t>(V) * H));

    for (int i = 0; i < cfg_single.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        b.add(p + "input_layernorm.weight", {H}, fp16_ones(H));
        b.add(p + "post_attention_layernorm.weight", {H}, fp16_ones(H));
        b.add(p + "mlp.gate_proj.weight", {Fm, H},
              R(static_cast<std::size_t>(Fm) * H));
        b.add(p + "mlp.up_proj.weight", {Fm, H},
              R(static_cast<std::size_t>(Fm) * H));
        b.add(p + "mlp.down_proj.weight", {H, Fm},
              R(static_cast<std::size_t>(H) * Fm));

        if (cfg_single.layer_types[static_cast<std::size_t>(i)] == q35::LayerType::Full) {
            b.add(p + "self_attn.q_proj.weight", {2 * q_d, H},
                  R(static_cast<std::size_t>(2 * q_d) * H));
            b.add(p + "self_attn.k_proj.weight", {kv_d, H},
                  R(static_cast<std::size_t>(kv_d) * H));
            b.add(p + "self_attn.v_proj.weight", {kv_d, H},
                  R(static_cast<std::size_t>(kv_d) * H));
            b.add(p + "self_attn.o_proj.weight", {H, q_d},
                  R(static_cast<std::size_t>(H) * q_d));
            b.add(p + "self_attn.q_norm.weight", {HD}, fp16_ones(HD));
            b.add(p + "self_attn.k_norm.weight", {HD}, fp16_ones(HD));
        } else {
            const std::string lp = p + "linear_attn.";
            b.add(lp + "A_log",   {H_lin}, R(static_cast<std::size_t>(H_lin)));
            b.add(lp + "conv1d.weight",
                  {conv_ch, 1, conv_kd},
                  R(static_cast<std::size_t>(conv_ch) * conv_kd));
            b.add(lp + "dt_bias", {H_lin}, R(static_cast<std::size_t>(H_lin)));
            b.add(lp + "in_proj_a.weight",   {H_lin, H},
                  R(static_cast<std::size_t>(H_lin) * H));
            b.add(lp + "in_proj_b.weight",   {H_lin, H},
                  R(static_cast<std::size_t>(H_lin) * H));
            b.add(lp + "in_proj_qkv.weight", {lin_qkv, H},
                  R(static_cast<std::size_t>(lin_qkv) * H));
            b.add(lp + "in_proj_z.weight",   {lin_vv, H},
                  R(static_cast<std::size_t>(lin_vv) * H));
            b.add(lp + "norm.weight",        {D_v}, fp16_ones(D_v));
            b.add(lp + "out_proj.weight",    {H, lin_vv},
                  R(static_cast<std::size_t>(H) * lin_vv));
        }
    }
    b.add("model.norm.weight", {H}, fp16_ones(H));

    auto path = std::filesystem::temp_directory_path() / "brolm_qwen35_pipeline_test.safetensors";
    b.write(path);

    try {
        auto file = st::File::open(path.string());
        const std::vector<int> seq = {5, 17, 2, 31, 9};
        const int Lseq = static_cast<int>(seq.size());
        std::vector<int64_t> mt(Lseq), mh(Lseq), mw(Lseq);
        for (int i = 0; i < Lseq; ++i) { mt[i] = mh[i] = mw[i] = i; }

        // 1. Single GPU Run on CUDA:0
        std::vector<float> single_logits;
        {
            bt::DeviceScope scope(bt::Device::cuda(0));
            q35::TextModel model(cfg_single);
            model.load_weights(file, "model.");
            auto cache = model.make_cache(16);

            bt::Tensor logits;
            model.forward(seq, mt, mh, mw, cache, logits);
            bt::sync_all();

            single_logits = bdtest::bd_download(logits);
        }

        // 2. Multi-GPU Pipeline Run (dual RTX 4090)
        std::vector<float> pipe_logits;
        {
            q35::TextModel model(cfg_pipe);
            model.load_weights(file, "model.");
            auto cache = model.make_cache(16);

            bt::Tensor logits;
            model.forward(seq, mt, mh, mw, cache, logits);
            bt::sync_all();

            pipe_logits = bdtest::bd_download(logits);
        }

        // 3. Compare outputs
        CHECK(single_logits.size() == pipe_logits.size());
        float max_diff = 0.0f;
        for (std::size_t i = 0; i < single_logits.size(); ++i) {
            float diff = std::abs(single_logits[i] - pipe_logits[i]);
            if (diff > max_diff) max_diff = diff;
        }
        std::cout << "[test_qwen35_pipeline_multigpu] Max difference between Single-GPU and 2-GPU Pipeline: "
                  << max_diff << std::endl;
        CHECK(max_diff < 1e-4f);

        // 4. Token-by-token decode on 2-GPU pipeline
        {
            q35::TextModel pipe_model(cfg_pipe);
            pipe_model.load_weights(file, "model.");
            auto cache = pipe_model.make_cache(16);

            for (int i = 0; i < Lseq; ++i) {
                std::vector<int> step_seq = {seq[i]};
                std::vector<int64_t> st_t = {i}, st_h = {i}, st_w = {i};
                bt::Tensor step_logits;
                pipe_model.forward(step_seq, st_t, st_h, st_w, cache, step_logits);
                bt::sync_all();

                if (i == Lseq - 1) {
                    std::vector<float> step_last = bdtest::bd_download(step_logits);
                    float diff_last = 0.0f;
                    for (int v = 0; v < V; ++v) {
                        float expected = pipe_logits[(Lseq - 1) * V + v];
                        float d = std::abs(step_last[v] - expected);
                        if (d > diff_last) diff_last = d;
                    }
                    std::cout << "[test_qwen35_pipeline_multigpu] Max diff on step decode: " << diff_last << std::endl;
                    CHECK(diff_last < 5e-3f);
                }
            }
        }

    } catch (const std::exception& e) {
        std::fprintf(stderr, "Exception: %s\n", e.what());
        CHECK(false);
    }

    std::filesystem::remove(path);
    if (g_failures == 0) {
        std::cout << "[test_qwen35_pipeline_multigpu] ALL TESTS PASSED!" << std::endl;
        return 0;
    }
    return 1;
}
