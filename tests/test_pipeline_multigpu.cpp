#include "brolm/qwen.h"
#include "brolm/detail/compute.h"
#include "brotensor/safetensors.h"
#include "brotensor/runtime.h"
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

namespace qwen = brolm::qwen;
namespace st   = brotensor::safetensors;
namespace bt   = brotensor;

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
    std::cout << "[test_pipeline_multigpu] Detected " << count << " CUDA device(s)" << std::endl;
    if (count < 2) {
        std::cout << "[test_pipeline_multigpu] Multi-GPU test requires at least 2 GPUs. Skipping." << std::endl;
        return 0;
    }

    qwen::Qwen3Config cfg_single;
    cfg_single.vocab_size          = 48;
    cfg_single.hidden_size         = 32;
    cfg_single.intermediate_size   = 64;
    cfg_single.num_hidden_layers   = 4;
    cfg_single.num_attention_heads = 4;
    cfg_single.num_key_value_heads = 2;
    cfg_single.head_dim            = 8;
    cfg_single.rms_norm_eps        = 1e-6f;
    cfg_single.rope_theta          = 1000000.0f;
    cfg_single.tie_word_embeddings = true;

    qwen::Qwen3Config cfg_pipe = cfg_single;
    cfg_pipe.pipeline_devices = { bt::Device::cuda(0), bt::Device::cuda(1) };

    const int V    = cfg_single.vocab_size;
    const int H    = cfg_single.hidden_size;
    const int F    = cfg_single.intermediate_size;
    const int HD   = cfg_single.head_dim;
    const int q_d  = cfg_single.num_attention_heads * HD;
    const int kv_d = cfg_single.num_key_value_heads * HD;

    Builder b;
    uint32_t seed = 42;
    auto R = [&](std::size_t n) { return fp16_rand(n, seed++); };

    b.add("model.embed_tokens.weight", {V, H}, R(static_cast<std::size_t>(V) * H));

    for (int i = 0; i < cfg_single.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        b.add(p + "input_layernorm.weight", {H}, fp16_ones(H));
        b.add(p + "self_attn.q_proj.weight", {q_d, H}, R(static_cast<std::size_t>(q_d) * H));
        b.add(p + "self_attn.k_proj.weight", {kv_d, H}, R(static_cast<std::size_t>(kv_d) * H));
        b.add(p + "self_attn.v_proj.weight", {kv_d, H}, R(static_cast<std::size_t>(kv_d) * H));
        b.add(p + "self_attn.o_proj.weight", {H, q_d}, R(static_cast<std::size_t>(H) * q_d));
        b.add(p + "self_attn.q_norm.weight", {HD}, fp16_ones(HD));
        b.add(p + "self_attn.k_norm.weight", {HD}, fp16_ones(HD));
        b.add(p + "post_attention_layernorm.weight", {H}, fp16_ones(H));
        b.add(p + "mlp.gate_proj.weight", {F, H}, R(static_cast<std::size_t>(F) * H));
        b.add(p + "mlp.up_proj.weight", {F, H}, R(static_cast<std::size_t>(F) * H));
        b.add(p + "mlp.down_proj.weight", {H, F}, R(static_cast<std::size_t>(H) * F));
    }
    b.add("model.norm.weight", {H}, fp16_ones(H));

    auto path = std::filesystem::temp_directory_path() / "brolm_pipeline_test.safetensors";
    b.write(path);

    try {
        auto file = st::File::open(path.string());
        const std::vector<int32_t> seq = {7, 13, 22, 35};
        const int Lseq = static_cast<int>(seq.size());

        // 1. Single GPU Run
        std::vector<float> single_logits;
        {
            qwen::Qwen3Model model(cfg_single);
            model.load_weights(file, "");
            model.allocate_cache(16);

            bt::Tensor logits;
            model.forward(seq.data(), Lseq, logits);
            bt::sync_all();

            single_logits = bdtest::bd_download(logits);
        }

        // 2. Multi-GPU Pipeline Run
        std::vector<float> pipe_logits;
        {
            qwen::Qwen3Model model(cfg_pipe);
            model.load_weights(file, "");
            model.allocate_cache(16);

            bt::Tensor logits;
            model.forward(seq.data(), Lseq, logits);
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
        std::cout << "[test_pipeline_multigpu] Max difference between Single-GPU and 2-GPU Pipeline: "
                  << max_diff << std::endl;
        CHECK(max_diff < 1e-4f);

        // 4. Token-by-token decode on multi-GPU pipeline
        {
            qwen::Qwen3Model pipe_model(cfg_pipe);
            pipe_model.load_weights(file, "");
            pipe_model.allocate_cache(16);

            for (int i = 0; i < Lseq; ++i) {
                bt::Tensor step_logits;
                pipe_model.forward_last(&seq[i], 1, step_logits);
                bt::sync_all();
                CHECK(pipe_model.cache_len() == i + 1);

                if (i == Lseq - 1) {
                    std::vector<float> step_last = bdtest::bd_download(step_logits);
                    float diff_last = 0.0f;
                    for (int v = 0; v < V; ++v) {
                        float expected = pipe_logits[(Lseq - 1) * V + v];
                        float d = std::abs(step_last[v] - expected);
                        if (d > diff_last) diff_last = d;
                    }
                    std::cout << "[test_pipeline_multigpu] Max diff on step decode: " << diff_last << std::endl;
                    CHECK(diff_last < 1e-4f);
                }
            }
        }

    } catch (const std::exception& e) {
        std::fprintf(stderr, "Exception: %s\n", e.what());
        CHECK(false);
    }

    std::filesystem::remove(path);
    if (g_failures == 0) {
        std::cout << "[test_pipeline_multigpu] ALL TESTS PASSED!" << std::endl;
        return 0;
    }
    return 1;
}
