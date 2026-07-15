// Ad-hoc CLI driver + parity harness backend for the LLM2Vec encoder.
//
// Loads an LLM2Vec (bidirectional LLaMA) checkpoint — config.json +
// model.safetensors, the merged layout scripts/convert-llm2vec.py writes — runs
// one bidirectional forward over a sequence of token ids read from a file, and
// dumps the per-token hidden states as raw little-endian float32. With --pool it
// dumps the masked-mean-pooled sentence embedding instead.
//
// This is the C++ side of scripts/llm2vec_parity.sh: the Python reference
// (scripts/llm2vec_ref.py) writes the ids + its own hidden states, this reads
// the ids and writes ours, and the shell script diffs the two. Force the CPU
// backend so the compute dtype is FP32 and matches the FP32 reference exactly:
//   BROTENSOR_DEFAULT_DEVICE=CPU brolm_llm2vec_encode ...
//
// Usage:
//   brolm_llm2vec_encode --weights-dir DIR --ids ids.i32 --out hidden.f32 [--pool]
//   brolm_llm2vec_encode --config config.json --weights model.safetensors \
//                        --ids ids.i32 --out hidden.f32
//
//   --weights-dir DIR   dir holding config.json + model.safetensors (or
//                       *.safetensors shards); shorthand for --config/--weights
//   --config PATH       config.json (default: <weights-dir>/config.json)
//   --weights PATH      model.safetensors file or a dir to scan for shards
//   --prefix STR        weight-name prefix (default none)
//   --ids PATH          raw little-endian int32 token ids (count = filesize/4)
//   --out PATH          raw little-endian float32 output
//   --pool              output the (hidden,1) masked-mean embedding, not (L,hidden)
//
// Not run by ctest — driven by scripts/llm2vec_parity.sh.

#include "brolm/llm2vec.h"

#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace l2v = brolm::llm2vec;
namespace bt  = brotensor;
namespace st  = brotensor::safetensors;
namespace fs  = std::filesystem;

namespace {

void usage(const char* argv0) {
    std::printf(
        "usage: %s --weights-dir DIR --ids ids.i32 --out hidden.f32 [--pool]\n"
        "       %s --config config.json --weights model.safetensors \\\n"
        "              --ids ids.i32 --out hidden.f32 [--prefix STR] [--pool]\n",
        argv0, argv0);
}

// Read a raw little-endian int32 file into a vector.
std::vector<int32_t> read_ids(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open ids file '" + path + "'");
    const std::streamsize n = f.tellg();
    if (n < 0 || n % 4 != 0)
        throw std::runtime_error("ids file size is not a multiple of 4 bytes");
    f.seekg(0);
    std::vector<int32_t> ids(static_cast<std::size_t>(n / 4));
    if (!ids.empty())
        f.read(reinterpret_cast<char*>(ids.data()), n);
    return ids;
}

// Write host float32 values as a raw little-endian f32 file.
void write_f32(const std::string& path, const std::vector<float>& v) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot open out file '" + path + "'");
    f.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(float)));
}

// Download a tensor to host FP32 regardless of its dtype (mirrors the tests'
// bd_download): FP32 verbatim, FP16 through fp16_bits_to_fp32.
std::vector<float> download_f32(const bt::Tensor& t) {
    bt::sync_all();
    if (t.dtype == bt::Dtype::FP16) {
        std::vector<uint16_t> bits = t.to_host_vector_fp16();
        std::vector<float> out(bits.size());
        for (std::size_t i = 0; i < bits.size(); ++i)
            out[i] = bt::fp16_bits_to_fp32(bits[i]);
        return out;
    }
    return t.to_host_vector();
}

// Collect the safetensors shard paths for a --weights argument: a single file
// verbatim, or every *.safetensors under a directory (sorted for determinism).
std::vector<std::string> resolve_shards(const std::string& weights_arg) {
    if (fs::is_regular_file(weights_arg)) return {weights_arg};
    if (!fs::is_directory(weights_arg))
        throw std::runtime_error("weights path not found: '" + weights_arg + "'");
    // Prefer a single model.safetensors when present.
    const fs::path single = fs::path(weights_arg) / "model.safetensors";
    if (fs::is_regular_file(single)) return {single.string()};
    std::vector<std::string> shards;
    for (const auto& e : fs::directory_iterator(weights_arg))
        if (e.path().extension() == ".safetensors")
            shards.push_back(e.path().string());
    std::sort(shards.begin(), shards.end());
    if (shards.empty())
        throw std::runtime_error("no *.safetensors under '" + weights_arg + "'");
    return shards;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    std::string weights_dir, config_arg, weights_arg, prefix, ids_path, out_path;
    bool pool = false;

    for (int i = 1; i < argc; ++i) {
        auto next = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s needs an argument\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };
        const char* a = argv[i];
        if      (!std::strcmp(a, "--weights-dir")) weights_dir = next(a);
        else if (!std::strcmp(a, "--config"))      config_arg  = next(a);
        else if (!std::strcmp(a, "--weights"))     weights_arg = next(a);
        else if (!std::strcmp(a, "--prefix"))      prefix      = next(a);
        else if (!std::strcmp(a, "--ids"))         ids_path    = next(a);
        else if (!std::strcmp(a, "--out"))         out_path    = next(a);
        else if (!std::strcmp(a, "--pool"))        pool        = true;
        else if (!std::strcmp(a, "--help") || !std::strcmp(a, "-h")) {
            usage(argv[0]); return 0;
        } else {
            std::fprintf(stderr, "error: unknown arg '%s'\n", a);
            usage(argv[0]); return 2;
        }
    }

    if (!weights_dir.empty()) {
        if (config_arg.empty())
            config_arg = (fs::path(weights_dir) / "config.json").string();
        if (weights_arg.empty()) weights_arg = weights_dir;
    }
    if (config_arg.empty() || weights_arg.empty() || ids_path.empty() ||
        out_path.empty()) {
        std::fprintf(stderr, "error: need --config/--weights (or --weights-dir), "
                             "--ids, and --out\n");
        usage(argv[0]);
        return 2;
    }

    try {
        bt::init();
        std::printf("[info] device=%s compute_dtype=%s\n",
                    bt::default_device() == bt::Device::CPU ? "CPU" : "GPU",
                    bt::compute_dtype() == bt::Dtype::FP16 ? "FP16" : "FP32");

        l2v::Config cfg = l2v::Config::load(config_arg);
        std::printf("[info] hidden=%d layers=%d q=%d kv=%d hd=%d vocab=%d\n",
                    cfg.hidden_size, cfg.num_hidden_layers,
                    cfg.num_attention_heads, cfg.num_key_value_heads,
                    cfg.head_dim, cfg.vocab_size);

        const std::vector<std::string> shard_paths = resolve_shards(weights_arg);
        std::vector<st::File> shard_files;
        shard_files.reserve(shard_paths.size());
        for (const auto& p : shard_paths) {
            std::printf("[load] %s\n", p.c_str());
            shard_files.push_back(st::File::open(p));
        }
        std::vector<const st::File*> shards;
        shards.reserve(shard_files.size());
        for (const auto& f : shard_files) shards.push_back(&f);

        l2v::Encoder enc(cfg);
        enc.load_weights(shards, prefix);
        std::printf("[info] weights loaded\n");

        const std::vector<int32_t> ids = read_ids(ids_path);
        const int L = static_cast<int>(ids.size());
        if (L == 0) throw std::runtime_error("ids file is empty");
        std::printf("[info] L=%d ids\n", L);

        std::vector<float> host;
        if (pool) {
            bt::Tensor emb;
            enc.encode_pooled(ids.data(), L, emb, /*pool_mask=*/nullptr);
            host = download_f32(emb);
            std::printf("[info] pooled embedding: (%d,1) -> %zu floats\n",
                        cfg.hidden_size, host.size());
        } else {
            bt::Tensor hidden;
            enc.encode(ids.data(), L, hidden);
            host = download_f32(hidden);
            std::printf("[info] hidden states: (%d,%d) -> %zu floats\n",
                        L, cfg.hidden_size, host.size());
        }

        write_f32(out_path, host);
        std::printf("[done] wrote %s (%zu floats)\n", out_path.c_str(),
                    host.size());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 1;
    }
    return 0;
}
