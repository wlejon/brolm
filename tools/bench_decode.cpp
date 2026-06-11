// brolm_bench_decode — prefill / decode throughput for the text decoders.
//
// Loads a checkpoint, runs a synthetic prompt through one prefill forward and
// a token-by-token decode loop (greedy sampling, the generate.h recipe
// inlined so prefill and decode are timed separately), and reports ms + tok/s
// for each. Repeats --reps times and reports the best rep, so the numbers are
// comparable across runs on a busy box.
//
// Usage:
//   brolm_bench_decode <model.gguf | qwen35-checkpoint-dir>
//                      [--prefill N] [--decode M] [--reps R]
//
// A .gguf path benches the Qwen3 dense decoder; a directory benches the
// Qwen3.5 hybrid text model (config.json + safetensors, text-only M-RoPE).
//
// Set BROLM_PROFILE=1 for the per-stage breakdown (printed after the run;
// the stage syncs serialise the stream, so throughput numbers from a
// profiling run are NOT comparable to a plain run).

#include "brolm/detail/generate.h"
#include "brolm/detail/profile.h"
#include "brolm/qwen.h"
#include "brolm/qwen35_config.h"
#include "brolm/qwen35_text.h"

#include "brotensor/gguf.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace bt = brotensor;

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

[[noreturn]] void usage() {
    std::fprintf(stderr,
                 "usage: brolm_bench_decode <model.gguf> [--prefill N] "
                 "[--decode M] [--reps R]\n");
    std::exit(2);
}

int arg_int(int argc, char** argv, int& i) {
    if (i + 1 >= argc) usage();
    return std::atoi(argv[++i]);
}

// Qwen3.5 hybrid text model from a HF checkpoint directory. Text-only
// generation: all three M-RoPE axes carry the sequential position.
int bench_qwen35(const std::string& dir, int prefill_n, int decode_n,
                 int reps) {
    namespace q35 = brolm::qwen35;
    namespace st  = brotensor::safetensors;
    namespace fs  = std::filesystem;

    const fs::path cfg_path = fs::path(dir) / "config.json";
    const fs::path st_path =
        fs::path(dir) / "model.safetensors-00001-of-00001.safetensors";
    if (!fs::exists(cfg_path) || !fs::exists(st_path)) {
        std::fprintf(stderr, "no config.json / safetensors under %s\n",
                     dir.c_str());
        return 2;
    }

    const double t_load0 = now_ms();
    auto cfg_full = q35::Qwen35Config::load(cfg_path.string());
    const auto& cfg = cfg_full.text;
    q35::TextModel model(cfg);
    auto file = st::File::open(st_path.string());
    model.load_weights(file);
    brotensor::sync_all();
    std::printf("loaded %s in %.0f ms  (layers=%d hidden=%d vocab=%d)\n",
                dir.c_str(), now_ms() - t_load0, cfg.num_hidden_layers,
                cfg.hidden_size, cfg.vocab_size);

    std::mt19937_64 prompt_rng(42);
    std::vector<int> prompt(static_cast<std::size_t>(prefill_n));
    for (int& id : prompt) {
        id = static_cast<int>(prompt_rng() %
                              static_cast<std::uint64_t>(cfg.vocab_size));
    }

    brolm::detail::SamplingParams greedy;
    greedy.temperature = 0.0f;

    auto positions = [](int start, int n) {
        std::vector<int64_t> p(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) p[static_cast<std::size_t>(i)] = start + i;
        return p;
    };

    double best_prefill_ms = 0.0;
    double best_decode_ms  = 0.0;

    for (int rep = -1; rep < reps; ++rep) {   // rep -1 = warm-up
        const int pn = (rep < 0) ? std::min(prefill_n, 8) : prefill_n;
        const int dn = (rep < 0) ? 4 : decode_n;
        auto cache = model.make_cache(prefill_n + decode_n);
        std::mt19937_64 rng(0);
        std::vector<int> ids(prompt.begin(), prompt.begin() + pn);

        brotensor::Tensor logits;
        const double t0 = now_ms();
        auto p = positions(0, pn);
        model.forward(ids, p, p, p, cache, logits);
        std::vector<float> row = brolm::detail::last_row_fp32(logits);
        int next = brolm::detail::sample_token(row.data(), cfg.vocab_size,
                                               greedy, rng);
        const double prefill_ms = now_ms() - t0;

        const double t1 = now_ms();
        for (int t = 0; t < dn; ++t) {
            auto sp = positions(pn + t, 1);
            std::vector<int> one{next};
            model.forward(one, sp, sp, sp, cache, logits);
            row  = brolm::detail::last_row_fp32(logits);
            next = brolm::detail::sample_token(row.data(), cfg.vocab_size,
                                               greedy, rng);
        }
        const double decode_ms = now_ms() - t1;
        if (rep < 0) {
            brolm::detail::profile::reset();
            continue;
        }

        std::printf(
            "rep %d: prefill %4d tok %8.1f ms (%7.1f tok/s)   "
            "decode %4d tok %8.1f ms (%6.2f ms/tok, %6.1f tok/s)\n",
            rep, pn, prefill_ms, 1000.0 * pn / prefill_ms,
            dn, decode_ms, decode_ms / dn, 1000.0 * dn / decode_ms);

        if (rep == 0 || prefill_ms < best_prefill_ms) {
            best_prefill_ms = prefill_ms;
        }
        if (rep == 0 || decode_ms < best_decode_ms) {
            best_decode_ms = decode_ms;
        }
    }

    std::printf("best: prefill %.1f tok/s   decode %.2f ms/tok (%.1f tok/s)\n",
                1000.0 * prefill_n / best_prefill_ms,
                best_decode_ms / decode_n, 1000.0 * decode_n / best_decode_ms);

    brolm::detail::profile::report();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string path;
    int prefill_n = 512;
    int decode_n  = 128;
    int reps      = 3;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--prefill") == 0) {
            prefill_n = arg_int(argc, argv, i);
        } else if (std::strcmp(argv[i], "--decode") == 0) {
            decode_n = arg_int(argc, argv, i);
        } else if (std::strcmp(argv[i], "--reps") == 0) {
            reps = arg_int(argc, argv, i);
        } else if (path.empty() && argv[i][0] != '-') {
            path = argv[i];
        } else {
            usage();
        }
    }
    if (path.empty() || prefill_n < 1 || decode_n < 1 || reps < 1) usage();

    bt::init();
    std::printf("device: %s\n",
                bt::default_device() == bt::Device::CPU ? "CPU" : "GPU");

    if (std::filesystem::is_directory(path)) {
        return bench_qwen35(path, prefill_n, decode_n, reps);
    }

    const double t_load0 = now_ms();
    bt::gguf::File f = bt::gguf::File::open(path);
    brolm::qwen::Qwen3Config cfg = brolm::qwen::Qwen3Config::from_gguf(f);
    brolm::qwen::Qwen3Model model(cfg);
    model.load_weights(f);
    bt::sync_all();
    std::printf("loaded %s in %.0f ms  (layers=%d hidden=%d vocab=%d)\n",
                path.c_str(), now_ms() - t_load0, cfg.num_hidden_layers,
                cfg.hidden_size, cfg.vocab_size);

    // Synthetic prompt: fixed-seed uniform ids. The token *values* don't
    // matter for throughput — every id costs the same embedding lookup and
    // the same per-layer math.
    std::mt19937_64 prompt_rng(42);
    std::vector<int32_t> prompt(static_cast<std::size_t>(prefill_n));
    for (int32_t& id : prompt) {
        id = static_cast<int32_t>(prompt_rng() %
                                  static_cast<std::uint64_t>(cfg.vocab_size));
    }

    brolm::detail::SamplingParams greedy;
    greedy.temperature = 0.0f;

    // Warm-up: a short prefill + a few decode steps so allocator pools and
    // any lazily-built state are primed before timing.
    {
        model.allocate_cache(prefill_n + decode_n);
        bt::Tensor logits;
        model.forward_last(prompt.data(), std::min(prefill_n, 8), logits);
        std::mt19937_64 rng(0);
        std::vector<float> row = brolm::detail::last_row_fp32(logits);
        int32_t next = static_cast<int32_t>(
            brolm::detail::sample_token(row.data(), cfg.vocab_size, greedy, rng));
        for (int t = 0; t < 4; ++t) {
            model.forward_last(&next, 1, logits);
            row  = brolm::detail::last_row_fp32(logits);
            next = static_cast<int32_t>(brolm::detail::sample_token(
                row.data(), cfg.vocab_size, greedy, rng));
        }
        bt::sync_all();
    }
    brolm::detail::profile::reset();   // don't count warm-up in the breakdown

    double best_prefill_ms = 0.0;
    double best_decode_ms  = 0.0;

    for (int rep = 0; rep < reps; ++rep) {
        model.allocate_cache(prefill_n + decode_n);   // also resets cache_len
        std::mt19937_64 rng(0);

        // Prefill: one forward over the whole prompt, then sample.
        bt::Tensor logits;
        const double t0 = now_ms();
        model.forward_last(prompt.data(), prefill_n, logits);
        std::vector<float> row = brolm::detail::last_row_fp32(logits);
        int32_t next = static_cast<int32_t>(brolm::detail::sample_token(
            row.data(), cfg.vocab_size, greedy, rng));
        const double prefill_ms = now_ms() - t0;

        // Decode: token-by-token, greedy.
        const double t1 = now_ms();
        for (int t = 0; t < decode_n; ++t) {
            model.forward_last(&next, 1, logits);
            row  = brolm::detail::last_row_fp32(logits);
            next = static_cast<int32_t>(brolm::detail::sample_token(
                row.data(), cfg.vocab_size, greedy, rng));
        }
        const double decode_ms = now_ms() - t1;

        std::printf(
            "rep %d: prefill %4d tok %8.1f ms (%7.1f tok/s)   "
            "decode %4d tok %8.1f ms (%6.2f ms/tok, %6.1f tok/s)\n",
            rep, prefill_n, prefill_ms, 1000.0 * prefill_n / prefill_ms,
            decode_n, decode_ms, decode_ms / decode_n,
            1000.0 * decode_n / decode_ms);

        if (rep == 0 || prefill_ms < best_prefill_ms) {
            best_prefill_ms = prefill_ms;
        }
        if (rep == 0 || decode_ms < best_decode_ms) {
            best_decode_ms = decode_ms;
        }
    }

    std::printf("best: prefill %.1f tok/s   decode %.2f ms/tok (%.1f tok/s)\n",
                1000.0 * prefill_n / best_prefill_ms,
                best_decode_ms / decode_n, 1000.0 * decode_n / best_decode_ms);

    brolm::detail::profile::report();
    return 0;
}
