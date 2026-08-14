// CLI driver for Qwen 3.5 / Qwen 3.8 hybrid text decoder.
//
// Supports single-GPU and multi-GPU pipeline parallelism across dual RTX 4090s.
// Supports both one-shot prompt completion and interactive REPL mode with streaming output.
//
// usage:
//   brolm_run_qwen35_text --dir <weights_dir> --prompt "Hello Qwen!"
//   brolm_run_qwen35_text --dir <weights_dir> --chat --interactive --pipeline

#include "brolm/qwen35_config.h"
#include "brolm/qwen35_text.h"
#include "brolm/qwen35_tokenizer.h"
#include "brolm/detail/generate.h"

#include "brotensor/gguf.h"
#include "brotensor/safetensors.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace q35 = brolm::qwen35;
namespace st  = brotensor::safetensors;
namespace bt  = brotensor;

namespace {

void usage(const char* argv0) {
    std::printf(
        "usage: %s (--dir PATH | --gguf FILE.gguf) [options]\n"
        "\n"
        "Options:\n"
        "  --gguf FILE.gguf Direct path to a GGUF model file (e.g. D:\\Qwen3.8-27B-Q6_K.gguf)\n"
        "  --dir PATH       Directory containing config.json, safetensors, vocab.json, merges.txt\n"
        "  --prompt TEXT    Prompt text (default: 'Tell me three facts about space.')\n"
        "  --max-new N      Maximum new tokens to generate (default: 128)\n"
        "  --temp T         Sampling temperature (<=0 => greedy, default: 0.0)\n"
        "  --top-k K        Top-k cutoff (0 = off, default: 0)\n"
        "  --top-p P        Nucleus top-p cutoff (>=1 = off, default: 1.0)\n"
        "  --seed S         RNG seed for sampling (default: 0)\n"
        "  --chat           Format prompt in Qwen chat template\n"
        "  --no-think       Bypass <think> reasoning mode in chat template\n"
        "  --pipeline       Enable multi-GPU pipeline parallelism across available CUDA devices\n"
        "  --interactive    Run interactive chat REPL mode\n"
        "  --help, -h       Show this help message\n",
        argv0);
}

std::vector<std::string> find_safetensors(const std::filesystem::path& dir) {
    std::vector<std::string> shards;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            shards.push_back(entry.path().string());
        }
    }
    std::sort(shards.begin(), shards.end());
    return shards;
}

std::string format_chat(const std::string& user_prompt, bool enable_thinking) {
    if (!enable_thinking) {
        return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" +
               user_prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    }
    return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" +
           user_prompt + "<|im_end|>\n<|im_start|>assistant\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    std::string model_path;
    std::string prompt = "Tell me three facts about space.";
    int max_new_tokens = 128;
    brolm::detail::SamplingParams sampling_params;
    sampling_params.temperature = 0.0f;
    sampling_params.top_k = 0;
    sampling_params.top_p = 1.0f;
    sampling_params.seed = 0;
    bool chat_mode = false;
    bool enable_thinking = true;
    bool pipeline_mode = false;
    bool interactive_mode = false;

    for (int i = 1; i < argc; ++i) {
        auto next = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s requires an argument\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };
        const char* a = argv[i];
        if (!std::strcmp(a, "--dir") || !std::strcmp(a, "--gguf")) model_path = next(a);
        else if (!std::strcmp(a, "--prompt")) prompt = next(a);
        else if (!std::strcmp(a, "--max-new")) max_new_tokens = std::atoi(next(a));
        else if (!std::strcmp(a, "--temp")) sampling_params.temperature = static_cast<float>(std::atof(next(a)));
        else if (!std::strcmp(a, "--top-k")) sampling_params.top_k = std::atoi(next(a));
        else if (!std::strcmp(a, "--top-p")) sampling_params.top_p = static_cast<float>(std::atof(next(a)));
        else if (!std::strcmp(a, "--seed")) sampling_params.seed = static_cast<uint64_t>(std::strtoull(next(a), nullptr, 10));
        else if (!std::strcmp(a, "--chat")) chat_mode = true;
        else if (!std::strcmp(a, "--no-think")) enable_thinking = false;
        else if (!std::strcmp(a, "--pipeline") || !std::strcmp(a, "--multi-gpu")) pipeline_mode = true;
        else if (!std::strcmp(a, "--interactive") || !std::strcmp(a, "-i")) interactive_mode = true;
        else if (!std::strcmp(a, "--help") || !std::strcmp(a, "-h")) { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "error: unknown argument '%s'\n", a); usage(argv[0]); return 2; }
    }

    if (model_path.empty()) {
        const char* env_dir = std::getenv("BROLM_QWEN35_DIR");
        if (env_dir) {
            model_path = env_dir;
        } else {
            std::fprintf(stderr, "error: --gguf FILE or --dir PATH is required (or set BROLM_QWEN35_DIR)\n");
            usage(argv[0]);
            return 1;
        }
    }

    bool is_gguf_file = (std::filesystem::is_regular_file(model_path) &&
                         std::filesystem::path(model_path).extension() == ".gguf");

    if (!is_gguf_file && !std::filesystem::is_directory(model_path)) {
        std::fprintf(stderr, "error: '%s' is not a valid directory or .gguf file\n", model_path.c_str());
        return 1;
    }

    try {
        bt::init();

        int num_gpus = bt::cuda_device_count();
        std::printf("[init] Detected %d CUDA device(s)\n", num_gpus);

        std::unique_ptr<q35::Tokenizer> tok;
        q35::Qwen35Config::Text cfg;
        std::unique_ptr<q35::TextModel> model;

        if (is_gguf_file) {
            std::printf("[load] Opening GGUF file: %s\n", model_path.c_str());
            auto gguf_f = bt::gguf::File::open(model_path);
            
            std::printf("[load] Parsing GGUF tokenizer...\n");
            tok = std::make_unique<q35::Tokenizer>(q35::Tokenizer::from_gguf(gguf_f));

            std::printf("[load] Parsing GGUF metadata config...\n");
            q35::Qwen35Config full_cfg = q35::Qwen35Config::from_gguf(gguf_f);
            cfg = full_cfg.text;

            if (pipeline_mode) {
                if (num_gpus >= 2) {
                    cfg.pipeline_devices = { bt::Device::cuda(0), bt::Device::cuda(1) };
                    std::printf("[pipeline] Enabled 2-GPU pipeline parallelism across cuda(0) and cuda(1)\n");
                } else {
                    std::printf("[pipeline] Warning: pipeline requested but only %d GPU(s) available. Using single device.\n", num_gpus);
                }
            }

            std::printf("[info] Model architecture: layers=%d, hidden=%d, full_interval=%d, vocab=%d\n",
                        cfg.num_hidden_layers, cfg.hidden_size, cfg.full_attention_interval, cfg.vocab_size);

            std::printf("[load] Instantiating TextModel and loading GGUF weights...\n");
            auto load_start = std::chrono::steady_clock::now();
            model = std::make_unique<q35::TextModel>(cfg);
            model->load_weights(gguf_f);
            auto load_end = std::chrono::steady_clock::now();
            double load_sec = std::chrono::duration<double>(load_end - load_start).count();
            std::printf("[load] Weights loaded in %.2f s\n", load_sec);
        } else {
            std::filesystem::path dir(model_path);
            // 1. Load Tokenizer
            std::filesystem::path vocab_p  = dir / "vocab.json";
            std::filesystem::path merges_p = dir / "merges.txt";
            if (!std::filesystem::exists(vocab_p) || !std::filesystem::exists(merges_p)) {
                std::fprintf(stderr, "error: vocab.json or merges.txt not found in %s\n", model_path.c_str());
                return 1;
            }
            std::printf("[load] Loading tokenizer from %s...\n", model_path.c_str());
            tok = std::make_unique<q35::Tokenizer>(q35::Tokenizer::load(vocab_p.string(), merges_p.string()));

            // 2. Load Configuration
            std::filesystem::path config_p = dir / "config.json";
            if (!std::filesystem::exists(config_p)) {
                std::fprintf(stderr, "error: config.json not found in %s\n", model_path.c_str());
                return 1;
            }
            std::printf("[load] Reading config: %s\n", config_p.string().c_str());
            q35::Qwen35Config full_cfg = q35::Qwen35Config::load(config_p.string());
            cfg = full_cfg.text;

            if (pipeline_mode) {
                if (num_gpus >= 2) {
                    cfg.pipeline_devices = { bt::Device::cuda(0), bt::Device::cuda(1) };
                    std::printf("[pipeline] Enabled 2-GPU pipeline parallelism across cuda(0) and cuda(1)\n");
                } else {
                    std::printf("[pipeline] Warning: pipeline requested but only %d GPU(s) available. Using single device.\n", num_gpus);
                }
            }

            std::printf("[info] Model architecture: layers=%d, hidden=%d, full_interval=%d, vocab=%d\n",
                        cfg.num_hidden_layers, cfg.hidden_size, cfg.full_attention_interval, cfg.vocab_size);

            // 3. Load Model Weights
            auto shard_paths = find_safetensors(dir);
            if (shard_paths.empty()) {
                std::fprintf(stderr, "error: no .safetensors files found in %s\n", model_path.c_str());
                return 1;
            }
            std::printf("[load] Opening %zu safetensors shard(s)...\n", shard_paths.size());
            std::vector<st::File> shard_files;
            std::vector<const st::File*> shard_ptrs;
            shard_files.reserve(shard_paths.size());
            shard_ptrs.reserve(shard_paths.size());
            for (const auto& sp : shard_paths) {
                shard_files.push_back(st::File::open(sp));
                shard_ptrs.push_back(&shard_files.back());
            }

            std::printf("[load] Instantiating and uploading TextModel weights...\n");
            auto load_start = std::chrono::steady_clock::now();
            model = std::make_unique<q35::TextModel>(cfg);
            if (shard_ptrs[0]->find("model.language_model.embed_tokens.weight")) {
                model->load_weights(shard_ptrs, "model.language_model.");
            } else if (shard_ptrs[0]->find("language_model.embed_tokens.weight")) {
                model->load_weights(shard_ptrs, "language_model.");
            } else if (shard_ptrs[0]->find("model.embed_tokens.weight")) {
                model->load_weights(shard_ptrs, "model.");
            } else {
                model->load_weights(shard_ptrs, "");
            }
            auto load_end = std::chrono::steady_clock::now();
            double load_sec = std::chrono::duration<double>(load_end - load_start).count();
            std::printf("[load] Weights loaded in %.2f s\n", load_sec);
        }

        // Helper to run generation for a given prompt string
        auto run_generation = [&](const std::string& input_text) {
            std::string prompt_formatted = (chat_mode ? format_chat(input_text, enable_thinking) : input_text);
            std::vector<int32_t> prompt_ids32 = tok->encode(prompt_formatted);
            if (prompt_ids32.empty()) {
                std::printf("[error] Empty prompt token sequence\n");
                return;
            }
            std::vector<int> prompt_ids(prompt_ids32.begin(), prompt_ids32.end());
            const int L = static_cast<int>(prompt_ids.size());

            std::printf("\n--- Prompt (%d tokens) ---\n%s\n--- Response ---\n", L, prompt_formatted.c_str());

            const int max_seq = L + max_new_tokens + 16;
            auto cache = model->make_cache(max_seq);

            std::mt19937_64 rng(sampling_params.seed);

            // 1. Prefill
            std::vector<int64_t> mrope_t(L), mrope_h(L), mrope_w(L);
            for (int i = 0; i < L; ++i) {
                mrope_t[i] = i;
                mrope_h[i] = i;
                mrope_w[i] = i;
            }

            auto t0 = std::chrono::steady_clock::now();
            bt::Tensor logits;
            model->forward(prompt_ids, mrope_t, mrope_h, mrope_w, cache, logits);
            bt::sync_all();
            auto t_prefill = std::chrono::steady_clock::now();

            std::vector<float> last_logits = brolm::detail::last_row_fp32(logits);
            int next_token = brolm::detail::sample_token(
                last_logits.data(), cfg.vocab_size, sampling_params, rng);

            int generated_count = 0;
            int cur_pos = L;

            const int im_end_id = tok->im_end_id();
            const int eos_id    = tok->eos_id();

            while (generated_count < max_new_tokens) {
                if (next_token == eos_id || next_token == im_end_id) {
                    break;
                }
                std::string token_str = tok->decode({next_token});
                std::printf("%s", token_str.c_str());
                std::fflush(stdout);
                generated_count++;

                // Step forward
                std::vector<int> step_tok = {next_token};
                std::vector<int64_t> st_t = {cur_pos}, st_h = {cur_pos}, st_w = {cur_pos};
                cur_pos++;

                model->forward(step_tok, st_t, st_h, st_w, cache, logits);
                bt::sync_all();

                last_logits = brolm::detail::last_row_fp32(logits);
                next_token = brolm::detail::sample_token(
                    last_logits.data(), cfg.vocab_size, sampling_params, rng);
            }
            auto t_end = std::chrono::steady_clock::now();
            std::printf("\n");

            double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill - t0).count();
            double decode_sec = std::chrono::duration<double>(t_end - t_prefill).count();
            double decode_tps = (decode_sec > 0 ? (generated_count / decode_sec) : 0.0);

            std::printf("\n[stats] Prefill: %.2f ms (%.1f tok/s) | Generated: %d tokens in %.2f s (%.2f tok/s)\n",
                        prefill_ms, (prefill_ms > 0 ? (L / (prefill_ms / 1000.0)) : 0.0),
                        generated_count, decode_sec, decode_tps);
        };

        if (interactive_mode) {
            std::printf("\n=== Interactive Qwen3.5 REPL (type 'exit' or 'quit' to end) ===\n");
            while (true) {
                std::printf("\nUser> ");
                std::string line;
                if (!std::getline(std::cin, line)) break;
                if (line == "exit" || line == "quit") break;
                if (line.empty()) continue;
                run_generation(line);
            }
        } else {
            run_generation(prompt);
        }

    } catch (const std::exception& e) {
        std::fprintf(stderr, "\nfatal: %s\n", e.what());
        return 1;
    }

    return 0;
}
