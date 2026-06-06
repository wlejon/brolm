// Ad-hoc CLI driver: prompt the Mistral 3.1 text decoder and print its reply.
//
// Loads a quantized Mistral text GGUF (Q4_K_M / Q6_K / Q8_0) + the native
// tekken tokenizer, encodes a prompt, runs greedy/sampled autoregressive
// generation, and prints the decoded completion. This is the end-to-end
// text path — tokenizer → DenseDecoder forward → sampler → detokenize — on
// the real 24B weights.
//
// The quant matmul/dequant kernels are GPU-only, so this needs a CUDA build:
//   cmake -B build-cuda -DBROTENSOR_WITH_CUDA=ON
//   cmake --build build-cuda --config Release --target brolm_run_mistral_text
//   ./build-cuda/tools/Release/brolm_run_mistral_text \
//       --gguf  weights/Mistral-Small-3.1-24B-Instruct-2503-GGUF/<file>.gguf \
//       --tok   weights/Mistral-Small-3.1-24B-Instruct-2503/tekken.json \
//       --prompt "The capital of France is" --max-new 40
//
// With no --gguf / --tok the defaults below point at the layout produced by
// scripts/download_mistral_gguf.sh + download_mistral.ps1.
//
// NOT run by ctest — it needs the ~14 GB checkpoint and a GPU. For eyeballing
// the full pipeline, like tools/run_qwen35_image.cpp.

#include "brolm/mistral3_config.h"
#include "brolm/mistral3_text.h"
#include "brolm/mistral3_generate.h"
#include "brolm/mistral_tokenizer.h"

#include "brotensor/gguf.h"
#include "brotensor/runtime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

namespace m3 = brolm::mistral3;
namespace bt = brotensor;

namespace {

// Find the first non-mmproj .gguf in a directory (mirrors the gated test's
// discovery), so --gguf can be a directory or omitted entirely.
std::string resolve_gguf(const std::string& arg) {
    namespace fs = std::filesystem;
    fs::path p = arg.empty()
        ? fs::path("weights") / "Mistral-Small-3.1-24B-Instruct-2503-GGUF"
        : fs::path(arg);
    if (fs::is_regular_file(p)) return p.string();
    if (fs::is_directory(p)) {
        for (const auto& e : fs::directory_iterator(p)) {
            if (e.path().extension() != ".gguf") continue;
            if (e.path().filename().string().find("mmproj") != std::string::npos)
                continue;
            return e.path().string();
        }
    }
    return {};
}

void usage(const char* argv0) {
    std::printf(
        "usage: %s [--gguf PATH|DIR] [--tok tekken.json] [--prompt TEXT]\n"
        "          [--max-new N] [--temp T] [--top-k K] [--top-p P]\n"
        "          [--seed S] [--chat] [--no-bos]\n"
        "\n"
        "  --gguf    Mistral text .gguf file, or a dir to scan (default:\n"
        "            weights/Mistral-Small-3.1-24B-Instruct-2503-GGUF/)\n"
        "  --tok     tekken.json (default: weights/Mistral-Small-3.1-24B-\n"
        "            Instruct-2503/tekken.json)\n"
        "  --prompt  prompt text (default: a short factual prompt)\n"
        "  --max-new max new tokens (default 48)\n"
        "  --temp    sampling temperature, <=0 => greedy (default 0 = greedy)\n"
        "  --top-k   top-k cutoff, 0 = off (default 0)\n"
        "  --top-p   nucleus top-p, >=1 = off (default 1)\n"
        "  --seed    RNG seed for sampling (default 0)\n"
        "  --chat    wrap the prompt in Mistral's [INST] chat template\n"
        "  --no-bos  do not prepend BOS (<s>) to the prompt\n",
        argv0);
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    std::string gguf_arg, tok_arg;
    std::string prompt = "The capital of France is";
    m3::GenerateOptions opts;
    opts.max_new_tokens       = 48;
    opts.sampling.temperature = 0.0f;  // greedy by default
    bool chat = false;
    bool add_bos = true;

    for (int i = 1; i < argc; ++i) {
        auto next = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s needs an argument\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };
        const char* a = argv[i];
        if (!std::strcmp(a, "--gguf"))        gguf_arg = next(a);
        else if (!std::strcmp(a, "--tok"))    tok_arg = next(a);
        else if (!std::strcmp(a, "--prompt")) prompt = next(a);
        else if (!std::strcmp(a, "--max-new")) opts.max_new_tokens = std::atoi(next(a));
        else if (!std::strcmp(a, "--temp"))   opts.sampling.temperature = static_cast<float>(std::atof(next(a)));
        else if (!std::strcmp(a, "--top-k"))  opts.sampling.top_k = std::atoi(next(a));
        else if (!std::strcmp(a, "--top-p"))  opts.sampling.top_p = static_cast<float>(std::atof(next(a)));
        else if (!std::strcmp(a, "--seed"))   opts.sampling.seed = static_cast<uint64_t>(std::strtoull(next(a), nullptr, 10));
        else if (!std::strcmp(a, "--chat"))   chat = true;
        else if (!std::strcmp(a, "--no-bos")) add_bos = false;
        else if (!std::strcmp(a, "--help") || !std::strcmp(a, "-h")) { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "error: unknown arg '%s'\n", a); usage(argv[0]); return 2; }
    }

    const std::string gguf_path = resolve_gguf(gguf_arg);
    if (gguf_path.empty()) {
        std::fprintf(stderr, "error: no Mistral .gguf found (pass --gguf, or run "
                             "scripts/download_mistral_gguf.sh)\n");
        return 1;
    }
    const std::string tok_path = tok_arg.empty()
        ? (std::filesystem::path("weights") /
           "Mistral-Small-3.1-24B-Instruct-2503" / "tekken.json").string()
        : tok_arg;
    if (!std::filesystem::exists(tok_path)) {
        std::fprintf(stderr, "error: tokenizer not found at %s (pass --tok)\n",
                     tok_path.c_str());
        return 1;
    }

    try {
        bt::init();
        if (bt::default_device() == bt::Device::CPU) {
            std::fprintf(stderr, "error: the Mistral GGUF quant path is GPU-only; "
                                 "build with -DBROTENSOR_WITH_CUDA=ON\n");
            return 1;
        }

        std::printf("[load] tokenizer: %s\n", tok_path.c_str());
        brolm::mistral::Tokenizer tok =
            brolm::mistral::Tokenizer::load(tok_path);

        std::printf("[load] gguf: %s\n", gguf_path.c_str());
        bt::gguf::File f = bt::gguf::File::open(gguf_path);
        m3::Mistral3Config cfg = m3::Mistral3Config::from_gguf(f);
        std::printf("[info] hidden=%d layers=%d q=%d kv=%d vocab=%d\n",
                    cfg.text.hidden_size, cfg.text.num_hidden_layers,
                    cfg.text.num_attention_heads, cfg.text.num_key_value_heads,
                    cfg.text.vocab_size);

        m3::TextModel model(cfg.text);
        model.load_weights(f);
        std::printf("[info] weights loaded\n");

        // Build the prompt string: chat template or raw, then BOS handling.
        std::string text = prompt;
        if (chat) {
            std::vector<std::pair<std::string, std::string>> msgs = {
                {"user", prompt}};
            text = tok.apply_chat_template(msgs, /*add_generation_prompt=*/true);
            add_bos = false;  // the template already emits a leading <s>
        }

        std::vector<int32_t> ids = tok.encode(text, add_bos);
        std::printf("[info] prompt tokens: %zu\n", ids.size());
        std::printf("\n--- prompt ---\n%s\n--- completion ---\n", prompt.c_str());

        std::vector<int32_t> out = m3::generate(model, ids, tok.eos_id(), opts);
        std::string reply = tok.decode(out);
        std::printf("%s\n", reply.c_str());
        std::printf("\n[info] generated %zu tokens\n", out.size());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 1;
    }
    return 0;
}
