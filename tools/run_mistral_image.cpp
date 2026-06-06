// Ad-hoc CLI driver: prompt Mistral 3.1 with an image + text and print the
// reply. The full VLM pipeline on real weights — Pixtral preprocess → vision
// tower → projector → splice into the text stream → Mistral decoder → generate.
//
// Loads the two llama.cpp GGUFs Mistral 3.1 ships: the quantized text decoder
// and the F16 mmproj/clip (vision tower + projector), plus the tekken
// tokenizer. The text quant path is GPU-only, so this needs a CUDA build:
//   cmake -B build-cuda -DBROTENSOR_WITH_CUDA=ON
//   cmake --build build-cuda --config Release --target brolm_run_mistral_image
//   ./build-cuda/tools/Release/brolm_run_mistral_image \
//       --image cat.jpg --prompt "What is in this image?"
//
// With no explicit paths it uses the layout from scripts/download_mistral_gguf.sh
// (text Q4_K_M + MMPROJ=1) and download_mistral.ps1 (tekken.json).
//
// NOT run by ctest — needs the checkpoints, a GPU, and an image file.

#include "brolm/mistral3_config.h"
#include "brolm/mistral3_vl.h"
#include "brolm/mistral3_preprocessor.h"
#include "brolm/mistral_tokenizer.h"
#include "brolm/detail/generate.h"

#include "broimage/buffer.h"
#include "broimage/decode.h"

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
namespace fs = std::filesystem;

namespace {

std::string resolve(const std::string& arg, const std::string& def_dir,
                    const char* must_contain, const char* must_not) {
    if (!arg.empty()) return arg;
    if (!fs::is_directory(def_dir)) return {};
    for (const auto& e : fs::directory_iterator(def_dir)) {
        if (e.path().extension() != ".gguf") continue;
        const std::string n = e.path().filename().string();
        if (must_contain && n.find(must_contain) == std::string::npos) continue;
        if (must_not && n.find(must_not) != std::string::npos) continue;
        return e.path().string();
    }
    return {};
}

// Encode a single special token string to its id (asserts a 1-token result).
int special_id(const brolm::mistral::Tokenizer& tok, const std::string& s) {
    std::vector<int32_t> ids = tok.encode(s, /*add_special=*/false);
    if (ids.size() != 1) {
        std::fprintf(stderr, "warning: '%s' did not encode to a single id (%zu)\n",
                     s.c_str(), ids.size());
    }
    return ids.empty() ? -1 : ids[0];
}

void usage(const char* a0) {
    std::printf("usage: %s --image PATH [--prompt TEXT] [--text-gguf F] [--mmproj F]\n"
                "          [--tok tekken.json] [--max-new N] [--temp T] [--seed S]\n", a0);
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    std::string text_arg, mmproj_arg, tok_arg, image_path;
    std::string prompt = "Describe this image.";
    brolm::detail::GenerateOptions opts;
    opts.max_new_tokens = 64;
    opts.sampling.temperature = 0.0f;

    for (int i = 1; i < argc; ++i) {
        auto next = [&](const char* fl) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "error: %s needs an argument\n", fl); std::exit(2); }
            return argv[++i];
        };
        const char* a = argv[i];
        if      (!std::strcmp(a, "--image"))     image_path = next(a);
        else if (!std::strcmp(a, "--prompt"))    prompt = next(a);
        else if (!std::strcmp(a, "--text-gguf")) text_arg = next(a);
        else if (!std::strcmp(a, "--mmproj"))    mmproj_arg = next(a);
        else if (!std::strcmp(a, "--tok"))       tok_arg = next(a);
        else if (!std::strcmp(a, "--max-new"))   opts.max_new_tokens = std::atoi(next(a));
        else if (!std::strcmp(a, "--temp"))      opts.sampling.temperature = static_cast<float>(std::atof(next(a)));
        else if (!std::strcmp(a, "--seed"))      opts.sampling.seed = static_cast<uint64_t>(std::strtoull(next(a), nullptr, 10));
        else if (!std::strcmp(a, "--help") || !std::strcmp(a, "-h")) { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "error: unknown arg '%s'\n", a); usage(argv[0]); return 2; }
    }
    if (image_path.empty()) { std::fprintf(stderr, "error: --image is required\n"); usage(argv[0]); return 2; }

    const std::string ggdir = "weights/Mistral-Small-3.1-24B-Instruct-2503-GGUF";
    const std::string text_path   = resolve(text_arg, ggdir, nullptr, "mmproj");
    const std::string mmproj_path = resolve(mmproj_arg, ggdir, "mmproj", nullptr);
    const std::string tok_path = tok_arg.empty()
        ? (fs::path("weights") / "Mistral-Small-3.1-24B-Instruct-2503" / "tokenizer.json.tekken").string()
        : tok_arg;
    if (text_path.empty() || mmproj_path.empty()) {
        std::fprintf(stderr, "error: need a text gguf and an mmproj gguf (pass --text-gguf/--mmproj)\n");
        return 1;
    }

    try {
        bt::init();
        if (bt::default_device() == bt::Device::CPU) {
            std::fprintf(stderr, "error: the text gguf quant path is GPU-only; build with -DBROTENSOR_WITH_CUDA=ON\n");
            return 1;
        }

        // tokenizer (tekken.json — the CLI accepts any path via --tok).
        std::string tokfile = tok_arg;
        if (tokfile.empty()) {
            const fs::path d = fs::path("weights") / "Mistral-Small-3.1-24B-Instruct-2503";
            for (const char* cand : {"tekken.json", "tokenizer.json"}) {
                if (fs::exists(d / cand)) { tokfile = (d / cand).string(); break; }
            }
        }
        if (tokfile.empty() || !fs::exists(tokfile)) { std::fprintf(stderr, "error: tokenizer not found (pass --tok tekken.json)\n"); return 1; }
        std::printf("[load] tokenizer: %s\n", tokfile.c_str());
        brolm::mistral::Tokenizer tok = brolm::mistral::Tokenizer::load(tokfile);

        // config from text gguf + vision from mmproj.
        std::printf("[load] text gguf: %s\n", text_path.c_str());
        bt::gguf::File text_f = bt::gguf::File::open(text_path);
        std::printf("[load] mmproj:    %s\n", mmproj_path.c_str());
        bt::gguf::File mmproj_f = bt::gguf::File::open(mmproj_path);

        m3::Mistral3Config cfg = m3::Mistral3Config::from_gguf(text_f);
        cfg.merge_vision_from_mmproj_gguf(mmproj_f);
        std::printf("[info] text hidden=%d layers=%d | vision hidden=%d layers=%d patch=%d merge=%d\n",
                    cfg.text.hidden_size, cfg.text.num_hidden_layers,
                    cfg.vision.hidden_size, cfg.vision.num_hidden_layers,
                    cfg.vision.patch_size, cfg.spatial_merge_size);

        m3::VLModel vl(cfg);
        vl.load_weights(text_f, mmproj_f);
        std::printf("[info] weights loaded\n");

        // image → CHW float [0,1] RGB. "--image synthetic[:WxH]" generates a
        // procedural pattern so the pipeline can be smoke-tested without a file.
        int W = 0, H = 0;
        std::vector<float> chw;
        if (image_path.rfind("synthetic", 0) == 0) {
            W = H = 448;
            if (auto colon = image_path.find(':'); colon != std::string::npos) {
                int w = 0, h = 0;
                if (std::sscanf(image_path.c_str() + colon + 1, "%dx%d", &w, &h) == 2 && w > 0 && h > 0) {
                    W = w; H = h;
                }
            }
            std::printf("[info] synthetic image %dx%d\n", W, H);
            chw.assign(static_cast<std::size_t>(3) * H * W, 0.0f);
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    const float r = static_cast<float>(x) / W;
                    const float g = static_cast<float>(y) / H;
                    const bool box = (x > W / 4 && x < 3 * W / 4 && y > H / 4 && y < 3 * H / 4);
                    chw[(static_cast<std::size_t>(0) * H + y) * W + x] = r;
                    chw[(static_cast<std::size_t>(1) * H + y) * W + x] = g;
                    chw[(static_cast<std::size_t>(2) * H + y) * W + x] = box ? 1.0f : 0.2f;
                }
        } else {
            broimage::Image img;
            std::string derr;
            if (!broimage::decode_file(image_path, img, &derr)) {
                std::fprintf(stderr, "error: failed to decode '%s': %s\n", image_path.c_str(), derr.c_str());
                return 1;
            }
            std::printf("[info] image %dx%d (%d ch)\n", img.width, img.height, img.channels);
            W = img.width; H = img.height;
            chw.assign(static_cast<std::size_t>(3) * H * W, 0.0f);
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x)
                    for (int c = 0; c < 3; ++c)
                        chw[(static_cast<std::size_t>(c) * H + y) * W + x] =
                            img.pixels[(static_cast<std::size_t>(y) * W + x) * img.channels + c] / 255.0f;
        }

        // preprocess → patches + grid.
        m3::PreprocessConfig pcfg = m3::PreprocessConfig::from_config(cfg);
        m3::PreprocessedImage pim = m3::preprocess_image(chw.data(), H, W, pcfg);
        std::printf("[info] preprocessed: grid %dx%d → %d image tokens\n",
                    pim.grid_h, pim.grid_w, pim.num_image_tokens());

        // build the prompt: <s>[INST] {image span} {text} [/INST]
        const int IMG = cfg.image_token_index;
        const int BRK = special_id(tok, "[IMG_BREAK]");
        const int END = special_id(tok, "[IMG_END]");
        const int INST = special_id(tok, "[INST]");
        const int EINST = special_id(tok, "[/INST]");

        std::vector<int32_t> ids;
        ids.push_back(tok.bos_id());
        ids.push_back(INST);
        std::vector<int> span = m3::build_image_token_span(pim, IMG, BRK, END);
        ids.insert(ids.end(), span.begin(), span.end());
        std::vector<int32_t> ptext = tok.encode(prompt, /*add_special=*/false);
        ids.insert(ids.end(), ptext.begin(), ptext.end());
        ids.push_back(EINST);
        std::printf("[info] prompt tokens: %zu (image span %zu)\n", ids.size(), span.size());

        std::printf("\n--- prompt ---\n%s\n--- reply ---\n", prompt.c_str());
        std::vector<int32_t> out = vl.generate(ids, {pim}, IMG, tok.eos_id(), opts);
        std::printf("%s\n\n[info] generated %zu tokens\n", tok.decode(out).c_str(), out.size());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 1;
    }
    return 0;
}
