// Q8_0 GGUF forward smoke test — regression cover for the embed/norm dequant
// path. Before upload_compute_dequant was added, Q8_0 token_embd.weight was
// passed through verbatim to embedding_lookup_forward, whose CUDA kernel
// silently reinterpreted Q8 bytes as FP32 and tagged the output as Q8_0;
// the very next op (RMSNorm) then threw a dtype-mismatch contract error.
//
// This test:
//   1. Finds a Q8_0 .gguf (env var BROLM_QWEN3_GGUF_Q8 or default path).
//   2. Loads the model; on FP32/CPU compute that load is expected to throw a
//      clear "quant requires FP16 compute" error — we just check load doesn't
//      silently succeed.
//   3. On FP16 (GPU) compute: runs a length-4 forward and asserts the logits
//      are all finite — proving the embed tensor was actually dequanted and
//      the downstream RMSNorm sees a real FP16 input.
//
// Skipped if no Q8 .gguf is present.

#include "brolm/qwen.h"
#include "brolm/detail/compute.h"

#include "brotensor/gguf.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace qwen = brolm::qwen;
namespace bt   = brotensor;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

namespace {

std::string find_q8_gguf() {
    if (const char* env = std::getenv("BROLM_QWEN3_GGUF_Q8")) {
        if (*env && std::filesystem::exists(env)) return env;
    }
    namespace fs = std::filesystem;
    const fs::path dir = fs::path("weights") / "Qwen3-0.6B-GGUF";
    if (!fs::is_directory(dir)) return {};
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() != ".gguf") continue;
        const std::string name = entry.path().filename().string();
        if (name.find("Q8_0") != std::string::npos ||
            name.find("q8_0") != std::string::npos) {
            return entry.path().string();
        }
    }
    return {};
}

bool all_finite(const std::vector<float>& v) {
    for (float x : v) if (!std::isfinite(x)) return false;
    return true;
}

}  // namespace

int main() {
    const std::string path = find_q8_gguf();
    if (path.empty()) {
        std::printf("[skip] no Qwen3 Q8_0 .gguf found "
                    "(set BROLM_QWEN3_GGUF_Q8 or download via "
                    "scripts/download_qwen3_gguf.sh)\n");
        return 0;
    }
    std::printf("[info] using %s\n", path.c_str());

    bt::init();
    bt::gguf::File f = bt::gguf::File::open(path);

    // Confirm the file actually carries the quant payload we expect — guards
    // against accidentally pointing the test at the dense file.
    bool any_q8 = false;
    for (const auto& info : f.tensors()) {
        if (info.dtype == bt::Dtype::Q8_0) { any_q8 = true; break; }
    }
    CHECK(any_q8);

    qwen::Qwen3Config cfg = qwen::Qwen3Config::from_gguf(f);
    qwen::Qwen3Model model(cfg);

    const bool fp16_compute = (brolm::compute_dtype() == bt::Dtype::FP16);

    if (!fp16_compute) {
        // Quant + FP32/CPU compute is unsupported (no CPU dequant op); the
        // load must throw a clear error rather than silently produce a
        // broken model.
        bool threw = false;
        try {
            model.load_weights(f);
        } catch (const std::exception&) {
            threw = true;
        }
        CHECK(threw);
        if (g_failures) {
            std::fprintf(stderr, "FAILED: %d check(s)\n", g_failures);
            return 1;
        }
        std::printf("OK (load-rejected on FP32 compute)\n");
        return 0;
    }

    // FP16/GPU compute path — load and forward must produce finite logits.
    try {
        model.load_weights(f);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "load_weights threw on FP16 compute: %s\n", e.what());
        return 1;
    }

    const int L_prefill = 4;
    const std::vector<int32_t> ids = {1, 2, 3, 4};
    model.allocate_cache(L_prefill + 1);

    bt::Tensor logits;
    model.forward(ids.data(), L_prefill, logits);
    CHECK(logits.rows == L_prefill);
    CHECK(logits.cols == cfg.vocab_size);
    std::vector<float> v = bdtest::bd_download(logits);
    CHECK(all_finite(v));

    if (g_failures) {
        std::fprintf(stderr, "FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
