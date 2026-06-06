// End-to-end Mistral 3.1 text-decoder GGUF smoke test — config + weights read
// out of a single llama.cpp .gguf file. Gated on the file being present; skips
// cleanly otherwise so the suite still runs without the ~14 GB checkpoint.
//
// Discovery order:
//   1. $BROLM_MISTRAL_GGUF — explicit path.
//   2. <repo>/weights/Mistral-Small-3.1-24B-Instruct-2503-GGUF/*.gguf — the
//      default produced by scripts/download_mistral_gguf.sh (prefers a non-
//      mmproj text model file).
//
// On any build this prints the gguf architecture, the parsed config dims, and
// a tensor-dtype histogram — enough to eyeball that the loader read the file
// correctly. The quant variants (Q4_K / Q6_K / Q8_0) require a GPU backend at
// load + forward time (brotensor's dequant / quant-matmul kernels are CUDA-
// only); on the CPU backend the test stops after the config + histogram and
// skips the load/forward.
//
// Checks (GPU backend):
//   1. Mistral3Config::from_gguf returns positive dims and a GQA-divisible head
//      split; head_dim is read from key_length (128, not hidden/heads).
//   2. TextModel::load_weights(gguf) succeeds.
//   3. Forward on a length-4 prompt produces (4, vocab) finite logits.
//   4. Prefill+decode equivalence: the same 4 tokens fed as 4 length-1 forwards
//      yield the same final-row logits as the single length-4 call.

#include "brolm/mistral3_config.h"
#include "brolm/mistral3_text.h"
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
#include <map>
#include <string>
#include <vector>

namespace m3 = brolm::mistral3;
namespace bt = brotensor;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

namespace {

std::string find_gguf() {
    if (const char* env = std::getenv("BROLM_MISTRAL_GGUF")) {
        if (*env && std::filesystem::exists(env)) return env;
    }
    namespace fs = std::filesystem;
    const fs::path dir =
        fs::path("weights") / "Mistral-Small-3.1-24B-Instruct-2503-GGUF";
    if (!fs::is_directory(dir)) return {};
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() != ".gguf") continue;
        const std::string name = entry.path().filename().string();
        // Skip the vision projector (mmproj) gguf — we want the text model.
        if (name.find("mmproj") != std::string::npos) continue;
        return entry.path().string();
    }
    return {};
}

const char* dtype_name(bt::Dtype d) {
    switch (d) {
        case bt::Dtype::FP32: return "FP32";
        case bt::Dtype::FP16: return "FP16";
        case bt::Dtype::BF16: return "BF16";
        case bt::Dtype::INT8: return "INT8";
        case bt::Dtype::INT32: return "INT32";
        case bt::Dtype::Q4_0: return "Q4_0";
        case bt::Dtype::Q4_1: return "Q4_1";
        case bt::Dtype::Q5_0: return "Q5_0";
        case bt::Dtype::Q5_1: return "Q5_1";
        case bt::Dtype::Q8_0: return "Q8_0";
        case bt::Dtype::Q2_K: return "Q2_K";
        case bt::Dtype::Q3_K: return "Q3_K";
        case bt::Dtype::Q4_K: return "Q4_K";
        case bt::Dtype::Q5_K: return "Q5_K";
        case bt::Dtype::Q6_K: return "Q6_K";
        default: return "?";
    }
}

// brolm dispatches only Q4_K / Q6_K / Q8_0 through brotensor's quant kernels.
bool brolm_supports(bt::Dtype d) {
    return !bt::dtype_is_quant(d) ||
           d == bt::Dtype::Q4_K || d == bt::Dtype::Q6_K || d == bt::Dtype::Q8_0;
}

bool all_finite(const std::vector<float>& v) {
    for (float x : v) if (!std::isfinite(x)) return false;
    return true;
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);  // unbuffered: survive a hard crash
    const std::string path = find_gguf();
    if (path.empty()) {
        std::printf("[skip] no Mistral .gguf found (set BROLM_MISTRAL_GGUF or "
                    "run scripts/download_mistral_gguf.sh)\n");
        return 0;
    }
    std::printf("[info] using %s\n", path.c_str());

    bt::init();

    bt::gguf::File f = bt::gguf::File::open(path);
    std::printf("[info] tensors=%zu  metadata keys=%zu\n",
                f.tensor_count(), f.metadata().size());
    if (const auto* arch = f.find_meta("general.architecture");
        arch && arch->type == bt::gguf::ValueType::String) {
        std::printf("[info] general.architecture = %s\n", arch->str.c_str());
    }

    // ── Config ────────────────────────────────────────────────────────────
    m3::Mistral3Config cfg;
    try {
        cfg = m3::Mistral3Config::from_gguf(f);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "from_gguf threw: %s\n", e.what());
        return 1;
    }
    const auto& t = cfg.text;
    std::printf("[info] cfg: hidden=%d ff=%d layers=%d q=%d kv=%d hd=%d vocab=%d "
                "eps=%g theta=%g tied=%d\n",
                t.hidden_size, t.intermediate_size, t.num_hidden_layers,
                t.num_attention_heads, t.num_key_value_heads, t.head_dim,
                t.vocab_size, static_cast<double>(t.rms_norm_eps),
                static_cast<double>(t.rope_theta), t.tie_word_embeddings);
    CHECK(t.hidden_size > 0);
    CHECK(t.intermediate_size > 0);
    CHECK(t.num_hidden_layers > 0);
    CHECK(t.num_attention_heads > 0);
    CHECK(t.num_key_value_heads > 0);
    CHECK(t.head_dim > 0);
    CHECK(t.num_attention_heads % t.num_key_value_heads == 0);
    CHECK(t.rms_norm_eps > 0.0f);
    CHECK(t.rope_theta > 0.0f);

    // ── Tensor-dtype histogram + support check ────────────────────────────
    std::map<bt::Dtype, int> hist;
    bool any_quant = false;
    bool all_supported = true;
    for (const auto& info : f.tensors()) {
        ++hist[info.dtype];
        if (bt::dtype_is_quant(info.dtype)) any_quant = true;
        if (!brolm_supports(info.dtype)) all_supported = false;
    }
    std::printf("[info] dtype histogram:");
    for (const auto& [d, n] : hist) {
        std::printf(" %s=%d", dtype_name(d), n);
    }
    std::printf("\n");
    CHECK(all_supported);  // every quant tensor must be Q4_K / Q6_K / Q8_0
    if (!all_supported) {
        std::fprintf(stderr, "[warn] file contains a quant dtype brolm cannot "
                             "dispatch — pick a Q4_K_M / Q6_K / Q8_0 quant\n");
    }

    // The quant path (dequant for the embedding, quant matmuls) is GPU-only.
    const bool can_run = !any_quant || bt::default_device() != bt::Device::CPU;
    if (!can_run) {
        std::printf("[skip] load/forward — quant tensors present and the CPU "
                    "backend has no quant kernels (build with "
                    "-DBROTENSOR_WITH_CUDA=ON to run the real load)\n");
        if (g_failures) { std::fprintf(stderr, "FAILED: %d\n", g_failures); return 1; }
        std::printf("OK (config-only)\n");
        return 0;
    }

    // ── Weights ───────────────────────────────────────────────────────────
    m3::TextModel model(cfg.text);
    try {
        model.load_weights(f);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "load_weights threw: %s\n", e.what());
        return 1;
    }
    std::printf("[info] weights loaded\n");

    // ── Forward ───────────────────────────────────────────────────────────
    const int L_prefill = 4;
    std::vector<int32_t> ids = {1, 2, 3, 4};

    bt::Tensor logits_a;
    std::vector<float> a;
    std::vector<float> last_row;
  try {
        model.allocate_cache(L_prefill + 4);
        std::printf("[info] cache allocated; running prefill forward...\n");
        model.forward(ids.data(), L_prefill, logits_a);
        bt::sync_all();
        std::printf("[info] prefill forward returned\n");
        CHECK(logits_a.rows == L_prefill);
        CHECK(logits_a.cols == t.vocab_size);
        a = bdtest::bd_download(logits_a);
        std::printf("[info] logits downloaded (%zu floats)\n", a.size());
        CHECK(all_finite(a));

        // ── Prefill vs streamed decode ────────────────────────────────────
        // Reuse the SAME model — a second 14 GB model would not fit alongside
        // the first on a 24 GB GPU. reset_cache() rewinds cache_len to 0 while
        // keeping the weights and the cache allocation, so the streamed decode
        // starts from an identical state to the prefill.
        model.reset_cache();
        std::printf("[info] cache reset; streaming decode...\n");
        for (int i = 0; i < L_prefill; ++i) {
            bt::Tensor row;
            model.forward(&ids[static_cast<std::size_t>(i)], 1, row);
            CHECK(row.rows == 1);
            CHECK(row.cols == t.vocab_size);
            last_row = bdtest::bd_download(row);
        }
        std::printf("[info] streaming decode done\n");
        CHECK(all_finite(last_row));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "forward/decode threw: %s\n", e.what());
        return 1;
    }

    const std::size_t V = static_cast<std::size_t>(t.vocab_size);
    const float tol = (brolm::compute_dtype() == bt::Dtype::FP16) ? 2e-1f : 1e-3f;
    std::size_t mismatches = 0;
    float max_abs = 0.0f;
    for (std::size_t j = 0; j < V; ++j) {
        const float ref = a[(L_prefill - 1) * V + j];
        const float got = last_row[j];
        const float diff = std::fabs(ref - got);
        if (diff > max_abs) max_abs = diff;
        if (diff > tol) ++mismatches;
    }
    std::printf("[info] prefill-vs-decode max_abs=%g mismatches=%zu/%zu\n",
                static_cast<double>(max_abs), mismatches, V);
    if (mismatches) ++g_failures;

    if (g_failures) { std::fprintf(stderr, "FAILED: %d check(s)\n", g_failures); return 1; }
    std::printf("OK\n");
    return 0;
}
