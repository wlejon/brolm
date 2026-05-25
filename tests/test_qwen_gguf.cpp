// End-to-end Qwen3 GGUF smoke test — tokenizer, config, and weights all read
// out of a single .gguf file. Gated on the file being present; skips cleanly
// otherwise so the suite still runs without large checkpoints downloaded.
//
// Discovery order:
//   1. $BROLM_QWEN3_GGUF — explicit path.
//   2. <repo>/weights/Qwen3-0.6B-GGUF/Qwen3-0.6B-f16.gguf — the default
//      produced by scripts/download_qwen3_gguf.sh.
//
// The dense FP16 variant is preferred because it runs on every backend; the
// quant variants (Q8_0 / Q4_K_M / Q6_K) currently require CUDA at forward
// time. If a quant file is the only one present, the test exercises the
// config / tokenizer / weight-load path and skips the forward.
//
// Checks:
//   1. Tokenizer roundtrip: encode(decode("hello world")) is non-empty and
//      decode(encode("hello world")) == "hello world".
//   2. ChatML special tokens resolve (<|im_start|>, <|im_end|>, <|endoftext|>).
//   3. Qwen3Config::from_gguf returns positive dimensions and num_attention_heads
//      divisible by num_key_value_heads (GQA invariant).
//   4. Qwen3Model::load_weights(gguf) succeeds.
//   5. Forward on a length-4 prompt produces (4, vocab) finite logits.
//   6. Prefill+decode equivalence: feeding the same 4 tokens as 4 length-1
//      forwards yields the same final-row logits as the single length-4 call.

#include "brolm/qwen.h"
#include "brolm/qwen_tokenizer.h"
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

std::string find_gguf() {
    if (const char* env = std::getenv("BROLM_QWEN3_GGUF")) {
        if (*env && std::filesystem::exists(env)) return env;
    }
    namespace fs = std::filesystem;
    const fs::path dir = fs::path("weights") / "Qwen3-0.6B-GGUF";
    if (!fs::is_directory(dir)) return {};

    // Prefer the dense (BF16 / F16) variant so the forward path runs on any
    // backend; fall back to any .gguf so quant-only downloads still exercise
    // the loader (with the forward skipped on CPU).
    std::string fallback;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() != ".gguf") continue;
        const std::string name = entry.path().filename().string();
        if (name.find("BF16") != std::string::npos ||
            name.find("bf16") != std::string::npos ||
            name.find("F16")  != std::string::npos ||
            name.find("f16")  != std::string::npos) {
            return entry.path().string();
        }
        if (fallback.empty()) fallback = entry.path().string();
    }
    return fallback;
}

bool all_finite(const std::vector<float>& v) {
    for (float x : v) if (!std::isfinite(x)) return false;
    return true;
}

}  // namespace

int main() {
    const std::string path = find_gguf();
    if (path.empty()) {
        std::printf("[skip] no Qwen3 .gguf found "
                    "(set BROLM_QWEN3_GGUF or run scripts/download_qwen3_gguf.sh)\n");
        return 0;
    }
    std::printf("[info] using %s\n", path.c_str());

    bt::init();

    bt::gguf::File f = bt::gguf::File::open(path);
    std::printf("[info] tensors=%zu  metadata keys=%zu\n",
                f.tensor_count(), f.metadata().size());

    // ── Tokenizer ─────────────────────────────────────────────────────────
    qwen::Tokenizer tok = qwen::Tokenizer::from_gguf(f);
    CHECK(tok.vocab_count() > 100000);
    CHECK(tok.merge_count() > 0);
    CHECK(tok.im_start_id() >= 0);
    CHECK(tok.im_end_id()   >= 0);

    {
        const std::string s = "hello world";
        auto ids = tok.encode(s);
        CHECK(!ids.empty());
        std::string back = tok.decode(ids);
        CHECK(back == s);
    }

    // ── Config ────────────────────────────────────────────────────────────
    qwen::Qwen3Config cfg = qwen::Qwen3Config::from_gguf(f);
    CHECK(cfg.vocab_size            > 0);
    CHECK(cfg.hidden_size           > 0);
    CHECK(cfg.intermediate_size     > 0);
    CHECK(cfg.num_hidden_layers     > 0);
    CHECK(cfg.num_attention_heads   > 0);
    CHECK(cfg.num_key_value_heads   > 0);
    CHECK(cfg.head_dim              > 0);
    CHECK(cfg.num_attention_heads % cfg.num_key_value_heads == 0);
    CHECK(cfg.rms_norm_eps > 0.0f);
    CHECK(cfg.rope_theta   > 0.0f);
    std::printf("[info] cfg: hidden=%d ff=%d layers=%d q=%d kv=%d hd=%d vocab=%d\n",
                cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers,
                cfg.num_attention_heads, cfg.num_key_value_heads,
                cfg.head_dim, cfg.vocab_size);

    // ── Weights ───────────────────────────────────────────────────────────
    qwen::Qwen3Model model(cfg);
    bool weight_load_ok = true;
    try {
        model.load_weights(f);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "load_weights threw: %s\n", e.what());
        weight_load_ok = false;
        ++g_failures;
    }

    // If any tensor was quantized and we're on the CPU backend, the forward
    // would throw at the first matmul (no CPU quant kernels). In that case
    // skip the forward checks but keep the config/tokenizer/load assertions.
    bool any_quant = false;
    for (const auto& info : f.tensors()) {
        if (bt::dtype_is_quant(info.dtype)) { any_quant = true; break; }
    }
    const bool can_forward = weight_load_ok && (!any_quant ||
                                                bt::default_device() != bt::Device::CPU);
    if (!can_forward) {
        std::printf("[skip] forward — quant tensors present and CPU backend "
                    "has no quant matmul kernels yet\n");
        if (g_failures) {
            std::fprintf(stderr, "FAILED: %d check(s)\n", g_failures);
            return 1;
        }
        std::printf("OK (load-only)\n");
        return 0;
    }

    // ── Forward ───────────────────────────────────────────────────────────
    const int L_prefill = 4;
    std::vector<int32_t> ids = {1, 2, 3, 4};
    while (static_cast<int>(ids.size()) < L_prefill) ids.push_back(5);
    ids.resize(L_prefill);

    model.allocate_cache(L_prefill + 4);

    bt::Tensor logits_a;
    model.forward(ids.data(), L_prefill, logits_a);
    CHECK(logits_a.rows == L_prefill);
    CHECK(logits_a.cols == cfg.vocab_size);
    std::vector<float> a = bdtest::bd_download(logits_a);
    CHECK(all_finite(a));

    // ── Prefill vs streamed decode ────────────────────────────────────────
    qwen::Qwen3Model model_b(cfg);
    model_b.load_weights(f);
    model_b.allocate_cache(L_prefill + 4);

    bt::Tensor row;
    std::vector<float> last_row;
    for (int i = 0; i < L_prefill; ++i) {
        model_b.forward(&ids[i], 1, row);
        CHECK(row.rows == 1);
        CHECK(row.cols == cfg.vocab_size);
        last_row = bdtest::bd_download(row);
    }
    CHECK(all_finite(last_row));

    // The last row of the prefill should agree with the final decode step
    // within a tight FP16 tolerance (these are FP16 GEMM paths on a GPU
    // backend and FP32 on CPU — pick a tolerance that covers both).
    const std::size_t V = static_cast<std::size_t>(cfg.vocab_size);
    const float tol = (brolm::compute_dtype() == bt::Dtype::FP16) ? 1e-1f : 1e-3f;
    std::size_t mismatches = 0;
    float max_abs = 0.0f;
    for (std::size_t j = 0; j < V; ++j) {
        const float ref = a[(L_prefill - 1) * V + j];
        const float got = last_row[j];
        const float diff = std::fabs(ref - got);
        if (diff > max_abs) max_abs = diff;
        if (diff > tol) ++mismatches;
    }
    if (mismatches) {
        std::fprintf(stderr,
                     "prefill vs decode: %zu/%zu vocab logits exceed tol %g "
                     "(max_abs=%g)\n",
                     mismatches, V, tol, max_abs);
        ++g_failures;
    }

    if (g_failures) {
        std::fprintf(stderr, "FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
