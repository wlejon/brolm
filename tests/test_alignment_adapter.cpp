// AlignmentAdapter training test.
//
// AlignmentAdapter is brolm's first trainable module — it has a backward pass
// and an Adam optimizer. The decisive test (#2) runs a real training loop on a
// fixed input + fixed random targets and asserts the loss collapses. If the
// backward gradient math is wrong the loss will not fall, so a passing loop is
// strong evidence the whole forward/backward/optimizer chain is correct.
//
// Also checks: forward shape + finiteness, deterministic init, and a
// save/load weight round-trip.

#include "brolm/alignment_adapter.h"
#include "brolm/detail/compute.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <system_error>
#include <vector>

namespace bt = brotensor;
namespace st = brotensor::safetensors;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

namespace {

// Deterministic splitmix64-based [-1, 1) host vector — input / target data.
std::vector<float> rand_vec(std::size_t n, uint64_t seed) {
    std::vector<float> out(n);
    uint64_t s = seed;
    for (std::size_t i = 0; i < n; ++i) {
        s += 0x9E3779B97F4A7C15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= (z >> 31);
        out[i] = (static_cast<float>(z >> 40) / 16777216.0f) * 2.0f - 1.0f;
    }
    return out;
}

}  // namespace

int main() {
    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }

    brolm::AlignmentAdapterConfig cfg;
    cfg.d_in    = 16;
    cfg.d_model = 24;
    cfg.d_cond  = 12;
    const int L = 5;

    // ── Test 1: forward shape + finiteness ──────────────────────────────
    {
        brolm::AlignmentAdapter adapter(cfg, /*init_seed=*/1234);
        std::vector<float> h = rand_vec(static_cast<std::size_t>(L) * cfg.d_in, 7);
        bt::Tensor H = bdtest::bd_upload(h, L, cfg.d_in);

        bt::Tensor text_emb, pooled;
        adapter.forward(H, text_emb, pooled);
        bt::sync_all();

        CHECK(text_emb.rows == L);
        CHECK(text_emb.cols == cfg.d_cond);
        CHECK(pooled.rows == 1);
        CHECK(pooled.cols == cfg.d_cond);
        CHECK(text_emb.dtype == brolm::compute_dtype());
        CHECK(pooled.dtype == brolm::compute_dtype());

        int nonfinite = 0;
        for (float v : bdtest::bd_download(text_emb))
            if (!bdtest::bd_finite(v)) ++nonfinite;
        for (float v : bdtest::bd_download(pooled))
            if (!bdtest::bd_finite(v)) ++nonfinite;
        CHECK(nonfinite == 0);
    }

    // ── Test 2: training reduces loss (the key correctness gate) ────────
    {
        brolm::AlignmentAdapter adapter(cfg, /*init_seed=*/99);

        std::vector<float> h = rand_vec(static_cast<std::size_t>(L) * cfg.d_in, 21);
        std::vector<float> tt = rand_vec(static_cast<std::size_t>(L) * cfg.d_cond, 33);
        std::vector<float> tp = rand_vec(static_cast<std::size_t>(cfg.d_cond), 44);

        bt::Tensor H        = bdtest::bd_upload(h, L, cfg.d_in);
        bt::Tensor tgt_text = bdtest::bd_upload(tt, L, cfg.d_cond);
        bt::Tensor tgt_pool = bdtest::bd_upload(tp, 1, cfg.d_cond);

        bt::Tensor text_emb, pooled, d_text, d_pool;

        const int steps = 200;
        float initial_loss = 0.0f, final_loss = 0.0f;
        std::vector<float> checkpoints;

        for (int it = 0; it < steps; ++it) {
            adapter.zero_grads();
            adapter.forward(H, text_emb, pooled);

            const float loss_text = bt::mse_vec_forward(text_emb, tgt_text);
            const float loss_pool = bt::mse_vec_forward(pooled, tgt_pool);
            const float loss = loss_text + loss_pool;

            if (it == 0) initial_loss = loss;
            final_loss = loss;
            if (it % 40 == 0) checkpoints.push_back(loss);

            bt::mse_vec_backward(text_emb, tgt_text, d_text);
            bt::mse_vec_backward(pooled, tgt_pool, d_pool);
            adapter.backward(d_text, d_pool, /*dH_out=*/nullptr);
            adapter.step(/*lr=*/1e-2f);
        }

        std::printf("alignment_adapter: training loss %.6f -> %.6f\n",
                    initial_loss, final_loss);

        // The decisive assertion: if the backward is wrong the loss won't fall.
        CHECK(std::isfinite(initial_loss));
        CHECK(std::isfinite(final_loss));
        CHECK(final_loss < 0.1f * initial_loss);

        // Loss must be non-increasing across coarse checkpoints (tiny noise ok).
        for (std::size_t i = 1; i < checkpoints.size(); ++i) {
            CHECK(checkpoints[i] <= checkpoints[i - 1] + 1e-4f);
        }
    }

    // ── Test 3: deterministic init ──────────────────────────────────────
    {
        brolm::AlignmentAdapter a1(cfg, /*init_seed=*/555);
        brolm::AlignmentAdapter a2(cfg, /*init_seed=*/555);

        std::vector<float> h = rand_vec(static_cast<std::size_t>(L) * cfg.d_in, 8);
        bt::Tensor H1 = bdtest::bd_upload(h, L, cfg.d_in);
        bt::Tensor H2 = bdtest::bd_upload(h, L, cfg.d_in);

        bt::Tensor t1, p1, t2, p2;
        a1.forward(H1, t1, p1);
        a2.forward(H2, t2, p2);
        bt::sync_all();

        CHECK(bdtest::bd_download(t1) == bdtest::bd_download(t2));
        CHECK(bdtest::bd_download(p1) == bdtest::bd_download(p2));
    }

    // ── Test 4: save / load round-trip ──────────────────────────────────
    {
        brolm::AlignmentAdapter trained(cfg, /*init_seed=*/77);

        // Train a few steps so the weights diverge from their init.
        std::vector<float> h = rand_vec(static_cast<std::size_t>(L) * cfg.d_in, 5);
        std::vector<float> tt = rand_vec(static_cast<std::size_t>(L) * cfg.d_cond, 6);
        std::vector<float> tp = rand_vec(static_cast<std::size_t>(cfg.d_cond), 9);
        bt::Tensor H        = bdtest::bd_upload(h, L, cfg.d_in);
        bt::Tensor tgt_text = bdtest::bd_upload(tt, L, cfg.d_cond);
        bt::Tensor tgt_pool = bdtest::bd_upload(tp, 1, cfg.d_cond);

        bt::Tensor text_emb, pooled, d_text, d_pool;
        for (int it = 0; it < 30; ++it) {
            trained.zero_grads();
            trained.forward(H, text_emb, pooled);
            bt::mse_vec_backward(text_emb, tgt_text, d_text);
            bt::mse_vec_backward(pooled, tgt_pool, d_pool);
            trained.backward(d_text, d_pool, nullptr);
            trained.step(1e-2f);
        }

        bt::Tensor ref_text, ref_pool;
        trained.forward(H, ref_text, ref_pool);
        bt::sync_all();
        std::vector<float> ref_text_v = bdtest::bd_download(ref_text);
        std::vector<float> ref_pool_v = bdtest::bd_download(ref_pool);

        auto path = std::filesystem::temp_directory_path() /
                    "brolm_alignment_adapter_test.safetensors";
        trained.save_weights(path.string());

        brolm::AlignmentAdapter loaded(cfg, /*init_seed=*/0);
        {
            auto file = st::File::open(path.string());
            loaded.load_weights(file, "");
        }

        bt::Tensor ld_text, ld_pool;
        loaded.forward(H, ld_text, ld_pool);
        bt::sync_all();
        std::vector<float> ld_text_v = bdtest::bd_download(ld_text);
        std::vector<float> ld_pool_v = bdtest::bd_download(ld_pool);

        CHECK(ref_text_v.size() == ld_text_v.size());
        CHECK(ref_pool_v.size() == ld_pool_v.size());
        // FP tolerance: save round-trips through F32, compute dtype may be FP16.
        float max_diff = 0.0f;
        for (std::size_t i = 0; i < ref_text_v.size(); ++i)
            max_diff = std::max(max_diff,
                                std::fabs(ref_text_v[i] - ld_text_v[i]));
        for (std::size_t i = 0; i < ref_pool_v.size(); ++i)
            max_diff = std::max(max_diff,
                                std::fabs(ref_pool_v[i] - ld_pool_v[i]));
        CHECK(max_diff < 1e-3f);

        std::error_code ec;
        std::filesystem::remove(path, ec);
    }

    if (g_failures == 0) {
        std::printf("alignment_adapter: OK\n");
    } else {
        std::fprintf(stderr, "alignment_adapter: %d failure(s)\n", g_failures);
    }
    return g_failures ? 1 : 0;
}
