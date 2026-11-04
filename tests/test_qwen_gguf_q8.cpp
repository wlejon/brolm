// GGUF quantized (Q8_0, Q4_K, Q6_K) weight upload, dequant, and linear smoke test.
//
// Verifies:
//   1. Synthetic GGUF fixture with Q8_0, Q4_K, and Q6_K tensors on CPU (FP32 compute):
//      - GgufSource::upload_compute_checked preserves Q8_0 / Q4_K / Q6_K dtype.
//      - GgufSource::upload_compute_dequant dequantizes Q8_0 / Q4_K / Q6_K to
//        compute_dtype() (FP32 on CPU, FP16 on GPU) and matches the reference.
//      - brolm::detail::linear_batched and brolm::detail::linear_vec work on CPU
//        with FP32 activations and biases across Q8_0, Q4_K, and Q6_K.
//   2. Real Qwen3 Q8_0 .gguf model load + length-4 forward (if present on disk).

#include "brolm/qwen.h"
#include "brolm/detail/compute.h"
#include "brolm/detail/weights.h"

#include "brotensor/gguf.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
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

void put_u32(std::vector<uint8_t>& b, uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(static_cast<uint8_t>(v >> (8 * i)));
}
void put_u64(std::vector<uint8_t>& b, uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back(static_cast<uint8_t>(v >> (8 * i)));
}
void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put_u64(b, s.size());
    for (char c : s) b.push_back(static_cast<uint8_t>(c));
}
std::size_t align32(std::size_t n) { return (n + 31u) & ~std::size_t{31u}; }

// Test Q8_0, Q4_K, and Q6_K upload_compute_checked, upload_compute_dequant,
// linear_batched, and linear_vec on CPU (FP32 activations).
void test_cpu_quant_gguf_ops() {
    const bt::Device prev_dev = bt::default_device();
    bt::set_default_device(bt::Device::CPU);
    CHECK(brolm::compute_dtype() == bt::Dtype::FP32);

    constexpr int rows = 4;
    constexpr int cols = 256;
    const std::size_t q8_bytes  = static_cast<std::size_t>(rows) * (cols / 32) * 34;
    const std::size_t q4k_bytes = static_cast<std::size_t>(rows) * (cols / 256) * 144;
    const std::size_t q6k_bytes = static_cast<std::size_t>(rows) * (cols / 256) * 210;

    // Build deterministic raw blocks for Q8_0, Q4_K, and Q6_K.
    std::vector<uint8_t> raw_q8(q8_bytes, 0);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < cols / 32; ++b) {
            uint8_t* blk = raw_q8.data() + (r * (cols / 32) + b) * 34;
            uint16_t d = bt::fp32_to_fp16_bits(0.125f);
            std::memcpy(blk, &d, 2);
            for (int i = 0; i < 32; ++i) {
                blk[2 + i] = static_cast<uint8_t>(static_cast<int8_t>((r + 1) * ((i % 5) - 2)));
            }
        }
    }

    std::vector<uint8_t> raw_q4k(q4k_bytes, 0);
    for (int r = 0; r < rows; ++r) {
        uint8_t* blk = raw_q4k.data() + r * 144;
        uint16_t d    = bt::fp32_to_fp16_bits(0.25f);
        uint16_t dmin = bt::fp32_to_fp16_bits(0.125f);
        std::memcpy(blk + 0, &d, 2);
        std::memcpy(blk + 2, &dmin, 2);
        for (int j = 0; j < 4; ++j) {
            blk[4 + j]     = static_cast<uint8_t>(r + 1); // sc[0..3]
            blk[4 + 4 + j] = static_cast<uint8_t>(1);     // m[0..3]
        }
        for (int j = 4; j < 8; ++j) {
            blk[4 + j + 4] = static_cast<uint8_t>((r + 1) | (1 << 4)); // sc[4..7], m[4..7] low 4 bits
        }
        for (int i = 0; i < 128; ++i) {
            uint8_t lo = static_cast<uint8_t>((i + r) % 7);
            uint8_t hi = static_cast<uint8_t>((i + r + 3) % 7);
            blk[16 + i] = static_cast<uint8_t>(lo | (hi << 4));
        }
    }

    std::vector<uint8_t> raw_q6k(q6k_bytes, 0);
    for (int r = 0; r < rows; ++r) {
        uint8_t* blk = raw_q6k.data() + r * 210;
        for (int i = 0; i < 128; ++i) {
            uint8_t lo = static_cast<uint8_t>((i + r) & 0x0F);
            uint8_t hi = static_cast<uint8_t>((i + 2 * r) & 0x0F);
            blk[i] = static_cast<uint8_t>(lo | (hi << 4));
        }
        for (int i = 0; i < 64; ++i) {
            blk[128 + i] = 0x55; // high2 = 1 for all quads -> val6 = raw4 + 16 - 32 = raw4 - 16
        }
        for (int i = 0; i < 16; ++i) {
            blk[192 + i] = static_cast<uint8_t>(static_cast<int8_t>(r + 1));
        }
        uint16_t d = bt::fp32_to_fp16_bits(0.0625f);
        std::memcpy(blk + 208, &d, 2);
    }

    // Write temporary GGUF file.
    auto tmp_path = std::filesystem::temp_directory_path() / "brolm_cpu_quant_test.gguf";
    {
        std::vector<uint8_t> buf;
        put_u32(buf, 0x46554747u); // "GGUF"
        put_u32(buf, 3);           // version
        put_u64(buf, 3);           // tensor_count
        put_u64(buf, 0);           // metadata_kv_count

        const std::size_t off_q8  = 0;
        const std::size_t off_q4k = align32(off_q8 + q8_bytes);
        const std::size_t off_q6k = align32(off_q4k + q4k_bytes);

        auto put_tensor_info = [&](const std::string& name, uint32_t ggml_type, std::size_t offset) {
            put_str(buf, name);
            put_u32(buf, 2);
            put_u64(buf, cols); // dim 0 (innermost)
            put_u64(buf, rows); // dim 1
            put_u32(buf, ggml_type);
            put_u64(buf, offset);
        };
        put_tensor_info("w_q8",  8,  off_q8);
        put_tensor_info("w_q4k", 12, off_q4k);
        put_tensor_info("w_q6k", 14, off_q6k);

        while (buf.size() % 32 != 0) buf.push_back(0);
        const std::size_t data_start = buf.size();

        buf.insert(buf.end(), raw_q8.begin(), raw_q8.end());
        while (buf.size() - data_start < off_q4k) buf.push_back(0);
        buf.insert(buf.end(), raw_q4k.begin(), raw_q4k.end());
        while (buf.size() - data_start < off_q6k) buf.push_back(0);
        buf.insert(buf.end(), raw_q6k.begin(), raw_q6k.end());

        std::ofstream ofs(tmp_path, std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    }

    {
        bt::gguf::File gf = bt::gguf::File::open(tmp_path.string());
        brolm::detail::weights::GgufSource src(
            {&gf}, [](std::string_view hf) { return std::string(hf); });

        std::vector<float> h_x(cols), h_X(2 * cols), h_bias(rows);
        for (int i = 0; i < cols; ++i) {
            h_x[i] = 0.01f * static_cast<float>((i % 13) - 6);
            h_X[i] = h_x[i];
            h_X[cols + i] = -0.5f * h_x[i];
        }
        for (int r = 0; r < rows; ++r) h_bias[r] = 0.1f * static_cast<float>(r + 1);

        bt::Tensor x_f32    = bt::Tensor::from_host_on(bt::Device::CPU, h_x.data(), cols, 1);
        bt::Tensor X_bd_f32 = bt::Tensor::from_host_on(bt::Device::CPU, h_X.data(), 2, cols);
        bt::Tensor bias_f32 = bt::Tensor::from_host_on(bt::Device::CPU, h_bias.data(), rows, 1);

        const char* names[3] = {"w_q8", "w_q4k", "w_q6k"};
        const bt::Dtype expected_dt[3] = {bt::Dtype::Q8_0, bt::Dtype::Q4_K, bt::Dtype::Q6_K};

        for (int t = 0; t < 3; ++t) {
            bt::Tensor W_quant;
            src.upload_compute_checked(names[t], rows, cols, W_quant, "test");
            CHECK(W_quant.device == bt::Device::CPU);
            CHECK(W_quant.dtype == expected_dt[t]);

            bt::Tensor W_deq;
            src.upload_compute_dequant(names[t], rows, cols, W_deq, "test");
            CHECK(W_deq.device == bt::Device::CPU);
            CHECK(W_deq.dtype == bt::Dtype::FP32);
            CHECK(W_deq.rows == rows && W_deq.cols == cols);

            // Reference dense FP32 linear vs quantized linear_vec and linear_batched
            bt::Tensor y_ref, Y_ref;
            brolm::detail::linear_vec(W_deq, &bias_f32, x_f32, y_ref);
            brolm::detail::linear_batched(W_deq, &bias_f32, X_bd_f32, Y_ref);

            bt::Tensor y_quant, Y_quant;
            brolm::detail::linear_vec(W_quant, &bias_f32, x_f32, y_quant);
            brolm::detail::linear_batched(W_quant, &bias_f32, X_bd_f32, Y_quant);

            CHECK(y_quant.dtype == bt::Dtype::FP32);
            CHECK(Y_quant.dtype == bt::Dtype::FP32);

            std::vector<float> hy_ref = y_ref.to_host_vector();
            std::vector<float> hy_q   = y_quant.to_host_vector();
            for (int r = 0; r < rows; ++r) {
                CHECK(std::fabs(hy_ref[r] - hy_q[r]) < 1e-4f);
            }

            std::vector<float> hY_ref = Y_ref.to_host_vector();
            std::vector<float> hY_q   = Y_quant.to_host_vector();
            for (int i = 0; i < 2 * rows; ++i) {
                CHECK(std::fabs(hY_ref[i] - hY_q[i]) < 1e-4f);
            }
        }
    }

    std::error_code ec;
    std::filesystem::remove(tmp_path, ec);
    bt::set_default_device(prev_dev);
}

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
    bt::init();

    // 1. Always verify CPU Q8_0 / Q4_K / Q6_K upload_compute_dequant,
    //    linear_batched, and linear_vec on a synthetic GGUF fixture.
    test_cpu_quant_gguf_ops();
    if (g_failures) {
        std::fprintf(stderr, "FAILED synthetic CPU GGUF quant checks: %d\n", g_failures);
        return 1;
    }
    std::printf("[ok] CPU Q8_0 / Q4_K / Q6_K upload_compute_dequant + linear_batched + linear_vec\n");

    // 2. Optional full-model Q8_0 GGUF test if checkpoint is present.
    const std::string path = find_q8_gguf();
    if (path.empty()) {
        std::printf("[skip] no Qwen3 Q8_0 .gguf found "
                    "(set BROLM_QWEN3_GGUF_Q8 or download via "
                    "scripts/download_qwen3_gguf.sh)\n");
        return 0;
    }
    std::printf("[info] using %s\n", path.c_str());

    bt::gguf::File f = bt::gguf::File::open(path);

    bool any_q8 = false;
    for (const auto& info : f.tensors()) {
        if (info.dtype == bt::Dtype::Q8_0) { any_q8 = true; break; }
    }
    CHECK(any_q8);

    qwen::Qwen3Config cfg = qwen::Qwen3Config::from_gguf(f);
    qwen::Qwen3Model model(cfg);

    try {
        model.load_weights(f);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "load_weights threw: %s\n", e.what());
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
