// Mistral 3.1 vision-language fusion test.
//
// Builds a tiny end-to-end VLM (text + Pixtral tower + projector) in one
// synthetic safetensors file with the real HF prefixes (language_model.* /
// vision_tower.* / multi_modal_projector.*), then checks the fusion wiring:
//   1. load + a 4×4-patch image (→ 2×2 = 4 image tokens) into a prompt whose
//      [IMG] span has exactly 4 placeholders;
//   2. fuse_embeds correctness — the fused stream equals, bitwise, a manual
//      build (text embeddings with the [IMG] rows overwritten by the projector
//      output) and leaves the non-[IMG] rows as the text embeddings;
//   3. image conditioning — two different images give different prefill logits
//      (the image actually flows into the decoder);
//   4. generation — greedy generate returns the right count, in-range,
//      deterministic across two instances;
//   5. a mismatched image-token count raises.

#include "brolm/mistral3_config.h"
#include "brolm/mistral3_vl.h"
#include "brolm/detail/compute.h"

#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace m3 = brolm::mistral3;
namespace st = brotensor::safetensors;
namespace bt = brotensor;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

namespace {

struct Builder {
    std::string entries;
    std::vector<uint8_t> payload;
    bool first = true;
    void add(const std::string& name, std::vector<int> shape,
             const std::vector<uint16_t>& bits) {
        std::size_t exp = 1; for (int d : shape) exp *= static_cast<std::size_t>(d);
        if (exp != bits.size()) { std::fprintf(stderr, "fixture mismatch %s\n", name.c_str()); std::abort(); }
        std::uint64_t s = payload.size();
        const std::uint8_t* b = reinterpret_cast<const std::uint8_t*>(bits.data());
        payload.insert(payload.end(), b, b + bits.size() * 2);
        std::uint64_t e = payload.size();
        if (!first) entries += ","; first = false;
        entries += "\"" + name + "\":{\"dtype\":\"F16\",\"shape\":[";
        for (std::size_t i = 0; i < shape.size(); ++i) { if (i) entries += ","; entries += std::to_string(shape[i]); }
        entries += "],\"data_offsets\":[" + std::to_string(s) + "," + std::to_string(e) + "]}";
    }
    void write(const std::filesystem::path& p) const {
        std::string h = "{" + entries + "}";
        std::uint64_t hs = h.size();
        std::ofstream f(p, std::ios::binary | std::ios::trunc);
        if (!f) std::abort();
        f.write(reinterpret_cast<const char*>(&hs), 8);
        f.write(h.data(), h.size());
        f.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    }
};

uint32_t g_seed = 1;
std::vector<uint16_t> R(std::size_t n) {
    std::vector<uint16_t> out(n);
    uint32_t s = g_seed++ * 2654435761u + 1u;
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        out[i] = bt::fp32_to_fp16_bits((static_cast<float>(s >> 8) / 16777216.0f - 0.5f) * 0.2f);
    }
    return out;
}
std::vector<uint16_t> ones(std::size_t n) { return std::vector<uint16_t>(n, bt::fp32_to_fp16_bits(1.0f)); }

}  // namespace

int main() {
    try { bt::init(); }
    catch (const std::exception& e) { std::fprintf(stderr, "init failed: %s\n", e.what()); return 1; }

    m3::Mistral3Config cfg;
    // text
    cfg.text.vocab_size = 48; cfg.text.hidden_size = 16; cfg.text.intermediate_size = 32;
    cfg.text.num_hidden_layers = 2; cfg.text.num_attention_heads = 2;
    cfg.text.num_key_value_heads = 1; cfg.text.head_dim = 8;
    cfg.text.rms_norm_eps = 1e-5f; cfg.text.rope_theta = 1e9f; cfg.text.tie_word_embeddings = false;
    // vision
    cfg.vision.hidden_size = 16; cfg.vision.num_attention_heads = 2; cfg.vision.head_dim = 8;
    cfg.vision.intermediate_size = 32; cfg.vision.num_hidden_layers = 2;
    cfg.vision.num_channels = 3; cfg.vision.patch_size = 4; cfg.vision.rope_theta = 1e4f;
    // glue
    cfg.spatial_merge_size = 2; cfg.image_token_index = 10; cfg.multimodal_projector_bias = false;

    const int V = cfg.text.vocab_size, H = cfg.text.hidden_size, F = cfg.text.intermediate_size;
    const int nq = cfg.text.num_attention_heads, nkv = cfg.text.num_key_value_heads, hd = cfg.text.head_dim;
    const int Dv = cfg.vision.hidden_size, Fv = cfg.vision.intermediate_size;
    const int C = cfg.vision.num_channels, P = cfg.vision.patch_size, CP = C * P * P;
    const int IMG = cfg.image_token_index;

    // ── build the single-file fixture ─────────────────────────────────────
    Builder b;
    // text under language_model.
    b.add("language_model.model.embed_tokens.weight", {V, H}, R(static_cast<std::size_t>(V) * H));
    for (int i = 0; i < cfg.text.num_hidden_layers; ++i) {
        const std::string p = "language_model.model.layers." + std::to_string(i) + ".";
        b.add(p + "input_layernorm.weight", {H}, ones(H));
        b.add(p + "self_attn.q_proj.weight", {nq * hd, H}, R(static_cast<std::size_t>(nq * hd) * H));
        b.add(p + "self_attn.k_proj.weight", {nkv * hd, H}, R(static_cast<std::size_t>(nkv * hd) * H));
        b.add(p + "self_attn.v_proj.weight", {nkv * hd, H}, R(static_cast<std::size_t>(nkv * hd) * H));
        b.add(p + "self_attn.o_proj.weight", {H, nq * hd}, R(static_cast<std::size_t>(H) * nq * hd));
        b.add(p + "post_attention_layernorm.weight", {H}, ones(H));
        b.add(p + "mlp.gate_proj.weight", {F, H}, R(static_cast<std::size_t>(F) * H));
        b.add(p + "mlp.up_proj.weight", {F, H}, R(static_cast<std::size_t>(F) * H));
        b.add(p + "mlp.down_proj.weight", {H, F}, R(static_cast<std::size_t>(H) * F));
    }
    b.add("language_model.model.norm.weight", {H}, ones(H));
    b.add("language_model.lm_head.weight", {V, H}, R(static_cast<std::size_t>(V) * H));
    // vision under vision_tower.
    b.add("vision_tower.patch_conv.weight", {Dv, CP}, R(static_cast<std::size_t>(Dv) * CP));
    b.add("vision_tower.ln_pre.weight", {Dv}, ones(Dv));
    for (int i = 0; i < cfg.vision.num_hidden_layers; ++i) {
        const std::string p = "vision_tower.transformer.layers." + std::to_string(i) + ".";
        b.add(p + "attention_norm.weight", {Dv}, ones(Dv));
        b.add(p + "attention.q_proj.weight", {Dv, Dv}, R(static_cast<std::size_t>(Dv) * Dv));
        b.add(p + "attention.k_proj.weight", {Dv, Dv}, R(static_cast<std::size_t>(Dv) * Dv));
        b.add(p + "attention.v_proj.weight", {Dv, Dv}, R(static_cast<std::size_t>(Dv) * Dv));
        b.add(p + "attention.o_proj.weight", {Dv, Dv}, R(static_cast<std::size_t>(Dv) * Dv));
        b.add(p + "ffn_norm.weight", {Dv}, ones(Dv));
        b.add(p + "feed_forward.gate_proj.weight", {Fv, Dv}, R(static_cast<std::size_t>(Fv) * Dv));
        b.add(p + "feed_forward.up_proj.weight", {Fv, Dv}, R(static_cast<std::size_t>(Fv) * Dv));
        b.add(p + "feed_forward.down_proj.weight", {Dv, Fv}, R(static_cast<std::size_t>(Dv) * Fv));
    }
    // projector under multi_modal_projector.
    b.add("multi_modal_projector.norm.weight", {Dv}, R(static_cast<std::size_t>(Dv)));
    b.add("multi_modal_projector.patch_merger.merging_layer.weight", {Dv, Dv * 4}, R(static_cast<std::size_t>(Dv) * Dv * 4));
    b.add("multi_modal_projector.linear_1.weight", {H, Dv}, R(static_cast<std::size_t>(H) * Dv));
    b.add("multi_modal_projector.linear_2.weight", {H, H}, R(static_cast<std::size_t>(H) * H));

    auto path = std::filesystem::temp_directory_path() / "brolm_mistral3_vl.safetensors";
    b.write(path);

    // Helper: build a PreprocessedImage with a 4×4 patch grid (→ 2×2 tokens).
    auto make_image = [&](uint32_t seed) {
        m3::PreprocessedImage im;
        im.grid_h = 4; im.grid_w = 4; im.merge = 2;
        const int N = 16;
        bt::Tensor patches = bt::Tensor::empty_on(bt::Device::CPU, N, CP, bt::Dtype::FP32);
        float* d = patches.host_f32_mut();
        uint32_t s = seed;
        for (int i = 0; i < N * CP; ++i) { s = s * 1664525u + 1013904223u; d[i] = (static_cast<float>(s >> 8) / 16777216.0f - 0.5f) * 0.5f; }
        im.patches = std::move(patches);
        return im;
    };

    try {
        auto file = st::File::open(path.string());

        // prompt: text [5,6] + image span [IMG IMG BREAK IMG IMG END] + [7].
        // 4 [IMG] placeholders at indices {2,3,5,6}.
        const int BRK = 12, END = 13;
        std::vector<int32_t> prompt = {5, 6, IMG, IMG, BRK, IMG, IMG, END, 7};
        const int Lp = static_cast<int>(prompt.size());
        std::vector<int> img_pos = {2, 3, 5, 6};

        m3::PreprocessedImage imgA = make_image(11);

        // ── 2. fuse_embeds correctness ────────────────────────────────────
        {
            m3::VLModel vl(cfg);
            vl.load_weights(file);

            bt::Tensor fused;
            vl.fuse_embeds(prompt, {imgA}, IMG, fused);
            bt::sync_all();
            CHECK(fused.rows == Lp);
            CHECK(fused.cols == H);
            std::vector<float> fv = bdtest::bd_download(fused);

            // Manual rebuild: text embeddings, then overwrite [IMG] rows with
            // the projector output.
            bt::Tensor temb;
            vl.text().embed_tokens(prompt.data(), Lp, temb);
            std::vector<float> tv = bdtest::bd_download(temb);
            bt::Tensor iemb;
            vl.image_embeddings(imgA.patches, imgA.grid_h, imgA.grid_w, iemb);
            bt::sync_all();
            CHECK(iemb.rows == imgA.num_image_tokens());
            CHECK(iemb.cols == H);
            std::vector<float> iv = bdtest::bd_download(iemb);

            std::vector<float> manual = tv;
            for (std::size_t k = 0; k < img_pos.size(); ++k) {
                const int row = img_pos[k];
                for (int c = 0; c < H; ++c)
                    manual[static_cast<std::size_t>(row) * H + c] = iv[k * H + static_cast<std::size_t>(c)];
            }
            CHECK(fv == manual);

            // Non-[IMG] rows must be the untouched text embeddings.
            bool text_rows_ok = true;
            for (int r = 0; r < Lp; ++r) {
                bool is_img = (r == 2 || r == 3 || r == 5 || r == 6);
                if (is_img) continue;
                for (int c = 0; c < H; ++c) {
                    const std::size_t i = static_cast<std::size_t>(r) * H + c;
                    if (fv[i] != tv[i]) text_rows_ok = false;
                }
            }
            CHECK(text_rows_ok);
        }

        // ── 3. image conditioning (different image → different prefill) ────
        {
            m3::PreprocessedImage imgB = make_image(999);
            auto prefill_logits = [&](const m3::PreprocessedImage& im) {
                m3::VLModel vl(cfg);
                vl.load_weights(file);
                vl.allocate_cache(Lp + 4);
                bt::Tensor fused; vl.fuse_embeds(prompt, {im}, IMG, fused);
                bt::Tensor logits; vl.text().forward_embeds(fused, Lp, logits);
                bt::sync_all();
                return bdtest::bd_download(logits);
            };
            std::vector<float> la = prefill_logits(imgA);
            std::vector<float> lb = prefill_logits(imgB);
            float max_abs = 0.0f;
            for (std::size_t i = 0; i < la.size(); ++i) max_abs = std::max(max_abs, std::fabs(la[i] - lb[i]));
            std::printf("mistral3_vl: imageA-vs-imageB prefill max_abs = %.3e\n", static_cast<double>(max_abs));
            CHECK(max_abs > 1e-3f);
        }

        // ── 4. generation: count, range, determinism ──────────────────────
        {
            brolm::detail::GenerateOptions opts;
            opts.max_new_tokens = 5; opts.stop_on_eos = false; opts.sampling.temperature = 0.0f;

            m3::VLModel vla(cfg); vla.load_weights(file);
            std::vector<int32_t> a = vla.generate(prompt, {imgA}, IMG, -1, opts);
            m3::VLModel vlb(cfg); vlb.load_weights(file);
            std::vector<int32_t> bb = vlb.generate(prompt, {imgA}, IMG, -1, opts);

            CHECK(static_cast<int>(a.size()) == 5);
            for (int32_t id : a) CHECK(id >= 0 && id < V);
            CHECK(a == bb);
        }

        // ── 5. mismatched image-token count raises ────────────────────────
        {
            m3::VLModel vl(cfg); vl.load_weights(file);
            bt::Tensor fused;
            bool threw = false;
            try { vl.fuse_embeds(prompt, {}, IMG, fused); }  // prompt has 4 [IMG], 0 images
            catch (const std::exception&) { threw = true; }
            CHECK(threw);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mistral3_vl test threw: %s\n", e.what());
        ++g_failures;
    }

    std::error_code ec; std::filesystem::remove(path, ec);

    if (g_failures == 0) std::printf("mistral3_vl: OK\n");
    else std::fprintf(stderr, "mistral3_vl: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
