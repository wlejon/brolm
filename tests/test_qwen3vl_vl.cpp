// Qwen3-VL top-level driver integration test.
//
// (a) Synthetic — always runs. Builds a temporary checkpoint directory with a
//     tiny config + vocab/merges + random-weight safetensors covering both
//     the vision tower (incl. DeepStack mergers) and the text backbone.
//     Loads via VLM::load_from_directory and generates a few text-only
//     tokens (no images supplied), exercising the prefill -> sample ->
//     decode loop and confirming both towers' weights load without error.
//
// (b) Real-checkpoint — gated on BROLM_QWEN3VL_DIR. Loads the real 4B VLM
//     checkpoint, builds a synthetic 224x224 gray image, runs a ChatML
//     prompt containing one image_pad triple, and decodes up to 16 greedy
//     tokens (exercising the DeepStack splice end to end). Asserts the
//     pipeline completes without NaN/crash and that the prompt's image_pad
//     expansion matches num_image_tokens. We do NOT assert any specific
//     output string — numerical parity with HF is future work.

#include "brolm/qwen3vl_config.h"
#include "brolm/qwen3vl_vl.h"
#include "brolm/detail/compute.h"

#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <system_error>
#include <vector>

namespace q3vl = brolm::qwen3vl;
namespace st   = brotensor::safetensors;
namespace bt   = brotensor;

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
             const std::vector<uint16_t>& fp16_bits) {
        std::size_t expected = 1;
        for (int d : shape) expected *= static_cast<std::size_t>(d);
        if (expected != fp16_bits.size()) {
            std::fprintf(stderr, "fixture: shape/data mismatch for %s\n",
                         name.c_str());
            std::abort();
        }
        std::uint64_t start = payload.size();
        const std::uint8_t* bytes =
            reinterpret_cast<const std::uint8_t*>(fp16_bits.data());
        payload.insert(payload.end(), bytes, bytes + fp16_bits.size() * 2);
        std::uint64_t end = payload.size();
        if (!first) entries += ",";
        first = false;
        entries += "\"" + name + "\":{\"dtype\":\"F16\",\"shape\":[";
        for (std::size_t i = 0; i < shape.size(); ++i) {
            if (i) entries += ",";
            entries += std::to_string(shape[i]);
        }
        entries += "],\"data_offsets\":[" + std::to_string(start) + "," +
                   std::to_string(end) + "]}";
    }

    void write(const std::filesystem::path& path) const {
        std::string header = "{" + entries + "}";
        std::uint64_t hdr_size = header.size();
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) std::abort();
        f.write(reinterpret_cast<const char*>(&hdr_size), 8);
        f.write(header.data(), header.size());
        f.write(reinterpret_cast<const char*>(payload.data()),
                static_cast<std::streamsize>(payload.size()));
    }
};

std::vector<uint16_t> fp16_rand(std::size_t n, uint32_t seed) {
    std::vector<uint16_t> out(n);
    uint32_t s = seed * 2654435761u + 1u;
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        float v = (static_cast<float>(s >> 8) / 16777216.0f - 0.5f) * 0.2f;
        out[i] = bt::fp32_to_fp16_bits(v);
    }
    return out;
}
std::vector<uint16_t> fp16_ones(std::size_t n) {
    return std::vector<uint16_t>(n, bt::fp32_to_fp16_bits(1.0f));
}
std::vector<uint16_t> fp16_zeros(std::size_t n) {
    return std::vector<uint16_t>(n, 0);
}

void utf8_encode(uint32_t cp, std::string& out) {
    if (cp < 0x80) { out += static_cast<char>(cp); }
    else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

void byte_to_unicode_strings(std::string out[256]) {
    bool self_map[256] = {false};
    auto mark = [&](int lo, int hi) {
        for (int b = lo; b <= hi; ++b) self_map[b] = true;
    };
    mark(33, 126);
    mark(161, 172);
    mark(174, 255);
    int next_cp = 256;
    for (int b = 0; b < 256; ++b) {
        uint32_t cp = self_map[b] ? static_cast<uint32_t>(b)
                                  : static_cast<uint32_t>(next_cp++);
        out[b].clear();
        utf8_encode(cp, out[b]);
    }
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (uc < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", uc);
            out += buf;
        } else {
            out += c;
        }
    }
    return out;
}

void write_synth_vocab(const std::filesystem::path& path) {
    std::string mapping[256];
    byte_to_unicode_strings(mapping);
    std::ofstream f(path);
    f << "{";
    for (int i = 0; i < 256; ++i) {
        if (i) f << ",";
        f << "\"" << json_escape(mapping[i]) << "\":" << i;
    }
    f << "}";
}

void write_empty_merges(const std::filesystem::path& path) {
    std::ofstream f(path);
    f << "#version: 0.2\n";
}

void write_synth_config(const std::filesystem::path& path,
                        const q3vl::Qwen3VLConfig& cfg) {
    std::ofstream f(path);
    f << "{";
    f << "\"image_token_id\":" << cfg.image_token_id << ",";
    f << "\"video_token_id\":" << cfg.video_token_id << ",";
    f << "\"vision_start_token_id\":" << cfg.vision_start_token_id << ",";
    f << "\"vision_end_token_id\":" << cfg.vision_end_token_id << ",";
    f << "\"tie_word_embeddings\":true,";
    f << "\"text_config\":{";
    f << "\"vocab_size\":" << cfg.text.vocab_size << ",";
    f << "\"hidden_size\":" << cfg.text.hidden_size << ",";
    f << "\"intermediate_size\":" << cfg.text.intermediate_size << ",";
    f << "\"num_hidden_layers\":" << cfg.text.num_hidden_layers << ",";
    f << "\"num_attention_heads\":" << cfg.text.num_attention_heads << ",";
    f << "\"num_key_value_heads\":" << cfg.text.num_key_value_heads << ",";
    f << "\"head_dim\":" << cfg.text.head_dim << ",";
    f << "\"rms_norm_eps\":1e-6,";
    f << "\"tie_word_embeddings\":true,";
    f << "\"max_position_embeddings\":2048,";
    f << "\"rope_theta\":" << cfg.text.rope.rope_theta << ",";
    f << "\"rope_scaling\":{\"rope_type\":\"default\",\"mrope_interleaved\":true,"
         "\"mrope_section\":[" << cfg.text.rope.mrope_section[0] << ","
         << cfg.text.rope.mrope_section[1] << ","
         << cfg.text.rope.mrope_section[2] << "]}";
    f << "},";
    f << "\"vision_config\":{";
    f << "\"depth\":" << cfg.vision.depth << ",";
    f << "\"hidden_size\":" << cfg.vision.hidden_size << ",";
    f << "\"num_heads\":" << cfg.vision.num_heads << ",";
    f << "\"intermediate_size\":" << cfg.vision.intermediate_size << ",";
    f << "\"in_channels\":" << cfg.vision.in_channels << ",";
    f << "\"patch_size\":" << cfg.vision.patch_size << ",";
    f << "\"temporal_patch_size\":" << cfg.vision.temporal_patch_size << ",";
    f << "\"spatial_merge_size\":" << cfg.vision.spatial_merge_size << ",";
    f << "\"out_hidden_size\":" << cfg.vision.out_hidden_size << ",";
    f << "\"num_position_embeddings\":" << cfg.vision.num_position_embeddings << ",";
    f << "\"deepstack_visual_indexes\":[";
    for (std::size_t i = 0; i < cfg.vision.deepstack_visual_indexes.size(); ++i) {
        if (i) f << ",";
        f << cfg.vision.deepstack_visual_indexes[i];
    }
    f << "]";
    f << "}";
    f << "}";
}

// Build a synthetic safetensors covering BOTH model.visual.* (incl. DeepStack
// mergers) and model.language_model.*.
void write_synth_safetensors(const std::filesystem::path& path,
                             const q3vl::Qwen3VLConfig& cfg) {
    Builder b;
    uint32_t seed = 1;
    auto R = [&](std::size_t n) { return fp16_rand(n, seed++); };

    // ── Vision tower ──────────────────────────────────────────────────────
    const auto& vc = cfg.vision;
    const int D     = vc.hidden_size;
    const int F     = vc.intermediate_size;
    const int C     = vc.in_channels;
    const int P     = vc.patch_size;
    const int Tps   = vc.temporal_patch_size;
    const int m     = vc.spatial_merge_size;
    const int Npos  = vc.num_position_embeddings;
    const int Hmrg  = D * m * m;
    const int Hout  = vc.out_hidden_size;
    const int qkv   = 3 * D;

    b.add("model.visual.patch_embed.proj.weight",
          {D, C, Tps, P, P},
          R(static_cast<std::size_t>(D) * C * Tps * P * P));
    b.add("model.visual.patch_embed.proj.bias", {D}, fp16_zeros(D));
    b.add("model.visual.pos_embed.weight", {Npos, D},
          R(static_cast<std::size_t>(Npos) * D));
    for (int i = 0; i < vc.depth; ++i) {
        const std::string p = "model.visual.blocks." + std::to_string(i) + ".";
        b.add(p + "norm1.weight", {D}, fp16_ones(D));
        b.add(p + "norm1.bias",   {D}, fp16_zeros(D));
        b.add(p + "norm2.weight", {D}, fp16_ones(D));
        b.add(p + "norm2.bias",   {D}, fp16_zeros(D));
        b.add(p + "attn.qkv.weight",  {qkv, D},
              R(static_cast<std::size_t>(qkv) * D));
        b.add(p + "attn.qkv.bias",  {qkv}, fp16_zeros(qkv));
        b.add(p + "attn.proj.weight", {D, D},
              R(static_cast<std::size_t>(D) * D));
        b.add(p + "attn.proj.bias", {D}, fp16_zeros(D));
        b.add(p + "mlp.linear_fc1.weight", {F, D},
              R(static_cast<std::size_t>(F) * D));
        b.add(p + "mlp.linear_fc1.bias", {F}, fp16_zeros(F));
        b.add(p + "mlp.linear_fc2.weight", {D, F},
              R(static_cast<std::size_t>(D) * F));
        b.add(p + "mlp.linear_fc2.bias", {D}, fp16_zeros(D));
    }
    b.add("model.visual.merger.norm.weight", {D}, fp16_ones(D));
    b.add("model.visual.merger.norm.bias",   {D}, fp16_zeros(D));
    b.add("model.visual.merger.linear_fc1.weight", {Hmrg, Hmrg},
          R(static_cast<std::size_t>(Hmrg) * Hmrg));
    b.add("model.visual.merger.linear_fc1.bias", {Hmrg}, fp16_zeros(Hmrg));
    b.add("model.visual.merger.linear_fc2.weight", {Hout, Hmrg},
          R(static_cast<std::size_t>(Hout) * Hmrg));
    b.add("model.visual.merger.linear_fc2.bias", {Hout}, fp16_zeros(Hout));

    for (std::size_t k = 0; k < vc.deepstack_visual_indexes.size(); ++k) {
        const std::string p =
            "model.visual.deepstack_merger_list." + std::to_string(k) + ".";
        b.add(p + "norm.weight", {Hmrg}, fp16_ones(Hmrg));
        b.add(p + "norm.bias",   {Hmrg}, fp16_zeros(Hmrg));
        b.add(p + "linear_fc1.weight", {Hmrg, Hmrg},
              R(static_cast<std::size_t>(Hmrg) * Hmrg));
        b.add(p + "linear_fc1.bias", {Hmrg}, fp16_zeros(Hmrg));
        b.add(p + "linear_fc2.weight", {Hout, Hmrg},
              R(static_cast<std::size_t>(Hout) * Hmrg));
        b.add(p + "linear_fc2.bias", {Hout}, fp16_zeros(Hout));
    }

    // ── Text backbone ─────────────────────────────────────────────────────
    const auto& tc = cfg.text;
    const int V    = tc.vocab_size;
    const int H    = tc.hidden_size;
    const int Fm   = tc.intermediate_size;
    const int HD   = tc.head_dim;
    const int n_q  = tc.num_attention_heads;
    const int n_kv = tc.num_key_value_heads;
    const int q_d  = n_q  * HD;
    const int kv_d = n_kv * HD;

    b.add("model.language_model.embed_tokens.weight", {V, H},
          R(static_cast<std::size_t>(V) * H));
    b.add("model.language_model.norm.weight", {H}, fp16_ones(H));

    for (int i = 0; i < tc.num_hidden_layers; ++i) {
        const std::string p =
            "model.language_model.layers." + std::to_string(i) + ".";
        b.add(p + "input_layernorm.weight",          {H}, fp16_ones(H));
        b.add(p + "post_attention_layernorm.weight", {H}, fp16_ones(H));
        b.add(p + "mlp.gate_proj.weight", {Fm, H},
              R(static_cast<std::size_t>(Fm) * H));
        b.add(p + "mlp.up_proj.weight",   {Fm, H},
              R(static_cast<std::size_t>(Fm) * H));
        b.add(p + "mlp.down_proj.weight", {H, Fm},
              R(static_cast<std::size_t>(H) * Fm));

        b.add(p + "self_attn.q_proj.weight", {q_d, H},
              R(static_cast<std::size_t>(q_d) * H));
        b.add(p + "self_attn.k_proj.weight", {kv_d, H},
              R(static_cast<std::size_t>(kv_d) * H));
        b.add(p + "self_attn.v_proj.weight", {kv_d, H},
              R(static_cast<std::size_t>(kv_d) * H));
        b.add(p + "self_attn.o_proj.weight", {H, q_d},
              R(static_cast<std::size_t>(H) * q_d));
        b.add(p + "self_attn.q_norm.weight", {HD}, fp16_ones(HD));
        b.add(p + "self_attn.k_norm.weight", {HD}, fp16_ones(HD));
    }
    b.write(path);
}

// Build a tiny synthetic VLM config. Vocab is set to 320 so the byte-fallback
// IDs (0..255) the synthetic vocab maps fit.
q3vl::Qwen3VLConfig make_synth_config() {
    q3vl::Qwen3VLConfig cfg;

    cfg.image_token_id        = 250;
    cfg.video_token_id        = 251;
    cfg.vision_start_token_id = 252;
    cfg.vision_end_token_id   = 253;

    auto& tc = cfg.text;
    tc.vocab_size          = 320;
    tc.hidden_size         = 64;
    tc.intermediate_size   = 128;
    tc.num_hidden_layers   = 4;
    tc.num_attention_heads = 2;
    tc.num_key_value_heads = 1;
    tc.head_dim            = 32;          // rotary_dim == 32 (full rotation)
    tc.rms_norm_eps        = 1e-6f;
    tc.tie_word_embeddings = true;
    tc.rope.rope_theta     = 1.0e7f;
    tc.rope.mrope_section  = {6, 5, 5};   // 2*(6+5+5)=32

    auto& vc = cfg.vision;
    vc.depth                   = 6;
    vc.hidden_size             = 32;
    vc.num_heads               = 4;
    vc.intermediate_size       = 64;
    vc.in_channels             = 3;
    vc.patch_size              = 16;
    vc.temporal_patch_size     = 2;
    vc.spatial_merge_size      = 2;
    vc.out_hidden_size         = tc.hidden_size;
    vc.num_position_embeddings = 256;
    vc.deepstack_visual_indexes = {1, 3};

    return cfg;
}

void run_synthetic() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "brolm_qwen3vl_vl_synth";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);

    auto cfg = make_synth_config();
    write_synth_config(dir / "config.json", cfg);
    write_synth_vocab (dir / "vocab.json");
    write_empty_merges(dir / "merges.txt");
    write_synth_safetensors(dir / "model.safetensors", cfg);

    try {
        q3vl::VLMConfig vcfg;
        vcfg.model_cfg      = cfg;
        vcfg.max_seq_len    = 64;
        vcfg.max_new_tokens = 3;
        vcfg.temperature    = 0.0f;
        vcfg.seed           = 0;
        vcfg.pp.patch_size          = cfg.vision.patch_size;
        vcfg.pp.temporal_patch_size = cfg.vision.temporal_patch_size;
        vcfg.pp.merge_size          = cfg.vision.spatial_merge_size;

        q3vl::VLM vlm(vcfg);
        vlm.load_from_directory(dir.string());
        std::printf("vl synthetic: loaded vision tower (incl. %d DeepStack "
                    "merger(s)) + text backbone\n",
                    static_cast<int>(cfg.vision.deepstack_visual_indexes.size()));

        // Text-only generation. The synthetic tokenizer maps every byte
        // through the GPT-2 byte-fallback alphabet; with no merges every
        // byte becomes one id.
        const std::string prompt = "Hi";
        std::vector<q3vl::ImageInput> no_images;
        std::vector<int> out_ids = vlm.generate_tokens(prompt, no_images);

        std::printf("vl synthetic: generated %d tokens\n",
                    static_cast<int>(out_ids.size()));
        CHECK(static_cast<int>(out_ids.size()) <= vcfg.max_new_tokens);
        CHECK(!out_ids.empty());
        for (int id : out_ids) {
            CHECK(id >= 0);
            CHECK(id < cfg.text.vocab_size);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "qwen3vl_vl synthetic threw: %s\n", e.what());
        ++g_failures;
    }

    fs::remove_all(dir, ec);
}

void run_real_checkpoint() {
    const char* dir_env = std::getenv("BROLM_QWEN3VL_DIR");
    if (!dir_env) {
        std::printf("[skip] BROLM_QWEN3VL_DIR not set\n");
        return;
    }
    const std::filesystem::path dir = dir_env;
    if (!std::filesystem::exists(dir / "config.json")) {
        std::printf("[skip] real checkpoint not found at %s\n", dir_env);
        return;
    }

    try {
        q3vl::VLMConfig vcfg;
        vcfg.max_seq_len    = 1024;
        vcfg.max_new_tokens = 16;
        vcfg.temperature    = 0.0f;
        vcfg.seed           = 0;

        q3vl::VLM vlm(vcfg);
        vlm.load_from_directory(dir.string());

        const int H = 224, W = 224, C = 3;
        std::vector<float> img(static_cast<std::size_t>(C) * H * W, 0.5f);
        q3vl::ImageInput inp;
        inp.pixels = img.data();
        inp.H = H; inp.W = W;
        std::vector<q3vl::ImageInput> images = {inp};

        const std::string prompt =
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>"
            "Describe the image.<|im_end|>\n"
            "<|im_start|>assistant\n";

        auto enc = vlm.tokenizer().encode(prompt);
        int pad_id = vlm.tokenizer().image_pad_id();
        int pad_count = 0;
        for (int32_t t : enc) if (t == pad_id) ++pad_count;
        std::printf("vl real: prompt tokens=%d, image_pad placeholders=%d, "
                    "pad_id=%d\n",
                    static_cast<int>(enc.size()), pad_count, pad_id);
        CHECK(pad_count == 1);

        std::vector<int> out_ids = vlm.generate_tokens(prompt, images);
        std::string out_text = vlm.tokenizer().decode(
            std::vector<int32_t>(out_ids.begin(), out_ids.end()));

        std::printf("vl real: generated %d token(s)\n",
                    static_cast<int>(out_ids.size()));
        std::printf("vl real: decoded = \"%s\"\n", out_text.c_str());

        CHECK(!out_ids.empty());
        CHECK(!out_text.empty());
        const int V = vlm.config().text.vocab_size;
        for (int id : out_ids) {
            CHECK(id >= 0);
            CHECK(id < V);
        }

        // ── parity vs HF transformers ───────────────────────────────────────
        //
        // Reference produced by Qwen/Qwen3-VL-4B-Instruct under transformers
        // 5.9.0 (Qwen3VLForConditionalGeneration.generate, dtype=float16,
        // greedy: do_sample=False, on a CUDA device), for this exact prompt
        // and a constant mid-gray 224x224 image (all pixels 0.5 in [0,1] —
        // i.e. uint8 128). brolm must reproduce the full 16-token greedy
        // continuation exactly.
        const std::vector<int> golden_ids = {
            785, 2168, 374, 264, 6437, 11, 13794, 17545, 9334, 13,
            2619, 525, 902, 41545, 1238, 6171,
        };
        const char* golden_text =
            "The image is a solid, uniform gray square. There are no "
            "discernible objects";
        CHECK(out_ids == golden_ids);
        CHECK(out_text == golden_text);
        std::printf("vl real: parity vs HF %s\n",
                    (out_ids == golden_ids && out_text == golden_text)
                        ? "MATCH" : "MISMATCH");

        // ── text-only parity case (no image — isolates the dense decoder +
        // full-rotary M-RoPE path from vision/DeepStack). Same HF setup as
        // above (transformers 5.9.0, float16, greedy).
        const std::string text_prompt =
            "<|im_start|>user\n"
            "The capital of France is<|im_end|>\n"
            "<|im_start|>assistant\n";
        std::vector<q3vl::ImageInput> no_images;
        std::vector<int> text_out_ids =
            vlm.generate_tokens(text_prompt, no_images);
        std::string text_out_text = vlm.tokenizer().decode(
            std::vector<int32_t>(text_out_ids.begin(), text_out_ids.end()));

        const std::vector<int> text_golden_ids = {
            785, 6722, 315, 9625, 374, 3070, 59604, 334, 382,
            59604, 374, 537, 1172, 279, 6722, 714,
        };
        const char* text_golden_text =
            "The capital of France is **Paris**.\n\nParis is not only the "
            "capital but";
        std::printf("vl real (text-only): decoded = \"%s\"\n",
                    text_out_text.c_str());
        CHECK(text_out_ids == text_golden_ids);
        CHECK(text_out_text == text_golden_text);
        std::printf("vl real (text-only): parity vs HF %s\n",
                    (text_out_ids == text_golden_ids &&
                     text_out_text == text_golden_text)
                        ? "MATCH" : "MISMATCH");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "qwen3vl_vl real-checkpoint threw: %s\n", e.what());
        ++g_failures;
    }
}

}  // namespace

int main() {
    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }
    run_synthetic();
    run_real_checkpoint();
    if (g_failures == 0) std::printf("qwen3vl_vl: OK\n");
    else std::fprintf(stderr, "qwen3vl_vl: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
