// Qwen3-VL top-level inference driver.

#include "brolm/qwen3vl_vl.h"

#include "brolm/detail/compute.h"
#include "brolm/qwen_generate.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace brolm::qwen3vl {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("qwen3vl::VLM: " + msg);
}

// Download the last (vocab,) row of a (T, vocab) logits tensor as FP32.
std::vector<float> last_row_fp32(const bt::Tensor& logits) {
    bt::Tensor last = bt::Tensor::view(
        logits.device,
        static_cast<char*>(logits.data) +
            static_cast<std::size_t>(logits.rows - 1) *
                static_cast<std::size_t>(logits.cols) *
                static_cast<std::size_t>(bt::dtype_size_bytes(logits.dtype)),
        1, logits.cols, logits.dtype);
    const std::size_t n = static_cast<std::size_t>(last.size());
    if (last.dtype == bt::Dtype::FP16) {
        std::vector<std::uint16_t> bits(n);
        last.copy_to_host_fp16(bits.data());
        std::vector<float> out(n);
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = bt::fp16_bits_to_fp32(bits[i]);
        }
        return out;
    }
    return last.to_host_vector();
}

bool row_finite(const float* p, int n) {
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(p[i])) return false;
    }
    return true;
}

}  // namespace

// ─── ctor / dtor ───────────────────────────────────────────────────────────

VLM::VLM(const VLMConfig& cfg) : cfg_(cfg) {}
VLM::~VLM() = default;

const Tokenizer& VLM::tokenizer() const {
    if (!tokenizer_) fail("tokenizer() called before load_from_directory");
    return *tokenizer_;
}

// ─── load_from_directory ───────────────────────────────────────────────────

void VLM::load_from_directory(const std::string& dir) {
    namespace fs = std::filesystem;
    const fs::path root = dir;
    if (!fs::exists(root) || !fs::is_directory(root)) {
        fail("checkpoint directory does not exist: " + dir);
    }

    // 1. Config.
    const fs::path cfg_path = root / "config.json";
    if (!fs::exists(cfg_path)) fail("missing config.json in " + dir);
    cfg_.model_cfg = Qwen3VLConfig::load(cfg_path.string());

    // 2. Tokenizer — vocab.json + merges.txt (the BPE loader's format).
    const fs::path vocab_path  = root / "vocab.json";
    const fs::path merges_path = root / "merges.txt";
    if (!fs::exists(vocab_path) || !fs::exists(merges_path)) {
        fail("missing vocab.json or merges.txt in " + dir);
    }
    tokenizer_ = std::make_unique<Tokenizer>(
        Tokenizer::load(vocab_path.string(), merges_path.string()));

    // 3. Safetensors shard(s). Accept the standard single `model.safetensors`,
    //    HF's sharded `model-NNNNN-of-MMMMM.safetensors` pattern, and the
    //    unusual `model.safetensors-NNNNN-of-MMMMM.safetensors` naming some
    //    Qwen releases use — anything starting with "model" and ending in
    //    ".safetensors".
    std::vector<fs::path> shard_paths;
    for (const auto& entry : fs::directory_iterator(root)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("model", 0) == 0 &&
            entry.path().extension() == ".safetensors") {
            shard_paths.push_back(entry.path());
        }
    }
    std::sort(shard_paths.begin(), shard_paths.end());
    if (shard_paths.empty()) {
        fail("no model*.safetensors file in " + dir);
    }

    std::vector<st::File> shards;
    shards.reserve(shard_paths.size());
    std::vector<const st::File*> shard_ptrs;
    shard_ptrs.reserve(shard_paths.size());
    for (const auto& p : shard_paths) {
        shards.emplace_back(st::File::open(p.string()));
    }
    for (const auto& f : shards) shard_ptrs.push_back(&f);

    // 4. Vision tower + text model. Both scan every shard for their tensors,
    //    so this is robust regardless of how the checkpoint's tensors are
    //    distributed across shard files.
    vision_ = std::make_unique<VisionTower>(cfg_.model_cfg.vision,
                                            cfg_.model_cfg.text.hidden_size);
    vision_->load_weights(shard_ptrs);

    text_ = std::make_unique<TextModel>(cfg_.model_cfg.text);
    text_->load_weights(shard_ptrs);

    // 5. KV cache sized for max_seq_len.
    cache_ = text_->make_cache(cfg_.max_seq_len);
    cache_allocated_ = true;
}

// ─── generate ──────────────────────────────────────────────────────────────

namespace {

// Expand each single <|image_pad|> token into N copies, where N is the
// num_image_tokens() count for the i-th image.
std::vector<int> expand_image_pad_(const std::vector<int>& tokens,
                                   const std::vector<PreprocessedImage>& images,
                                   int image_pad_id) {
    std::size_t pad_count = 0;
    for (int t : tokens) if (t == image_pad_id) ++pad_count;
    if (pad_count != images.size()) {
        throw std::runtime_error(
            "qwen3vl::VLM: prompt has " + std::to_string(pad_count) +
            " <|image_pad|> token(s) but " + std::to_string(images.size()) +
            " image(s) supplied");
    }

    std::vector<int> out;
    std::size_t expanded_extra = 0;
    for (const auto& im : images) {
        expanded_extra += static_cast<std::size_t>(im.num_image_tokens());
    }
    out.reserve(tokens.size() + expanded_extra);

    std::size_t img_idx = 0;
    for (int t : tokens) {
        if (t == image_pad_id) {
            const int n = images[img_idx++].num_image_tokens();
            for (int k = 0; k < n; ++k) out.push_back(image_pad_id);
        } else {
            out.push_back(t);
        }
    }
    return out;
}

// Locate the start row of each image_pad run in `expanded`.
std::vector<int> image_pad_run_starts_(const std::vector<int>& expanded,
                                       const std::vector<PreprocessedImage>& images,
                                       int image_pad_id) {
    std::vector<int> starts;
    starts.reserve(images.size());
    std::size_t img_idx = 0;
    const int T = static_cast<int>(expanded.size());
    int i = 0;
    while (i < T) {
        if (expanded[static_cast<std::size_t>(i)] == image_pad_id) {
            int j = i;
            while (j < T &&
                   expanded[static_cast<std::size_t>(j)] == image_pad_id) {
                ++j;
            }
            if (img_idx >= images.size()) {
                throw std::runtime_error(
                    "qwen3vl::VLM: image_pad run count mismatch (internal)");
            }
            const int expected_len = images[img_idx++].num_image_tokens();
            if (j - i != expected_len) {
                throw std::runtime_error(
                    "qwen3vl::VLM: image_pad run length " +
                    std::to_string(j - i) +
                    " != num_image_tokens " +
                    std::to_string(expected_len));
            }
            starts.push_back(i);
            i = j;
        } else {
            ++i;
        }
    }
    if (img_idx != images.size()) {
        throw std::runtime_error(
            "qwen3vl::VLM: fewer image_pad runs than images (internal)");
    }
    return starts;
}

// Splice the (n, hidden) vision-tower output into `embeds` starting at row
// `dst_row`.
void splice_vision_(bt::Tensor& embeds, int dst_row,
                    const bt::Tensor& vision_out) {
    const int n      = vision_out.rows;
    const int hidden = vision_out.cols;
    if (embeds.cols != hidden) {
        throw std::runtime_error(
            "qwen3vl::VLM: splice hidden mismatch " +
            std::to_string(embeds.cols) + " vs " + std::to_string(hidden));
    }
    if (embeds.dtype != vision_out.dtype) {
        throw std::runtime_error("qwen3vl::VLM: splice dtype mismatch");
    }
    bt::copy_d2d(vision_out, /*src_off=*/0,
                embeds, /*dst_off=*/dst_row * hidden,
                n * hidden);
}

// Upload an FP32 host buffer (3, H, W) preprocessor input, then run the
// preprocessor + vision tower, returning the per-image post-merger token
// tensor plus the DeepStack feature list at compute dtype.
bt::Tensor run_vision_one_(VisionTower& tower,
                           const PreprocessConfig& pp,
                           const ImageInput& img,
                           PreprocessedImage& pp_out,
                           std::vector<bt::Tensor>& deepstack_out) {
    if (!img.pixels || img.H <= 0 || img.W <= 0) {
        throw std::runtime_error("qwen3vl::VLM: invalid ImageInput");
    }
    pp_out = preprocess_image(img.pixels, img.H, img.W, pp);

    const float* patch_src = pp_out.patches.host_f32();
    const int N          = pp_out.patches.rows;
    const int patch_cols = pp_out.patches.cols;
    bt::Tensor patches_dev;
    if (brolm::compute_dtype() == bt::Dtype::FP16) {
        std::vector<std::uint16_t> bits(
            static_cast<std::size_t>(N) * patch_cols);
        for (std::size_t i = 0; i < bits.size(); ++i) {
            bits[i] = bt::fp32_to_fp16_bits(patch_src[i]);
        }
        patches_dev = bt::Tensor::from_host_fp16(bits.data(), N, patch_cols);
    } else {
        patches_dev = bt::Tensor::from_host(patch_src, N, patch_cols);
    }

    bt::Tensor out;
    tower.forward(patches_dev, pp_out.grid_t, pp_out.grid_h, pp_out.grid_w,
                 out, deepstack_out);
    return out;
}

}  // namespace

std::vector<int> VLM::generate_tokens(const std::string& prompt,
                                      const std::vector<ImageInput>& images) {
    return generate_tokens(prompt, images, TokenCallback{});
}

void VLM::set_generation(int max_new_tokens, float temperature, int top_k,
                         float top_p, uint64_t seed) {
    cfg_.max_new_tokens = max_new_tokens;
    cfg_.temperature    = temperature;
    cfg_.top_k          = top_k;
    cfg_.top_p          = top_p;
    cfg_.seed           = seed;
}

std::vector<int> VLM::generate_tokens(const std::string& prompt,
                                      const std::vector<ImageInput>& images,
                                      const TokenCallback& on_token) {
    if (!tokenizer_ || !text_ || !vision_) {
        fail("generate_tokens called before load_from_directory");
    }
    const int pad_id = tokenizer_->image_pad_id();
    if (pad_id < 0 && !images.empty()) {
        fail("tokenizer is missing the <|image_pad|> id");
    }
    const int im_end_id = tokenizer_->im_end_id();
    const int eot_id    = tokenizer_->endoftext_id();

    // 1. Tokenize prompt.
    auto encoded = tokenizer_->encode(prompt);
    std::vector<int> tokens(encoded.begin(), encoded.end());

    // 2. Preprocess + run vision tower for every image (main embedding +
    //    DeepStack feature list).
    std::vector<PreprocessedImage> pp_results;
    pp_results.reserve(images.size());
    std::vector<bt::Tensor> vision_outs;
    vision_outs.reserve(images.size());
    std::vector<std::vector<bt::Tensor>> deepstack_outs;
    deepstack_outs.reserve(images.size());
    for (const auto& img : images) {
        PreprocessedImage pp_one;
        std::vector<bt::Tensor> ds_one;
        bt::Tensor out = run_vision_one_(*vision_, cfg_.pp, img, pp_one, ds_one);
        pp_results.push_back(std::move(pp_one));
        vision_outs.push_back(std::move(out));
        deepstack_outs.push_back(std::move(ds_one));
    }

    // 3. Expand each <|image_pad|> placeholder to its post-merger run length.
    std::vector<int> expanded = expand_image_pad_(tokens, pp_results, pad_id);
    const int T = static_cast<int>(expanded.size());
    if (T <= 0) fail("generate: empty prompt");
    if (T > cfg_.max_seq_len) {
        fail("generate: expanded prompt (" + std::to_string(T) +
             ") exceeds max_seq_len (" + std::to_string(cfg_.max_seq_len) + ")");
    }

    // 4. Embed via the text model's tied table.
    bt::Tensor embeds = text_->embed_tokens(expanded);

    // 5. Splice each image's main-merger output into its image_pad run, and
    //    build the DeepStack splice list for the same row ranges.
    std::vector<int> run_starts =
        image_pad_run_starts_(expanded, pp_results, pad_id);
    std::vector<DeepstackSplice> deepstack;
    deepstack.reserve(images.size());
    for (std::size_t i = 0; i < images.size(); ++i) {
        splice_vision_(embeds, run_starts[i], vision_outs[i]);
        DeepstackSplice sp;
        sp.row_start = run_starts[i];
        sp.per_layer = std::move(deepstack_outs[i]);
        deepstack.push_back(std::move(sp));
    }

    // 6. M-RoPE positions for the expanded sequence.
    MRopePositions mp = build_mrope_position_ids(
        expanded, pp_results,
        cfg_.model_cfg.image_token_id,
        cfg_.model_cfg.vision_start_token_id);

    // 7. Prefill. Reset the cache len fields back to 0 for a fresh forward.
    if (!cache_allocated_) {
        cache_ = text_->make_cache(cfg_.max_seq_len);
        cache_allocated_ = true;
    }
    for (auto& lc : cache_) lc.len = 0;

    bt::Tensor logits;
    text_->forward_embeds(embeds, mp.t, mp.h, mp.w, cache_, logits, deepstack);
    bt::sync_all();

    // 8. Sample loop. Use the existing qwen_generate sampler for parity.
    qwen::SamplingParams sp;
    sp.temperature = cfg_.temperature;
    sp.top_k       = cfg_.top_k;
    sp.top_p       = cfg_.top_p;
    sp.seed        = cfg_.seed;
    std::mt19937_64 rng(cfg_.seed);

    const int vocab = cfg_.model_cfg.text.vocab_size;
    std::vector<int> generated;
    if (cfg_.max_new_tokens <= 0) return generated;
    generated.reserve(static_cast<std::size_t>(cfg_.max_new_tokens));

    auto stop_token = [&](int t) {
        if (im_end_id >= 0 && t == im_end_id) return true;
        if (eot_id    >= 0 && t == eot_id)    return true;
        return false;
    };

    std::vector<float> row = last_row_fp32(logits);
    if (!row_finite(row.data(), vocab)) {
        fail("prefill produced non-finite logits");
    }
    int next = qwen::sample_token(row.data(), vocab, sp, rng);

    // Position advancement during decode: HF advances all three axes by
    // (max(t,h,w) + 1) of the prefill.
    int64_t next_pos = static_cast<int64_t>(T) + mp.delta;

    int steps_remaining = cfg_.max_new_tokens;
    while (steps_remaining > 0 && !stop_token(next)) {
        generated.push_back(next);
        --steps_remaining;
        if (on_token && !on_token(next)) break;
        if (steps_remaining == 0) break;

        // Embed the just-sampled token and forward one step. No images
        // appear in a single-token decode step, so no DeepStack splices.
        std::vector<int> one = {next};
        bt::Tensor one_embed = text_->embed_tokens(one);
        std::vector<int64_t> mt = {next_pos};
        std::vector<int64_t> mh = {next_pos};
        std::vector<int64_t> mw = {next_pos};
        ++next_pos;

        bt::Tensor step_logits;
        text_->forward_embeds(one_embed, mt, mh, mw, cache_, step_logits);
        bt::sync_all();
        row = last_row_fp32(step_logits);
        if (!row_finite(row.data(), vocab)) {
            fail("decode step produced non-finite logits");
        }
        next = qwen::sample_token(row.data(), vocab, sp, rng);
    }

    return generated;
}

std::string VLM::generate(const std::string& prompt,
                          const std::vector<ImageInput>& images) {
    std::vector<int> ids = generate_tokens(prompt, images);
    if (!tokenizer_) return std::string();
    std::vector<int32_t> ids32(ids.begin(), ids.end());
    return tokenizer_->decode(ids32);
}

}  // namespace brolm::qwen3vl
