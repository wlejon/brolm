#include "brolm/mistral3_vl.h"

#include "brolm/detail/compute.h"
#include "brolm/detail/grammar_decode.h"
#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace brolm::mistral3 {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

namespace {
[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("mistral3::VLModel: " + msg);
}
}  // namespace

VLModel::VLModel(const Mistral3Config& cfg)
    : cfg_(cfg), text_(cfg.text), vision_(cfg.vision), projector_(cfg) {}

VLModel::~VLModel() = default;

void VLModel::load_weights(const st::File& f) {
    text_.load_weights(f, "language_model.");
    vision_.load_weights(f, "vision_tower.");
    projector_.load_weights(f, "multi_modal_projector.");
}

void VLModel::load_weights(const bt::gguf::File& text_gguf,
                           const bt::gguf::File& mmproj_gguf) {
    text_.load_weights(text_gguf);        // ggml `token_embd` / `blk.*` / `output`
    vision_.load_weights(mmproj_gguf);    // ggml `v.*`
    projector_.load_weights(mmproj_gguf); // ggml `mm.*`
}

void VLModel::allocate_cache(int max_seq_len) { text_.allocate_cache(max_seq_len); }
void VLModel::reset_cache() { text_.reset_cache(); }
int  VLModel::cache_len() const { return text_.cache_len(); }

// ─── image embeddings ────────────────────────────────────────────────────────

void VLModel::image_embeddings(const bt::Tensor& patches_host,
                               int grid_h, int grid_w, bt::Tensor& out) {
    if (patches_host.device != bt::Device::CPU || patches_host.dtype != bt::Dtype::FP32) {
        fail("image_embeddings: patches must be CPU FP32 (preprocessor output)");
    }
    // Upload patches to the compute device/dtype, then tower → projector.
    bt::Tensor patches_dev = detail::upload_host(
        patches_host.host_f32(), patches_host.rows, patches_host.cols);
    vision_.forward(patches_dev, grid_h, grid_w, vis_feat_);
    projector_.forward(vis_feat_, grid_h, grid_w, out);
}

// ─── fuse ────────────────────────────────────────────────────────────────────

void VLModel::fuse_embeds(const std::vector<int32_t>& prompt_ids,
                          const std::vector<PreprocessedImage>& images,
                          int image_token_id, bt::Tensor& fused_out) {
    const int L = static_cast<int>(prompt_ids.size());
    if (L <= 0) fail("fuse_embeds: empty prompt");
    const int H = cfg_.text.hidden_size;

    // Text embeddings for every token.
    text_.embed_tokens(prompt_ids.data(), L, fused_out);
    if (fused_out.rows != L || fused_out.cols != H) {
        fail("fuse_embeds: unexpected embedding shape");
    }

    // [IMG] positions in prompt order.
    std::vector<int> positions;
    positions.reserve(prompt_ids.size());
    for (int i = 0; i < L; ++i) {
        if (prompt_ids[static_cast<std::size_t>(i)] == image_token_id) positions.push_back(i);
    }

    std::size_t total_img_tokens = 0;
    for (const auto& im : images) total_img_tokens += static_cast<std::size_t>(im.num_image_tokens());
    if (total_img_tokens != positions.size()) {
        fail("fuse_embeds: image-token count (" + std::to_string(total_img_tokens) +
             ") does not match the number of image placeholders (" +
             std::to_string(positions.size()) + ")");
    }
    if (images.empty()) return;  // text-only: nothing to splice

    // For each image, project its patches and copy the rows onto the next run
    // of [IMG] positions. copy_d2d moves one (H,) row at a time on the device.
    std::size_t pos_idx = 0;
    bt::Tensor img_emb;
    for (const auto& im : images) {
        image_embeddings(im.patches, im.grid_h, im.grid_w, img_emb);
        const int Li = im.num_image_tokens();
        if (img_emb.rows != Li || img_emb.cols != H) {
            fail("fuse_embeds: projector output shape mismatch");
        }
        for (int r = 0; r < Li; ++r) {
            const int dst_row = positions[pos_idx++];
            bt::copy_d2d(img_emb, r * H, fused_out, dst_row * H, H);
        }
    }
}

// ─── generate ────────────────────────────────────────────────────────────────

std::vector<int32_t> VLModel::generate(
    const std::vector<int32_t>& prompt_ids,
    const std::vector<PreprocessedImage>& images,
    int image_token_id, int eos_id,
    const brolm::detail::GenerateOptions& opts,
    const std::vector<std::string>* token_text) {
    std::vector<int32_t> generated;
    const int L = static_cast<int>(prompt_ids.size());
    if (L <= 0 || opts.max_new_tokens <= 0) return generated;

    const int vocab = cfg_.text.vocab_size;
    // Throws before any device work when a grammar comes without token text.
    brolm::detail::GrammarDecode gd(opts.grammar, token_text);
    allocate_cache(L + opts.max_new_tokens);

    std::mt19937_64 rng(opts.sampling.seed);
    const bool stop = opts.stop_on_eos && eos_id >= 0;
    const int grammar_eos = stop ? eos_id : -1;
    // Penalties read the whole context (prompt + generated), as detail::generate.
    std::vector<int32_t> context = prompt_ids;
    static const std::string kNoText;
    auto text_of = [&](int id) -> const std::string& {
        return token_text && id >= 0 && static_cast<size_t>(id) < token_text->size()
            ? (*token_text)[static_cast<size_t>(id)] : kNoText;
    };

    // Prefill: build the fused image+text stream, run forward_embeds, sample
    // the first new token from the last logits row; then decode text-only,
    // one token at a time.
    bt::Tensor fused;
    fuse_embeds(prompt_ids, images, image_token_id, fused);
    bt::Tensor logits;
    text_.forward_embeds(fused, L, logits);
    while (true) {
        std::vector<float> row = brolm::detail::last_row_fp32(logits);
        if (!gd.mask(row.data(), vocab, grammar_eos)) break;
        const int next = brolm::detail::sample_token(row.data(), vocab, opts.sampling, rng,
                                                     context.data(), static_cast<int>(context.size()));
        if (stop && next == eos_id) break;
        generated.push_back(static_cast<int32_t>(next));
        context.push_back(static_cast<int32_t>(next));
        gd.accept(next);
        if (opts.on_token && !opts.on_token(static_cast<int32_t>(next), text_of(next))) break;
        if (static_cast<int>(generated.size()) >= opts.max_new_tokens) break;
        const int32_t cur = generated.back();
        text_.forward(&cur, 1, logits);
    }
    return generated;
}

}  // namespace brolm::mistral3
