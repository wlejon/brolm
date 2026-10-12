// BioCLIP (ViT-B/16) and BioCLIP 2 (ViT-L/14) parity against open_clip
// (tests/ref/gen_bioclip_fixture.py -> tests/fixtures/bioclip{,-2}.bin) on the
// CPU and on the default GPU when there is one.
//
// Weights are the open_clip releases converted by scripts/convert-openclip.py:
// BROLM_BIOCLIP_DIR / BROLM_BIOCLIP2_DIR, else the brovisionml-data sibling
// (../brovisionml-data/bioclip, ../brovisionml-data/bioclip-2). Each model's
// section skips (passes) when its weights or fixture are absent.
//
// Per model and device: the tokenizer's ids for each prompt; the vision tower
// truncated to its first block and in full (post-LN CLS feature); the text
// tower's final hidden states up to EOT and its 1-layer truncation; the
// projected, normalised embeddings through CLIPScorer (the path bro.lm's
// ClipModel takes, preprocessing included); the cosine similarities and the
// softmax at the checkpoint's logit scale. Tolerances are per stage: FP32 on
// the CPU is held to ~1e-4 on raw features, a GPU computes in FP16 and is
// held to what half precision through 12-24 residual blocks allows.
#include "brolm/clip.h"
#include "brolm/clip_image.h"
#include "brolm/clip_score.h"
#include "brolm/detail/compute.h"
#include "brolm/tokenizer.h"
#include "test_compute.h"

#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace bt = brotensor;
namespace st = brotensor::safetensors;

namespace {

int g_failures = 0;

struct Rec {
    int code = 0;
    std::vector<int> dims;
    std::vector<float> f;
    std::vector<int> i;
    std::string s;
};

using Fixture = std::map<std::string, Rec>;

Fixture read_fixture(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    Fixture out;
    char magic[8];
    in.read(magic, 8);
    if (!in || std::memcmp(magic, "BIOCFIX1", 8) != 0) throw std::runtime_error("bad fixture magic");
    auto rd = [&](void* d, std::size_t n) {
        in.read(static_cast<char*>(d), static_cast<std::streamsize>(n));
        if (!in) throw std::runtime_error("truncated fixture");
    };
    int count = 0;
    rd(&count, 4);
    for (int r = 0; r < count; ++r) {
        int nl = 0;
        rd(&nl, 4);
        std::string name(static_cast<std::size_t>(nl), '\0');
        rd(name.data(), name.size());
        Rec rec;
        int nd = 0;
        rd(&rec.code, 4);
        rd(&nd, 4);
        rec.dims.resize(static_cast<std::size_t>(nd));
        rd(rec.dims.data(), 4u * static_cast<std::size_t>(nd));
        std::size_t n = 1;
        for (int d : rec.dims) n *= static_cast<std::size_t>(d);
        if (rec.code == 0) { rec.f.resize(n); rd(rec.f.data(), 4 * n); }
        else if (rec.code == 1) { rec.i.resize(n); rd(rec.i.data(), 4 * n); }
        else { rec.s.resize(n); rd(rec.s.data(), n); }
        out.emplace(name, std::move(rec));
    }
    return out;
}

const Rec& need(const Fixture& fx, const std::string& name) {
    auto it = fx.find(name);
    if (it == fx.end()) throw std::runtime_error("fixture has no record " + name);
    return it->second;
}

std::vector<std::string> split_lines(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string line;
    while (std::getline(ss, line)) out.push_back(line);
    return out;
}

double max_abs(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e30;
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::fabs(static_cast<double>(a[i]) - b[i]));
    return m;
}

void report(const char* what, double v, double tol) {
    std::printf("    %-44s %.3e  (tol %.0e)\n", what, v, tol);
    if (!(v <= tol)) {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s = %g > %g\n", what, v, tol);
    }
}

// Per-stage tolerances for one device class.
struct Tol {
    double pooled_l1, pooled, hidden, embed, sim, prob;
};
// Measured on halo (Radeon 8060S, Vulkan): CPU worst 1.2e-4 (BioCLIP 2's
// 24-block pooled feature), GPU worst 5.7e-2 on the same stage, 1.3e-3 on an
// embedding; the bounds leave about 2x headroom.
constexpr Tol kCpuTol{1e-4, 5e-4, 5e-4, 2e-5, 2e-5, 1e-4};
constexpr Tol kGpuTol{2e-2, 1e-1, 1e-1, 4e-3, 4e-3, 2e-2};

struct Model {
    brolm::clip::TextEncoderConfig text;
    brolm::clip_image::ImageEncoderConfig vision;
    int embed_dim = 0;
};

Model model_from_fixture(const Fixture& fx) {
    Model m;
    const auto& v = need(fx, "arch/vision").i;
    const auto& t = need(fx, "arch/text").i;
    const auto act = brolm::clip::activation_from_name(need(fx, "arch/hidden_act").s);
    m.vision.image_size = v[0];
    m.vision.patch_size = v[1];
    m.vision.hidden_dim = v[2];
    m.vision.num_heads = v[3];
    m.vision.num_layers = v[4];
    m.vision.intermediate_dim = v[5];
    m.vision.activation = act;
    m.text.vocab_size = t[0];
    m.text.max_position = t[1];
    m.text.hidden_dim = t[2];
    m.text.num_heads = t[3];
    m.text.num_layers = t[4];
    m.text.intermediate_dim = t[5];
    m.text.activation = act;
    m.embed_dim = need(fx, "arch/embed_dim").i[0];
    return m;
}

std::vector<float> row(const std::vector<float>& m, int r, int cols) {
    return {m.begin() + static_cast<std::ptrdiff_t>(r) * cols,
            m.begin() + static_cast<std::ptrdiff_t>(r + 1) * cols};
}

void run_device(const fs::path& dir, const Fixture& fx, bt::Device dev, const Tol& tol) {
    std::printf("  == %s ==\n", bt::to_string(dev).c_str());
    bt::DeviceScope scope(dev);
    const Model m = model_from_fixture(fx);
    auto f = st::File::open((dir / "model.safetensors").string());

    // Vision: 1-layer truncation and the full tower, from the reference's
    // normalised pixels (no resampling in between).
    const auto& px = need(fx, "image/pixels").f;
    const int D = m.vision.hidden_dim;
    bt::Tensor pixels = brolm::detail::upload_host(px.data(), 1, static_cast<int>(px.size()));
    {
        auto cfg1 = m.vision;
        cfg1.num_layers = 1;
        brolm::clip_image::ImageEncoder enc1(cfg1);
        enc1.load_weights(f, "vision_model.");
        bt::Tensor cls;
        enc1.forward(pixels, cls);
        report("vision block 0 pooled (max abs)",
               max_abs(bdtest::bd_download(cls), need(fx, "image/pooled_l1").f), tol.pooled_l1);
    }
    brolm::clip_image::ImageEncoder venc(m.vision);
    venc.load_weights(f, "vision_model.");
    {
        bt::Tensor cls;
        venc.forward(pixels, cls);
        std::vector<float> got = bdtest::bd_download(cls);
        if (static_cast<int>(got.size()) != D) throw std::runtime_error("vision pooled width");
        report("vision pooled (max abs)", max_abs(got, need(fx, "image/pooled").f), tol.pooled);
    }

    // Text: tokenizer ids, the final hidden states up to EOT, a 1-layer
    // truncation pooled at EOT.
    auto tok = brolm::clip::Tokenizer::load((dir / "vocab.json").string(),
                                            (dir / "merges.txt").string());
    brolm::clip::TextEncoder tenc(m.text);
    tenc.load_weights(f, "text_model.");
    auto cfg1 = m.text;
    cfg1.num_layers = 1;
    brolm::clip::TextEncoder tenc1(cfg1);
    tenc1.load_weights(f, "text_model.");
    const auto prompts = split_lines(need(fx, "prompt_texts").s);
    const int Dt = m.text.hidden_dim;
    double d_ids = 0, d_hidden = 0, d_l1 = 0;
    for (std::size_t p = 0; p < prompts.size(); ++p) {
        const std::string P = "prompt/" + std::to_string(p) + "/";
        const auto& ref_ids = need(fx, P + "ids").i;
        const auto ids = tok.encode(prompts[p]);
        const int n = static_cast<int>(ref_ids.size());
        for (int i = 0; i < n; ++i)
            if (ids[static_cast<std::size_t>(i)] != ref_ids[static_cast<std::size_t>(i)]) d_ids += 1;
        bt::Tensor out;
        tenc.forward(ids.data(), out);
        std::vector<float> h = bdtest::bd_download(out);
        h.resize(static_cast<std::size_t>(n) * Dt);
        d_hidden = std::max(d_hidden, max_abs(h, need(fx, P + "hidden").f));
        tenc1.forward(ids.data(), out);
        const std::vector<float> h1 = bdtest::bd_download(out);
        d_l1 = std::max(d_l1, max_abs(row(h1, n - 1, Dt), need(fx, P + "pooled_l1").f));
    }
    report("tokenizer ids differing", d_ids, 0);
    report("text block 0 at EOT (max abs)", d_l1, tol.pooled_l1);
    report("text hidden up to EOT (max abs)", d_hidden, tol.hidden);

    // Embeddings through CLIPScorer, the bro.lm ClipModel path: the image goes
    // in as [-1, 1] RGB and is resized (an identity at 224) and normalised by
    // brolm itself.
    brolm::clip_score::Config scfg;
    scfg.projection_dim = m.embed_dim;
    brolm::clip_score::CLIPScorer scorer(tok, tenc, venc, scfg);
    scorer.load_projections(f);
    const auto& rgb = need(fx, "image/rgb");
    const int S = rgb.dims[0];
    std::vector<float> chw(static_cast<std::size_t>(3) * S * S);
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < S * S; ++i)
            chw[static_cast<std::size_t>(c) * S * S + i] =
                static_cast<float>(rgb.i[static_cast<std::size_t>(i) * 3 + c]) / 127.5f - 1.0f;
    const std::vector<float> img = scorer.encode_image(chw, S, S);
    report("image embedding (max abs)", max_abs(img, need(fx, "image/embed").f), tol.embed);

    // encode_images: a batch of crops (the fixture image, mirrored, a
    // non-square crop of it, a smooth ramp) against encode_image of each.
    // The batch runs the same arithmetic a row at a time, so the CPU is held
    // to bit-identical. A GPU is held to the embedding tolerance: Vulkan's
    // cooperative-matrix GEMM rounds a row by where it sits in its tile (a row
    // at offset 257 instead of 0 differs by an FP16 ulp in a few elements),
    // and halo measured 2.4e-4 (BioCLIP 2) / 1.2e-3 (BioCLIP) here.
    {
        std::vector<std::vector<float>> ims(4);
        std::vector<std::pair<int, int>> hw = {{S, S}, {S, S}, {S / 2, S}, {S, S}};
        ims[0] = chw;
        ims[1].resize(chw.size());
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < S; ++y)
                for (int x = 0; x < S; ++x)
                    ims[1][(static_cast<std::size_t>(c) * S + y) * S + x] =
                        chw[(static_cast<std::size_t>(c) * S + y) * S + (S - 1 - x)];
        for (int c = 0; c < 3; ++c)   // the top half: (S/2) x S
            ims[2].insert(ims[2].end(), chw.begin() + static_cast<std::ptrdiff_t>(c) * S * S,
                          chw.begin() + static_cast<std::ptrdiff_t>(c) * S * S + (S / 2) * S);
        ims[3].resize(chw.size());
        for (std::size_t i = 0; i < ims[3].size(); ++i)
            ims[3][i] = static_cast<float>(i % (S * 3)) / (S * 1.5f) - 1.0f;
        std::vector<brolm::clip_score::CLIPScorer::ImageRef> refs;
        for (std::size_t i = 0; i < ims.size(); ++i) refs.push_back({ims[i].data(), hw[i].first, hw[i].second});
        const auto batch = scorer.encode_images(refs);
        double d = 0;
        bool sizes = batch.size() == ims.size();
        for (std::size_t i = 0; i < ims.size() && sizes; ++i) {
            const auto one = scorer.encode_image(ims[i], hw[i].first, hw[i].second);
            sizes = sizes && one.size() == batch[i].size();
            if (sizes) d = std::max(d, max_abs(batch[i], one));
        }
        const double btol = dev.type == bt::DeviceType::CPU ? 0.0 : tol.embed;
        report("encode_images batch of 4 vs one at a time", sizes ? d : 1e9, btol);
        if (sizes) report("batch: image 0 vs encode_image (max abs)", max_abs(batch[0], img), btol);
    }

    double d_text = 0, d_sim = 0, d_prob = 0;
    std::vector<double> logits;
    const double scale = need(fx, "logit_scale").f[0];
    for (std::size_t p = 0; p < prompts.size(); ++p) {
        scorer.set_prompt(prompts[p]);
        const auto& t = scorer.text_feature();
        d_text = std::max(d_text, max_abs(t, need(fx, "prompt/" + std::to_string(p) + "/embed").f));
        double s = 0;
        for (std::size_t i = 0; i < t.size(); ++i) s += static_cast<double>(t[i]) * img[i];
        d_sim = std::max(d_sim, std::fabs(s - need(fx, "sim").f[p]));
        logits.push_back(s * scale);
    }
    const double mx = *std::max_element(logits.begin(), logits.end());
    double z = 0;
    for (double l : logits) z += std::exp(l - mx);
    for (std::size_t p = 0; p < logits.size(); ++p)
        d_prob = std::max(d_prob, std::fabs(std::exp(logits[p] - mx) / z - need(fx, "probs").f[p]));
    report("text embeddings (max abs)", d_text, tol.embed);
    report("cosine similarity (max abs)", d_sim, tol.sim);
    report("softmax over prompts (max abs)", d_prob, tol.prob);

    // The checkpoint's logit_scale is the log of the reference's scale.
    const st::TensorView* ls = f.find("logit_scale");
    if (ls && ls->dtype == st::Dtype::F32 && ls->nbytes == 4) {
        float raw = 0;
        std::memcpy(&raw, ls->data, 4);
        report("exp(logit_scale) (abs)", std::fabs(std::exp(static_cast<double>(raw)) - scale), 1e-3);
    }
}

fs::path model_dir(const char* env, const char* sibling) {
    const char* e = std::getenv(env);
    if (e && e[0]) return e;
    return fs::path(BROLM_SIBLING_DIR) / "brovisionml-data" / sibling;
}

void run_model(const char* name, const char* env, const char* sibling) {
    const fs::path dir = model_dir(env, sibling);
    const fs::path fixture = fs::path(BROLM_TEST_FIXTURE_DIR) / (std::string(name) + ".bin");
    std::printf("=== %s parity vs open_clip ===\n", name);
    if (!fs::exists(dir / "model.safetensors") || !fs::exists(fixture)) {
        std::printf("[skip] need %s and %s (scripts/convert-openclip.py, "
                    "tests/ref/gen_bioclip_fixture.py; or set %s)\n",
                    (dir / "model.safetensors").string().c_str(), fixture.string().c_str(), env);
        return;
    }
    const Fixture fx = read_fixture(fixture);
    run_device(dir, fx, bt::Device::CPU, kCpuTol);
    const bt::Device gpu = bt::default_device();
    if (gpu != bt::Device::CPU) run_device(dir, fx, gpu, kGpuTol);
    else std::printf("  (no GPU backend: CPU only)\n");
}

}  // namespace

int main() {
    bt::init();
    try {
        run_model("bioclip", "BROLM_BIOCLIP_DIR", "bioclip");
        run_model("bioclip-2", "BROLM_BIOCLIP2_DIR", "bioclip-2");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: exception: %s\n", e.what());
        return 1;
    }
    if (g_failures) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("All BioCLIP parity checks passed.\n");
    return 0;
}
