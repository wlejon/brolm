// bro.lm.loadModernBert / ModernBertModel: the ModernBERT bidirectional text
// encoder (brolm/modernbert.h) with its byte-level / Metaspace BPE tokenizer
// (brolm/laya_tokenizer.h), bound the way T5Model is: encode() returns the
// last hidden state as a Float32Array, optionally pooled.
//
// A checkpoint is either a Hugging Face ModernBERT directory (config.json,
// tokenizer.json, model.safetensors with "model." tensors) or a Laya
// checkpoint, whose encoder lives under encoder/config.json, tokenizer/ and
// the "encoder." tensors of model.safetensors.

#include "host_lm_internal.h"

#include <brolm/detail/weights.h>
#include <brolm/laya_tokenizer.h>
#include <brolm/modernbert.h>

namespace brolm::api {

HostClass g_modernBertModelClass;

namespace {

struct HostModernBert {
    std::unique_ptr<brolm::modernbert::ModernBertModel> enc;
    std::unique_ptr<brolm::laya::LayaTokenizer> tok;
    int maxLength = 8192;
    brotensor::Device device = brotensor::Device::CPU;
};

HostModernBert* hostModernBertOf(Value v) {
    if (!ev::isObject(v)) return nullptr;
    return static_cast<HostModernBert*>(g_modernBertModelClass.unwrap(v));
}

// The first existing path of `candidates`, or "".
std::string firstExisting(std::initializer_list<std::filesystem::path> candidates) {
    for (const auto& p : candidates)
        if (std::filesystem::exists(p)) return p.generic_string();
    return {};
}

// [CLS] text [SEP], the text cut so the whole fits in maxLen (HF truncation
// keeps both specials).
std::vector<int32_t> wrapSpecial(const brolm::laya::LayaTokenizer& tok, std::vector<int32_t> body,
                                 int maxLen, bool addSpecial) {
    const std::size_t room = static_cast<std::size_t>(std::max(0, addSpecial ? maxLen - 2 : maxLen));
    if (body.size() > room) body.resize(room);
    if (!addSpecial) return body;
    std::vector<int32_t> ids;
    ids.reserve(body.size() + 2);
    ids.push_back(tok.cls_token_id());
    ids.insert(ids.end(), body.begin(), body.end());
    ids.push_back(tok.sep_token_id());
    return ids;
}

struct EncodeArgs {
    std::vector<int32_t> ids;
    std::string pooling = "none";
};

// encode / tokenize arguments: (text | ids, opts?). A string is tokenized
// and wrapped per opts.addSpecialTokens (default true) and opts.maxLength;
// an id array is taken as is (already wrapped).
bool readEncodeArgs(HostModernBert& w, const char* what, std::span<const Value> a, EncodeArgs& out,
                    std::string& err) {
    if (a.empty()) {
        err = std::string(what) + "(text, opts?): text string or id array required";
        return false;
    }
    int maxLen = w.maxLength;
    bool addSpecial = true;
    if (a.size() >= 2 && ev::isObject(a[1])) {
        double d = 0;
        if (propNumber(a[1], "maxLength", d)) {
            if (!(d >= 1) || d > w.maxLength) {
                err = std::string(what) + ": opts.maxLength must be in [1, " + std::to_string(w.maxLength) + "]";
                return false;
            }
            maxLen = static_cast<int>(d);
        }
        propBool(a[1], "addSpecialTokens", addSpecial);
        std::string pooling;
        if (propString(a[1], "pooling", pooling)) {
            if (pooling != "none" && pooling != "cls" && pooling != "mean") {
                err = std::string(what) + ": opts.pooling must be 'none', 'cls' or 'mean'";
                return false;
            }
            out.pooling = pooling;
        }
    }
    if (ev::isString(a[0])) {
        out.ids = wrapSpecial(*w.tok, w.tok->encode(ev::toUtf8(a[0])), maxLen, addSpecial);
    } else if (ev::isObject(a[0])) {
        out.ids = readInt32Array(a[0]);
        if (static_cast<int>(out.ids.size()) > maxLen) {
            err = std::string(what) + ": " + std::to_string(out.ids.size()) + " ids exceed maxLength " +
                  std::to_string(maxLen);
            return false;
        }
        const int V = w.enc->config().vocab_size;
        for (int32_t id : out.ids) {
            if (id < 0 || id >= V) {
                err = std::string(what) + ": id " + std::to_string(id) + " is outside the vocabulary (" +
                      std::to_string(V) + ")";
                return false;
            }
        }
    } else {
        err = std::string(what) + ": text must be a string or an id array";
        return false;
    }
    if (out.ids.empty()) {
        err = std::string(what) + ": nothing to encode";
        return false;
    }
    return true;
}

void decorateModernBert(ObjectBuilder& b) {
    b.accessor("family", [](Value, std::span<const Value>) { return ev::fromUtf8("modernbert"); });
    b.accessor("hiddenSize", [](Value self, std::span<const Value>) {
        auto* w = hostModernBertOf(self);
        return ev::fromDouble(w && w->enc ? w->enc->config().hidden_size : 0);
    });
    b.accessor("numLayers", [](Value self, std::span<const Value>) {
        auto* w = hostModernBertOf(self);
        return ev::fromDouble(w && w->enc ? w->enc->config().num_hidden_layers : 0);
    });
    b.accessor("vocabSize", [](Value self, std::span<const Value>) {
        auto* w = hostModernBertOf(self);
        return ev::fromDouble(w && w->enc ? w->enc->config().vocab_size : 0);
    });
    b.accessor("maxLength", [](Value self, std::span<const Value>) {
        auto* w = hostModernBertOf(self);
        return ev::fromDouble(w ? w->maxLength : 0);
    });
    b.accessor("clsId", [](Value self, std::span<const Value>) {
        auto* w = hostModernBertOf(self);
        return ev::fromDouble(w && w->tok ? w->tok->cls_token_id() : -1);
    });
    b.accessor("sepId", [](Value self, std::span<const Value>) {
        auto* w = hostModernBertOf(self);
        return ev::fromDouble(w && w->tok ? w->tok->sep_token_id() : -1);
    });
    b.accessor("padId", [](Value self, std::span<const Value>) {
        auto* w = hostModernBertOf(self);
        return ev::fromDouble(w && w->tok ? w->tok->pad_token_id() : -1);
    });
    b.accessor("maskId", [](Value self, std::span<const Value>) {
        auto* w = hostModernBertOf(self);
        return ev::fromDouble(w && w->tok ? w->tok->mask_token_id() : -1);
    });

    b.def("tokenize", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostModernBertOf(self);
        if (!w || !w->enc || !w->tok) return ev::throwTypeError("tokenize: not a ModernBertModel");
        if (a.empty() || !ev::isString(a[0]))
            return ev::throwTypeError("tokenize(text, opts?): text string required");
        EncodeArgs args;
        std::string err;
        try {
            if (!readEncodeArgs(*w, "tokenize", a, args, err)) return ev::throwTypeError(err);
        } catch (const std::exception& e) {
            return ev::throwError(std::string("tokenize: ") + e.what());
        }
        return makeInt32Array(args.ids.data(), args.ids.size());
    });

    b.def("encode", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostModernBertOf(self);
        if (!w || !w->enc || !w->tok) return ev::throwTypeError("encode: not a ModernBertModel");
        EncodeArgs args;
        std::string err;
        std::vector<float> host;
        int D = 0;
        try {
            if (!readEncodeArgs(*w, "encode", a, args, err)) return ev::throwTypeError(err);
            D = w->enc->config().hidden_size;
            brotensor::DeviceScope scope(w->device);
            brotensor::Tensor out;
            w->enc->forward(args.ids.data(), static_cast<int>(args.ids.size()), out);
            brotensor::sync_all();
            host = downloadFloats(out);
        } catch (const std::exception& e) {
            return ev::throwError(std::string("encode: ") + e.what());
        }
        const std::size_t L = args.ids.size();

        ObjectBuilder res;
        res.set("data", makeFloat32Array(host.data(), host.size()));
        res.set("length", ev::fromDouble(static_cast<double>(L)));
        res.set("dim", ev::fromDouble(D));
        res.set("ids", makeInt32Array(args.ids.data(), args.ids.size()));
        if (args.pooling != "none") {
            std::vector<float> pooled(static_cast<std::size_t>(D), 0.0f);
            if (args.pooling == "cls") {
                std::copy(host.begin(), host.begin() + D, pooled.begin());
            } else {
                for (std::size_t r = 0; r < L; ++r)
                    for (int c = 0; c < D; ++c) pooled[static_cast<std::size_t>(c)] += host[r * D + c];
                for (float& v : pooled) v /= static_cast<float>(L);
            }
            res.set("pooled", makeFloat32Array(pooled.data(), pooled.size()));
        }
        return res.build();
    });
}

}  // namespace

void registerLMModernBertClass() {
    g_modernBertModelClass.install("ModernBertModel", 0, nullptr, decorateModernBert);
}

// loadModernBert(dir, { device, configPath, tokenizerPath, weightsPath,
//                       prefix, maxLength, onReady, onError })
Value js_loadModernBert(Value, std::span<const Value> a) {
    if (a.empty() || !ev::isString(a[0]))
        return ev::throwTypeError("loadModernBert(checkpointDir, opts?): dir string required");
    const std::filesystem::path dir = resolvePath(ev::toUtf8(a[0]));

    std::string configPath, tokPath, weightsPath, prefix;
    bool havePrefix = false;
    int maxLength = 0;
    brotensor::init();
    brotensor::Device dev = autoDevice();
    if (a.size() >= 2 && ev::isObject(a[1])) {
        std::string err;
        if (!parseDeviceOpt(a[1], dev, err)) return ev::throwTypeError("loadModernBert: " + err);
        if (propString(a[1], "configPath", configPath)) configPath = resolvePath(configPath);
        if (propString(a[1], "tokenizerPath", tokPath)) tokPath = resolvePath(tokPath);
        if (propString(a[1], "weightsPath", weightsPath)) weightsPath = resolvePath(weightsPath);
        havePrefix = propString(a[1], "prefix", prefix);
        double d = 0;
        if (propNumber(a[1], "maxLength", d)) {
            if (!(d >= 2)) return ev::throwRangeError("loadModernBert: opts.maxLength must be >= 2");
            maxLength = static_cast<int>(d);
        }
    }
    // The HF layout first, then a Laya checkpoint's encoder.
    if (configPath.empty()) configPath = firstExisting({dir / "config.json", dir / "encoder" / "config.json"});
    if (tokPath.empty()) tokPath = firstExisting({dir / "tokenizer.json", dir / "tokenizer" / "tokenizer.json"});
    if (weightsPath.empty()) weightsPath = firstExisting({dir / "model.safetensors"});
    if (configPath.empty())
        return ev::throwError("loadModernBert: no config.json or encoder/config.json in " + dir.generic_string());
    if (tokPath.empty())
        return ev::throwError("loadModernBert: no tokenizer.json or tokenizer/tokenizer.json in " + dir.generic_string());
    if (weightsPath.empty())
        return ev::throwError("loadModernBert: no model.safetensors in " + dir.generic_string());

    struct Built { std::unique_ptr<HostModernBert> w; };
    return runLoad<Built>(a, 1, "loadModernBert",
        [configPath, tokPath, weightsPath, prefix, havePrefix, maxLength, dev](Built& b) {
            auto w = std::make_unique<HostModernBert>();
            w->device = dev;
            brolm::modernbert::Config cfg = brolm::modernbert::Config::load(configPath);
            cfg.validate();
            w->maxLength = maxLength > 0 ? std::min(maxLength, cfg.max_position_embeddings)
                                         : cfg.max_position_embeddings;
            w->tok = std::make_unique<brolm::laya::LayaTokenizer>(brolm::laya::LayaTokenizer::load(tokPath));
            for (const int32_t id : {w->tok->cls_token_id(), w->tok->sep_token_id()})
                if (id < 0 || id >= cfg.vocab_size)
                    throw std::runtime_error("tokenizer special id " + std::to_string(id) +
                                             " is outside the encoder vocabulary");

            brotensor::safetensors::File st = brotensor::safetensors::File::open(weightsPath);
            std::string pfx = prefix;
            if (!havePrefix) {
                bool found = false;
                for (const char* p : {"model.", "encoder.", ""}) {
                    if (st.find(std::string(p) + "embeddings.tok_embeddings.weight")) {
                        pfx = p;
                        found = true;
                        break;
                    }
                }
                if (!found)
                    throw std::runtime_error("no embeddings.tok_embeddings.weight under 'model.', 'encoder.' or '' in " +
                                             weightsPath + " (pass opts.prefix)");
            }
            brotensor::DeviceScope scope(dev);
            w->enc = std::make_unique<brolm::modernbert::ModernBertModel>(std::move(cfg));
            brolm::detail::weights::SafetensorsSource src({&st});
            w->enc->load_weights(src, pfx);
            b.w = std::move(w);
        },
        [](Built& b) { return g_modernBertModelClass.createInstance(std::move(b.w)); });
}

}  // namespace brolm::api
