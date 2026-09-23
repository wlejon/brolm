#include "host_lm_internal.h"

namespace brolm::api {

HostClass g_qwen35ModelClass;
HostClass g_qwen3VLModelClass;

// ═══════════════════════════════════════════════════════════════════════════
// Creators and Unwrappers
// ═══════════════════════════════════════════════════════════════════════════

HostQwen35Model* hostQwen35ModelOf(Value v) {
    if (!ev::isObject(v)) return nullptr;
    return static_cast<HostQwen35Model*>(g_qwen35ModelClass.unwrap(v));
}

Value makeQwen35ModelValue(std::unique_ptr<brolm::qwen35::VLM> vlm, int maxSeqLen, brotensor::Device dev) {
    auto w = std::make_unique<HostQwen35Model>();
    w->vlm = std::move(vlm);
    w->maxSeqLen = maxSeqLen;
    w->device = dev;
    return g_qwen35ModelClass.createInstance(std::move(w));
}

HostQwen3VLModel* hostQwen3VLModelOf(Value v) {
    if (!ev::isObject(v)) return nullptr;
    return static_cast<HostQwen3VLModel*>(g_qwen3VLModelClass.unwrap(v));
}

Value makeQwen3VLModelValue(std::unique_ptr<brolm::qwen3vl::VLM> vlm, int maxSeqLen, brotensor::Device dev) {
    auto w = std::make_unique<HostQwen3VLModel>();
    w->vlm = std::move(vlm);
    w->maxSeqLen = maxSeqLen;
    w->device = dev;
    return g_qwen3VLModelClass.createInstance(std::move(w));
}

// ═══════════════════════════════════════════════════════════════════════════
// The shared VLM surface (Qwen3.5 and Qwen3-VL drive one driver shape)
// ═══════════════════════════════════════════════════════════════════════════

template <class HostT, class ImageInput>
struct VlmOps {
    HostT* (*of)(Value);
    const char* className;
    const char* family;
};

// generate(prompt, opts?) / generateStream(prompt, opts, onToken): the prompt
// is a string (the driver owns tokenization) or ids decoded back to one.
template <class HostT, class ImageInput>
static Value vlmGenerate(const VlmOps<HostT, ImageInput>& ops, const char* what, bool stream,
                         Value self, std::span<const Value> a) {
    HostT* w = ops.of(self);
    if (!w || !w->vlm) return ev::throwTypeError(std::string(what) + ": not a " + ops.className);
    if (a.empty()) return ev::throwTypeError(std::string(what) + "(prompt, opts?): prompt required");

    std::string prompt;
    if (ev::isString(a[0])) {
        prompt = ev::toUtf8(a[0]);
    } else if (stream && ev::isObject(a[0])) {
        std::vector<int32_t> ids = readInt32Array(a[0]);
        if (ids.empty()) return ev::throwTypeError(std::string(what) + ": promptIds must be non-empty");
        prompt = w->vlm->tokenizer().decode(ids);
    } else {
        return ev::throwTypeError(std::string(what) + ": prompt must be a string");
    }

    // Options sit at a[1] unless it is the callback; the callback is a[2],
    // a[1] (generateStream(prompt, onToken)) or opts.onToken.
    brolm::qwen::GenerateOptions opts;
    std::vector<VlmImage> images;
    ev::Persistent tokenCb;
    size_t optsIdx = SIZE_MAX;
    if (a.size() >= 3 && ev::isFunction(a[2])) {
        tokenCb.set(a[2]);
        if (ev::isObject(a[1])) optsIdx = 1;
    } else if (a.size() >= 2 && ev::isFunction(a[1])) {
        tokenCb.set(a[1]);
    } else if (a.size() >= 2 && ev::isObject(a[1])) {
        optsIdx = 1;
        tokenCb.set(ev::getProperty(a[1], "onToken"));
    }
    if (optsIdx != SIZE_MAX) {
        opts = parseGenerateOptions(a[optsIdx]);
        std::string imgErr;
        if (!readVlmImages(a[optsIdx], images, imgErr))
            return ev::throwTypeError(imgErr);
    }
    if (stream && !ev::isFunction(tokenCb.get()))
        return ev::throwTypeError(std::string(what) + ": onToken callback must be a function");

    std::function<bool(int)> hook;
    if (ev::isFunction(tokenCb.get())) {
        hook = [&tokenCb](int id) -> bool {
            const Value arg = ev::fromDouble(id);
            ev::CallResult r = ev::call(tokenCb.get(), ev::undefined(), std::span<const Value>(&arg, 1));
            if (r.thrown) return false;
            return ev::isUndefined(r.value) || ev::toBool(r.value);
        };
    }

    BusyClaim claim(w->generating);
    if (!claim.ok()) return ev::throwError(std::string(what) + ": " + kBusyMessage);
    try {
        brotensor::DeviceScope scope(w->device);
        w->vlm->set_generation(opts.max_new_tokens, opts.sampling.temperature,
                               opts.sampling.top_k, opts.sampling.top_p,
                               opts.sampling.seed, opts.sampling.min_p,
                               opts.sampling.repetition_penalty,
                               opts.sampling.frequency_penalty,
                               opts.sampling.presence_penalty,
                               opts.stop_on_eos);
        std::vector<ImageInput> inputs;
        inputs.reserve(images.size());
        for (auto& im : images) inputs.push_back(ImageInput{ im.chw.data(), im.H, im.W });

        std::vector<int> ids = hook ? w->vlm->generate_tokens(prompt, inputs, hook)
                                    : w->vlm->generate_tokens(prompt, inputs);
        std::vector<int32_t> i32(ids.begin(), ids.end());
        return makeInt32Array(i32.data(), i32.size());
    } catch (const std::exception& e) {
        return ev::throwError(std::string(what) + ": " + e.what());
    }
}

template <class HostT, class ImageInput>
static void decorateVlm(ObjectBuilder& b, const VlmOps<HostT, ImageInput>& ops) {
    const std::string family = ops.family;
    b.accessor("family", [family](Value, std::span<const Value>) {
        return ev::fromUtf8(family);
    });
    b.accessor("vocabSize", [ops](Value self, std::span<const Value>) {
        HostT* w = ops.of(self);
        return ev::fromDouble(w && w->vlm ? w->vlm->config().text.vocab_size : 0);
    });
    b.accessor("hiddenSize", [ops](Value self, std::span<const Value>) {
        HostT* w = ops.of(self);
        return ev::fromDouble(w && w->vlm ? w->vlm->config().text.hidden_size : 0);
    });
    b.accessor("numLayers", [ops](Value self, std::span<const Value>) {
        HostT* w = ops.of(self);
        return ev::fromDouble(w && w->vlm ? w->vlm->config().text.num_hidden_layers : 0);
    });
    b.accessor("maxSeqLen", [ops](Value self, std::span<const Value>) {
        HostT* w = ops.of(self);
        return ev::fromDouble(w ? w->maxSeqLen : 4096);
    });
    b.accessor("eosId", [ops](Value self, std::span<const Value>) {
        HostT* w = ops.of(self);
        return ev::fromDouble(w && w->vlm ? w->vlm->tokenizer().eos_id() : 151645);
    });
    b.accessor("imEndId", [ops](Value self, std::span<const Value>) {
        HostT* w = ops.of(self);
        return ev::fromDouble(w && w->vlm ? w->vlm->tokenizer().im_end_id() : 151645);
    });
    b.accessor("endoftextId", [ops](Value self, std::span<const Value>) {
        HostT* w = ops.of(self);
        return ev::fromDouble(w && w->vlm ? w->vlm->tokenizer().endoftext_id() : 151643);
    });
    b.accessor("busy", [ops](Value self, std::span<const Value>) {
        HostT* w = ops.of(self);
        return ev::fromBool(w && w->generating.load(std::memory_order_acquire));
    });

    b.def("encode", 1, [ops](Value self, std::span<const Value> a) -> Value {
        HostT* w = ops.of(self);
        if (!w || !w->vlm) return ev::throwTypeError(std::string("encode: not a ") + ops.className);
        if (a.empty() || !ev::isString(a[0]))
            return ev::throwTypeError("encode(text, addSpecial?): text string required");
        std::string text = ev::toUtf8(a[0]);
        bool addSpecial = (a.size() >= 2) && ev::toBool(a[1]);
        try {
            auto ids = w->vlm->tokenizer().encode(text, addSpecial);
            return makeInt32Array(ids.data(), ids.size());
        } catch (const std::exception& e) {
            return ev::throwError(std::string("encode: ") + e.what());
        }
    });

    b.def("decode", 1, [ops](Value self, std::span<const Value> a) -> Value {
        HostT* w = ops.of(self);
        if (!w || !w->vlm) return ev::throwTypeError(std::string("decode: not a ") + ops.className);
        if (a.empty()) return ev::throwTypeError("decode(ids): ids required");
        std::vector<int32_t> ids = readInt32Array(a[0]);
        try {
            return ev::fromUtf8(w->vlm->tokenizer().decode(ids));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("decode: ") + e.what());
        }
    });

    b.def("generate", 1, [ops](Value self, std::span<const Value> a) -> Value {
        return vlmGenerate(ops, "generate", false, self, a);
    });
    b.def("generateStream", 2, [ops](Value self, std::span<const Value> a) -> Value {
        return vlmGenerate(ops, "generateStream", true, self, a);
    });
}

static const VlmOps<HostQwen35Model, brolm::qwen35::ImageInput> kQwen35Ops{
    &hostQwen35ModelOf, "Qwen35Model", "qwen35"};
static const VlmOps<HostQwen3VLModel, brolm::qwen3vl::ImageInput> kQwen3VLOps{
    &hostQwen3VLModelOf, "Qwen3VLModel", "qwen3vl"};

// ═══════════════════════════════════════════════════════════════════════════
// Registration and Loader Functions
// ═══════════════════════════════════════════════════════════════════════════

void registerLMVLClasses() {
    g_qwen35ModelClass.install("Qwen35Model", 0, nullptr,
                               [](ObjectBuilder& b) { decorateVlm(b, kQwen35Ops); });
    g_qwen3VLModelClass.install("Qwen3VLModel", 0, nullptr,
                                [](ObjectBuilder& b) { decorateVlm(b, kQwen3VLOps); });
}

// loadQwen35 / loadQwen3VL (dir, { device, maxSeqLen, onReady, onError }).
template <class VLM, class VLMConfig>
static Value loadVlm(std::span<const Value> a, const char* what,
                     Value (*makeValue)(std::unique_ptr<VLM>, int, brotensor::Device)) {
    if (a.empty() || !ev::isString(a[0]))
        return ev::throwTypeError(std::string(what) + "(checkpointDir, opts?): dir string required");
    const std::string dir = resolvePath(ev::toUtf8(a[0]));

    brotensor::init();
    brotensor::Device dev = autoDevice();
    int maxSeqLen = 4096;
    if (a.size() >= 2 && ev::isObject(a[1])) {
        std::string err;
        if (!parseDeviceOpt(a[1], dev, err))
            return ev::throwTypeError(err);
        double d = 0;
        if (propNumber(a[1], "maxSeqLen", d)) maxSeqLen = static_cast<int>(d);
    }
    if (maxSeqLen <= 0)
        return ev::throwTypeError(std::string(what) + ": opts.maxSeqLen must be > 0");

    struct Built { std::unique_ptr<VLM> vlm; };
    return runLoad<Built>(a, 1, what,
        [dir, dev, maxSeqLen](Built& b) {
            VLMConfig vcfg;
            vcfg.max_seq_len = maxSeqLen;
            brotensor::DeviceScope scope(dev);
            b.vlm = std::make_unique<VLM>(vcfg);
            b.vlm->load_from_directory(dir);
        },
        [makeValue, maxSeqLen, dev](Built& b) { return makeValue(std::move(b.vlm), maxSeqLen, dev); });
}

Value js_loadQwen35(Value, std::span<const Value> a) {
    return loadVlm<brolm::qwen35::VLM, brolm::qwen35::VLMConfig>(a, "loadQwen35", &makeQwen35ModelValue);
}

Value js_loadQwen3VL(Value, std::span<const Value> a) {
    return loadVlm<brolm::qwen3vl::VLM, brolm::qwen3vl::VLMConfig>(a, "loadQwen3VL", &makeQwen3VLModelValue);
}

} // namespace brolm::api
