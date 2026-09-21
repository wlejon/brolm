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
// Qwen35Model Decorator
// ═══════════════════════════════════════════════════════════════════════════

static void decorateQwen35Model(ObjectBuilder& b) {
    b.accessor("family", [](Value, std::span<const Value>) {
        return ev::fromUtf8("qwen35");
    });
    b.accessor("vocabSize", [](Value self, std::span<const Value>) {
        auto* w = hostQwen35ModelOf(self);
        return ev::fromDouble(w && w->vlm ? w->vlm->config().text.vocab_size : 0);
    });
    b.accessor("hiddenSize", [](Value self, std::span<const Value>) {
        auto* w = hostQwen35ModelOf(self);
        return ev::fromDouble(w && w->vlm ? w->vlm->config().text.hidden_size : 0);
    });
    b.accessor("numLayers", [](Value self, std::span<const Value>) {
        auto* w = hostQwen35ModelOf(self);
        return ev::fromDouble(w && w->vlm ? w->vlm->config().text.num_hidden_layers : 0);
    });
    b.accessor("maxSeqLen", [](Value self, std::span<const Value>) {
        auto* w = hostQwen35ModelOf(self);
        return ev::fromDouble(w ? w->maxSeqLen : 4096);
    });
    b.accessor("eosId", [](Value self, std::span<const Value>) {
        auto* w = hostQwen35ModelOf(self);
        return ev::fromDouble(w && w->vlm ? w->vlm->tokenizer().eos_id() : 151645);
    });
    b.accessor("imEndId", [](Value self, std::span<const Value>) {
        auto* w = hostQwen35ModelOf(self);
        return ev::fromDouble(w && w->vlm ? w->vlm->tokenizer().im_end_id() : 151645);
    });
    b.accessor("endoftextId", [](Value self, std::span<const Value>) {
        auto* w = hostQwen35ModelOf(self);
        return ev::fromDouble(w && w->vlm ? w->vlm->tokenizer().endoftext_id() : 151643);
    });

    b.def("encode", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostQwen35ModelOf(self);
        if (!w || !w->vlm) return ev::throwTypeError("encode: not a Qwen35Model");
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

    b.def("decode", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostQwen35ModelOf(self);
        if (!w || !w->vlm) return ev::throwTypeError("decode: not a Qwen35Model");
        if (a.empty()) return ev::throwTypeError("decode(ids): ids required");
        std::vector<int32_t> ids = readInt32Array(a[0]);
        try {
            return ev::fromUtf8(w->vlm->tokenizer().decode(ids));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("decode: ") + e.what());
        }
    });

    b.def("generate", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostQwen35ModelOf(self);
        if (!w || !w->vlm) return ev::throwTypeError("generate: not a Qwen35Model");
        if (a.empty() || !ev::isString(a[0]))
            return ev::throwTypeError("generate(prompt, opts?): prompt string required");

        std::string prompt = ev::toUtf8(a[0]);
        brolm::qwen::GenerateOptions opts;
        std::vector<VlmImage> images;
        Value onTokenVal = ev::undefined();

        if (a.size() >= 2 && ev::isObject(a[1])) {
            opts = parseGenerateOptions(a[1]);
            onTokenVal = ev::getProperty(a[1], "onToken");
            std::string imgErr;
            if (!readVlmImages(a[1], images, imgErr))
                return ev::throwTypeError(imgErr);
        }

        brolm::qwen35::VLM::TokenCallback hook;
        if (ev::isFunction(onTokenVal)) {
            ev::Persistent tokenCb(onTokenVal);
            hook = [tokenCb](int id) -> bool {
                Value arg = ev::fromDouble(id);
                ev::CallResult r = ev::call(tokenCb.get(), ev::undefined(), std::span<const Value>(&arg, 1));
                if (r.thrown) return false;
                return ev::isUndefined(r.value) || ev::toBool(r.value);
            };
        }

        try {
            brotensor::DeviceScope scope(w->device);
            w->vlm->set_generation(opts.max_new_tokens, opts.sampling.temperature,
                                   opts.sampling.top_k, opts.sampling.top_p,
                                   opts.sampling.seed, opts.sampling.min_p,
                                   opts.sampling.repetition_penalty,
                                   opts.sampling.frequency_penalty,
                                   opts.sampling.presence_penalty,
                                   opts.stop_on_eos);
            std::vector<brolm::qwen35::ImageInput> inputs;
            inputs.reserve(images.size());
            for (auto& im : images)
                inputs.push_back(brolm::qwen35::ImageInput{ im.chw.data(), im.H, im.W });

            std::vector<int> ids;
            if (hook) {
                ids = w->vlm->generate_tokens(prompt, inputs, hook);
            } else {
                ids = w->vlm->generate_tokens(prompt, inputs);
            }
            std::vector<int32_t> i32(ids.begin(), ids.end());
            return makeInt32Array(i32.data(), i32.size());
        } catch (const std::exception& e) {
            return ev::throwError(std::string("generate: ") + e.what());
        }
    });

    b.def("generateStream", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostQwen35ModelOf(self);
        if (!w || !w->vlm) return ev::throwTypeError("generateStream: not a Qwen35Model");
        if (a.empty()) return ev::throwTypeError("generateStream: prompt required");

        std::string prompt;
        if (ev::isString(a[0])) {
            prompt = ev::toUtf8(a[0]);
        } else if (ev::isObject(a[0])) {
            std::vector<int32_t> pIds = readInt32Array(a[0]);
            if (pIds.empty()) return ev::throwTypeError("generateStream: promptIds must be non-empty");
            prompt = w->vlm->tokenizer().decode(pIds);
        } else {
            return ev::throwTypeError("generateStream: prompt must be a string or Int32Array");
        }

        brolm::qwen::GenerateOptions opts;
        std::vector<VlmImage> images;
        Value onTokenVal = ev::undefined();

        if (a.size() >= 3 && ev::isFunction(a[2])) {
            onTokenVal = a[2];
            if (ev::isObject(a[1])) {
                opts = parseGenerateOptions(a[1]);
                std::string imgErr;
                if (!readVlmImages(a[1], images, imgErr))
                    return ev::throwTypeError(imgErr);
            }
        } else if (a.size() >= 2 && ev::isFunction(a[1])) {
            onTokenVal = a[1];
        } else if (a.size() >= 2 && ev::isObject(a[1])) {
            opts = parseGenerateOptions(a[1]);
            onTokenVal = ev::getProperty(a[1], "onToken");
            std::string imgErr;
            if (!readVlmImages(a[1], images, imgErr))
                return ev::throwTypeError(imgErr);
        }

        if (!ev::isFunction(onTokenVal))
            return ev::throwTypeError("generateStream: onToken callback must be a function");

        ev::Persistent tokenCb(onTokenVal);
        brolm::qwen35::VLM::TokenCallback hook = [tokenCb](int id) -> bool {
            Value arg = ev::fromDouble(id);
            ev::CallResult r = ev::call(tokenCb.get(), ev::undefined(), std::span<const Value>(&arg, 1));
            if (r.thrown) return false;
            return ev::isUndefined(r.value) || ev::toBool(r.value);
        };

        try {
            brotensor::DeviceScope scope(w->device);
            w->vlm->set_generation(opts.max_new_tokens, opts.sampling.temperature,
                                   opts.sampling.top_k, opts.sampling.top_p,
                                   opts.sampling.seed, opts.sampling.min_p,
                                   opts.sampling.repetition_penalty,
                                   opts.sampling.frequency_penalty,
                                   opts.sampling.presence_penalty,
                                   opts.stop_on_eos);
            std::vector<brolm::qwen35::ImageInput> inputs;
            inputs.reserve(images.size());
            for (auto& im : images)
                inputs.push_back(brolm::qwen35::ImageInput{ im.chw.data(), im.H, im.W });

            auto ids = w->vlm->generate_tokens(prompt, inputs, hook);
            std::vector<int32_t> i32(ids.begin(), ids.end());
            return makeInt32Array(i32.data(), i32.size());
        } catch (const std::exception& e) {
            return ev::throwError(std::string("generateStream: ") + e.what());
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Qwen3VLModel Decorator
// ═══════════════════════════════════════════════════════════════════════════

static void decorateQwen3VLModel(ObjectBuilder& b) {
    b.accessor("family", [](Value, std::span<const Value>) {
        return ev::fromUtf8("qwen3vl");
    });
    b.accessor("vocabSize", [](Value self, std::span<const Value>) {
        auto* w = hostQwen3VLModelOf(self);
        return ev::fromDouble(w && w->vlm ? w->vlm->config().text.vocab_size : 0);
    });
    b.accessor("hiddenSize", [](Value self, std::span<const Value>) {
        auto* w = hostQwen3VLModelOf(self);
        return ev::fromDouble(w && w->vlm ? w->vlm->config().text.hidden_size : 0);
    });
    b.accessor("numLayers", [](Value self, std::span<const Value>) {
        auto* w = hostQwen3VLModelOf(self);
        return ev::fromDouble(w && w->vlm ? w->vlm->config().text.num_hidden_layers : 0);
    });
    b.accessor("maxSeqLen", [](Value self, std::span<const Value>) {
        auto* w = hostQwen3VLModelOf(self);
        return ev::fromDouble(w ? w->maxSeqLen : 4096);
    });
    b.accessor("eosId", [](Value self, std::span<const Value>) {
        auto* w = hostQwen3VLModelOf(self);
        return ev::fromDouble(w && w->vlm ? w->vlm->tokenizer().eos_id() : 151645);
    });
    b.accessor("imEndId", [](Value self, std::span<const Value>) {
        auto* w = hostQwen3VLModelOf(self);
        return ev::fromDouble(w && w->vlm ? w->vlm->tokenizer().im_end_id() : 151645);
    });
    b.accessor("endoftextId", [](Value self, std::span<const Value>) {
        auto* w = hostQwen3VLModelOf(self);
        return ev::fromDouble(w && w->vlm ? w->vlm->tokenizer().endoftext_id() : 151643);
    });

    b.def("encode", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostQwen3VLModelOf(self);
        if (!w || !w->vlm) return ev::throwTypeError("encode: not a Qwen3VLModel");
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

    b.def("decode", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostQwen3VLModelOf(self);
        if (!w || !w->vlm) return ev::throwTypeError("decode: not a Qwen3VLModel");
        if (a.empty()) return ev::throwTypeError("decode(ids): ids required");
        std::vector<int32_t> ids = readInt32Array(a[0]);
        try {
            return ev::fromUtf8(w->vlm->tokenizer().decode(ids));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("decode: ") + e.what());
        }
    });

    b.def("generate", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostQwen3VLModelOf(self);
        if (!w || !w->vlm) return ev::throwTypeError("generate: not a Qwen3VLModel");
        if (a.empty() || !ev::isString(a[0]))
            return ev::throwTypeError("generate(prompt, opts?): prompt string required");

        std::string prompt = ev::toUtf8(a[0]);
        brolm::qwen::GenerateOptions opts;
        std::vector<VlmImage> images;
        Value onTokenVal = ev::undefined();

        if (a.size() >= 2 && ev::isObject(a[1])) {
            opts = parseGenerateOptions(a[1]);
            onTokenVal = ev::getProperty(a[1], "onToken");
            std::string imgErr;
            if (!readVlmImages(a[1], images, imgErr))
                return ev::throwTypeError(imgErr);
        }

        brolm::qwen3vl::VLM::TokenCallback hook;
        if (ev::isFunction(onTokenVal)) {
            ev::Persistent tokenCb(onTokenVal);
            hook = [tokenCb](int id) -> bool {
                Value arg = ev::fromDouble(id);
                ev::CallResult r = ev::call(tokenCb.get(), ev::undefined(), std::span<const Value>(&arg, 1));
                if (r.thrown) return false;
                return ev::isUndefined(r.value) || ev::toBool(r.value);
            };
        }

        try {
            brotensor::DeviceScope scope(w->device);
            w->vlm->set_generation(opts.max_new_tokens, opts.sampling.temperature,
                                   opts.sampling.top_k, opts.sampling.top_p,
                                   opts.sampling.seed, opts.sampling.min_p,
                                   opts.sampling.repetition_penalty,
                                   opts.sampling.frequency_penalty,
                                   opts.sampling.presence_penalty,
                                   opts.stop_on_eos);
            std::vector<brolm::qwen3vl::ImageInput> inputs;
            inputs.reserve(images.size());
            for (auto& im : images)
                inputs.push_back(brolm::qwen3vl::ImageInput{ im.chw.data(), im.H, im.W });

            std::vector<int> ids;
            if (hook) {
                ids = w->vlm->generate_tokens(prompt, inputs, hook);
            } else {
                ids = w->vlm->generate_tokens(prompt, inputs);
            }
            std::vector<int32_t> i32(ids.begin(), ids.end());
            return makeInt32Array(i32.data(), i32.size());
        } catch (const std::exception& e) {
            return ev::throwError(std::string("generate: ") + e.what());
        }
    });

    b.def("generateStream", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostQwen3VLModelOf(self);
        if (!w || !w->vlm) return ev::throwTypeError("generateStream: not a Qwen3VLModel");
        if (a.empty()) return ev::throwTypeError("generateStream: prompt required");

        std::string prompt;
        if (ev::isString(a[0])) {
            prompt = ev::toUtf8(a[0]);
        } else if (ev::isObject(a[0])) {
            std::vector<int32_t> pIds = readInt32Array(a[0]);
            if (pIds.empty()) return ev::throwTypeError("generateStream: promptIds must be non-empty");
            prompt = w->vlm->tokenizer().decode(pIds);
        } else {
            return ev::throwTypeError("generateStream: prompt must be a string or Int32Array");
        }

        brolm::qwen::GenerateOptions opts;
        std::vector<VlmImage> images;
        Value onTokenVal = ev::undefined();

        if (a.size() >= 3 && ev::isFunction(a[2])) {
            onTokenVal = a[2];
            if (ev::isObject(a[1])) {
                opts = parseGenerateOptions(a[1]);
                std::string imgErr;
                if (!readVlmImages(a[1], images, imgErr))
                    return ev::throwTypeError(imgErr);
            }
        } else if (a.size() >= 2 && ev::isFunction(a[1])) {
            onTokenVal = a[1];
        } else if (a.size() >= 2 && ev::isObject(a[1])) {
            opts = parseGenerateOptions(a[1]);
            onTokenVal = ev::getProperty(a[1], "onToken");
            std::string imgErr;
            if (!readVlmImages(a[1], images, imgErr))
                return ev::throwTypeError(imgErr);
        }

        if (!ev::isFunction(onTokenVal))
            return ev::throwTypeError("generateStream: onToken callback must be a function");

        ev::Persistent tokenCb(onTokenVal);
        brolm::qwen3vl::VLM::TokenCallback hook = [tokenCb](int id) -> bool {
            Value arg = ev::fromDouble(id);
            ev::CallResult r = ev::call(tokenCb.get(), ev::undefined(), std::span<const Value>(&arg, 1));
            if (r.thrown) return false;
            return ev::isUndefined(r.value) || ev::toBool(r.value);
        };

        try {
            brotensor::DeviceScope scope(w->device);
            w->vlm->set_generation(opts.max_new_tokens, opts.sampling.temperature,
                                   opts.sampling.top_k, opts.sampling.top_p,
                                   opts.sampling.seed, opts.sampling.min_p,
                                   opts.sampling.repetition_penalty,
                                   opts.sampling.frequency_penalty,
                                   opts.sampling.presence_penalty,
                                   opts.stop_on_eos);
            std::vector<brolm::qwen3vl::ImageInput> inputs;
            inputs.reserve(images.size());
            for (auto& im : images)
                inputs.push_back(brolm::qwen3vl::ImageInput{ im.chw.data(), im.H, im.W });

            auto ids = w->vlm->generate_tokens(prompt, inputs, hook);
            std::vector<int32_t> i32(ids.begin(), ids.end());
            return makeInt32Array(i32.data(), i32.size());
        } catch (const std::exception& e) {
            return ev::throwError(std::string("generateStream: ") + e.what());
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Registration and Loader Functions
// ═══════════════════════════════════════════════════════════════════════════

void registerLMVLClasses() {
    g_qwen35ModelClass.install("Qwen35Model", 0, nullptr, decorateQwen35Model);
    g_qwen3VLModelClass.install("Qwen3VLModel", 0, nullptr, decorateQwen3VLModel);
}

Value js_loadQwen35(Value, std::span<const Value> a) {
    if (a.empty() || !ev::isString(a[0]))
        return ev::throwTypeError("loadQwen35(checkpointDir, opts?): dir string required");
    std::string dir = ev::toUtf8(a[0]);

    brotensor::init();
    brotensor::Device dev = autoDevice();
    int maxSeqLen = 4096;
    if (a.size() >= 2 && ev::isObject(a[1])) {
        std::string err;
        if (!parseDeviceOpt(a[1], dev, err))
            return ev::throwTypeError(err);
        Value msl = ev::getProperty(a[1], "maxSeqLen");
        if (ev::isNumber(msl)) maxSeqLen = static_cast<int>(ev::toDouble(msl));
    }

    try {
        brolm::qwen35::VLMConfig vcfg;
        vcfg.max_seq_len = maxSeqLen;
        std::unique_ptr<brolm::qwen35::VLM> vlm;
        {
            brotensor::DeviceScope scope(dev);
            vlm = std::make_unique<brolm::qwen35::VLM>(vcfg);
            vlm->load_from_directory(dir);
        }
        return makeQwen35ModelValue(std::move(vlm), maxSeqLen, dev);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("loadQwen35: ") + e.what());
    }
}

Value js_loadQwen3VL(Value, std::span<const Value> a) {
    if (a.empty() || !ev::isString(a[0]))
        return ev::throwTypeError("loadQwen3VL(checkpointDir, opts?): dir string required");
    std::string dir = ev::toUtf8(a[0]);

    brotensor::init();
    brotensor::Device dev = autoDevice();
    int maxSeqLen = 4096;
    if (a.size() >= 2 && ev::isObject(a[1])) {
        std::string err;
        if (!parseDeviceOpt(a[1], dev, err))
            return ev::throwTypeError(err);
        Value msl = ev::getProperty(a[1], "maxSeqLen");
        if (ev::isNumber(msl)) maxSeqLen = static_cast<int>(ev::toDouble(msl));
    }

    try {
        brolm::qwen3vl::VLMConfig vcfg;
        vcfg.max_seq_len = maxSeqLen;
        std::unique_ptr<brolm::qwen3vl::VLM> vlm;
        {
            brotensor::DeviceScope scope(dev);
            vlm = std::make_unique<brolm::qwen3vl::VLM>(vcfg);
            vlm->load_from_directory(dir);
        }
        return makeQwen3VLModelValue(std::move(vlm), maxSeqLen, dev);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("loadQwen3VL: ") + e.what());
    }
}

} // namespace brolm::api
