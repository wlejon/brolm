#include "host_lm_internal.h"
#include "brolm/detail/json.h"
#include <fstream>

namespace brolm::api {

HostClass g_clipModelClass;
HostClass g_nllbModelClass;
HostClass g_t5ModelClass;

// ═══════════════════════════════════════════════════════════════════════════
// Creators and Unwrappers
// ═══════════════════════════════════════════════════════════════════════════

HostClipModel* hostClipModelOf(Value v) {
    if (!ev::isObject(v)) return nullptr;
    return static_cast<HostClipModel*>(g_clipModelClass.unwrap(v));
}

Value makeClipModelValue(std::unique_ptr<HostClipModel> clip) {
    return g_clipModelClass.createInstance(std::move(clip));
}

HostNllbModel* hostNllbModelOf(Value v) {
    if (!ev::isObject(v)) return nullptr;
    return static_cast<HostNllbModel*>(g_nllbModelClass.unwrap(v));
}

Value makeNllbModelValue(std::unique_ptr<brolm::nllb::Translator> tr, brotensor::Device dev) {
    auto w = std::make_unique<HostNllbModel>();
    w->tr = std::move(tr);
    w->device = dev;
    return g_nllbModelClass.createInstance(std::move(w));
}

HostT5Model* hostT5ModelOf(Value v) {
    if (!ev::isObject(v)) return nullptr;
    return static_cast<HostT5Model*>(g_t5ModelClass.unwrap(v));
}

Value makeT5ModelValue(std::unique_ptr<HostT5Model> t5) {
    return g_t5ModelClass.createInstance(std::move(t5));
}

// ═══════════════════════════════════════════════════════════════════════════
// ClipModel Decorator
// ═══════════════════════════════════════════════════════════════════════════

static void decorateClipModel(ObjectBuilder& b) {
    b.accessor("projectionDim", [](Value self, std::span<const Value>) {
        auto* w = hostClipModelOf(self);
        return ev::fromDouble(w && w->scorer ? w->scorer->config().projection_dim : 768);
    });

    b.def("encodeText", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostClipModelOf(self);
        if (!w || !w->scorer) return ev::throwTypeError("encodeText: not a ClipModel");
        if (a.empty()) return ev::throwTypeError("encodeText(text): text string or array required");

        std::vector<std::string> prompts;
        if (ev::isString(a[0])) {
            prompts.push_back(ev::toUtf8(a[0]));
        } else if (ev::isObject(a[0])) {
            Value lenVal = ev::getProperty(a[0], "length");
            if (ev::isNumber(lenVal)) {
                uint32_t len = static_cast<uint32_t>(ev::toDouble(lenVal));
                for (uint32_t i = 0; i < len; ++i) {
                    Value item = ev::getElement(a[0], i);
                    if (ev::isString(item)) prompts.push_back(ev::toUtf8(item));
                }
            }
        }
        if (prompts.empty())
            return ev::throwTypeError("encodeText: text must be a non-empty string or array of strings");

        try {
            brotensor::DeviceScope scope(w->device);
            if (prompts.size() == 1 && ev::isString(a[0])) {
                w->scorer->set_prompt(prompts[0]);
                const auto& feat = w->scorer->text_feature();
                return makeFloat32Array(feat.data(), feat.size());
            }

            return hostArrayOf(prompts.size(), [&](size_t i) {
                w->scorer->set_prompt(prompts[i]);
                const auto& feat = w->scorer->text_feature();
                return makeFloat32Array(feat.data(), feat.size());
            });
        } catch (const std::exception& e) {
            return ev::throwError(std::string("encodeText: ") + e.what());
        }
    });

    b.def("encodeImage", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostClipModelOf(self);
        if (!w || !w->scorer) return ev::throwTypeError("encodeImage: not a ClipModel");
        if (a.empty()) return ev::throwTypeError("encodeImage(image): image required");

        std::vector<uint8_t> rgba;
        int width = 0, height = 0;
        std::string err;
        if (!readImageArg(a[0], rgba, width, height, err))
            return ev::throwTypeError(err);

        try {
            brotensor::DeviceScope scope(w->device);
            std::vector<float> nchw = rgbaToNchwSigned(rgba, width, height);
            std::vector<float> feat = w->scorer->encode_image(nchw, height, width);
            return makeFloat32Array(feat.data(), feat.size());
        } catch (const std::exception& e) {
            return ev::throwError(std::string("encodeImage: ") + e.what());
        }
    });

    auto scoreFn = [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostClipModelOf(self);
        if (!w || !w->scorer) return ev::throwTypeError("score: not a ClipModel");
        if (a.size() < 2) return ev::throwTypeError("score(text, image): text and image required");

        std::vector<uint8_t> rgba;
        int width = 0, height = 0;
        std::string err;
        if (!readImageArg(a[1], rgba, width, height, err))
            return ev::throwTypeError(err);

        std::vector<std::string> prompts;
        if (ev::isString(a[0])) {
            prompts.push_back(ev::toUtf8(a[0]));
        } else if (ev::isObject(a[0])) {
            Value lenVal = ev::getProperty(a[0], "length");
            if (ev::isNumber(lenVal)) {
                uint32_t len = static_cast<uint32_t>(ev::toDouble(lenVal));
                for (uint32_t i = 0; i < len; ++i) {
                    Value item = ev::getElement(a[0], i);
                    if (ev::isString(item)) prompts.push_back(ev::toUtf8(item));
                }
            }
        }
        if (prompts.empty())
            return ev::throwTypeError("score: text must be a string or array of strings");

        try {
            brotensor::DeviceScope scope(w->device);
            std::vector<float> nchw = rgbaToNchwSigned(rgba, width, height);

            if (prompts.size() == 1 && ev::isString(a[0])) {
                w->scorer->set_prompt(prompts[0]);
                float s = w->scorer->score(nchw, height, width);
                return ev::fromDouble(s);
            }

            return hostArrayOf(prompts.size(), [&](size_t i) {
                w->scorer->set_prompt(prompts[i]);
                float s = w->scorer->score(nchw, height, width);
                return ev::fromDouble(s);
            });
        } catch (const std::exception& e) {
            return ev::throwError(std::string("score: ") + e.what());
        }
    };

    b.def("score", 2, scoreFn);
    b.def("similarity", 2, scoreFn);
}

// ═══════════════════════════════════════════════════════════════════════════
// NllbModel Decorator
// ═══════════════════════════════════════════════════════════════════════════

static void decorateNllbModel(ObjectBuilder& b) {
    b.accessor("family", [](Value, std::span<const Value>) {
        return ev::fromUtf8("nllb");
    });
    b.accessor("vocabSize", [](Value self, std::span<const Value>) {
        auto* w = hostNllbModelOf(self);
        return ev::fromDouble(w && w->tr ? w->tr->config().vocab_size : 256206);
    });
    b.accessor("dModel", [](Value self, std::span<const Value>) {
        auto* w = hostNllbModelOf(self);
        return ev::fromDouble(w && w->tr ? w->tr->config().d_model : 1024);
    });
    b.accessor("encoderLayers", [](Value self, std::span<const Value>) {
        auto* w = hostNllbModelOf(self);
        return ev::fromDouble(w && w->tr ? w->tr->config().encoder_layers : 12);
    });
    b.accessor("decoderLayers", [](Value self, std::span<const Value>) {
        auto* w = hostNllbModelOf(self);
        return ev::fromDouble(w && w->tr ? w->tr->config().decoder_layers : 12);
    });
    b.accessor("languageCount", [](Value self, std::span<const Value>) {
        auto* w = hostNllbModelOf(self);
        return ev::fromDouble(w && w->tr ? static_cast<double>(w->tr->tokenizer().language_count()) : 200.0);
    });

    b.def("hasLanguage", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostNllbModelOf(self);
        if (!w || !w->tr) return ev::throwTypeError("hasLanguage: not an NllbModel");
        if (a.empty() || !ev::isString(a[0])) return ev::fromBool(false);
        return ev::fromBool(w->tr->tokenizer().has_lang(ev::toUtf8(a[0])));
    });

    b.def("translate", 3, [](Value self, std::span<const Value> a) -> Value {
        ev::Persistent selfRoot(self);  // an async job keeps the model alive
        auto* w = hostNllbModelOf(self);
        if (!w || !w->tr) return ev::throwTypeError("translate: not an NllbModel");
        if (a.size() < 3 || !ev::isString(a[0]) || !ev::isString(a[1]) || !ev::isString(a[2]))
            return ev::throwTypeError("translate(text, srcLang, tgtLang, opts?): three strings required");

        std::string text = ev::toUtf8(a[0]);
        std::string src = ev::toUtf8(a[1]);
        std::string tgt = ev::toUtf8(a[2]);

        if (!w->tr->tokenizer().has_lang(src))
            return ev::throwTypeError("translate: unknown source language: " + src);
        if (!w->tr->tokenizer().has_lang(tgt))
            return ev::throwTypeError("translate: unknown target language: " + tgt);

        brolm::nllb::BeamOptions bopts;
        ev::Persistent onDone, onError;
        if (a.size() >= 4 && ev::isObject(a[3])) {
            double d = 0;
            if (propNumber(a[3], "numBeams", d)) bopts.num_beams = static_cast<int>(d);
            if (propNumber(a[3], "maxNewTokens", d)) bopts.max_new_tokens = static_cast<int>(d);
            if (propNumber(a[3], "lengthPenalty", d)) bopts.length_penalty = static_cast<float>(d);
            onDone.set(ev::getProperty(a[3], "onDone"));
            onError.set(ev::getProperty(a[3], "onError"));
        }
        if (bopts.num_beams <= 0) return ev::throwRangeError("translate: opts.numBeams must be > 0");
        if (bopts.max_new_tokens <= 0) return ev::throwRangeError("translate: opts.maxNewTokens must be > 0");

        BusyClaim claim(w->translating);
        if (!claim.ok()) return ev::throwError("translate: a translation is already in flight on this model");

        // opts.onDone(text): the beam search runs on a worker thread and the
        // call returns an AsyncHandle; onDone fires on this thread's tick
        // (nothing after cancel(); a failure goes to onError).
        if (ev::isFunction(onDone.get())) {
            auto job = newAsyncJob();
            job->keep.set(selfRoot.get());
            job->onDone.set(onDone.get());
            if (ev::isFunction(onError.get())) job->onError.set(onError.get());
            job->busy = claim.detach();
            auto out = std::make_shared<std::string>();
            job->finish = [out](AsyncJob& j) {
                if (j.core->cancelled()) return;
                if (!j.core->error.empty()) {
                    reportJobError(j, "translate: " + j.core->error);
                    return;
                }
                ev::Persistent s(ev::fromUtf8(*out));
                callRooted(j.onDone, {&s});
            };
            return launchAsyncJob(job, [w, out, text, src, tgt, bopts](AsyncCore&) {
                brotensor::DeviceScope scope(w->device);
                *out = w->tr->translate(text, src, tgt, bopts);
            });
        }

        try {
            brotensor::DeviceScope scope(w->device);
            std::string out = w->tr->translate(text, src, tgt, bopts);
            return ev::fromUtf8(out);
        } catch (const std::exception& e) {
            return ev::throwError(std::string("translate: ") + e.what());
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// T5Model Decorator
// ═══════════════════════════════════════════════════════════════════════════

static void decorateT5Model(ObjectBuilder& b) {
    b.accessor("dModel", [](Value self, std::span<const Value>) {
        auto* w = hostT5ModelOf(self);
        return ev::fromDouble(w ? w->dModel : 4096);
    });
    b.accessor("maxLength", [](Value self, std::span<const Value>) {
        auto* w = hostT5ModelOf(self);
        return ev::fromDouble(w ? w->maxLength : 512);
    });
    b.accessor("padId", [](Value self, std::span<const Value>) {
        auto* w = hostT5ModelOf(self);
        return ev::fromDouble(w && w->tok ? w->tok->pad_id() : 0);
    });
    b.accessor("eosId", [](Value self, std::span<const Value>) {
        auto* w = hostT5ModelOf(self);
        return ev::fromDouble(w && w->tok ? w->tok->eos_id() : 1);
    });
    b.accessor("vocabCount", [](Value self, std::span<const Value>) {
        auto* w = hostT5ModelOf(self);
        return ev::fromDouble(w && w->tok ? static_cast<double>(w->tok->vocab_count()) : 32128.0);
    });

    b.def("encode", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostT5ModelOf(self);
        if (!w || !w->enc || !w->tok) return ev::throwTypeError("encode: not a T5Model");
        if (a.empty() || !ev::isString(a[0]))
            return ev::throwTypeError("encode(text, opts?): text string required");

        std::string text = ev::toUtf8(a[0]);
        int maxLen = w->maxLength;
        if (a.size() >= 2 && ev::isObject(a[1])) {
            Value ml = ev::getProperty(a[1], "maxLength");
            if (ev::isNumber(ml)) maxLen = static_cast<int>(ev::toDouble(ml));
        }
        if (maxLen <= 0) maxLen = 512;

        try {
            std::vector<int32_t> ids = w->tok->encode(text, maxLen);
            const int L = static_cast<int>(ids.size());
            brotensor::DeviceScope scope(w->device);
            brotensor::Tensor out;
            w->enc->forward(ids.data(), L, out, w->tok->pad_id());
            brotensor::sync_all();
            std::vector<float> host = downloadFloats(out);

            ObjectBuilder res;
            res.set("data", makeFloat32Array(host.data(), host.size()));
            res.set("length", ev::fromDouble(L));
            res.set("dim", ev::fromDouble(w->dModel));
            res.set("ids", makeInt32Array(ids.data(), ids.size()));
            return res.build();
        } catch (const std::exception& e) {
            return ev::throwError(std::string("encode: ") + e.what());
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Registration and Loader Functions
// ═══════════════════════════════════════════════════════════════════════════

void registerLMClipClasses() {
    g_clipModelClass.install("ClipModel", 0, nullptr, decorateClipModel);
    g_nllbModelClass.install("NllbModel", 0, nullptr, decorateNllbModel);
    g_t5ModelClass.install("T5Model", 0, nullptr, decorateT5Model);
}

namespace json = brolm::detail::json;

static void parseTextConfigJson(const json::Value& tc, brolm::clip::TextEncoderConfig& cfg) {
    if (!tc.is_object()) return;
    if (tc.contains("vocab_size")) cfg.vocab_size = tc.get_int("vocab_size", cfg.vocab_size);
    else if (tc.contains("vocabSize")) cfg.vocab_size = tc.get_int("vocabSize", cfg.vocab_size);

    if (tc.contains("max_position_embeddings")) cfg.max_position = tc.get_int("max_position_embeddings", cfg.max_position);
    else if (tc.contains("max_position")) cfg.max_position = tc.get_int("max_position", cfg.max_position);
    else if (tc.contains("maxPosition")) cfg.max_position = tc.get_int("maxPosition", cfg.max_position);

    if (tc.contains("hidden_size")) cfg.hidden_dim = tc.get_int("hidden_size", cfg.hidden_dim);
    else if (tc.contains("hidden_dim")) cfg.hidden_dim = tc.get_int("hidden_dim", cfg.hidden_dim);
    else if (tc.contains("hiddenDim")) cfg.hidden_dim = tc.get_int("hiddenDim", cfg.hidden_dim);

    if (tc.contains("num_attention_heads")) cfg.num_heads = tc.get_int("num_attention_heads", cfg.num_heads);
    else if (tc.contains("num_heads")) cfg.num_heads = tc.get_int("num_heads", cfg.num_heads);
    else if (tc.contains("numHeads")) cfg.num_heads = tc.get_int("numHeads", cfg.num_heads);

    if (tc.contains("num_hidden_layers")) cfg.num_layers = tc.get_int("num_hidden_layers", cfg.num_layers);
    else if (tc.contains("num_layers")) cfg.num_layers = tc.get_int("num_layers", cfg.num_layers);
    else if (tc.contains("numLayers")) cfg.num_layers = tc.get_int("numLayers", cfg.num_layers);

    if (tc.contains("intermediate_size")) cfg.intermediate_dim = tc.get_int("intermediate_size", cfg.intermediate_dim);
    else if (tc.contains("intermediate_dim")) cfg.intermediate_dim = tc.get_int("intermediate_dim", cfg.intermediate_dim);
    else if (tc.contains("intermediateDim")) cfg.intermediate_dim = tc.get_int("intermediateDim", cfg.intermediate_dim);

    if (tc.contains("layer_norm_eps")) cfg.layer_norm_eps = tc.get_float("layer_norm_eps", cfg.layer_norm_eps);
    else if (tc.contains("layerNormEps")) cfg.layer_norm_eps = tc.get_float("layerNormEps", cfg.layer_norm_eps);

    if (tc.contains("eos_token_id")) cfg.eos_token_id = tc.get_int("eos_token_id", cfg.eos_token_id);
    else if (tc.contains("eosTokenId")) cfg.eos_token_id = tc.get_int("eosTokenId", cfg.eos_token_id);
}

static void parseVisionConfigJson(const json::Value& vc, brolm::clip_image::ImageEncoderConfig& cfg) {
    if (!vc.is_object()) return;
    if (vc.contains("image_size")) cfg.image_size = vc.get_int("image_size", cfg.image_size);
    else if (vc.contains("imageSize")) cfg.image_size = vc.get_int("imageSize", cfg.image_size);

    if (vc.contains("patch_size")) cfg.patch_size = vc.get_int("patch_size", cfg.patch_size);
    else if (vc.contains("patchSize")) cfg.patch_size = vc.get_int("patchSize", cfg.patch_size);

    if (vc.contains("num_channels")) cfg.in_channels = vc.get_int("num_channels", cfg.in_channels);
    else if (vc.contains("in_channels")) cfg.in_channels = vc.get_int("in_channels", cfg.in_channels);
    else if (vc.contains("inChannels")) cfg.in_channels = vc.get_int("inChannels", cfg.in_channels);

    if (vc.contains("hidden_size")) cfg.hidden_dim = vc.get_int("hidden_size", cfg.hidden_dim);
    else if (vc.contains("hidden_dim")) cfg.hidden_dim = vc.get_int("hidden_dim", cfg.hidden_dim);
    else if (vc.contains("hiddenDim")) cfg.hidden_dim = vc.get_int("hiddenDim", cfg.hidden_dim);

    if (vc.contains("num_attention_heads")) cfg.num_heads = vc.get_int("num_attention_heads", cfg.num_heads);
    else if (vc.contains("num_heads")) cfg.num_heads = vc.get_int("num_heads", cfg.num_heads);
    else if (vc.contains("numHeads")) cfg.num_heads = vc.get_int("numHeads", cfg.num_heads);

    if (vc.contains("num_hidden_layers")) cfg.num_layers = vc.get_int("num_hidden_layers", cfg.num_layers);
    else if (vc.contains("num_layers")) cfg.num_layers = vc.get_int("num_layers", cfg.num_layers);
    else if (vc.contains("numLayers")) cfg.num_layers = vc.get_int("numLayers", cfg.num_layers);

    if (vc.contains("intermediate_size")) cfg.intermediate_dim = vc.get_int("intermediate_size", cfg.intermediate_dim);
    else if (vc.contains("intermediate_dim")) cfg.intermediate_dim = vc.get_int("intermediate_dim", cfg.intermediate_dim);
    else if (vc.contains("intermediateDim")) cfg.intermediate_dim = vc.get_int("intermediateDim", cfg.intermediate_dim);

    if (vc.contains("layer_norm_eps")) cfg.layer_norm_eps = vc.get_float("layer_norm_eps", cfg.layer_norm_eps);
    else if (vc.contains("layerNormEps")) cfg.layer_norm_eps = vc.get_float("layerNormEps", cfg.layer_norm_eps);
}

static void parseClipConfigJson(const json::Value& root,
                                brolm::clip::TextEncoderConfig& textCfg,
                                brolm::clip_image::ImageEncoderConfig& visionCfg,
                                brolm::clip_score::Config& scoreCfg) {
    if (!root.is_object()) return;
    if (root.contains("projection_dim")) scoreCfg.projection_dim = root.get_int("projection_dim", scoreCfg.projection_dim);
    else if (root.contains("projectionDim")) scoreCfg.projection_dim = root.get_int("projectionDim", scoreCfg.projection_dim);

    if (root.contains("text_config")) parseTextConfigJson(root.at("text_config"), textCfg);
    else if (root.contains("text_model")) parseTextConfigJson(root.at("text_model"), textCfg);
    else if (root.contains("text")) parseTextConfigJson(root.at("text"), textCfg);
    else parseTextConfigJson(root, textCfg);

    if (root.contains("vision_config")) parseVisionConfigJson(root.at("vision_config"), visionCfg);
    else if (root.contains("vision_model")) parseVisionConfigJson(root.at("vision_model"), visionCfg);
    else if (root.contains("vision")) parseVisionConfigJson(root.at("vision"), visionCfg);
    else parseVisionConfigJson(root, visionCfg);
}

// The first of `keys` present as a number on `obj`.
static bool numAny(Value obj, std::initializer_list<std::string_view> keys, double& out) {
    if (!ev::isObject(obj)) return false;
    ev::Persistent o(obj);
    for (std::string_view k : keys)
        if (propNumber(o.get(), k, out)) return true;
    return false;
}

static void parseTextConfigJs(Value tc, brolm::clip::TextEncoderConfig& cfg) {
    if (!ev::isObject(tc)) return;
    ev::Persistent o(tc);
    double d = 0;
    if (numAny(o.get(), {"vocabSize", "vocab_size"}, d)) cfg.vocab_size = static_cast<int>(d);
    if (numAny(o.get(), {"maxPosition", "max_position", "max_position_embeddings"}, d)) cfg.max_position = static_cast<int>(d);
    if (numAny(o.get(), {"hiddenDim", "hidden_dim", "hidden_size"}, d)) cfg.hidden_dim = static_cast<int>(d);
    if (numAny(o.get(), {"numHeads", "num_heads", "num_attention_heads"}, d)) cfg.num_heads = static_cast<int>(d);
    if (numAny(o.get(), {"numLayers", "num_layers", "num_hidden_layers"}, d)) cfg.num_layers = static_cast<int>(d);
    if (numAny(o.get(), {"intermediateDim", "intermediate_dim", "intermediate_size"}, d)) cfg.intermediate_dim = static_cast<int>(d);
    if (numAny(o.get(), {"layerNormEps", "layer_norm_eps"}, d)) cfg.layer_norm_eps = static_cast<float>(d);
    if (numAny(o.get(), {"eosTokenId", "eos_token_id"}, d)) cfg.eos_token_id = static_cast<int>(d);
}

static void parseVisionConfigJs(Value vc, brolm::clip_image::ImageEncoderConfig& cfg) {
    if (!ev::isObject(vc)) return;
    ev::Persistent o(vc);
    double d = 0;
    if (numAny(o.get(), {"imageSize", "image_size"}, d)) cfg.image_size = static_cast<int>(d);
    if (numAny(o.get(), {"patchSize", "patch_size"}, d)) cfg.patch_size = static_cast<int>(d);
    if (numAny(o.get(), {"inChannels", "in_channels", "num_channels"}, d)) cfg.in_channels = static_cast<int>(d);
    if (numAny(o.get(), {"hiddenDim", "hidden_dim", "hidden_size"}, d)) cfg.hidden_dim = static_cast<int>(d);
    if (numAny(o.get(), {"numHeads", "num_heads", "num_attention_heads"}, d)) cfg.num_heads = static_cast<int>(d);
    if (numAny(o.get(), {"numLayers", "num_layers", "num_hidden_layers"}, d)) cfg.num_layers = static_cast<int>(d);
    if (numAny(o.get(), {"intermediateDim", "intermediate_dim", "intermediate_size"}, d)) cfg.intermediate_dim = static_cast<int>(d);
    if (numAny(o.get(), {"layerNormEps", "layer_norm_eps"}, d)) cfg.layer_norm_eps = static_cast<float>(d);
}

// The first of `keys` present as an object on `root`, or undefined.
static Value objAny(Value root, std::initializer_list<std::string_view> keys) {
    ev::Persistent o(root);
    for (std::string_view k : keys) {
        Value v = ev::getProperty(o.get(), k);
        if (ev::isObject(v)) return v;
    }
    return ev::undefined();
}

static void parseClipConfigJs(Value root,
                              brolm::clip::TextEncoderConfig& textCfg,
                              brolm::clip_image::ImageEncoderConfig& visionCfg,
                              brolm::clip_score::Config& scoreCfg) {
    if (!ev::isObject(root)) return;
    ev::Persistent r(root);
    double d = 0;
    if (numAny(r.get(), {"projectionDim", "projection_dim"}, d)) scoreCfg.projection_dim = static_cast<int>(d);

    ev::Persistent tc(objAny(r.get(), {"textConfig", "text_config", "text"}));
    parseTextConfigJs(ev::isObject(tc.get()) ? tc.get() : r.get(), textCfg);

    ev::Persistent vc(objAny(r.get(), {"visionConfig", "vision_config", "vision"}));
    parseVisionConfigJs(ev::isObject(vc.get()) ? vc.get() : r.get(), visionCfg);
}

Value js_loadClip(Value, std::span<const Value> a) {
    if (a.empty() || !ev::isObject(a[0]))
        return ev::throwTypeError("loadClip(opts): opts object required");

    // a[0] is a rooted argument slot: current across every read below.
    const Value& opts = a[0];
    std::string vocab, merges;
    if (!propString(opts, "vocabPath", vocab) || !propString(opts, "mergesPath", merges))
        return ev::throwTypeError("loadClip: opts.vocabPath and opts.mergesPath required");
    vocab = resolvePath(vocab);
    merges = resolvePath(merges);

    std::string weights;
    if (propString(opts, "weightsPath", weights)) weights = resolvePath(weights);

    std::string textPath = weights, imagePath = weights, projPath = weights;
    std::string s;
    if (propString(opts, "textPath", s)) textPath = resolvePath(s);
    if (propString(opts, "imagePath", s)) imagePath = resolvePath(s);
    if (propString(opts, "projectionPath", s)) projPath = resolvePath(s);

    if (textPath.empty() || imagePath.empty() || projPath.empty())
        return ev::throwTypeError("loadClip: opts.weightsPath or opts.textPath+imagePath+projectionPath required");

    std::string textPrefix = "text_model.", visionPrefix = "vision_model.", projPrefix = "";
    propString(opts, "textPrefix", textPrefix);
    propString(opts, "visionPrefix", visionPrefix);
    propString(opts, "projectionPrefix", projPrefix);

    brotensor::init();
    brotensor::Device dev = autoDevice();
    std::string err;
    if (!parseDeviceOpt(opts, dev, err))
        return ev::throwTypeError(err);

    brolm::clip::TextEncoderConfig textCfg;
    brolm::clip_image::ImageEncoderConfig visionCfg;
    brolm::clip_score::Config scoreCfg;

    std::string configPath;
    if (propString(opts, "configPath", configPath) || propString(opts, "config", configPath))
        configPath = resolvePath(configPath);

    if (configPath.empty()) {
        std::vector<std::string> probeCandidates = { weights, textPath, vocab };
        for (const auto& p : probeCandidates) {
            if (!p.empty()) {
                std::filesystem::path parent = std::filesystem::path(p).parent_path();
                std::filesystem::path cand = parent / "config.json";
                if (std::filesystem::exists(cand)) {
                    configPath = cand.string();
                    break;
                }
            }
        }
    }

    if (!configPath.empty() && std::filesystem::exists(configPath)) {
        std::ifstream ifs(configPath);
        if (ifs.is_open()) {
            std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
            auto rootJson = brolm::detail::json::parse(content);
            parseClipConfigJson(rootJson, textCfg, visionCfg, scoreCfg);
        }
    }

    Value cVal = ev::getProperty(opts, "config");
    if (ev::isObject(cVal)) {
        parseClipConfigJs(cVal, textCfg, visionCfg, scoreCfg);
    }

    Value pdVal = ev::getProperty(opts, "projectionDim");
    if (!ev::isNumber(pdVal)) pdVal = ev::getProperty(opts, "projection_dim");
    if (ev::isNumber(pdVal)) scoreCfg.projection_dim = static_cast<int>(ev::toDouble(pdVal));

    try {
        auto w = std::make_unique<HostClipModel>();
        w->device = dev;
        brotensor::DeviceScope scope(dev);

        w->tok = std::make_unique<brolm::clip::Tokenizer>(
            brolm::clip::Tokenizer::load(vocab, merges));

        w->text = std::make_unique<brolm::clip::TextEncoder>(textCfg);
        {
            auto f = brotensor::safetensors::File::open(textPath);
            w->text->load_weights(f, textPrefix);
        }

        w->image = std::make_unique<brolm::clip_image::ImageEncoder>(visionCfg);
        {
            auto f = brotensor::safetensors::File::open(imagePath);
            w->image->load_weights(f, visionPrefix);
        }

        w->scorer = std::make_unique<brolm::clip_score::CLIPScorer>(
            *w->tok, *w->text, *w->image, scoreCfg);
        {
            auto f = brotensor::safetensors::File::open(projPath);
            w->scorer->load_projections(f, projPrefix);
        }

        return makeClipModelValue(std::move(w));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("loadClip: ") + e.what());
    }
}

Value js_loadNllb(Value, std::span<const Value> a) {
    if (a.empty() || !ev::isString(a[0]))
        return ev::throwTypeError("loadNllb(checkpointDir, opts?): dir string required");
    const std::string dir = resolvePath(ev::toUtf8(a[0]));

    brotensor::init();
    brotensor::Device dev = autoDevice();
    if (a.size() >= 2 && ev::isObject(a[1])) {
        std::string err;
        if (!parseDeviceOpt(a[1], dev, err))
            return ev::throwTypeError(err);
    }

    struct Built { std::unique_ptr<brolm::nllb::Translator> tr; };
    return runLoad<Built>(a, 1, "loadNllb",
        [dir, dev](Built& b) {
            brotensor::DeviceScope scope(dev);
            b.tr = std::make_unique<brolm::nllb::Translator>(brolm::nllb::Translator::load(dir));
        },
        [dev](Built& b) { return makeNllbModelValue(std::move(b.tr), dev); });
}

Value js_loadT5(Value, std::span<const Value> a) {
    if (a.empty() || !ev::isObject(a[0]))
        return ev::throwTypeError("loadT5(opts): opts object required");

    // a[0] is a rooted argument slot: current across every read below.
    const Value& opts = a[0];
    std::string tokPath;
    if (!propString(opts, "tokenizerPath", tokPath))
        return ev::throwTypeError("loadT5: opts.tokenizerPath (tokenizer.json) required");
    tokPath = resolvePath(tokPath);

    std::string gguf, weights;
    if (propString(opts, "ggufPath", gguf)) gguf = resolvePath(gguf);
    if (propString(opts, "weightsPath", weights)) weights = resolvePath(weights);

    std::vector<std::string> shards = readStringArray(ev::getProperty(opts, "shards"));
    for (auto& s : shards) s = resolvePath(s);

    if (gguf.empty() && weights.empty() && shards.empty())
        return ev::throwTypeError("loadT5: opts.ggufPath, opts.weightsPath, or opts.shards required");

    std::string prefix = "";
    propString(opts, "prefix", prefix);

    int maxLength = 512;
    double d = 0;
    if (propNumber(opts, "maxLength", d)) maxLength = static_cast<int>(d);
    if (maxLength <= 0)
        return ev::throwTypeError("loadT5: opts.maxLength must be > 0");
    // INT8 (W8A16) attention/FFN weights; honoured on a GPU backend.
    bool quantize = false;
    propBool(opts, "quantizeWeights", quantize);

    // Architecture overrides for safetensors/shard loads (a GGUF carries its own).
    brolm::t5::T5Config cfgOverride;
    {
        ev::Persistent cv(ev::getProperty(opts, "config"));
        if (ev::isObject(cv.get())) {
            if (propNumber(cv.get(), "vocabSize", d)) cfgOverride.vocab_size = static_cast<int>(d);
            if (propNumber(cv.get(), "dModel", d))    cfgOverride.d_model = static_cast<int>(d);
            if (propNumber(cv.get(), "dFf", d))       cfgOverride.d_ff = static_cast<int>(d);
            if (propNumber(cv.get(), "dKv", d))       cfgOverride.d_kv = static_cast<int>(d);
            if (propNumber(cv.get(), "numHeads", d))  cfgOverride.num_heads = static_cast<int>(d);
            if (propNumber(cv.get(), "numLayers", d)) cfgOverride.num_layers = static_cast<int>(d);
        }
    }

    brotensor::init();
    brotensor::Device dev = autoDevice();
    std::string err;
    if (!parseDeviceOpt(opts, dev, err))
        return ev::throwTypeError(err);

    try {
        auto w = std::make_unique<HostT5Model>();
        w->device = dev;
        w->maxLength = maxLength;
        brotensor::DeviceScope scope(dev);

        w->tok = std::make_unique<brolm::t5::Tokenizer>(
            brolm::t5::Tokenizer::load(tokPath));

        if (!gguf.empty()) {
            brotensor::gguf::File f = brotensor::gguf::File::open(gguf);
            brolm::t5::T5Config cfg = brolm::t5::T5Config::from_gguf(f);
            cfg.quantize_weights = quantize;
            w->enc = std::make_unique<brolm::t5::TextEncoder>(cfg);
            w->enc->load_weights(f);
            w->dModel = cfg.d_model;
        } else {
            brolm::t5::T5Config cfg = cfgOverride;
            cfg.quantize_weights = quantize;
            w->enc = std::make_unique<brolm::t5::TextEncoder>(cfg);

            if (!shards.empty()) {
                std::vector<brotensor::safetensors::File> files;
                files.reserve(shards.size());
                for (const auto& p : shards)
                    files.push_back(brotensor::safetensors::File::open(p));
                std::vector<const brotensor::safetensors::File*> ptrs;
                ptrs.reserve(files.size());
                for (const auto& f : files) ptrs.push_back(&f);
                w->enc->load_weights(ptrs, prefix);
            } else {
                auto f = brotensor::safetensors::File::open(weights);
                w->enc->load_weights(f, prefix);
            }
            w->dModel = cfg.d_model;
        }

        return makeT5ModelValue(std::move(w));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("loadT5: ") + e.what());
    }
}

} // namespace brolm::api
