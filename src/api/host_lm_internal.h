#pragma once

#include "api.h"
#include "host_class.h"
#include "object_builder.h"
#include "arg_reader.h"
#include "embed/embed.h"

#include <brotensor/tensor.h>
#include <brotensor/runtime.h>
#include <brotensor/gguf.h>
#include <brotensor/safetensors.h>

#include <brolm/qwen.h>
#include <brolm/qwen_generate.h>
#include <brolm/qwen_tokenizer.h>
#include <brolm/mistral_tokenizer.h>
#include <brolm/mistral3_text.h>
#include <brolm/mistral3_config.h>
#include <brolm/gemma_tokenizer.h>
#include <brolm/gemma2.h>
#include <brolm/gemma2_config.h>
#include <brolm/llama3_tokenizer.h>
#include <brolm/qwen35_vl.h>
#include <brolm/qwen3vl_vl.h>
#include <brolm/clip.h>
#include <brolm/clip_image.h>
#include <brolm/clip_score.h>
#include <brolm/nllb.h>
#include <brolm/t5.h>
#include <brolm/tokenizer_t5.h>
#include <brolm/sampler.h>
#include <brolm/laya.h>
#include <brolm/laya_scheduler.h>
#include <brolm/detail/generate.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace brolm::api {

// ═══════════════════════════════════════════════════════════════════════════
// Handles and Wrappers
// ═══════════════════════════════════════════════════════════════════════════

struct HostAsyncHandle {
    std::atomic<bool> cancelled{false};
    std::atomic<bool> finished{false};
    std::atomic<uint64_t> event_seq{0};
    std::mutex cvMutex;
    std::condition_variable cv;
    std::thread worker;

    void notify() {
        event_seq.fetch_add(1, std::memory_order_release);
        cv.notify_all();
    }

    ~HostAsyncHandle() {
        cancelled.store(true, std::memory_order_release);
        notify();
        if (worker.joinable()) {
            worker.join();
        }
    }
};

// Single-owner claim on a model's `generating`/`translating` flag: a decode
// drives the model's KV-cache in place, so a second overlapping call (a sync
// call while a bro.lm.generate job runs on a worker, or two jobs) would
// interleave writes into it. The claim is taken on the JS thread; an async
// job carries the flag and releases it on the JS thread before its onDone.
class BusyClaim {
public:
    explicit BusyClaim(std::atomic<bool>& f) {
        bool expected = false;
        if (f.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) flag_ = &f;
    }
    BusyClaim(const BusyClaim&) = delete;
    BusyClaim& operator=(const BusyClaim&) = delete;
    ~BusyClaim() { release(); }
    bool ok() const { return flag_ != nullptr; }
    // Hand the claim to someone else (an async job) without releasing it.
    std::atomic<bool>* detach() { auto* f = flag_; flag_ = nullptr; return f; }
    void release() {
        if (flag_) flag_->store(false, std::memory_order_release);
        flag_ = nullptr;
    }
private:
    std::atomic<bool>* flag_ = nullptr;
};

inline const char* kBusyMessage = "a generation is already in flight on this model";

// Tokenizers are shared with the model they were loaded beside (the model's
// grammar mask needs each token's text), so the handle holds a shared_ptr.
struct HostQwenTokenizer {
    std::shared_ptr<brolm::qwen::Tokenizer> tok;
};

struct HostMistralTokenizer {
    std::shared_ptr<brolm::mistral::Tokenizer> tok;
};

struct HostGemmaTokenizer {
    std::shared_ptr<brolm::gemma::Tokenizer> tok;
};

struct HostLlama3Tokenizer {
    std::shared_ptr<brolm::llama3::Tokenizer> tok;
};

struct LMDecoder {
    virtual ~LMDecoder() = default;
    virtual const char* family()    const = 0;
    virtual int  vocabSize()  const = 0;
    virtual int  hiddenSize() const = 0;
    virtual int  numLayers()  const = 0;
    virtual int  maxSeqLen()  const = 0;
    virtual int  cacheLen()   const = 0;
    virtual void allocateCache(int n) = 0;
    virtual void resetCache() = 0;
    virtual void forward(const int32_t* ids, int L, brotensor::Tensor& out) = 0;
    // Logits of the LAST position only: what the decode loop samples from.
    virtual void forwardLast(const int32_t* ids, int L, brotensor::Tensor& out) = 0;
};

struct QwenDecoder final : LMDecoder {
    brolm::qwen::Qwen3Model m;
    explicit QwenDecoder(const brolm::qwen::Qwen3Config& cfg) : m(cfg) {}
    const char* family()    const override { return "qwen3"; }
    int  vocabSize()  const override { return m.config().vocab_size; }
    int  hiddenSize() const override { return m.config().hidden_size; }
    int  numLayers()  const override { return m.config().num_hidden_layers; }
    int  maxSeqLen()  const override { return m.config().max_position_embeddings; }
    int  cacheLen()   const override { return m.cache_len(); }
    void allocateCache(int n) override { m.allocate_cache(n); }
    void resetCache() override { m.reset_cache(); }
    void forward(const int32_t* ids, int L, brotensor::Tensor& out) override {
        m.forward(ids, L, out);
    }
    void forwardLast(const int32_t* ids, int L, brotensor::Tensor& out) override {
        m.forward_last(ids, L, out);
    }
};

struct MistralDecoder final : LMDecoder {
    brolm::mistral3::TextModel m;
    explicit MistralDecoder(const brolm::mistral3::Mistral3Config::Text& cfg) : m(cfg) {}
    const char* family()    const override { return "mistral3"; }
    int  vocabSize()  const override { return m.config().vocab_size; }
    int  hiddenSize() const override { return m.config().hidden_size; }
    int  numLayers()  const override { return m.config().num_hidden_layers; }
    int  maxSeqLen()  const override { return m.config().max_position_embeddings; }
    int  cacheLen()   const override { return m.cache_len(); }
    void allocateCache(int n) override { m.allocate_cache(n); }
    void resetCache() override { m.reset_cache(); }
    void forward(const int32_t* ids, int L, brotensor::Tensor& out) override {
        m.forward(ids, L, out);
    }
    void forwardLast(const int32_t* ids, int L, brotensor::Tensor& out) override {
        m.forward_last(ids, L, out);
    }
};

struct GemmaDecoder final : LMDecoder {
    brolm::gemma::Gemma2Model m;
    explicit GemmaDecoder(const brolm::gemma::Gemma2Config& cfg) : m(cfg) {}
    const char* family()    const override { return "gemma2"; }
    int  vocabSize()  const override { return m.config().vocab_size; }
    int  hiddenSize() const override { return m.config().hidden_size; }
    int  numLayers()  const override { return m.config().num_hidden_layers; }
    int  maxSeqLen()  const override { return m.config().max_position_embeddings; }
    int  cacheLen()   const override { return m.cache_len(); }
    void allocateCache(int n) override { m.allocate_cache(n); }
    void resetCache() override { m.reset_cache(); }
    void forward(const int32_t* ids, int L, brotensor::Tensor& out) override {
        m.forward(ids, L, out);
    }
    void forwardLast(const int32_t* ids, int L, brotensor::Tensor& out) override {
        m.forward_last(ids, L, out);
    }
};

struct HostLMModel {
    std::unique_ptr<LMDecoder> model;
    bool weights_loaded = false;
    brotensor::Device device = brotensor::Device::CPU;
    std::atomic<bool> generating{false};

    // The paired tokenizer's end-of-turn id: the eosId generate() stops on
    // when opts.eosId is absent (-1 when the model came without one).
    int defaultEos = -1;
    // id -> token text, from the paired tokenizer; null when there is none.
    // Used to build `vocabPieces` for grammar-constrained decoding.
    std::function<std::string(int32_t)> tokenText;
    std::once_flag vocabOnce;
    std::vector<std::string> vocabPieces;  // built on first grammar use

    const std::vector<std::string>& pieces() {
        std::call_once(vocabOnce, [this] {
            if (!tokenText || !model) return;
            const int n = model->vocabSize();
            vocabPieces.resize(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) vocabPieces[static_cast<size_t>(i)] = tokenText(i);
        });
        return vocabPieces;
    }
};

struct HostQwen35Model {
    std::unique_ptr<brolm::qwen35::VLM> vlm;
    int maxSeqLen = 4096;
    brotensor::Device device = brotensor::Device::CPU;
    std::atomic<bool> generating{false};
};

struct HostQwen3VLModel {
    std::unique_ptr<brolm::qwen3vl::VLM> vlm;
    int maxSeqLen = 4096;
    brotensor::Device device = brotensor::Device::CPU;
    std::atomic<bool> generating{false};
};

struct HostClipModel {
    std::unique_ptr<brolm::clip::Tokenizer>          tok;
    std::unique_ptr<brolm::clip::TextEncoder>        text;
    std::unique_ptr<brolm::clip_image::ImageEncoder> image;
    std::unique_ptr<brolm::clip_score::CLIPScorer>   scorer;
    brotensor::Device device = brotensor::Device::CPU;
};

struct HostNllbModel {
    std::unique_ptr<brolm::nllb::Translator> tr;
    brotensor::Device device = brotensor::Device::CPU;
    std::atomic<bool> translating{false};
};

struct HostT5Model {
    std::unique_ptr<brolm::t5::Tokenizer>   tok;
    std::unique_ptr<brolm::t5::TextEncoder> enc;
    int               maxLength = 512;
    int               dModel    = 4096;
    brotensor::Device device    = brotensor::Device::CPU;
};

// ═══════════════════════════════════════════════════════════════════════════
// HostClass declarations
// ═══════════════════════════════════════════════════════════════════════════

extern HostClass g_asyncHandleClass;
extern HostClass g_qwenTokenizerClass;
extern HostClass g_mistralTokenizerClass;
extern HostClass g_gemmaTokenizerClass;
extern HostClass g_llama3TokenizerClass;
extern HostClass g_lmModelClass;
extern HostClass g_qwen35ModelClass;
extern HostClass g_qwen3VLModelClass;
extern HostClass g_clipModelClass;
extern HostClass g_nllbModelClass;
extern HostClass g_t5ModelClass;
extern HostClass g_layaModelClass;
extern HostClass g_grammarClass;
extern HostClass g_modernBertModelClass;

// ═══════════════════════════════════════════════════════════════════════════
// Conversions and Helpers
// ═══════════════════════════════════════════════════════════════════════════

inline brotensor::Device autoDevice() {
    if (brotensor::is_available(brotensor::Device::CUDA))  return brotensor::Device::CUDA;
    if (brotensor::is_available(brotensor::Device::Metal)) return brotensor::Device::Metal;
    return brotensor::Device::CPU;
}

inline const char* deviceName(brotensor::Device d) {
    switch (d.type) {
        case brotensor::DeviceType::CUDA:  return "CUDA";
        case brotensor::DeviceType::Metal: return "Metal";
        case brotensor::DeviceType::CPU:   return "CPU";
    }
    return "?";
}

inline bool parseDeviceOpt(Value opts, brotensor::Device& out, std::string& err) {
    if (!ev::isObject(opts)) return true;
    Value v = ev::getProperty(opts, "device");
    if (ev::isUndefined(v) || ev::isNull(v)) return true;
    if (!ev::isString(v)) {
        err = "opts.device must be a string ('cpu', 'cuda', or 'metal')";
        return false;
    }
    std::string sv = ev::toUtf8(v);
    if (sv == "cpu")   { out = brotensor::Device::CPU;   return true; }
    if (sv == "cuda")  { out = brotensor::Device::CUDA;  return true; }
    if (sv == "metal") { out = brotensor::Device::Metal; return true; }
    err = "opts.device must be 'cpu', 'cuda', or 'metal' (got '" + sv + "')";
    return false;
}

// ── Rooted property reads ────────────────────────────────────────────────
// getProperty allocates (it interns the key) and may run a getter, so under
// the embed.h GC contract every Value the caller holds is stale after it.
// These read ONE property of `obj` (which must be current at the call) and
// consume it before returning; a caller reading several properties of an
// object that is not an args[] slot holds it in an ev::Persistent and passes
// .get() each time.
inline bool propNumber(Value obj, std::string_view key, double& out) {
    if (!ev::isObject(obj)) return false;
    Value v = ev::getProperty(obj, key);
    if (!ev::isNumber(v)) return false;
    out = ev::toDouble(v);
    return true;
}

inline bool propString(Value obj, std::string_view key, std::string& out) {
    if (!ev::isObject(obj)) return false;
    Value v = ev::getProperty(obj, key);
    if (!ev::isString(v)) return false;
    out = ev::toUtf8(v);
    return true;
}

// A present (non-undefined, non-null) property's truthiness.
inline bool propBool(Value obj, std::string_view key, bool& out) {
    if (!ev::isObject(obj)) return false;
    Value v = ev::getProperty(obj, key);
    if (ev::isUndefined(v) || ev::isNull(v)) return false;
    out = ev::toBool(v);
    return true;
}

inline bool propFloat(Value obj, std::string_view camel, std::string_view snake, float& out) {
    double d = 0;
    if (propNumber(obj, camel, d) || propNumber(obj, snake, d)) {
        out = static_cast<float>(d);
        return true;
    }
    return false;
}

// Strings of an array-like (a plain array of strings), non-strings skipped.
inline std::vector<std::string> readStringArray(Value arr) {
    std::vector<std::string> out;
    if (!ev::isObject(arr)) return out;
    ev::Persistent a(arr);
    double len = 0;
    if (!propNumber(a.get(), "length", len)) return out;
    const uint32_t n = static_cast<uint32_t>(len);
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        Value e = ev::getElement(a.get(), i);
        if (ev::isString(e)) out.push_back(ev::toUtf8(e));
    }
    return out;
}

const brolm::Grammar* hostGrammarOf(Value v);  // native_lm_grammar.cpp

inline brolm::qwen::GenerateOptions parseGenerateOptions(Value v) {
    brolm::qwen::GenerateOptions o;
    if (!ev::isObject(v)) return o;
    ev::Persistent opts(v);
    double d = 0;
    if (propNumber(opts.get(), "maxNewTokens", d)) o.max_new_tokens = static_cast<int>(d);
    bool b = false;
    if (propBool(opts.get(), "stopOnEos", b)) o.stop_on_eos = b;

    ev::Persistent s(ev::getProperty(opts.get(), "sampling"));
    if (ev::isObject(s.get())) {
        if (propNumber(s.get(), "temperature", d)) o.sampling.temperature = static_cast<float>(d);
        if (propNumber(s.get(), "topK", d)) o.sampling.top_k = static_cast<int>(d);
        if (propNumber(s.get(), "topP", d)) o.sampling.top_p = static_cast<float>(d);
        Value seed = ev::getProperty(s.get(), "seed");
        if (ev::isNumber(seed)) o.sampling.seed = static_cast<uint64_t>(ev::toDouble(seed));
        else if (ev::isBigInt(seed)) o.sampling.seed = ev::toUint64(seed);
    }

    // The penalty knobs are read from opts.sampling first, then from opts.
    auto samplingFloat = [&](const char* camel, const char* snake, float& target) {
        if (propFloat(s.get(), camel, snake, target)) return;
        propFloat(opts.get(), camel, snake, target);
    };
    samplingFloat("minP", "min_p", o.sampling.min_p);
    samplingFloat("repetitionPenalty", "repetition_penalty", o.sampling.repetition_penalty);
    samplingFloat("frequencyPenalty", "frequency_penalty", o.sampling.frequency_penalty);
    samplingFloat("presencePenalty", "presence_penalty", o.sampling.presence_penalty);
    samplingFloat("dryMultiplier", "dry_multiplier", o.sampling.dry_multiplier);
    samplingFloat("dryBase", "dry_base", o.sampling.dry_base);
    float f = 0;
    if (propFloat(s.get(), "penaltyLastN", "penalty_last_n", f) ||
        propFloat(opts.get(), "penaltyLastN", "penalty_last_n", f))
        o.sampling.penalty_last_n = static_cast<int>(f);
    if (propFloat(s.get(), "dryAllowedLength", "dry_allowed_length", f) ||
        propFloat(opts.get(), "dryAllowedLength", "dry_allowed_length", f))
        o.sampling.dry_allowed_length = static_cast<int>(f);

    // opts.grammar: a bro.lm.Grammar. The caller keeps the grammar object
    // alive for the call (a sync call holds it in args; bro.lm.generate roots
    // it in the job), and generation clones its state, so the template is
    // never advanced.
    o.grammar = hostGrammarOf(ev::getProperty(opts.get(), "grammar"));

    return o;
}

inline std::vector<int32_t> readInt32Array(Value v) {
    std::vector<int32_t> out;
    if (!ev::isObject(v)) return out;
    auto tinfo = ev::typedArrayInfo(v);
    if (tinfo.data && tinfo.bytesPerElement == 4) {
        const int32_t* src = reinterpret_cast<const int32_t*>(tinfo.data);
        out.assign(src, src + tinfo.elementCount);
        return out;
    }
    if (tinfo.data) {  // another typed array: convert element by element
        ev::Persistent arr(v);
        out.reserve(tinfo.elementCount);
        for (uint32_t i = 0; i < tinfo.elementCount; ++i) {
            Value elem = ev::getElement(arr.get(), i);
            out.push_back(ev::isNumber(elem) ? static_cast<int32_t>(ev::toDouble(elem)) : 0);
        }
        return out;
    }
    ev::Persistent arr(v);
    double len = 0;
    if (propNumber(arr.get(), "length", len)) {
        const uint32_t n = static_cast<uint32_t>(len);
        out.reserve(n);
        for (uint32_t i = 0; i < n; ++i) {
            Value elem = ev::getElement(arr.get(), i);
            out.push_back(ev::isNumber(elem) ? static_cast<int32_t>(ev::toDouble(elem)) : 0);
        }
    }
    return out;
}

inline Value makeInt32Array(const int32_t* data, size_t count) {
    ev::Persistent view(ev::createTypedArray(ev::elements::Int32, static_cast<uint32_t>(count)));
    if (!ev::isObject(view.get())) return ev::undefined();
    if (data && count > 0) {
        ev::fillTypedArray(view.get(), std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(data), count * sizeof(int32_t)));
    }
    return view.get();
}

inline Value makeFloat32Array(const float* data, size_t count) {
    ev::Persistent view(ev::createTypedArray(ev::elements::Float32, static_cast<uint32_t>(count)));
    if (!ev::isObject(view.get())) return ev::undefined();
    if (data && count > 0) {
        ev::fillTypedArray(view.get(), std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(data), count * sizeof(float)));
    }
    return view.get();
}

inline std::vector<float> downloadFloats(const brotensor::Tensor& t) {
    if (t.dtype == brotensor::Dtype::FP16) {
        std::vector<uint16_t> bits = t.to_host_vector_fp16();
        std::vector<float> out(bits.size());
        for (size_t i = 0; i < bits.size(); ++i)
            out[i] = brotensor::fp16_bits_to_fp32(bits[i]);
        return out;
    }
    return t.to_host_vector();
}

inline std::vector<float> lastRowFp32(const brotensor::Tensor& logits) {
    return brolm::detail::last_row_fp32(logits);
}

inline std::vector<std::pair<std::string, std::string>> readChatMessages(Value v) {
    std::vector<std::pair<std::string, std::string>> out;
    if (!ev::isObject(v)) return out;
    ev::Persistent arr(v);
    double len = 0;
    if (!propNumber(arr.get(), "length", len)) return out;
    const uint32_t n = static_cast<uint32_t>(len);
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        ev::Persistent elem(ev::getElement(arr.get(), i));
        std::string role, content;
        propString(elem.get(), "role", role);
        propString(elem.get(), "content", content);
        out.emplace_back(std::move(role), std::move(content));
    }
    return out;
}

inline bool readImageArg(Value val, std::vector<uint8_t>& rgba, int& w, int& h, std::string& err) {
    if (!ev::isObject(val)) {
        err = "image must be an object with { data, width, height }";
        return false;
    }
    ev::Persistent img(val);
    double wd = 0, hd = 0;
    if (!propNumber(img.get(), "width", wd) || !propNumber(img.get(), "height", hd)) {
        err = "image { width, height } must be numbers";
        return false;
    }
    w = static_cast<int>(wd);
    h = static_cast<int>(hd);
    if (w <= 0 || h <= 0) {
        err = "image { width, height } must be positive";
        return false;
    }
    // The data pointer is consumed (copied) before any further allocation.
    Value dataVal = ev::getProperty(img.get(), "data");
    auto tinfo = ev::typedArrayInfo(dataVal);
    const size_t need = static_cast<size_t>(w) * h * 4;
    if (tinfo.data && tinfo.byteLength >= need) {
        rgba.assign(tinfo.data, tinfo.data + need);
        return true;
    }
    auto abinfo = ev::arrayBufferInfo(dataVal);
    if (abinfo.data && abinfo.byteLength >= need) {
        rgba.assign(abinfo.data, abinfo.data + need);
        return true;
    }
    err = "image.data must be a Uint8Array/Uint8ClampedArray/ArrayBuffer of size at least width*height*4";
    return false;
}

inline std::vector<float> rgbaToNchwSigned(const std::vector<uint8_t>& rgba, int w, int h) {
    const int plane = w * h;
    std::vector<float> out(static_cast<size_t>(3) * plane);
    for (int i = 0; i < plane; ++i)
        for (int c = 0; c < 3; ++c)
            out[static_cast<size_t>(c) * plane + i] =
                rgba[static_cast<size_t>(4) * i + c] / 255.0f * 2.0f - 1.0f;
    return out;
}

inline std::vector<float> rgbaToChwUnit(const std::vector<uint8_t>& rgba, int w, int h) {
    const int plane = w * h;
    std::vector<float> out(static_cast<size_t>(3) * plane);
    for (int i = 0; i < plane; ++i)
        for (int c = 0; c < 3; ++c)
            out[static_cast<size_t>(c) * plane + i] =
                rgba[static_cast<size_t>(4) * i + c] / 255.0f;
    return out;
}

struct VlmImage {
    std::vector<float> chw;
    int H = 0;
    int W = 0;
};

inline bool readVlmImages(Value opts, std::vector<VlmImage>& images, std::string& err) {
    if (!ev::isObject(opts)) return true;
    ev::Persistent imgs(ev::getProperty(opts, "images"));
    if (ev::isUndefined(imgs.get()) || ev::isNull(imgs.get())) return true;

    auto parseOne = [&](Value v) -> bool {
        std::vector<uint8_t> rgba;
        int w = 0, h = 0;
        if (!readImageArg(v, rgba, w, h, err)) return false;
        images.push_back({ rgbaToChwUnit(rgba, w, h), h, w });
        return true;
    };

    // An array of images (an image itself has no numeric `length`).
    double count = 0;
    if (propNumber(imgs.get(), "length", count)) {
        for (uint32_t i = 0; i < static_cast<uint32_t>(count); ++i) {
            if (!parseOne(ev::getElement(imgs.get(), i))) return false;
        }
        return true;
    }
    return parseOne(imgs.get());
}

// ═══════════════════════════════════════════════════════════════════════════
// Function & Object Creators / Unwrappers
// ═══════════════════════════════════════════════════════════════════════════

HostAsyncHandle* hostAsyncHandleOf(Value v);
Value makeAsyncHandleValue(std::shared_ptr<HostAsyncHandle> h);

// ── Background jobs (native_lm_model.cpp) ───────────────────────────────────
// A job's work runs on its own thread and touches no JS value; everything it
// shares with the JS thread lives in AsyncCore. The AsyncJob itself (the JS
// callbacks, the roots that keep a model alive, the finish step) is owned by
// the launching JS thread and ticked there by tickLMAsync / AsyncHandle.wait.
struct AsyncCore {
    std::shared_ptr<HostAsyncHandle> handle;
    std::mutex mu;
    std::vector<int32_t> pendingTokens;  // produced, not yet delivered to onToken
    std::string error;                   // set by the worker on a throw
    std::atomic<bool> done{false};

    bool cancelled() const { return handle->cancelled.load(std::memory_order_acquire); }
    void pushToken(int32_t id) {
        { std::lock_guard<std::mutex> lk(mu); pendingTokens.push_back(id); }
        handle->notify();
    }
};

struct AsyncJob {
    std::shared_ptr<AsyncCore> core;
    std::thread worker;
    ev::Persistent onToken, onDone, onError;
    ev::Persistent keep;                // the model (or other object) the work borrows
    std::atomic<bool>* busy = nullptr;  // a BusyClaim handed over, released before finish
    // JS thread, once, after the worker joined: deliver the outcome.
    std::function<void(AsyncJob&)> finish;
    bool ticking = false;  // being delivered by an outer tick (a callback re-entered)

    ~AsyncJob() {
        if (core) core->handle->cancelled.store(true, std::memory_order_release);
        if (worker.joinable()) worker.join();
        if (busy) busy->store(false, std::memory_order_release);
    }
};

std::shared_ptr<AsyncJob> newAsyncJob();
// Start `work` on the job's thread and return its AsyncHandle.
Value launchAsyncJob(std::shared_ptr<AsyncJob> job, std::function<void(AsyncCore&)> work);
// Call a JS callback held in a Persistent with already-rooted arguments.
void callRooted(const ev::Persistent& fn, std::initializer_list<const ev::Persistent*> args);
// Cancel and join this thread's jobs (shutdownLM).
void shutdownLMJobs();

// Report a background failure: opts.onError(message) when given, else stderr
// (a failure is never silent).
void reportJobError(AsyncJob& job, const std::string& message);

// A loader's shared shape. Synchronous unless opts.onReady is a function:
// then `build` runs on a worker thread, the call returns an AsyncHandle, and
// on a later tick onReady(wrap(built)) fires on this thread, or
// onError(message) on a failure (nothing after cancel()). `build` touches no
// JS value; `wrap` runs on the JS thread and makes the handles.
template <class Built>
Value runLoad(std::span<const Value> a, size_t optsIdx, const char* what,
              std::function<void(Built&)> build, std::function<Value(Built&)> wrap) {
    ev::Persistent onReady, onError;
    if (a.size() > optsIdx && ev::isObject(a[optsIdx])) {
        onReady.set(ev::getProperty(a[optsIdx], "onReady"));
        onError.set(ev::getProperty(a[optsIdx], "onError"));
    }
    if (!ev::isFunction(onReady.get())) {
        try {
            Built b;
            build(b);
            return wrap(b);
        } catch (const std::exception& e) {
            return ev::throwError(std::string(what) + ": " + e.what());
        }
    }
    auto job = newAsyncJob();
    job->onDone.set(onReady.get());
    if (ev::isFunction(onError.get())) job->onError.set(onError.get());
    auto built = std::make_shared<Built>();
    std::string label = what;
    job->finish = [built, wrap, label](AsyncJob& j) {
        if (j.core->cancelled()) return;
        if (!j.core->error.empty()) {
            reportJobError(j, label + ": " + j.core->error);
            return;
        }
        ev::Persistent result(wrap(*built));
        callRooted(j.onDone, {&result});
    };
    return launchAsyncJob(job, [built, build](AsyncCore&) { build(*built); });
}

HostQwenTokenizer* hostQwenTokenizerOf(Value v);
Value makeQwenTokenizerValue(std::shared_ptr<brolm::qwen::Tokenizer> tok);

HostMistralTokenizer* hostMistralTokenizerOf(Value v);
Value makeMistralTokenizerValue(std::shared_ptr<brolm::mistral::Tokenizer> tok);

HostGemmaTokenizer* hostGemmaTokenizerOf(Value v);
Value makeGemmaTokenizerValue(std::shared_ptr<brolm::gemma::Tokenizer> tok);

HostLlama3Tokenizer* hostLlama3TokenizerOf(Value v);
Value makeLlama3TokenizerValue(std::shared_ptr<brolm::llama3::Tokenizer> tok);

HostLMModel* hostLMModelOf(Value v);

HostQwen35Model* hostQwen35ModelOf(Value v);
Value makeQwen35ModelValue(std::unique_ptr<brolm::qwen35::VLM> vlm, int maxSeqLen, brotensor::Device dev);

HostQwen3VLModel* hostQwen3VLModelOf(Value v);
Value makeQwen3VLModelValue(std::unique_ptr<brolm::qwen3vl::VLM> vlm, int maxSeqLen, brotensor::Device dev);

HostClipModel* hostClipModelOf(Value v);
Value makeClipModelValue(std::unique_ptr<HostClipModel> clip);

HostNllbModel* hostNllbModelOf(Value v);
Value makeNllbModelValue(std::unique_ptr<brolm::nllb::Translator> tr, brotensor::Device dev);

HostT5Model* hostT5ModelOf(Value v);
Value makeT5ModelValue(std::unique_ptr<HostT5Model> t5);

// A LayaModel handle: the request scheduler over one replica per device.
Value makeLayaModelValue(std::shared_ptr<brolm::laya::Scheduler> sched);

// Laya promise delivery (native_lm_laya_async.cpp). layaTrackPromise roots
// `promise` on the calling JS thread and returns a post handle any thread
// may call with a settle function; tickLayaAsync (from tickLMAsync) runs the
// queued settles on that JS thread with their promise. `keepalive` lives
// until the promise settles.
struct LayaMailbox;
struct LayaPost {
    std::shared_ptr<LayaMailbox> box;
    uint64_t id = 0;
    void operator()(std::function<void(const ev::Persistent& promise)> settle) const;
};
LayaPost layaTrackPromise(Value promise, std::shared_ptr<void> keepalive);
void layaTrackLoad(Value promise, std::shared_ptr<brolm::laya::Scheduler> sched);
void layaRegisterScheduler(const std::shared_ptr<brolm::laya::Scheduler>& sched);
Value layaRejectWith(Value promise, const std::string& message);
bool tickLayaAsync();
void shutdownLaya();

// Path resolution for file loaders (api.h setPathResolver).
void setPathResolver(std::function<std::string(const std::string&)> resolver);
std::string resolvePath(const std::string& path);

// ═══════════════════════════════════════════════════════════════════════════
// Registration Functions
// ═══════════════════════════════════════════════════════════════════════════

void registerLMTokenizerClasses();
void registerLMModelClasses();
void registerLMVLClasses();
void registerLMClipClasses();
void registerLMLayaClasses();
void registerLMGrammarClass();
void registerLMModernBertClass();

// ═══════════════════════════════════════════════════════════════════════════
// bro.lm Namespace Bindings
// ═══════════════════════════════════════════════════════════════════════════

Value js_init(Value, std::span<const Value>);
Value js_loadQwen(Value, std::span<const Value>);
Value js_loadMistral(Value, std::span<const Value>);
Value js_loadGemma2(Value, std::span<const Value>);
Value js_loadQwen35(Value, std::span<const Value>);
Value js_loadQwen3VL(Value, std::span<const Value>);
Value js_loadNllb(Value, std::span<const Value>);
Value js_loadTokenizer(Value, std::span<const Value>);
Value js_loadLlama3Tokenizer(Value, std::span<const Value>);
Value js_loadClip(Value, std::span<const Value>);
Value js_loadT5(Value, std::span<const Value>);
Value js_loadModernBert(Value, std::span<const Value>);
Value js_loadLaya(Value, std::span<const Value>);
Value js_loadLayaAsync(Value, std::span<const Value>);
Value js_lm_generate(Value, std::span<const Value>);
Value js_lm_tick(Value, std::span<const Value>);

} // namespace brolm::api
