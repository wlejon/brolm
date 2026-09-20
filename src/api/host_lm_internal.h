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
        if (worker.joinable()) {
            worker.join();
        }
    }
};

struct HostQwenTokenizer {
    std::unique_ptr<brolm::qwen::Tokenizer> tok;
};

struct HostMistralTokenizer {
    std::unique_ptr<brolm::mistral::Tokenizer> tok;
};

struct HostGemmaTokenizer {
    std::unique_ptr<brolm::gemma::Tokenizer> tok;
};

struct HostLlama3Tokenizer {
    std::unique_ptr<brolm::llama3::Tokenizer> tok;
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
};

struct HostLMModel {
    std::unique_ptr<LMDecoder> model;
    bool weights_loaded = false;
    brotensor::Device device = brotensor::Device::CPU;
    std::atomic<bool> generating{false};
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

inline brolm::qwen::GenerateOptions parseGenerateOptions(Value v) {
    brolm::qwen::GenerateOptions o;
    if (!ev::isObject(v)) return o;
    Value maxTokens = ev::getProperty(v, "maxNewTokens");
    if (ev::isNumber(maxTokens)) {
        o.max_new_tokens = static_cast<int>(ev::toDouble(maxTokens));
    }
    Value stopOnEos = ev::getProperty(v, "stopOnEos");
    if (!ev::isUndefined(stopOnEos) && !ev::isNull(stopOnEos)) {
        o.stop_on_eos = ev::toBool(stopOnEos);
    }
    Value s = ev::getProperty(v, "sampling");
    if (ev::isObject(s)) {
        Value temp = ev::getProperty(s, "temperature");
        if (ev::isNumber(temp)) o.sampling.temperature = static_cast<float>(ev::toDouble(temp));
        Value topK = ev::getProperty(s, "topK");
        if (ev::isNumber(topK)) o.sampling.top_k = static_cast<int>(ev::toDouble(topK));
        Value topP = ev::getProperty(s, "topP");
        if (ev::isNumber(topP)) o.sampling.top_p = static_cast<float>(ev::toDouble(topP));
        Value seed = ev::getProperty(s, "seed");
        if (ev::isNumber(seed)) o.sampling.seed = static_cast<uint64_t>(ev::toDouble(seed));
        else if (ev::isBigInt(seed)) o.sampling.seed = ev::toUint64(seed);
    }
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
    Value lenVal = ev::getProperty(v, "length");
    if (ev::isNumber(lenVal)) {
        uint32_t len = static_cast<uint32_t>(ev::toDouble(lenVal));
        out.reserve(len);
        for (uint32_t i = 0; i < len; ++i) {
            Value elem = ev::getElement(v, i);
            out.push_back(static_cast<int32_t>(ev::toDouble(elem)));
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
    Value lenVal = ev::getProperty(v, "length");
    if (!ev::isNumber(lenVal)) return out;
    uint32_t n = static_cast<uint32_t>(ev::toDouble(lenVal));
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        Value elem = ev::getElement(v, i);
        std::string role, content;
        if (ev::isObject(elem)) {
            Value r = ev::getProperty(elem, "role");
            if (ev::isString(r)) role = ev::toUtf8(r);
            Value c = ev::getProperty(elem, "content");
            if (ev::isString(c)) content = ev::toUtf8(c);
        }
        out.emplace_back(std::move(role), std::move(content));
    }
    return out;
}

inline bool readImageArg(Value val, std::vector<uint8_t>& rgba, int& w, int& h, std::string& err) {
    if (!ev::isObject(val)) {
        err = "image must be an object with { data, width, height }";
        return false;
    }
    Value wVal = ev::getProperty(val, "width");
    Value hVal = ev::getProperty(val, "height");
    if (!ev::isNumber(wVal) || !ev::isNumber(hVal)) {
        err = "image { width, height } must be numbers";
        return false;
    }
    w = static_cast<int>(ev::toDouble(wVal));
    h = static_cast<int>(ev::toDouble(hVal));
    if (w <= 0 || h <= 0) {
        err = "image { width, height } must be positive";
        return false;
    }
    Value dataVal = ev::getProperty(val, "data");
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
    Value imgs = ev::getProperty(opts, "images");
    if (ev::isUndefined(imgs) || ev::isNull(imgs)) return true;

    auto parseOne = [&](Value v) -> bool {
        std::vector<uint8_t> rgba;
        int w = 0, h = 0;
        if (!readImageArg(v, rgba, w, h, err)) return false;
        images.push_back({ rgbaToChwUnit(rgba, w, h), h, w });
        return true;
    };

    Value lenVal = ev::getProperty(imgs, "length");
    if (ev::isNumber(lenVal)) {
        uint32_t count = static_cast<uint32_t>(ev::toDouble(lenVal));
        for (uint32_t i = 0; i < count; ++i) {
            Value item = ev::getElement(imgs, i);
            if (!parseOne(item)) return false;
        }
        return true;
    }
    return parseOne(imgs);
}

// ═══════════════════════════════════════════════════════════════════════════
// Function & Object Creators / Unwrappers
// ═══════════════════════════════════════════════════════════════════════════

HostAsyncHandle* hostAsyncHandleOf(Value v);
Value makeAsyncHandleValue(std::shared_ptr<HostAsyncHandle> h);

HostQwenTokenizer* hostQwenTokenizerOf(Value v);
Value makeQwenTokenizerValue(std::unique_ptr<brolm::qwen::Tokenizer> tok);

HostMistralTokenizer* hostMistralTokenizerOf(Value v);
Value makeMistralTokenizerValue(std::unique_ptr<brolm::mistral::Tokenizer> tok);

HostGemmaTokenizer* hostGemmaTokenizerOf(Value v);
Value makeGemmaTokenizerValue(std::unique_ptr<brolm::gemma::Tokenizer> tok);

HostLlama3Tokenizer* hostLlama3TokenizerOf(Value v);
Value makeLlama3TokenizerValue(std::unique_ptr<brolm::llama3::Tokenizer> tok);

HostLMModel* hostLMModelOf(Value v);
Value makeLMModelValue(std::unique_ptr<LMDecoder> dec, brotensor::Device dev);

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

// ═══════════════════════════════════════════════════════════════════════════
// Registration Functions
// ═══════════════════════════════════════════════════════════════════════════

void registerLMTokenizerClasses();
void registerLMModelClasses();
void registerLMVLClasses();
void registerLMClipClasses();

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
Value js_lm_generate(Value, std::span<const Value>);
Value js_lm_tick(Value, std::span<const Value>);

} // namespace brolm::api
