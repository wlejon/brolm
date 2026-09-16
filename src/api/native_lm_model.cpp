#include "host_lm_internal.h"

#include <chrono>

namespace brolm::api {

HostClass g_asyncHandleClass;
HostClass g_lmModelClass;

// ═══════════════════════════════════════════════════════════════════════════
// Creators and Unwrappers
// ═══════════════════════════════════════════════════════════════════════════

HostAsyncHandle* hostAsyncHandleOf(Value v) {
    if (!ev::isObject(v)) return nullptr;
    return static_cast<HostAsyncHandle*>(g_asyncHandleClass.unwrap(v));
}

Value makeAsyncHandleValue(std::shared_ptr<HostAsyncHandle> h) {
    auto wrapper = std::make_unique<std::shared_ptr<HostAsyncHandle>>(h);
    return g_asyncHandleClass.make(wrapper.release(), [](void* p) {
        delete static_cast<std::shared_ptr<HostAsyncHandle>*>(p);
    });
}

HostLMModel* hostLMModelOf(Value v) {
    if (!ev::isObject(v)) return nullptr;
    return static_cast<HostLMModel*>(g_lmModelClass.unwrap(v));
}

Value makeLMModelValue(std::unique_ptr<LMDecoder> dec, brotensor::Device dev) {
    auto w = std::make_unique<HostLMModel>();
    w->model = std::move(dec);
    w->weights_loaded = true;
    w->device = dev;
    return g_lmModelClass.createInstance(std::move(w));
}

// ═══════════════════════════════════════════════════════════════════════════
// Async Job Management
// ═══════════════════════════════════════════════════════════════════════════

struct AsyncLmJob {
    std::shared_ptr<HostAsyncHandle> handle;
    ev::Persistent onTokenCb;
    ev::Persistent onDoneCb;
    ev::Persistent onErrorCb;
    ev::Persistent modelRef;
    std::mutex queueMutex;
    std::vector<int32_t> pendingTokens;
    std::vector<int32_t> allGenerated;
    std::string errorMessage;
    std::atomic<bool> done{false};
    std::thread worker;

    ~AsyncLmJob() {
        if (worker.joinable()) worker.join();
    }
};

static std::vector<std::shared_ptr<AsyncLmJob>> s_activeJobs;
static std::mutex s_jobsMutex;

void tickLMAsync() {
    std::vector<std::shared_ptr<AsyncLmJob>> jobsToTick;
    {
        std::lock_guard<std::mutex> lock(s_jobsMutex);
        jobsToTick = s_activeJobs;
    }
    if (jobsToTick.empty()) return;

    std::vector<std::shared_ptr<AsyncLmJob>> remaining;
    for (auto& job : jobsToTick) {
        std::vector<int32_t> tokens;
        {
            std::lock_guard<std::mutex> qLock(job->queueMutex);
            tokens.swap(job->pendingTokens);
        }

        if (ev::isFunction(job->onTokenCb.get())) {
            for (int32_t tok : tokens) {
                Value arg = ev::fromDouble(tok);
                ev::call(job->onTokenCb.get(), ev::undefined(), std::span<const Value>(&arg, 1));
            }
        }

        if (job->done.load(std::memory_order_acquire)) {
            if (job->worker.joinable()) job->worker.join();

            // Deliver any last tokens
            {
                std::lock_guard<std::mutex> qLock(job->queueMutex);
                tokens.swap(job->pendingTokens);
            }
            if (ev::isFunction(job->onTokenCb.get())) {
                for (int32_t tok : tokens) {
                    Value arg = ev::fromDouble(tok);
                    ev::call(job->onTokenCb.get(), ev::undefined(), std::span<const Value>(&arg, 1));
                }
            }

            if (!job->errorMessage.empty()) {
                if (ev::isFunction(job->onErrorCb.get())) {
                    Value errVal = ev::fromUtf8(job->errorMessage);
                    ev::call(job->onErrorCb.get(), ev::undefined(), std::span<const Value>(&errVal, 1));
                }
            } else if (!job->handle->cancelled.load(std::memory_order_acquire)) {
                if (ev::isFunction(job->onDoneCb.get())) {
                    Value arrVal = makeInt32Array(job->allGenerated.data(), job->allGenerated.size());
                    ev::call(job->onDoneCb.get(), ev::undefined(), std::span<const Value>(&arrVal, 1));
                }
            }
        } else {
            remaining.push_back(job);
        }
    }

    {
        std::lock_guard<std::mutex> lock(s_jobsMutex);
        s_activeJobs = std::move(remaining);
    }
}

Value js_lm_tick(Value, std::span<const Value>) {
    tickLMAsync();
    return ev::undefined();
}

// ═══════════════════════════════════════════════════════════════════════════
// AsyncHandle Decorator
// ═══════════════════════════════════════════════════════════════════════════

static void decorateAsyncHandle(ObjectBuilder& b) {
    b.def("cancel", 0, [](Value self, std::span<const Value>) -> Value {
        auto* wp = static_cast<std::shared_ptr<HostAsyncHandle>*>(g_asyncHandleClass.unwrap(self));
        if (wp && *wp) {
            (*wp)->cancelled.store(true, std::memory_order_release);
        }
        return ev::undefined();
    });

    b.def("wait", 0, [](Value self, std::span<const Value>) -> Value {
        auto* wp = static_cast<std::shared_ptr<HostAsyncHandle>*>(g_asyncHandleClass.unwrap(self));
        if (wp && *wp) {
            while (!(*wp)->finished.load(std::memory_order_acquire)) {
                tickLMAsync();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            tickLMAsync();
        }
        return ev::undefined();
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// LMModel Autoregressive Generation Core
// ═══════════════════════════════════════════════════════════════════════════

static std::vector<int32_t> runDecode(LMDecoder& model,
                                      const std::vector<int32_t>& prompt,
                                      int eos_id,
                                      const brolm::qwen::GenerateOptions& opts,
                                      const std::function<bool(int32_t)>& onToken = nullptr,
                                      const std::atomic<bool>* cancel = nullptr) {
    std::vector<int32_t> generated;
    if (prompt.empty() || opts.max_new_tokens <= 0) return generated;

    const int vocab = model.vocabSize();
    model.allocateCache(static_cast<int>(prompt.size()) + opts.max_new_tokens);
    std::mt19937_64 rng(opts.sampling.seed);
    const bool stop = opts.stop_on_eos && eos_id >= 0;

    brotensor::Tensor logits;
    model.forward(prompt.data(), static_cast<int>(prompt.size()), logits);
    std::vector<float> row = lastRowFp32(logits);
    int next = brolm::qwen::sample_token(row.data(), vocab, opts.sampling, rng);

    while (true) {
        if (cancel && cancel->load(std::memory_order_acquire)) break;
        if (stop && next == eos_id) break;
        generated.push_back(static_cast<int32_t>(next));

        if (onToken) {
            bool keep = onToken(static_cast<int32_t>(next));
            if (!keep) break;
        }

        if (static_cast<int>(generated.size()) >= opts.max_new_tokens) break;
        int32_t cur = generated.back();
        model.forward(&cur, 1, logits);
        row = lastRowFp32(logits);
        next = brolm::qwen::sample_token(row.data(), vocab, opts.sampling, rng);
    }
    return generated;
}

// ═══════════════════════════════════════════════════════════════════════════
// LMModel Decorator
// ═══════════════════════════════════════════════════════════════════════════

static void decorateLMModel(ObjectBuilder& b) {
    b.accessor("family", [](Value self, std::span<const Value>) {
        auto* w = hostLMModelOf(self);
        return ev::fromUtf8(w && w->model ? w->model->family() : "qwen3");
    });
    b.accessor("vocabSize", [](Value self, std::span<const Value>) {
        auto* w = hostLMModelOf(self);
        return ev::fromDouble(w && w->model ? w->model->vocabSize() : 0);
    });
    b.accessor("hiddenSize", [](Value self, std::span<const Value>) {
        auto* w = hostLMModelOf(self);
        return ev::fromDouble(w && w->model ? w->model->hiddenSize() : 0);
    });
    b.accessor("numLayers", [](Value self, std::span<const Value>) {
        auto* w = hostLMModelOf(self);
        return ev::fromDouble(w && w->model ? w->model->numLayers() : 0);
    });
    b.accessor("maxSeqLen", [](Value self, std::span<const Value>) {
        auto* w = hostLMModelOf(self);
        return ev::fromDouble(w && w->model ? w->model->maxSeqLen() : 0);
    });
    b.accessor("cacheLen", [](Value self, std::span<const Value>) {
        auto* w = hostLMModelOf(self);
        return ev::fromDouble(w && w->model ? w->model->cacheLen() : 0);
    });

    b.def("allocateCache", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostLMModelOf(self);
        if (!w || !w->model) return ev::throwTypeError("allocateCache: not an LMModel");
        if (a.empty() || !ev::isNumber(a[0]))
            return ev::throwTypeError("allocateCache(maxSeqLen): number required");
        int32_t n = static_cast<int32_t>(ev::toDouble(a[0]));
        w->model->allocateCache(n);
        return ev::undefined();
    });

    b.def("resetCache", 0, [](Value self, std::span<const Value>) -> Value {
        auto* w = hostLMModelOf(self);
        if (!w || !w->model) return ev::throwTypeError("resetCache: not an LMModel");
        w->model->resetCache();
        return ev::undefined();
    });

    b.def("forward", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostLMModelOf(self);
        if (!w || !w->model) return ev::throwTypeError("forward: not an LMModel");
        if (a.empty()) return ev::throwTypeError("forward(promptIds): promptIds required");
        std::vector<int32_t> prompt = readInt32Array(a[0]);
        if (prompt.empty()) return ev::throwTypeError("forward: promptIds cannot be empty");

        try {
            brotensor::DeviceScope scope(w->device);
            brotensor::Tensor logits;
            w->model->forward(prompt.data(), static_cast<int>(prompt.size()), logits);
            brotensor::sync_all();
            std::vector<float> host = downloadFloats(logits);
            return makeFloat32Array(host.data(), host.size());
        } catch (const std::exception& e) {
            return ev::throwError(std::string("forward: ") + e.what());
        }
    });

    b.def("generate", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostLMModelOf(self);
        if (!w || !w->model) return ev::throwTypeError("generate: not an LMModel");
        if (!w->weights_loaded) return ev::throwError("generate: weights not loaded");
        if (a.empty()) return ev::throwTypeError("generate(promptIds, opts?): promptIds required");

        std::vector<int32_t> prompt = readInt32Array(a[0]);
        if (prompt.empty()) return ev::throwTypeError("generate: promptIds must be non-empty");

        brolm::qwen::GenerateOptions opts;
        int eos_id = -1;
        if (a.size() >= 2 && ev::isObject(a[1])) {
            opts = parseGenerateOptions(a[1]);
            Value eosVal = ev::getProperty(a[1], "eosId");
            if (ev::isNumber(eosVal)) eos_id = static_cast<int>(ev::toDouble(eosVal));
        }

        try {
            brotensor::DeviceScope scope(w->device);
            auto ids = runDecode(*w->model, prompt, eos_id, opts);
            return makeInt32Array(ids.data(), ids.size());
        } catch (const std::exception& e) {
            return ev::throwError(std::string("generate: ") + e.what());
        }
    });

    b.def("generateStream", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostLMModelOf(self);
        if (!w || !w->model) return ev::throwTypeError("generateStream: not an LMModel");
        if (!w->weights_loaded) return ev::throwError("generateStream: weights not loaded");
        if (a.size() < 2) return ev::throwTypeError("generateStream(promptIds, opts, onToken): required args missing");

        std::vector<int32_t> prompt = readInt32Array(a[0]);
        if (prompt.empty()) return ev::throwTypeError("generateStream: promptIds must be non-empty");

        brolm::qwen::GenerateOptions opts;
        int eos_id = -1;
        if (ev::isObject(a[1])) {
            opts = parseGenerateOptions(a[1]);
            Value eosVal = ev::getProperty(a[1], "eosId");
            if (ev::isNumber(eosVal)) eos_id = static_cast<int>(ev::toDouble(eosVal));
        }

        Value onTokenVal = (a.size() >= 3) ? a[2] : a[1];
        if (!ev::isFunction(onTokenVal))
            return ev::throwTypeError("generateStream: onToken callback must be a function");

        ev::Persistent tokenCb(onTokenVal);
        std::function<bool(int32_t)> hook = [&tokenCb](int32_t id) -> bool {
            Value arg = ev::fromDouble(id);
            ev::CallResult r = ev::call(tokenCb.get(), ev::undefined(), std::span<const Value>(&arg, 1));
            if (r.thrown) return false;
            return ev::isUndefined(r.value) || ev::toBool(r.value);
        };

        try {
            brotensor::DeviceScope scope(w->device);
            auto ids = runDecode(*w->model, prompt, eos_id, opts, hook);
            return makeInt32Array(ids.data(), ids.size());
        } catch (const std::exception& e) {
            return ev::throwError(std::string("generateStream: ") + e.what());
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Model Builders and Loaders
// ═══════════════════════════════════════════════════════════════════════════

static void buildQwen(const std::string& path, brotensor::Device dev,
                      std::unique_ptr<HostLMModel>& mw_out,
                      std::unique_ptr<brolm::qwen::Tokenizer>& tw_out) {
    brotensor::gguf::File f = brotensor::gguf::File::open(path);
    auto cfg = brolm::qwen::Qwen3Config::from_gguf(f);

    auto mw = std::make_unique<HostLMModel>();
    mw->device = dev;
    {
        brotensor::DeviceScope scope(dev);
        auto dec = std::make_unique<QwenDecoder>(cfg);
        dec->m.load_weights(f);
        mw->model = std::move(dec);
    }
    mw->weights_loaded = true;

    auto tw = std::make_unique<brolm::qwen::Tokenizer>(
        brolm::qwen::Tokenizer::from_gguf(f));

    mw_out = std::move(mw);
    tw_out = std::move(tw);
}

static void buildMistral(const std::string& ggufPath, const std::string& tokPath,
                         brotensor::Device dev,
                         std::unique_ptr<HostLMModel>& mw_out,
                         std::unique_ptr<brolm::mistral::Tokenizer>& tw_out) {
    brotensor::gguf::File f = brotensor::gguf::File::open(ggufPath);
    auto cfg = brolm::mistral3::Mistral3Config::from_gguf(f);

    auto mw = std::make_unique<HostLMModel>();
    mw->device = dev;
    {
        brotensor::DeviceScope scope(dev);
        auto dec = std::make_unique<MistralDecoder>(cfg.text);
        dec->m.load_weights(f);
        mw->model = std::move(dec);
    }
    mw->weights_loaded = true;

    auto tw = std::make_unique<brolm::mistral::Tokenizer>(
        brolm::mistral::Tokenizer::load(tokPath));

    mw_out = std::move(mw);
    tw_out = std::move(tw);
}

static void buildGemma2(const std::string& dir, brotensor::Device dev,
                        std::unique_ptr<HostLMModel>& mw_out,
                        std::unique_ptr<brolm::gemma::Tokenizer>& tw_out) {
    namespace fs = std::filesystem;
    auto cfg = brolm::gemma::Gemma2Config::from_safetensors_dir(dir);

    std::vector<std::string> shard_paths;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".safetensors")
            shard_paths.push_back(entry.path().string());
    }
    if (shard_paths.empty())
        throw std::runtime_error("no *.safetensors shards in " + dir);
    std::sort(shard_paths.begin(), shard_paths.end());

    std::vector<brotensor::safetensors::File> files;
    files.reserve(shard_paths.size());
    for (const auto& sp : shard_paths)
        files.push_back(brotensor::safetensors::File::open(sp));
    std::vector<const brotensor::safetensors::File*> shard_ptrs;
    shard_ptrs.reserve(files.size());
    for (const auto& f : files) shard_ptrs.push_back(&f);

    auto mw = std::make_unique<HostLMModel>();
    mw->device = dev;
    {
        brotensor::DeviceScope scope(dev);
        auto dec = std::make_unique<GemmaDecoder>(cfg);
        dec->m.load_weights(shard_ptrs);
        mw->model = std::move(dec);
    }
    mw->weights_loaded = true;

    const fs::path tok_json = fs::path(dir) / "tokenizer.json";
    if (!fs::exists(tok_json))
        throw std::runtime_error("Gemma-2 requires " + tok_json.string());
    auto tw = std::make_unique<brolm::gemma::Tokenizer>(
        brolm::gemma::Tokenizer::load(tok_json.string()));

    mw_out = std::move(mw);
    tw_out = std::move(tw);
}

// ═══════════════════════════════════════════════════════════════════════════
// Public LM Functions
// ═══════════════════════════════════════════════════════════════════════════

void registerLMModelClasses() {
    g_asyncHandleClass.install("AsyncHandle", 0, nullptr, decorateAsyncHandle);
    g_lmModelClass.install("LMModel", 0, nullptr, decorateLMModel);
}

Value js_init(Value, std::span<const Value>) {
    try {
        brotensor::init();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("bro.lm.init: ") + e.what());
    }
    return ev::undefined();
}

Value js_loadQwen(Value, std::span<const Value> a) {
    if (a.empty() || !ev::isString(a[0]))
        return ev::throwTypeError("loadQwen(ggufPath, opts?): path string required");
    std::string path = ev::toUtf8(a[0]);

    brotensor::init();
    brotensor::Device dev = autoDevice();
    if (a.size() >= 2) {
        std::string err;
        if (!parseDeviceOpt(a[1], dev, err))
            return ev::throwTypeError(err);
    }

    try {
        std::unique_ptr<HostLMModel> mw;
        std::unique_ptr<brolm::qwen::Tokenizer> tw;
        buildQwen(path, dev, mw, tw);

        ObjectBuilder res;
        res.set("model", g_lmModelClass.createInstance(std::move(mw)));
        res.set("tokenizer", makeQwenTokenizerValue(std::move(tw)));
        return res.build();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("loadQwen: ") + e.what());
    }
}

Value js_loadMistral(Value, std::span<const Value> a) {
    if (a.empty() || !ev::isString(a[0]))
        return ev::throwTypeError("loadMistral(ggufPath, opts): path string required");
    std::string path = ev::toUtf8(a[0]);

    std::string tokPath;
    brotensor::init();
    brotensor::Device dev = autoDevice();
    if (a.size() >= 2) {
        if (ev::isObject(a[1])) {
            Value tp = ev::getProperty(a[1], "tokenizerPath");
            if (ev::isString(tp)) tokPath = ev::toUtf8(tp);
            std::string err;
            if (!parseDeviceOpt(a[1], dev, err))
                return ev::throwTypeError(err);
        } else if (ev::isString(a[1])) {
            tokPath = ev::toUtf8(a[1]);
        }
    }

    if (tokPath.empty()) {
        std::filesystem::path p(path);
        std::filesystem::path tekken = p.parent_path() / "tekken.json";
        if (std::filesystem::exists(tekken)) tokPath = tekken.string();
    }
    if (tokPath.empty())
        return ev::throwTypeError("loadMistral: opts.tokenizerPath (tekken.json) required");

    try {
        std::unique_ptr<HostLMModel> mw;
        std::unique_ptr<brolm::mistral::Tokenizer> tw;
        buildMistral(path, tokPath, dev, mw, tw);

        ObjectBuilder res;
        res.set("model", g_lmModelClass.createInstance(std::move(mw)));
        res.set("tokenizer", makeMistralTokenizerValue(std::move(tw)));
        return res.build();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("loadMistral: ") + e.what());
    }
}

Value js_loadGemma2(Value, std::span<const Value> a) {
    if (a.empty() || !ev::isString(a[0]))
        return ev::throwTypeError("loadGemma2(modelDir, opts?): dir string required");
    std::string dir = ev::toUtf8(a[0]);

    brotensor::init();
    brotensor::Device dev = autoDevice();
    if (a.size() >= 2 && ev::isObject(a[1])) {
        std::string err;
        if (!parseDeviceOpt(a[1], dev, err))
            return ev::throwTypeError(err);
    }

    try {
        std::unique_ptr<HostLMModel> mw;
        std::unique_ptr<brolm::gemma::Tokenizer> tw;
        buildGemma2(dir, dev, mw, tw);

        ObjectBuilder res;
        res.set("model", g_lmModelClass.createInstance(std::move(mw)));
        res.set("tokenizer", makeGemmaTokenizerValue(std::move(tw)));
        return res.build();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("loadGemma2: ") + e.what());
    }
}

Value js_lm_generate(Value, std::span<const Value> a) {
    if (a.size() < 2)
        return ev::throwTypeError("generate(model, prompt, opts?): model and prompt required");

    auto handle = std::make_shared<HostAsyncHandle>();
    auto job = std::make_shared<AsyncLmJob>();
    job->handle = handle;
    job->modelRef.set(a[0]);

    Value optsVal = (a.size() >= 3 && ev::isObject(a[2])) ? a[2] : ev::undefined();
    if (ev::isObject(optsVal)) {
        Value onTok = ev::getProperty(optsVal, "onToken");
        if (ev::isFunction(onTok)) job->onTokenCb.set(onTok);
        Value onD = ev::getProperty(optsVal, "onDone");
        if (ev::isFunction(onD)) job->onDoneCb.set(onD);
        Value onErr = ev::getProperty(optsVal, "onError");
        if (ev::isFunction(onErr)) job->onErrorCb.set(onErr);
    }

    if (auto* w = hostLMModelOf(a[0])) {
        std::vector<int32_t> prompt = readInt32Array(a[1]);
        if (prompt.empty())
            return ev::throwTypeError("generate: promptIds must be non-empty for LMModel");
        brolm::qwen::GenerateOptions opts = parseGenerateOptions(optsVal);
        int eos_id = -1;
        if (ev::isObject(optsVal)) {
            Value evV = ev::getProperty(optsVal, "eosId");
            if (ev::isNumber(evV)) eos_id = static_cast<int>(ev::toDouble(evV));
        }

        job->worker = std::thread([job, w, prompt = std::move(prompt), eos_id, opts]() {
            try {
                brotensor::DeviceScope scope(w->device);
                auto onTokenHook = [job](int32_t tok) -> bool {
                    {
                        std::lock_guard<std::mutex> qLock(job->queueMutex);
                        job->pendingTokens.push_back(tok);
                    }
                    return !job->handle->cancelled.load(std::memory_order_acquire);
                };
                job->allGenerated = runDecode(*w->model, prompt, eos_id, opts, onTokenHook, &job->handle->cancelled);
            } catch (const std::exception& e) {
                job->errorMessage = e.what();
            }
            job->handle->finished.store(true, std::memory_order_release);
            job->done.store(true, std::memory_order_release);
        });

        {
            std::lock_guard<std::mutex> lock(s_jobsMutex);
            s_activeJobs.push_back(job);
        }
        return makeAsyncHandleValue(handle);
    }

    if (auto* q35 = hostQwen35ModelOf(a[0])) {
        if (!ev::isString(a[1]))
            return ev::throwTypeError("generate: prompt must be a string for Qwen35Model");
        std::string prompt = ev::toUtf8(a[1]);
        brolm::qwen::GenerateOptions opts = parseGenerateOptions(optsVal);
        std::vector<VlmImage> images;
        std::string imgErr;
        if (!readVlmImages(optsVal, images, imgErr))
            return ev::throwTypeError(imgErr);

        job->worker = std::thread([job, q35, prompt = std::move(prompt), opts, imgs = std::move(images)]() {
            try {
                brotensor::DeviceScope scope(q35->device);
                q35->vlm->set_generation(opts.max_new_tokens, opts.sampling.temperature,
                                         opts.sampling.top_k, opts.sampling.top_p,
                                         opts.sampling.seed);
                std::vector<brolm::qwen35::ImageInput> inputs;
                inputs.reserve(imgs.size());
                for (auto& im : imgs)
                    inputs.push_back(brolm::qwen35::ImageInput{ im.chw.data(), im.H, im.W });

                auto onTokenHook = [job](int tok) -> bool {
                    {
                        std::lock_guard<std::mutex> qLock(job->queueMutex);
                        job->pendingTokens.push_back(tok);
                    }
                    return !job->handle->cancelled.load(std::memory_order_acquire);
                };
                auto ids = q35->vlm->generate_tokens(prompt, inputs, onTokenHook);
                job->allGenerated.assign(ids.begin(), ids.end());
            } catch (const std::exception& e) {
                job->errorMessage = e.what();
            }
            job->handle->finished.store(true, std::memory_order_release);
            job->done.store(true, std::memory_order_release);
        });

        {
            std::lock_guard<std::mutex> lock(s_jobsMutex);
            s_activeJobs.push_back(job);
        }
        return makeAsyncHandleValue(handle);
    }

    if (auto* qvl = hostQwen3VLModelOf(a[0])) {
        if (!ev::isString(a[1]))
            return ev::throwTypeError("generate: prompt must be a string for Qwen3VLModel");
        std::string prompt = ev::toUtf8(a[1]);
        brolm::qwen::GenerateOptions opts = parseGenerateOptions(optsVal);
        std::vector<VlmImage> images;
        std::string imgErr;
        if (!readVlmImages(optsVal, images, imgErr))
            return ev::throwTypeError(imgErr);

        job->worker = std::thread([job, qvl, prompt = std::move(prompt), opts, imgs = std::move(images)]() {
            try {
                brotensor::DeviceScope scope(qvl->device);
                qvl->vlm->set_generation(opts.max_new_tokens, opts.sampling.temperature,
                                         opts.sampling.top_k, opts.sampling.top_p,
                                         opts.sampling.seed);
                std::vector<brolm::qwen3vl::ImageInput> inputs;
                inputs.reserve(imgs.size());
                for (auto& im : imgs)
                    inputs.push_back(brolm::qwen3vl::ImageInput{ im.chw.data(), im.H, im.W });

                auto onTokenHook = [job](int tok) -> bool {
                    {
                        std::lock_guard<std::mutex> qLock(job->queueMutex);
                        job->pendingTokens.push_back(tok);
                    }
                    return !job->handle->cancelled.load(std::memory_order_acquire);
                };
                auto ids = qvl->vlm->generate_tokens(prompt, inputs, onTokenHook);
                job->allGenerated.assign(ids.begin(), ids.end());
            } catch (const std::exception& e) {
                job->errorMessage = e.what();
            }
            job->handle->finished.store(true, std::memory_order_release);
            job->done.store(true, std::memory_order_release);
        });

        {
            std::lock_guard<std::mutex> lock(s_jobsMutex);
            s_activeJobs.push_back(job);
        }
        return makeAsyncHandleValue(handle);
    }

    return ev::throwTypeError("generate: arg 0 must be LMModel, Qwen35Model, or Qwen3VLModel");
}

} // namespace brolm::api
