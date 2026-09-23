#include "host_lm_internal.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <optional>

namespace brolm::api {

HostClass g_asyncHandleClass;
HostClass g_lmModelClass;

// ═══════════════════════════════════════════════════════════════════════════
// Creators and Unwrappers
// ═══════════════════════════════════════════════════════════════════════════

HostAsyncHandle* hostAsyncHandleOf(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* wp = static_cast<std::shared_ptr<HostAsyncHandle>*>(g_asyncHandleClass.unwrap(v));
    return (wp && *wp) ? wp->get() : nullptr;
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

// ═══════════════════════════════════════════════════════════════════════════
// Background jobs
// ═══════════════════════════════════════════════════════════════════════════

// Per JS thread: a job's callbacks are Persistents of the realm that
// launched it and must be delivered there, so a Worker's jobs are ticked by
// the Worker's own bro.lm.tick() / wait() and never by the main thread's.
static thread_local std::vector<std::shared_ptr<AsyncJob>> t_jobs;

std::shared_ptr<AsyncJob> newAsyncJob() {
    auto job = std::make_shared<AsyncJob>();
    job->core = std::make_shared<AsyncCore>();
    job->core->handle = std::make_shared<HostAsyncHandle>();
    return job;
}

Value launchAsyncJob(std::shared_ptr<AsyncJob> job, std::function<void(AsyncCore&)> work) {
    std::shared_ptr<AsyncCore> core = job->core;
    job->worker = std::thread([core, work = std::move(work)]() {
        try {
            work(*core);
        } catch (const std::exception& e) {
            core->error = e.what();
            if (core->error.empty()) core->error = "unknown error";
        } catch (...) {
            core->error = "unknown error";
        }
        core->done.store(true, std::memory_order_release);
        core->handle->notify();
    });
    t_jobs.push_back(job);
    return makeAsyncHandleValue(core->handle);
}

void callRooted(const ev::Persistent& fn, std::initializer_list<const ev::Persistent*> args) {
    if (!ev::isFunction(fn.get())) return;
    // Slot reads only, with no allocation between them and the call.
    std::vector<Value> argv;
    argv.reserve(args.size());
    for (const ev::Persistent* p : args) argv.push_back(p->get());
    ev::call(fn.get(), ev::undefined(), std::span<const Value>(argv.data(), argv.size()));
}

void reportJobError(AsyncJob& job, const std::string& message) {
    if (ev::isFunction(job.onError.get())) {
        ev::Persistent msg(ev::fromUtf8(message));
        callRooted(job.onError, {&msg});
    } else {
        std::fprintf(stderr, "bro.lm: %s\n", message.c_str());
    }
}

static void deliverTokens(AsyncJob& job) {
    std::vector<int32_t> tokens;
    {
        std::lock_guard<std::mutex> lk(job.core->mu);
        tokens.swap(job.core->pendingTokens);
    }
    if (!ev::isFunction(job.onToken.get())) return;
    for (int32_t tok : tokens) {
        const Value arg = ev::fromDouble(tok);
        ev::call(job.onToken.get(), ev::undefined(), std::span<const Value>(&arg, 1));
    }
}

void tickLMAsync() {
    tickLayaAsync();  // LayaModel promises (native_lm_laya_async.cpp)
    if (t_jobs.empty()) return;

    // A copy: a callback may launch a job (appending to t_jobs) or re-enter
    // this tick through AsyncHandle.wait().
    const std::vector<std::shared_ptr<AsyncJob>> jobs = t_jobs;
    for (const auto& job : jobs) {
        if (job->ticking) continue;
        job->ticking = true;
        deliverTokens(*job);
        if (!job->core->done.load(std::memory_order_acquire)) {
            job->ticking = false;
            continue;
        }
        if (job->worker.joinable()) job->worker.join();
        deliverTokens(*job);
        t_jobs.erase(std::remove(t_jobs.begin(), t_jobs.end(), job), t_jobs.end());

        // Release the model BEFORE finish, so an onDone that starts the next
        // generation on the same model succeeds; the finished work's output
        // lives in the job, which a new claim cannot disturb.
        if (job->busy) {
            job->busy->store(false, std::memory_order_release);
            job->busy = nullptr;
        }
        if (job->finish) job->finish(*job);

        job->onToken.set(ev::undefined());
        job->onDone.set(ev::undefined());
        job->onError.set(ev::undefined());
        job->keep.set(ev::undefined());
        job->core->handle->finished.store(true, std::memory_order_release);
        job->core->handle->notify();
    }
}

void shutdownLMJobs() {
    for (auto& job : t_jobs) job->core->handle->cancelled.store(true, std::memory_order_release);
    for (auto& job : t_jobs) {
        if (job->worker.joinable()) job->worker.join();
    }
    t_jobs.clear();
}

Value js_lm_tick(Value, std::span<const Value>) {
    tickLMAsync();
    return ev::undefined();
}

// ═══════════════════════════════════════════════════════════════════════════
// AsyncHandle Decorator
// ═══════════════════════════════════════════════════════════════════════════

static std::shared_ptr<HostAsyncHandle> asyncHandleOf(Value self) {
    auto* wp = static_cast<std::shared_ptr<HostAsyncHandle>*>(g_asyncHandleClass.unwrap(self));
    return (wp && *wp) ? *wp : nullptr;
}

static void decorateAsyncHandle(ObjectBuilder& b) {
    b.accessor("done", [](Value self, std::span<const Value>) {
        auto h = asyncHandleOf(self);
        return ev::fromBool(h && h->finished.load(std::memory_order_acquire));
    });
    b.accessor("cancelled", [](Value self, std::span<const Value>) {
        auto h = asyncHandleOf(self);
        return ev::fromBool(h && h->cancelled.load(std::memory_order_acquire));
    });

    b.def("cancel", 0, [](Value self, std::span<const Value>) -> Value {
        if (auto h = asyncHandleOf(self)) {
            h->cancelled.store(true, std::memory_order_release);
            h->notify();
        }
        return ev::undefined();
    });

    // Block until the job has finished and its callbacks have run.
    b.def("wait", 0, [](Value self, std::span<const Value>) -> Value {
        auto h = asyncHandleOf(self);
        if (!h) return ev::undefined();
        while (!h->finished.load(std::memory_order_acquire)) {
            tickLMAsync();
            if (h->finished.load(std::memory_order_acquire)) break;
            const uint64_t seq = h->event_seq.load(std::memory_order_acquire);
            std::unique_lock<std::mutex> lock(h->cvMutex);
            h->cv.wait_for(lock, std::chrono::milliseconds(5), [&]() {
                return h->finished.load(std::memory_order_acquire) ||
                       h->event_seq.load(std::memory_order_acquire) != seq;
            });
        }
        return ev::undefined();
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// LMModel Autoregressive Generation Core
// ═══════════════════════════════════════════════════════════════════════════

// The decode loop: prefill, then one token at a time. Repetition / frequency /
// presence / DRY penalties read the whole context (prompt + generated), and
// opts.grammar masks every step against the paired tokenizer's token text.
static std::vector<int32_t> runDecode(HostLMModel& hm,
                                      const std::vector<int32_t>& prompt,
                                      int eos_id,
                                      const brolm::qwen::GenerateOptions& opts,
                                      const std::function<bool(int32_t)>& onToken = nullptr,
                                      const std::atomic<bool>* cancel = nullptr) {
    LMDecoder& model = *hm.model;
    std::vector<int32_t> generated;
    if (prompt.empty() || opts.max_new_tokens <= 0) return generated;

    const int vocab = model.vocabSize();
    std::optional<brolm::Grammar> grammar;
    const std::vector<std::string>* pieces = nullptr;
    if (opts.grammar) {
        pieces = &hm.pieces();
        if (pieces->empty())
            throw std::runtime_error("opts.grammar needs the model's paired tokenizer (load it with bro.lm.loadQwen/loadMistral/loadGemma2)");
        grammar = opts.grammar->clone();
    }

    model.allocateCache(static_cast<int>(prompt.size()) + opts.max_new_tokens);
    std::mt19937_64 rng(opts.sampling.seed);
    const bool stop = opts.stop_on_eos && eos_id >= 0;
    std::vector<int32_t> context = prompt;

    brotensor::Tensor logits;
    model.forwardLast(prompt.data(), static_cast<int>(prompt.size()), logits);
    while (true) {
        if (cancel && cancel->load(std::memory_order_acquire)) break;
        std::vector<float> row = lastRowFp32(logits);
        // The mask also excludes tokens with no text (Grammar::mask_logits);
        // the stop token passes only when it stops, and a row with nothing
        // left allowed ends the decode (the sampler would return a masked id).
        if (grammar) {
            grammar->mask_logits(row.data(), vocab, *pieces, stop ? eos_id : -1);
            if (!brolm::detail::any_allowed(row.data(), vocab)) break;
        }
        const int next = brolm::qwen::sample_token(row.data(), vocab, opts.sampling, rng,
                                                   context.data(), static_cast<int>(context.size()));
        if (stop && next == eos_id) break;
        if (grammar) grammar->accept((*pieces)[static_cast<size_t>(next)]);
        generated.push_back(static_cast<int32_t>(next));
        context.push_back(static_cast<int32_t>(next));

        if (onToken && !onToken(static_cast<int32_t>(next))) break;
        if (static_cast<int>(generated.size()) >= opts.max_new_tokens) break;
        const int32_t cur = generated.back();
        model.forwardLast(&cur, 1, logits);
    }
    return generated;
}

// opts.eosId when given, else the paired tokenizer's end-of-turn id.
static int eosFor(const HostLMModel& w, Value opts) {
    double d = 0;
    if (propNumber(opts, "eosId", d)) return static_cast<int>(d);
    return w.defaultEos;
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
    b.accessor("eosId", [](Value self, std::span<const Value>) {
        auto* w = hostLMModelOf(self);
        return ev::fromDouble(w ? w->defaultEos : -1);
    });
    b.accessor("busy", [](Value self, std::span<const Value>) {
        auto* w = hostLMModelOf(self);
        return ev::fromBool(w && w->generating.load(std::memory_order_acquire));
    });

    b.def("allocateCache", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostLMModelOf(self);
        if (!w || !w->model) return ev::throwTypeError("allocateCache: not an LMModel");
        if (a.empty() || !ev::isNumber(a[0]))
            return ev::throwTypeError("allocateCache(maxSeqLen): number required");
        const int32_t n = static_cast<int32_t>(ev::toDouble(a[0]));
        if (n <= 0) return ev::throwRangeError("allocateCache: maxSeqLen must be > 0");
        BusyClaim claim(w->generating);
        if (!claim.ok()) return ev::throwError(std::string("allocateCache: ") + kBusyMessage);
        try {
            brotensor::DeviceScope scope(w->device);
            w->model->allocateCache(n);
        } catch (const std::exception& e) {
            return ev::throwError(std::string("allocateCache: ") + e.what());
        }
        return ev::undefined();
    });

    b.def("resetCache", 0, [](Value self, std::span<const Value>) -> Value {
        auto* w = hostLMModelOf(self);
        if (!w || !w->model) return ev::throwTypeError("resetCache: not an LMModel");
        BusyClaim claim(w->generating);
        if (!claim.ok()) return ev::throwError(std::string("resetCache: ") + kBusyMessage);
        w->model->resetCache();
        return ev::undefined();
    });

    b.def("forward", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostLMModelOf(self);
        if (!w || !w->model) return ev::throwTypeError("forward: not an LMModel");
        if (a.empty()) return ev::throwTypeError("forward(promptIds): promptIds required");
        std::vector<int32_t> prompt = readInt32Array(a[0]);
        if (prompt.empty()) return ev::throwTypeError("forward: promptIds cannot be empty");
        BusyClaim claim(w->generating);
        if (!claim.ok()) return ev::throwError(std::string("forward: ") + kBusyMessage);

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
        int eos_id = w->defaultEos;
        if (a.size() >= 2 && ev::isObject(a[1])) {
            opts = parseGenerateOptions(a[1]);
            eos_id = eosFor(*w, a[1]);
        }

        BusyClaim claim(w->generating);
        if (!claim.ok()) return ev::throwError(std::string("generate: ") + kBusyMessage);
        try {
            brotensor::DeviceScope scope(w->device);
            auto ids = runDecode(*w, prompt, eos_id, opts);
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

        // generateStream(ids, onToken) or generateStream(ids, opts, onToken).
        const size_t cbIdx = (a.size() >= 3 && ev::isFunction(a[2])) ? 2 : 1;
        if (!ev::isFunction(a[cbIdx]))
            return ev::throwTypeError("generateStream: onToken callback must be a function");
        ev::Persistent tokenCb(a[cbIdx]);

        brolm::qwen::GenerateOptions opts;
        int eos_id = w->defaultEos;
        if (cbIdx == 2 && ev::isObject(a[1])) {
            opts = parseGenerateOptions(a[1]);
            eos_id = eosFor(*w, a[1]);
        }

        std::function<bool(int32_t)> hook = [&tokenCb](int32_t id) -> bool {
            const Value arg = ev::fromDouble(id);
            ev::CallResult r = ev::call(tokenCb.get(), ev::undefined(), std::span<const Value>(&arg, 1));
            if (r.thrown) return false;
            return ev::isUndefined(r.value) || ev::toBool(r.value);
        };

        BusyClaim claim(w->generating);
        if (!claim.ok()) return ev::throwError(std::string("generateStream: ") + kBusyMessage);
        try {
            brotensor::DeviceScope scope(w->device);
            auto ids = runDecode(*w, prompt, eos_id, opts, hook);
            return makeInt32Array(ids.data(), ids.size());
        } catch (const std::exception& e) {
            return ev::throwError(std::string("generateStream: ") + e.what());
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Model Builders and Loaders
// ═══════════════════════════════════════════════════════════════════════════

// A decoder + the tokenizer it was loaded beside, built off the JS thread.
template <class Tok>
struct BuiltLM {
    std::unique_ptr<HostLMModel> model;
    std::shared_ptr<Tok> tok;
};

// Pair the model with its tokenizer: the default eos and the token text the
// grammar mask reads.
template <class Tok>
static void pairTokenizer(HostLMModel& mw, const std::shared_ptr<Tok>& tok) {
    mw.defaultEos = tok->eos_id();
    // A tokenizer that can tell control tokens apart gives them no text, so
    // the grammar mask never lets one through as though it were text.
    if constexpr (requires { tok->token_text(int32_t{0}); })
        mw.tokenText = [tok](int32_t id) { return tok->token_text(id); };
    else
        mw.tokenText = [tok](int32_t id) { return tok->decode(std::vector<int32_t>{id}); };
}

static void buildQwen(const std::string& path, brotensor::Device dev, BuiltLM<brolm::qwen::Tokenizer>& out) {
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

    out.tok = std::make_shared<brolm::qwen::Tokenizer>(brolm::qwen::Tokenizer::from_gguf(f));
    pairTokenizer(*mw, out.tok);
    out.model = std::move(mw);
}

static void buildMistral(const std::string& ggufPath, const std::string& tokPath,
                         brotensor::Device dev, BuiltLM<brolm::mistral::Tokenizer>& out) {
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

    out.tok = std::make_shared<brolm::mistral::Tokenizer>(brolm::mistral::Tokenizer::load(tokPath));
    pairTokenizer(*mw, out.tok);
    out.model = std::move(mw);
}

static void buildGemma2(const std::string& dir, brotensor::Device dev, BuiltLM<brolm::gemma::Tokenizer>& out) {
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

    const fs::path tok_json = fs::path(dir) / "tokenizer.json";
    if (!fs::exists(tok_json))
        throw std::runtime_error("Gemma-2 requires " + tok_json.string());

    auto mw = std::make_unique<HostLMModel>();
    mw->device = dev;
    {
        brotensor::DeviceScope scope(dev);
        auto dec = std::make_unique<GemmaDecoder>(cfg);
        dec->m.load_weights(shard_ptrs);
        mw->model = std::move(dec);
    }
    mw->weights_loaded = true;

    out.tok = std::make_shared<brolm::gemma::Tokenizer>(brolm::gemma::Tokenizer::load(tok_json.string()));
    pairTokenizer(*mw, out.tok);
    out.model = std::move(mw);
}

// { model, tokenizer } for a built pair.
template <class Tok, class MakeTok>
static Value wrapPair(BuiltLM<Tok>& b, MakeTok makeTok) {
    ObjectBuilder res;
    res.set("model", g_lmModelClass.createInstance(std::move(b.model)));
    res.set("tokenizer", makeTok(b.tok));
    return res.build();
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
    const std::string path = resolvePath(ev::toUtf8(a[0]));

    brotensor::init();
    brotensor::Device dev = autoDevice();
    if (a.size() >= 2) {
        std::string err;
        if (!parseDeviceOpt(a[1], dev, err))
            return ev::throwTypeError(err);
    }

    using B = BuiltLM<brolm::qwen::Tokenizer>;
    return runLoad<B>(a, 1, "loadQwen",
        [path, dev](B& b) { buildQwen(path, dev, b); },
        [](B& b) { return wrapPair(b, [](auto& t) { return makeQwenTokenizerValue(t); }); });
}

Value js_loadMistral(Value, std::span<const Value> a) {
    if (a.empty() || !ev::isString(a[0]))
        return ev::throwTypeError("loadMistral(ggufPath, opts): path string required");
    const std::string path = resolvePath(ev::toUtf8(a[0]));

    std::string tokPath;
    brotensor::init();
    brotensor::Device dev = autoDevice();
    if (a.size() >= 2) {
        if (ev::isObject(a[1])) {
            propString(a[1], "tokenizerPath", tokPath);
            std::string err;
            if (!parseDeviceOpt(a[1], dev, err))
                return ev::throwTypeError(err);
        } else if (ev::isString(a[1])) {
            tokPath = ev::toUtf8(a[1]);
        }
    }

    if (!tokPath.empty()) {
        tokPath = resolvePath(tokPath);
    } else {
        std::filesystem::path tekken = std::filesystem::path(path).parent_path() / "tekken.json";
        if (std::filesystem::exists(tekken)) tokPath = tekken.string();
    }
    if (tokPath.empty())
        return ev::throwTypeError("loadMistral: opts.tokenizerPath (tekken.json) required");

    using B = BuiltLM<brolm::mistral::Tokenizer>;
    return runLoad<B>(a, 1, "loadMistral",
        [path, tokPath, dev](B& b) { buildMistral(path, tokPath, dev, b); },
        [](B& b) { return wrapPair(b, [](auto& t) { return makeMistralTokenizerValue(t); }); });
}

Value js_loadGemma2(Value, std::span<const Value> a) {
    if (a.empty() || !ev::isString(a[0]))
        return ev::throwTypeError("loadGemma2(modelDir, opts?): dir string required");
    const std::string dir = resolvePath(ev::toUtf8(a[0]));

    brotensor::init();
    brotensor::Device dev = autoDevice();
    if (a.size() >= 2 && ev::isObject(a[1])) {
        std::string err;
        if (!parseDeviceOpt(a[1], dev, err))
            return ev::throwTypeError(err);
    }

    using B = BuiltLM<brolm::gemma::Tokenizer>;
    return runLoad<B>(a, 1, "loadGemma2",
        [dir, dev](B& b) { buildGemma2(dir, dev, b); },
        [](B& b) { return wrapPair(b, [](auto& t) { return makeGemmaTokenizerValue(t); }); });
}

// ═══════════════════════════════════════════════════════════════════════════
// bro.lm.generate(model, prompt, opts) — background generation
// ═══════════════════════════════════════════════════════════════════════════
//
// Runs the decode loop on a worker thread; returns an AsyncHandle (cancel(),
// wait(), done). opts.onToken(id) fires per token and opts.onDone(ids, info)
// once at the end, both on this thread's tick; info is { cancelled, error? }.
// A failure goes to opts.onError(message) when given, else to onDone's info.

struct GenResult {
    std::vector<int32_t> ids;
};

static void finishGenerate(const std::shared_ptr<GenResult>& res, AsyncJob& j) {
    const std::string& err = j.core->error;
    if (!err.empty() && ev::isFunction(j.onError.get())) {
        ev::Persistent msg(ev::fromUtf8("generate: " + err));
        callRooted(j.onError, {&msg});
        return;
    }
    if (!ev::isFunction(j.onDone.get())) {
        if (!err.empty()) std::fprintf(stderr, "bro.lm.generate: %s\n", err.c_str());
        return;
    }
    ev::Persistent ids(makeInt32Array(res->ids.data(), res->ids.size()));
    ev::Persistent info(ev::createObject());
    info.set(ev::setProperty(info.get(), "cancelled", ev::fromBool(j.core->cancelled())));
    if (!err.empty()) {
        ev::Persistent e(ev::fromUtf8(err));
        info.set(ev::setProperty(info.get(), "error", e.get()));
    }
    callRooted(j.onDone, {&ids, &info});
}

// The VLM families share one driver shape: set_generation + generate_tokens.
template <class VLM, class ImageInput, class HostT>
static Value launchVlmGenerate(HostT* w, std::shared_ptr<AsyncJob> job, std::string prompt,
                               brolm::qwen::GenerateOptions opts, std::vector<VlmImage> images) {
    // opts.grammar: the VLM copies it here on the JS thread (the caller holds
    // the model's busy claim, so no decode is running), and the worker never
    // reads the JS Grammar object. nullptr clears a previous call's grammar.
    w->vlm->set_grammar(opts.grammar);
    opts.grammar = nullptr;
    auto res = std::make_shared<GenResult>();
    job->finish = [res](AsyncJob& j) { finishGenerate(res, j); };
    return launchAsyncJob(job, [w, res, prompt = std::move(prompt), opts,
                                imgs = std::move(images)](AsyncCore& core) {
        brotensor::DeviceScope scope(w->device);
        w->vlm->set_generation(opts.max_new_tokens, opts.sampling.temperature,
                               opts.sampling.top_k, opts.sampling.top_p,
                               opts.sampling.seed, opts.sampling.min_p,
                               opts.sampling.repetition_penalty,
                               opts.sampling.frequency_penalty,
                               opts.sampling.presence_penalty,
                               opts.stop_on_eos);
        std::vector<ImageInput> inputs;
        inputs.reserve(imgs.size());
        for (auto& im : imgs) inputs.push_back(ImageInput{ im.chw.data(), im.H, im.W });
        w->vlm->generate_tokens(prompt, inputs, [&core, &res](int tok) -> bool {
            res->ids.push_back(static_cast<int32_t>(tok));
            core.pushToken(static_cast<int32_t>(tok));
            return !core.cancelled();
        });
    });
}

Value js_lm_generate(Value, std::span<const Value> a) {
    if (a.size() < 2)
        return ev::throwTypeError("generate(model, prompt, opts?): model and prompt required");

    const bool haveOpts = a.size() >= 3 && ev::isObject(a[2]);
    auto job = newAsyncJob();
    job->keep.set(a[0]);  // the model stays alive until the job is delivered
    if (haveOpts) {
        job->onToken.set(ev::getProperty(a[2], "onToken"));
        job->onDone.set(ev::getProperty(a[2], "onDone"));
        job->onError.set(ev::getProperty(a[2], "onError"));
    }
    const Value noOpts = ev::undefined();
    const Value& optsVal = haveOpts ? a[2] : noOpts;

    if (auto* w = hostLMModelOf(a[0])) {
        std::vector<int32_t> prompt = readInt32Array(a[1]);
        if (prompt.empty())
            return ev::throwTypeError("generate: promptIds must be non-empty for LMModel");
        brolm::qwen::GenerateOptions opts = parseGenerateOptions(optsVal);
        const int eos_id = haveOpts ? eosFor(*w, a[2]) : w->defaultEos;
        // The worker decodes against its own copy of the grammar, taken here
        // on the JS thread, so the JS Grammar object is never read off-thread.
        std::shared_ptr<brolm::Grammar> grammar;
        if (opts.grammar) {
            grammar = std::make_shared<brolm::Grammar>(opts.grammar->clone());
            opts.grammar = grammar.get();
        }

        BusyClaim claim(w->generating);
        if (!claim.ok()) return ev::throwError(std::string("generate: ") + kBusyMessage);
        job->busy = claim.detach();

        auto res = std::make_shared<GenResult>();
        job->finish = [res](AsyncJob& j) { finishGenerate(res, j); };
        return launchAsyncJob(job, [w, res, grammar, prompt = std::move(prompt), eos_id, opts](AsyncCore& core) {
            brotensor::DeviceScope scope(w->device);
            res->ids = runDecode(*w, prompt, eos_id, opts, [&core](int32_t tok) -> bool {
                core.pushToken(tok);
                return !core.cancelled();
            }, &core.handle->cancelled);
        });
    }

    auto readVlmArgs = [&](const char* family, std::string& prompt, brolm::qwen::GenerateOptions& opts,
                           std::vector<VlmImage>& images, std::string& err) -> bool {
        if (!ev::isString(a[1])) {
            err = std::string("generate: prompt must be a string for ") + family;
            return false;
        }
        prompt = ev::toUtf8(a[1]);
        opts = parseGenerateOptions(optsVal);
        if (opts.max_new_tokens <= 0) {
            err = "generate: opts.maxNewTokens must be > 0";
            return false;
        }
        return readVlmImages(optsVal, images, err);
    };

    if (auto* q35 = hostQwen35ModelOf(a[0])) {
        std::string prompt, err;
        brolm::qwen::GenerateOptions opts;
        std::vector<VlmImage> images;
        if (!readVlmArgs("Qwen35Model", prompt, opts, images, err)) return ev::throwTypeError(err);
        BusyClaim claim(q35->generating);
        if (!claim.ok()) return ev::throwError(std::string("generate: ") + kBusyMessage);
        job->busy = claim.detach();
        return launchVlmGenerate<brolm::qwen35::VLM, brolm::qwen35::ImageInput>(
            q35, job, std::move(prompt), opts, std::move(images));
    }

    if (auto* qvl = hostQwen3VLModelOf(a[0])) {
        std::string prompt, err;
        brolm::qwen::GenerateOptions opts;
        std::vector<VlmImage> images;
        if (!readVlmArgs("Qwen3VLModel", prompt, opts, images, err)) return ev::throwTypeError(err);
        BusyClaim claim(qvl->generating);
        if (!claim.ok()) return ev::throwError(std::string("generate: ") + kBusyMessage);
        job->busy = claim.detach();
        return launchVlmGenerate<brolm::qwen3vl::VLM, brolm::qwen3vl::ImageInput>(
            qvl, job, std::move(prompt), opts, std::move(images));
    }

    return ev::throwTypeError("generate: arg 0 must be LMModel, Qwen35Model, or Qwen3VLModel");
}

} // namespace brolm::api
