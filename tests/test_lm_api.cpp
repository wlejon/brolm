// Standalone test for brolm_api, the bronze-runtime JavaScript binding.
// No bro engine, no window, no weights: a fresh bronze realm, installLM(),
// then the mount points and the argument validation of every loader checked
// from both the embed API and a compiled script. A loader given nothing, a
// non-string, or a path that does not exist must throw; never null, never a
// silent success.

#include "api/api.h"
#include "host_lm_internal.h"
#include "embed/embed.h"
#include "eval/eval.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

namespace ev = bronze::embed;
using Value = bronze::Value;

#define TEST_CHECK(cond) do { \
    if (!(cond)) { \
        std::cerr << "CHECK FAILED: " #cond " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while (0)

static std::string errorName(Value thrown) {
    return ev::isObject(thrown) ? ev::toUtf8(ev::getProperty(thrown, "name")) : std::string();
}

static std::string errorMessage(Value thrown) {
    return ev::isObject(thrown) ? ev::toUtf8(ev::getProperty(thrown, "message")) : ev::toUtf8(thrown);
}

static const char* kLoaders[] = {
    "loadQwen", "loadMistral", "loadGemma2", "loadModel", "loadQwen35",
    "loadQwen3VL", "loadNllb", "loadTokenizer", "loadLlama3Tokenizer",
    "loadClip", "loadClipModel", "loadT5", "loadLaya", "loadModernBert",
};

static const char* kClasses[] = {
    "QwenTokenizer", "MistralTokenizer", "GemmaTokenizer", "Llama3Tokenizer",
    "LMModel", "Qwen35Model", "Qwen3VLModel", "ClipModel", "NllbModel",
    "T5Model", "LayaModel", "AsyncHandle", "Grammar", "ModernBertModel",
};

static Value lmNamespace() {
    ev::GlobalValue broG = ev::globalValue("bro");
    TEST_CHECK(broG.found);
    TEST_CHECK(ev::isObject(broG.value));
    Value lm = ev::getProperty(broG.value, "lm");
    TEST_CHECK(ev::isObject(lm));
    return lm;
}

// Every Value the C++ half of this test holds across an allocating embed
// call (getProperty included: it interns its key) rides in a Persistent, so
// the test itself keeps the embed.h GC contract and the gcstress variant
// checks the binding rather than the harness.
static void test_mounts() {
    std::cout << "[1/3] mount points..." << std::endl;

    ev::Persistent lm(lmNamespace());

    for (const char* name : kLoaders) {
        if (!ev::isFunction(ev::getProperty(lm.get(), name))) {
            std::cerr << "missing bro.lm." << name << std::endl;
            std::exit(1);
        }
    }
    TEST_CHECK(ev::isFunction(ev::getProperty(lm.get(), "init")));
    TEST_CHECK(ev::isFunction(ev::getProperty(lm.get(), "generate")));
    TEST_CHECK(ev::isFunction(ev::getProperty(lm.get(), "tick")));

    // Each class constructor sits on bro.lm and on globalThis, and is the
    // same function in both places.
    for (const char* name : kClasses) {
        ev::Persistent onNs(ev::getProperty(lm.get(), name));
        ev::GlobalValue onGlobal = ev::globalValue(name);
        if (!ev::isFunction(onNs.get()) || !onGlobal.found || !ev::isFunction(onGlobal.value)) {
            std::cerr << "missing constructor " << name << std::endl;
            std::exit(1);
        }
    }

    ev::Persistent q35Proto(ev::getProperty(ev::globalValue("Qwen35Model").value, "prototype"));
    TEST_CHECK(ev::isFunction(ev::getProperty(q35Proto.get(), "generateStream")));

    ev::Persistent qvlProto(ev::getProperty(ev::globalValue("Qwen3VLModel").value, "prototype"));
    TEST_CHECK(ev::isFunction(ev::getProperty(qvlProto.get(), "generateStream")));

    // tick() with nothing pending is a no-op.
    ev::Persistent tickFn(ev::getProperty(lm.get(), "tick"));
    ev::CallResult tick = ev::call(tickFn.get(), lm.get(), {});
    TEST_CHECK(!tick.thrown);
}

static void expectThrows(const ev::Persistent& lm, const char* fnName,
                         const std::vector<ev::Persistent>& rootedArgs, const char* expectName) {
    ev::Persistent fn(ev::getProperty(lm.get(), fnName));
    std::vector<Value> args;  // slot reads, after the last allocation
    for (const auto& p : rootedArgs) args.push_back(p.get());
    ev::CallResult r = ev::call(fn.get(), lm.get(), std::span<const Value>(args.data(), args.size()));
    if (!r.thrown) {
        std::cerr << "bro.lm." << fnName << " did not throw (returned "
                  << (ev::isNull(r.value) ? "null" : ev::isUndefined(r.value) ? "undefined" : "a value")
                  << ")" << std::endl;
        std::exit(1);
    }
    std::string name = errorName(r.value);
    if (expectName && name != expectName) {
        std::cerr << "bro.lm." << fnName << " threw " << name << " (" << errorMessage(r.value)
                  << "), expected " << expectName << std::endl;
        std::exit(1);
    }
}

static void test_loader_validation() {
    std::cout << "[2/3] loaders reject bad arguments..." << std::endl;

    ev::Persistent lm(lmNamespace());

    // No arguments at all: every loader throws a TypeError.
    for (const char* name : kLoaders) {
        expectThrows(lm, name, {}, "TypeError");
    }

    // A number where a path string (or options object) belongs.
    for (const char* name : kLoaders) {
        std::vector<ev::Persistent> args;
        args.emplace_back(ev::fromDouble(42.0));
        expectThrows(lm, name, args, "TypeError");
    }

    // A path that does not exist: the loader reaches the file and the
    // failure surfaces as a thrown Error carrying the loader's name.
    ev::Persistent missing(ev::fromUtf8("./does-not-exist/model.gguf"));
    auto callWithMissing = [&](const char* name) {
        ev::Persistent fn(ev::getProperty(lm.get(), name));
        TEST_CHECK(ev::isFunction(fn.get()));
        const Value arg = missing.get();
        return ev::call(fn.get(), lm.get(), std::span<const Value>(&arg, 1));
    };
    for (const char* name : {"loadQwen", "loadTokenizer", "loadLlama3Tokenizer", "loadLaya", "loadModernBert"}) {
        // (loadTokenizer / loadLlama3Tokenizer: construction from a
        // nonexistent tokenizer.json.)
        ev::CallResult r = callWithMissing(name);
        TEST_CHECK(r.thrown);
        TEST_CHECK(errorMessage(r.value).find(name) != std::string::npos);
    }
    // loadLayaAsync never throws: a bad path is a rejected promise.
    {
        ev::CallResult r = callWithMissing("loadLayaAsync");
        TEST_CHECK(!r.thrown);
        TEST_CHECK(ev::isPromise(r.value));
    }

    // loadTokenizer({}) : an options object with none of the doc'd keys.
    {
        std::vector<ev::Persistent> args;
        args.emplace_back(ev::createObject());
        expectThrows(lm, "loadTokenizer", args, "TypeError");
    }

    // generate() without a model, and with something that is not a model.
    expectThrows(lm, "generate", {}, "TypeError");
    {
        std::vector<ev::Persistent> args;
        args.emplace_back(ev::createObject());
        args.emplace_back(ev::fromUtf8("hello"));
        expectThrows(lm, "generate", args, "TypeError");
    }

    // The handle classes are not bare-constructible: they only come out of
    // a loader.
    for (const char* name : kClasses) {
        Value ctor = ev::globalValue(name).value;
        ev::CallResult r = ev::construct(ctor, {});
        if (!r.thrown || errorName(r.value) != "TypeError") {
            std::cerr << "new " << name << "() should throw TypeError" << std::endl;
            std::exit(1);
        }
    }
}

static void test_script() {
    std::cout << "[3/3] validation via bronze eval..." << std::endl;

    const char* script = R"JS(
        (function() {
            const lm = bro.lm;
            if (typeof lm !== "object") throw new Error("bro.lm missing");
            if (globalThis.QwenTokenizer !== lm.QwenTokenizer) throw new Error("QwenTokenizer global is not bro.lm.QwenTokenizer");
            if (globalThis.LMModel !== lm.LMModel) throw new Error("LMModel global is not bro.lm.LMModel");
            if (globalThis.LayaModel !== lm.LayaModel) throw new Error("LayaModel global is not bro.lm.LayaModel");

            function expectType(fn, name) {
                let caught = null;
                try { fn(); } catch (e) { caught = e; }
                if (caught === null) throw new Error(name + " did not throw");
                if (!(caught instanceof TypeError)) throw new Error(name + " threw " + caught.name + ": " + caught.message);
            }
            function expectThrows(fn, name) {
                let caught = null;
                try { fn(); } catch (e) { caught = e; }
                if (caught === null) throw new Error(name + " did not throw");
                if (!(caught instanceof Error)) throw new Error(name + " threw a non-Error: " + caught);
                return caught;
            }

            expectType(() => lm.loadQwen(), "loadQwen()");
            expectType(() => lm.loadQwen(null), "loadQwen(null)");
            expectType(() => lm.loadQwen({ path: "x.gguf" }), "loadQwen(object)");
            expectType(() => lm.loadMistral("m.gguf"), "loadMistral without tokenizerPath");
            expectType(() => lm.loadGemma2(7), "loadGemma2(number)");
            expectType(() => lm.loadQwen35(), "loadQwen35()");
            expectType(() => lm.loadQwen3VL(undefined), "loadQwen3VL(undefined)");
            expectType(() => lm.loadTokenizer(), "loadTokenizer()");
            expectType(() => lm.loadTokenizer({ vocabPath: "v.json" }), "loadTokenizer missing mergesPath");
            expectType(() => lm.loadLaya(), "loadLaya()");
            expectType(() => lm.loadLaya(null), "loadLaya(null)");
            expectType(() => lm.generate(), "generate()");
            expectType(() => new QwenTokenizer(), "new QwenTokenizer()");
            expectType(() => new LMModel(), "new LMModel()");
            expectType(() => new LayaModel(), "new LayaModel()");

            const e1 = expectThrows(() => lm.loadQwen("./does-not-exist/model.gguf"), "loadQwen(missing)");
            if (e1.message.indexOf("loadQwen") < 0) throw new Error("loadQwen(missing) message: " + e1.message);
            const e2 = expectThrows(() => lm.loadTokenizer("./does-not-exist/tokenizer.json"), "loadTokenizer(missing)");
            if (e2.message.indexOf("loadTokenizer") < 0) throw new Error("loadTokenizer(missing) message: " + e2.message);
            const e3 = expectThrows(() => lm.loadTokenizer({ vocabPath: "./nope/vocab.json", mergesPath: "./nope/merges.txt" }), "loadTokenizer(missing pair)");
            if (e3.message.indexOf("loadTokenizer") < 0) throw new Error("loadTokenizer(missing pair) message: " + e3.message);
            const e4 = expectThrows(() => lm.loadLaya("./does-not-exist"), "loadLaya(missing)");
            if (e4.message.indexOf("loadLaya") < 0) throw new Error("loadLaya(missing) message: " + e4.message);

            if (lm.tick() !== undefined) throw new Error("tick() should return undefined");
            return "SUCCESS";
        })()
    )JS";

    ev::CallResult res = bronze::eval::evalScript(script);
    if (res.thrown) {
        std::cerr << "eval threw: " << ev::toUtf8(res.value) << std::endl;
        std::exit(1);
    }
    TEST_CHECK(ev::toUtf8(res.value) == "SUCCESS");
}

static void test_async_handle() {
    std::cout << "[4/4] async handle lifecycle & wait..." << std::endl;

    auto handle = std::make_shared<brolm::api::HostAsyncHandle>();
    ev::Persistent hVal(brolm::api::makeAsyncHandleValue(handle));
    TEST_CHECK(ev::isObject(hVal.get()));

    // cancel()
    ev::Persistent cancelFn(ev::getProperty(hVal.get(), "cancel"));
    TEST_CHECK(ev::isFunction(cancelFn.get()));
    TEST_CHECK(!handle->cancelled.load());
    ev::call(cancelFn.get(), hVal.get(), {});
    TEST_CHECK(handle->cancelled.load());
    TEST_CHECK(ev::toBool(ev::getProperty(hVal.get(), "cancelled")));
    TEST_CHECK(!ev::toBool(ev::getProperty(hVal.get(), "done")));

    // wait() blocks until finished
    ev::Persistent waitFn(ev::getProperty(hVal.get(), "wait"));
    TEST_CHECK(ev::isFunction(waitFn.get()));
    TEST_CHECK(!handle->finished.load());

    std::thread backgroundWorker([handle]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        handle->finished.store(true, std::memory_order_release);
    });

    ev::call(waitFn.get(), hVal.get(), {});
    TEST_CHECK(handle->finished.load());
    if (backgroundWorker.joinable()) backgroundWorker.join();
    TEST_CHECK(ev::toBool(ev::getProperty(hVal.get(), "done")));
}

// bro.lm.Grammar (CPU only, no weights) and the class brand: a method called
// on a handle of another class is a TypeError, never a payload cast.
static void test_grammar() {
    std::cout << "[grammar] Grammar factories, state, brands..." << std::endl;

    const char* script = R"JS(
        (function() {
            const G = bro.lm.Grammar;
            if (typeof G !== "function" || G !== globalThis.Grammar) throw new Error("bro.lm.Grammar missing");
            function expectType(fn, name) {
                let caught = null;
                try { fn(); } catch (e) { caught = e; }
                if (!(caught instanceof TypeError)) throw new Error(name + ": expected TypeError, got " + caught);
            }

            const re = G.regex("[0-9]+");
            if (!(re instanceof G)) throw new Error("regex() is not a Grammar");
            if (!re.canAccept("12") || re.canAccept("x")) throw new Error("regex canAccept");
            if (re.isAccepted()) throw new Error("empty regex state must not be accepted");
            if (!re.accept("4") || !re.isAccepted()) throw new Error("regex accept");
            const copy = re.clone();
            re.reset();
            if (re.isAccepted() || !copy.isAccepted()) throw new Error("reset / clone");

            const r3 = G.regex("[0-9]{3}");
            if (!r3.accept("0") || r3.isAccepted()) throw new Error("regex {3}: accepted after one digit");
            if (!r3.accept("12") || !r3.isAccepted() || r3.canAccept("3")) throw new Error("regex {3}: three digits");

            const ch = G.choice(["yes", "no"]);
            if (!ch.canAccept("ye") || ch.canAccept("maybe")) throw new Error("choice");
            const ex = G.exact("hi");
            if (!ex.accept("hi") || !ex.isAccepted()) throw new Error("exact");
            const js = G.jsonObject();
            if (!js.canAccept("{") || js.canAccept("[")) throw new Error("jsonObject");
            if (!G.jsonBoolean().canAccept("tru")) throw new Error("jsonBoolean");
            if (!G.jsonInteger().canAccept("-3")) throw new Error("jsonInteger");
            const sch = G.jsonSchema({ type: "object", properties: { a: { type: "integer" } }, required: ["a"] });
            if (!sch.canAccept("{") || sch.canAccept("7")) throw new Error("jsonSchema(object)");
            const bnf = G.bnf('root ::= "a" | "b"');
            if (!bnf.canAccept("a") || bnf.canAccept("c")) throw new Error("bnf");

            expectType(() => G.regex(), "regex()");
            expectType(() => G.choice([]), "choice([])");
            expectType(() => G.jsonSchema(5), "jsonSchema(5)");
            expectType(() => new G(), "new Grammar()");

            // Brands: a Grammar is not an LMModel / AsyncHandle / tokenizer.
            expectType(() => bro.lm.LMModel.prototype.generate.call(re, [1, 2]), "LMModel.generate on a Grammar");
            expectType(() => bro.lm.QwenTokenizer.prototype.encode.call(re, "x"), "encode on a Grammar");
            expectType(() => bro.lm.Grammar.prototype.accept.call({}, "x"), "accept on a plain object");
            expectType(() => bro.lm.generate(re, [1, 2]), "bro.lm.generate(grammar)");
            if (bro.lm.AsyncHandle.prototype.cancel.call(re) !== undefined) throw new Error("cancel on a Grammar");
            return "SUCCESS";
        })()
    )JS";
    ev::CallResult res = bronze::eval::evalScript(script);
    if (res.thrown) {
        std::cerr << "grammar eval threw: " << errorMessage(res.value) << std::endl;
        std::exit(1);
    }
    TEST_CHECK(ev::toUtf8(res.value) == "SUCCESS");
}

// GemmaTokenizer's control-token surface from JS, over a synthetic
// tokenizer.json (no weights): isSpecial(id) and decode(ids, skipSpecial).
static void test_gemma_tokenizer_binding() {
    std::cout << "[gemma] GemmaTokenizer isSpecial / decode(ids, skipSpecial)..." << std::endl;

    // vocab: 0:<pad> 1:<eos> 2:<bos> 3:<unk> 4:▁ 5:a 6:c 7:t 8:ca 9:cat
    //        10:▁cat 11:<start_of_turn>(special) 12:<unused0>(not special)
    const auto path = std::filesystem::temp_directory_path() / "brolm_api_gemma_tokenizer.json";
    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        const std::string M = "\xE2\x96\x81";  // U+2581 metaspace
        f << "{\"added_tokens\":["
          << "{\"id\":0,\"content\":\"<pad>\",\"special\":true},"
          << "{\"id\":1,\"content\":\"<eos>\",\"special\":true},"
          << "{\"id\":2,\"content\":\"<bos>\",\"special\":true},"
          << "{\"id\":3,\"content\":\"<unk>\",\"special\":true},"
          << "{\"id\":11,\"content\":\"<start_of_turn>\",\"special\":true},"
          << "{\"id\":12,\"content\":\"<unused0>\",\"special\":false}],"
          << "\"model\":{\"type\":\"BPE\",\"unk_token\":\"<unk>\",\"byte_fallback\":true,"
          << "\"fuse_unk\":false,\"vocab\":{"
          << "\"<pad>\":0,\"<eos>\":1,\"<bos>\":2,\"<unk>\":3,"
          << "\"" << M << "\":4,\"a\":5,\"c\":6,\"t\":7,\"ca\":8,\"cat\":9,"
          << "\"" << M << "cat\":10,\"<start_of_turn>\":11,\"<unused0>\":12},"
          << "\"merges\":[\"c a\",\"ca t\",\"" << M << " cat\"]}}";
    }
    auto tok = std::make_shared<brolm::gemma::Tokenizer>(brolm::gemma::Tokenizer::load(path.string()));
    std::filesystem::remove(path);
    ev::Persistent tokVal(brolm::api::makeGemmaTokenizerValue(tok));
    ev::registerGlobal("__gemmaTok", tokVal.get());

    const char* script = R"JS(
        (function() {
            const t = globalThis.__gemmaTok;
            if (!(t instanceof bro.lm.GemmaTokenizer)) throw new Error("not a GemmaTokenizer");
            if (!t.isSpecial(t.bosId) || !t.isSpecial(t.eosId) || !t.isSpecial(11))
                throw new Error("control tokens must be special");
            if (t.isSpecial(5) || t.isSpecial(10)) throw new Error("text pieces are not special");
            if (t.isSpecial(12)) throw new Error("an added token marked special:false is not special");
            if (t.isSpecial(9999) || t.isSpecial(-1) || t.isSpecial(1.5) || t.isSpecial(NaN))
                throw new Error("ids outside the vocabulary are not special");
            let caught = null;
            try { t.isSpecial("1"); } catch (e) { caught = e; }
            if (!(caught instanceof TypeError)) throw new Error("isSpecial(string) must throw TypeError");

            const ids = new Int32Array([2, 11, 5, 10, 12, 1]);
            const full = t.decode(ids);
            if (full.indexOf("<bos>") < 0 || full.indexOf("<start_of_turn>") < 0 || full.indexOf("<eos>") < 0)
                throw new Error("decode(ids) keeps control tokens, got " + JSON.stringify(full));
            if (t.decode(ids, false) !== full) throw new Error("decode(ids, false) is decode(ids)");
            const skipped = t.decode(ids, true);
            if (skipped !== "a cat<unused0>")
                throw new Error("decode(ids, true) drops only control tokens, got " + JSON.stringify(skipped));
            return "SUCCESS";
        })()
    )JS";
    ev::CallResult res = bronze::eval::evalScript(script);
    if (res.thrown) {
        std::cerr << "gemma tokenizer eval threw: " << errorMessage(res.value) << std::endl;
        std::exit(1);
    }
    TEST_CHECK(ev::toUtf8(res.value) == "SUCCESS");
}

// Weights-gated: the generation surface against Qwen3-0.6B (and NLLB /
// Qwen3.5 when present). $BROLM_QWEN3_GGUF, else <repo>/weights/Qwen3-0.6B-GGUF.
static std::string findWeights(const char* env, const std::string& rel) {
    if (const char* e = std::getenv(env); e && *e && std::filesystem::exists(e)) return e;
    const std::filesystem::path p = std::filesystem::path(BROLM_SOURCE_DIR) / rel;
    return std::filesystem::exists(p) ? p.generic_string() : std::string();
}

static bool evalUntilDone(const std::string& launch, const char* poll, int timeoutS, std::string& state) {
    ev::CallResult res = bronze::eval::evalScript(launch);
    if (res.thrown) {
        state = "launch threw: " + errorMessage(res.value);
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        brolm::api::tickLMAsync();
        ev::drainMicrotasks();
        ev::CallResult s = bronze::eval::evalScript(poll);
        state = s.thrown ? "poll threw: " + errorMessage(s.value) : ev::toUtf8(s.value);
        if (state != "WAIT") return state == "DONE";
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(timeoutS)) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

static void test_generation_weights() {
    std::cout << "[weights] generation surface..." << std::endl;

    std::string gguf = findWeights("BROLM_QWEN3_GGUF", "weights/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf");
    if (gguf.empty()) gguf = findWeights("BROLM_QWEN3_GGUF", "weights/Qwen3-0.6B-GGUF/Qwen3-0.6B-BF16.gguf");
    if (gguf.empty()) {
        std::cout << "  Skipping: no Qwen3-0.6B GGUF (set BROLM_QWEN3_GGUF)" << std::endl;
    } else {
        std::string script = R"JS(
            (function() {
                const { model, tokenizer } = bro.lm.loadQwen(")JS" + gguf + R"JS(");
                if (model.eosId !== tokenizer.imEndId) throw new Error("model.eosId " + model.eosId);
                if (!(tokenizer.mergeCount > 0)) throw new Error("mergeCount");
                const prompt = tokenizer.encode(tokenizer.applyChatTemplate(
                    [{ role: "user", content: "Pick a number between 100 and 999. /no_think" }], true));

                // Grammar-constrained: exactly three digits, then the grammar
                // lets end-of-turn through.
                const g = bro.lm.Grammar.regex("[0-9]{3}");
                const ids = model.generate(prompt, { maxNewTokens: 12, grammar: g, sampling: { temperature: 0 } });
                const text = tokenizer.decode(ids).replace("<|im_end|>", "");
                if (!/^[0-9]{3}$/.test(text)) throw new Error("grammar output: " + JSON.stringify(text));
                if (g.isAccepted()) throw new Error("generation advanced the caller's Grammar");

                // Penalties reach the sampler: a strong repetition penalty
                // changes a greedy run that repeats.
                const plain = model.generate(prompt, { maxNewTokens: 24, sampling: { temperature: 0 } });
                const pen = model.generate(prompt, { maxNewTokens: 24, sampling: { temperature: 0, repetitionPenalty: 5 } });
                if (plain.length === pen.length && plain.every((v, i) => v === pen[i]))
                    throw new Error("repetitionPenalty had no effect");

                // Background generation claims the model: a sync call throws.
                const st = globalThis.__gen = { tokens: 0, done: null, busyThrew: false };
                const h = bro.lm.generate(model, prompt, {
                    maxNewTokens: 8, sampling: { temperature: 0 },
                    onToken: () => { st.tokens++; },
                    onDone: (all, info) => { st.done = { n: all.length, info }; },
                });
                try { model.generate(prompt, { maxNewTokens: 1 }); } catch (e) { st.busyThrew = /in flight/.test(e.message); }
                h.wait();
                if (!h.done || !st.done || st.done.info.cancelled !== false) throw new Error("onDone: " + JSON.stringify(st.done));
                if (st.tokens !== st.done.n || st.done.n < 1) throw new Error("onToken count " + st.tokens + " vs " + st.done.n);
                if (!st.busyThrew) throw new Error("sync generate during a job did not throw busy");
                if (model.busy) throw new Error("model still busy after the job");
                model.generate(prompt, { maxNewTokens: 1 });  // the claim is released

                // Async load: onReady on a later tick.
                const ld = globalThis.__load = { pair: null, err: null };
                const lh = bro.lm.loadQwen(")JS" + gguf + R"JS(", {
                    onReady: (p) => { ld.pair = p; }, onError: (e) => { ld.err = e; } });
                if (!(lh instanceof bro.lm.AsyncHandle)) throw new Error("async loadQwen did not return an AsyncHandle");
                lh.wait();
                if (ld.err || !ld.pair || !(ld.pair.model instanceof bro.lm.LMModel)) throw new Error("onReady: " + ld.err);
                return "SUCCESS";
            })()
        )JS";
        ev::CallResult res = bronze::eval::evalScript(script);
        if (res.thrown) {
            std::cerr << "qwen weights eval threw: " << errorMessage(res.value) << std::endl;
            std::exit(1);
        }
        TEST_CHECK(ev::toUtf8(res.value) == "SUCCESS");
        std::cout << "  qwen3: grammar, penalties, background generate, async load OK" << std::endl;
    }

    const std::string nllb = findWeights("BROLM_NLLB_DIR", "weights/nllb-200-distilled-600M");
    if (nllb.empty()) {
        std::cout << "  Skipping NLLB: no weights/nllb-200-distilled-600M" << std::endl;
    } else {
        std::string launch = R"JS(
            (function() {
                const m = bro.lm.loadNllb(")JS" + nllb + R"JS(");
                const st = globalThis.__nllb = { out: null, err: null, sync: m.translate("Hello, world!", "eng_Latn", "fra_Latn") };
                m.translate("Good morning.", "eng_Latn", "fra_Latn",
                            { numBeams: 2, onDone: (t) => { st.out = t; }, onError: (e) => { st.err = e; } });
                return "LAUNCHED";
            })()
        )JS";
        std::string state;
        const bool ok = evalUntilDone(launch,
            "(function(){ const s = globalThis.__nllb; return s.err ? 'BAD ' + s.err : (s.out ? 'DONE' : 'WAIT'); })()",
            120, state);
        if (!ok) {
            std::cerr << "nllb async: " << state << std::endl;
            std::exit(1);
        }
        std::cout << "  nllb: sync + async translate OK" << std::endl;
    }

    const std::string q35 = findWeights("BROLM_QWEN35_DIR", "weights/Qwen3.5-0.8B");
    if (q35.empty()) {
        std::cout << "  Skipping Qwen3.5: no weights/Qwen3.5-0.8B" << std::endl;
    } else {
        std::string launch = R"JS(
            (function() {
                const m = bro.lm.loadQwen35(")JS" + q35 + R"JS(", { maxSeqLen: 512 });
                const prompt = "<|im_start|>user\nIs the sky blue? Answer yes or no.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

                // Grammar-constrained VLM decode (sync): the reply is exactly
                // one of the choices, whatever the model would have said.
                const g = bro.lm.Grammar.choice(["maybe", "certainly not"]);
                const ids = m.generate(prompt, { maxNewTokens: 8, grammar: g, sampling: { temperature: 0 } });
                const text = m.decode(ids);
                if (text !== "maybe" && text !== "certainly not") throw new Error("vlm grammar output: " + JSON.stringify(text));
                if (g.isAccepted()) throw new Error("generation advanced the caller's Grammar");
                // No grammar on the next call: the previous one does not linger.
                const free = m.decode(m.generate(prompt, { maxNewTokens: 4, sampling: { temperature: 0 } }));
                if (free === text) throw new Error("grammar lingered into an unconstrained call: " + JSON.stringify(free));

                // The same through bro.lm.generate on a worker.
                const st = globalThis.__q35 = { ids: null, err: null, text: null };
                bro.lm.generate(m, prompt, {
                    maxNewTokens: 8, sampling: { temperature: 0 }, grammar: bro.lm.Grammar.regex("[0-9]{2}"),
                    onDone: (ids, info) => { st.ids = ids; st.info = info; st.text = m.decode(ids); },
                    onError: (e) => { st.err = e; },
                });
                return "LAUNCHED";
            })()
        )JS";
        std::string state;
        const bool ok = evalUntilDone(launch,
            "(function(){ const s = globalThis.__q35; return s.err ? 'BAD ' + s.err : (s.ids ? (/^[0-9]{2}$/.test(s.text) ? 'DONE' : 'BAD ' + JSON.stringify(s.text)) : 'WAIT'); })()",
            300, state);
        if (!ok) {
            std::cerr << "qwen35 async: " << state << std::endl;
            std::exit(1);
        }
        std::cout << "  qwen3.5: grammar (sync + bro.lm.generate) OK" << std::endl;
    }
}

// Weights-gated: bro.lm.loadModernBert over a Laya checkpoint's encoder,
// against Hugging Face's ModernBertModel (tests/ref/gen_modernbert_encode.py
// writes the fixture): the same [CLS] ... [SEP] ids, and hidden states that
// match row for row. LAYA_MODEL_DIR, else D:/projects/laya.
static void test_modernbert() {
    std::cout << "[weights] ModernBERT encoder..." << std::endl;

    const char* env_dir = std::getenv("LAYA_MODEL_DIR");
    std::string model_dir = (env_dir && env_dir[0]) ? env_dir : "D:/projects/laya";
    std::replace(model_dir.begin(), model_dir.end(), '\\', '/');
    const std::filesystem::path fixture = std::filesystem::path(BROLM_SOURCE_DIR) / "tests/ref/modernbert_encode.json";
    if (!std::filesystem::exists(model_dir + "/model.safetensors") ||
        !std::filesystem::exists(model_dir + "/encoder/config.json")) {
        std::cout << "  Skipping: no Laya checkpoint at " << model_dir << std::endl;
        return;
    }
    std::ifstream f(fixture, std::ios::binary);
    TEST_CHECK(f.good());
    const std::string ref((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    // Handed over as a string for JSON.parse: thousands of numbers as one
    // array literal overflow the compiler's stack.
    std::string refLit;
    for (char c : ref) {
        if (c == '\\' || c == '\'') refLit += '\\';
        if (c == '\n' || c == '\r') continue;
        refLit += c;
    }
    std::string script = "globalThis.__mbRef = JSON.parse('" + refLit + "');\n" + R"JS(
        (function() {
            const m = bro.lm.loadModernBert(")JS" + model_dir + R"JS(");
            if (!(m instanceof bro.lm.ModernBertModel)) throw new Error("not a ModernBertModel");
            const ref = globalThis.__mbRef;
            if (m.hiddenSize !== ref.hidden_size || m.numLayers !== 28 || m.maxLength !== 8192)
                throw new Error("accessors " + m.hiddenSize + " " + m.numLayers + " " + m.maxLength);
            const D = m.hiddenSize;
            const cmp = (got, want, what) => {
                let dot = 0, ng = 0, nw = 0, maxAbs = 0;
                for (let i = 0; i < want.length; ++i) {
                    dot += got[i] * want[i]; ng += got[i] * got[i]; nw += want[i] * want[i];
                    maxAbs = Math.max(maxAbs, Math.abs(got[i] - want[i]));
                }
                const cos = dot / Math.sqrt(ng * nw);
                if (!(cos > 0.999) || !(maxAbs < 0.08)) throw new Error(what + ": cos " + cos + " maxAbs " + maxAbs);
                return maxAbs;
            };
            let worst = 0;
            for (const c of ref.cases) {
                const toks = m.tokenize(c.text);
                if (toks.length !== c.ids.length || toks.some((v, i) => v !== c.ids[i]))
                    throw new Error("ids differ for " + JSON.stringify(c.text.slice(0, 20)) + ": " + Array.from(toks).slice(0, 8));
                const r = m.encode(c.text, { pooling: "mean" });
                if (r.length !== c.ids.length || r.dim !== D || r.data.length !== r.length * D)
                    throw new Error("shape " + r.length + "x" + r.dim);
                const L = r.length;
                worst = Math.max(worst, cmp(r.data.subarray(0, D), c.first, "row 0"));
                worst = Math.max(worst, cmp(r.data.subarray((L - 1) * D, L * D), c.last, "row L-1"));
                worst = Math.max(worst, cmp(r.pooled, c.mean, "mean"));
                // Ids in, the same states out; cls pooling is row 0.
                const byIds = m.encode(toks, { pooling: "cls" });
                for (let i = 0; i < D; ++i)
                    if (byIds.pooled[i] !== r.data[i]) throw new Error("ids path / cls pooling differ at " + i);
                // Ids in a Float32Array are values, not reinterpreted bits.
                const fIds = m.encode(new Float32Array(toks)).ids;
                if (fIds.some((v, i) => v !== toks[i])) throw new Error("Float32Array ids read as bits");
            }
            // addSpecialTokens: false drops [CLS]/[SEP]; maxLength keeps both.
            const bare = m.tokenize("Hello, world!", { addSpecialTokens: false });
            if (bare.length !== ref.cases[0].ids.length - 2 || bare[0] === m.clsId) throw new Error("addSpecialTokens");
            const cut = m.tokenize(ref.cases[2].text, { maxLength: 16 });
            if (cut.length !== 16 || cut[0] !== m.clsId || cut[15] !== m.sepId) throw new Error("maxLength cut");
            // Bad arguments.
            const expectThrow = (fn, re) => { try { fn(); } catch (e) { if (re.test(e.message)) return; throw e; } throw new Error("no throw: " + re); };
            expectThrow(() => m.encode([m.clsId, 1 << 20]), /outside the vocabulary/);
            expectThrow(() => m.encode("x", { pooling: "max" }), /pooling/);
            expectThrow(() => m.encode("x", { maxLength: 0 }), /maxLength/);
            // A method on the wrong receiver is refused, not reinterpreted.
            expectThrow(() => bro.lm.ModernBertModel.prototype.encode.call(bro.lm.Grammar.exact("a"), "x"), /not a ModernBertModel/);
            return "SUCCESS worst |d| " + worst.toFixed(4);
        })()
    )JS";
    ev::CallResult res = bronze::eval::evalScript(script);
    if (res.thrown) {
        std::cerr << "modernbert eval threw: " << errorMessage(res.value) << std::endl;
        std::exit(1);
    }
    const std::string out = ev::toUtf8(res.value);
    TEST_CHECK(out.rfind("SUCCESS", 0) == 0);
    std::cout << "  modernbert: " << out << std::endl;
}

static void test_laya() {
    std::cout << "[5/5] laya inference test..." << std::endl;

    const char* env_dir = std::getenv("LAYA_MODEL_DIR");
    std::string model_dir = (env_dir && env_dir[0]) ? env_dir : "D:/projects/laya";
    std::replace(model_dir.begin(), model_dir.end(), '\\', '/');

    if (!std::filesystem::exists(model_dir + "/model.safetensors")) {
        std::cout << "Skipping Laya inference test: checkpoint not found at " << model_dir << std::endl;
        return;
    }

    std::string script = R"JS(
        (function() {
            const laya = bro.lm.loadLaya(")JS" + model_dir + R"JS(");
            if (!laya || typeof laya.predict !== "function") {
                throw new Error("loadLaya failed or predict is not a function");
            }

            const state = {
                from: "user@acme.com",
                subject: "Duplicate charge on invoice #4411",
                body: "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
            };

            const questions = {
                department: {
                    type: "choice",
                    instructions: "Which department should handle this request?",
                    criteria: {
                        billing: "invoices, payments, refunds",
                        technical: "bugs, outages, system errors",
                        sales: "pricing, new contracts",
                        other: "everything else"
                    }
                },
                urgency: {
                    type: "score",
                    instructions: "How urgent is this request?",
                    criteria: ["not urgent", "soon", "critical deadline or blocking issue"]
                },
                churn_risk: {
                    type: "noul",
                    instructions: "Does the user threaten to cancel or leave?"
                }
            };

            const res = laya.predict(state, questions);
            if (typeof res !== "object" || res === null) {
                throw new Error("predict result must be an object");
            }
            if (res.model !== "rl-agent") {
                throw new Error("expected model to be rl-agent, got: " + res.model);
            }
            if (typeof res.usage !== "object" || typeof res.usage.input_tokens !== "number") {
                throw new Error("expected usage.input_tokens number");
            }
            if (res.usage.input_tokens <= 0) {
                throw new Error("input_tokens should be > 0");
            }
            if (typeof res.answers !== "object" || res.answers === null) {
                throw new Error("expected answers object");
            }

            const dept = res.answers.department;
            if (!dept || dept.type !== "choice") throw new Error("department must have type choice");
            if (dept.choice !== "billing") throw new Error("expected department choice billing, got: " + dept.choice);
            if (typeof dept.confidence !== "number" || dept.confidence <= 0) {
                throw new Error("expected positive confidence");
            }
            if (typeof dept.probabilities !== "object" || typeof dept.probabilities.billing !== "number") {
                throw new Error("expected probabilities.billing number");
            }
            if (typeof dept.rl_agent !== "object" || typeof dept.rl_agent.act_probability !== "number") {
                throw new Error("expected rl_agent.act_probability");
            }

            const urg = res.answers.urgency;
            if (!urg || urg.type !== "score") throw new Error("urgency must have type score");
            if (typeof urg.score !== "number") throw new Error("urgency score must be number");
            if (typeof urg.legend !== "object" || urg.legend["0"] !== "not urgent") {
                throw new Error("urgency legend missing or incorrect");
            }
            if (typeof urg.probabilities !== "object" || typeof urg.probabilities["0"] !== "number") {
                throw new Error("urgency probabilities missing");
            }

            const churn = res.answers.churn_risk;
            if (!churn || churn.type !== "noul") throw new Error("churn_risk must have type noul");
            if (typeof churn.noul !== "number") throw new Error("churn_risk noul must be number");
            if (typeof churn.rl_agent !== "object" || typeof churn.rl_agent.act_probability !== "number") {
                throw new Error("expected churn rl_agent.act_probability");
            }

            // Raw logits + temperature + act logits are exposed for refitting.
            if (!(dept.logits instanceof Float32Array) || dept.logits.length !== 4) {
                throw new Error("expected department.logits Float32Array(4)");
            }
            if (typeof dept.temperature !== "number" || !(dept.rl_agent.act_logits instanceof Float32Array)) {
                throw new Error("expected temperature and rl_agent.act_logits");
            }

            // An object state is serialised like Python's json.dumps: the same
            // text passed as a string must give the same logits. Same question
            // set, so the packed batch is identical too (results can move by
            // ~1e-2 with batch composition: split-K GEMM plans follow the
            // packed row count).
            const pyText = '{"from": "user@acme.com", "subject": "Duplicate charge on invoice #4411", ' +
                '"body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."}';
            const resText = laya.predict(pyText, questions);
            for (let k = 0; k < 4; ++k) {
                if (Math.abs(resText.answers.department.logits[k] - dept.logits[k]) > 1e-3) {
                    throw new Error("object state and json.dumps text disagree at logit " + k);
                }
            }

            // Options that cannot fit raise instead of answering a truncated set.
            const many = [];
            for (let i = 0; i < 200; ++i) many.push("option " + i);
            let threw = false;
            try {
                laya.predict("x", { huge: { type: "choice", instructions: "pick", criteria: many } });
            } catch (e) {
                threw = String(e.message).indexOf("options do not fit") >= 0;
            }
            if (!threw) throw new Error("expected options-do-not-fit error");

            // Per-call limits: raising headMaxLen/maxLen makes them fit.
            const big = laya.predict("x", { huge: { type: "choice", instructions: "pick", criteria: many } },
                                     { maxLen: 2048, headMaxLen: 1536 });
            if (big.answers.huge.logits.length !== 200) throw new Error("expected 200 logits with raised limits");

            // truncateLeft keeps the newest tokens: a long state whose tail
            // differs must change the answer's logits only under truncateLeft.
            const filler = "The weather report was unremarkable today. ".repeat(80);
            const q1 = { angry: { type: "noul", instructions: "Is the customer angry?" } };
            const rA = laya.predict(filler + "I am furious, cancel everything now!", q1, { truncateLeft: true });
            const rB = laya.predict(filler + "Thanks, all good.", q1, { truncateLeft: true });
            const rC = laya.predict(filler + "I am furious, cancel everything now!", q1);
            const rD = laya.predict(filler + "Thanks, all good.", q1);
            if (Math.abs(rA.answers.angry.logits[1] - rB.answers.angry.logits[1]) < 1e-3) {
                throw new Error("truncateLeft did not keep the state tail");
            }
            if (Math.abs(rC.answers.angry.logits[1] - rD.answers.angry.logits[1]) > 1e-3) {
                throw new Error("default truncation should drop the state tail");
            }

            const cfg = laya.config();
            if (cfg.max_len !== 512 || cfg.head_max_len !== 192 || cfg.temperature.length !== 3) {
                throw new Error("config() mismatch");
            }
            if (!Array.isArray(cfg.devices) || cfg.devices.length !== 1 || !(cfg.tokenBudget >= 512)) {
                throw new Error("config() scheduler fields mismatch");
            }
            globalThis.__laya = laya;

            // Also test constructor form: new bro.lm.LayaModel(...)
            const layaCtor = new bro.lm.LayaModel(")JS" + model_dir + R"JS(");
            const res2 = layaCtor.predict(JSON.stringify(state), questions);
            if (!res2 || !res2.answers || res2.answers.department.choice !== "billing") {
                throw new Error("new LayaModel() predict failed");
            }

            return "SUCCESS";
        })()
    )JS";

    ev::CallResult res = bronze::eval::evalScript(script);
    if (res.thrown) {
        std::cerr << "laya eval threw: " << ev::toUtf8(res.value) << std::endl;
        std::exit(1);
    }
    TEST_CHECK(ev::toUtf8(res.value) == "SUCCESS");

    // predictAsync: many requests in flight from one realm share forwards;
    // promises settle on the LM tick (the engine's frame pump in bro).
    std::string launch = R"JS(
        (function() {
            const laya = globalThis.__laya;
            const qs = {
                department: { type: "choice", instructions: "Which department should handle this request?",
                              criteria: { billing: "invoices, payments, refunds", technical: "bugs, outages",
                                          sales: "pricing, new contracts", other: "everything else" } },
                churn_risk: { type: "noul", instructions: "Does the user threaten to cancel or leave?" }
            };
            const st = globalThis.__async = { done: 0, bad: null, shared: 0, rejected: false, loaded: null };
            laya.resetStats();
            for (let i = 0; i < 24; ++i) {
                const state = { id: i, body: "We were billed twice for March, refund the duplicate or we cancel." };
                laya.predictAsync(state, qs, { priority: i % 3, deadlineMs: 100 }).then((r) => {
                    if (r.answers.department.choice !== "billing") st.bad = "wrong choice " + r.answers.department.choice;
                    if (typeof r.timing.totalMs !== "number" || r.timing.forwards < 1) st.bad = "timing missing";
                    if (typeof r.answers.churn_risk.confidence !== "number") st.bad = "noul confidence missing";
                    if (r.timing.batchRequests > 1) st.shared++;
                    st.done++;
                }, (e) => { st.bad = String(e); });
            }
            const many = [];
            for (let i = 0; i < 200; ++i) many.push("option " + i);
            laya.predictAsync("x", { huge: { type: "choice", instructions: "pick", criteria: many } })
                .then(() => { st.bad = "expected a rejection"; },
                      (e) => { st.rejected = String(e.message).indexOf("options do not fit") >= 0; });
            bro.lm.loadLayaAsync(")JS" + model_dir + R"JS(", { prewarm: false }).then((m) => {
                st.loaded = m.predict("Please refund my duplicate charge.", qs).answers.department.choice;
                m.dispose();
            }, (e) => { st.bad = "loadLayaAsync: " + e; });
            return "LAUNCHED";
        })()
    )JS";
    res = bronze::eval::evalScript(launch);
    TEST_CHECK(!res.thrown && ev::toUtf8(res.value) == "LAUNCHED");

    const auto t0 = std::chrono::steady_clock::now();
    std::string state;
    for (;;) {
        brolm::api::tickLMAsync();
        ev::drainMicrotasks();
        ev::CallResult s = bronze::eval::evalScript(
            "(function(){ const s = globalThis.__async; return s.bad ? 'BAD ' + s.bad : "
            "(s.done === 24 && s.rejected && s.loaded ? 'DONE' : 'WAIT'); })()");
        state = ev::toUtf8(s.value);
        if (state != "WAIT") break;
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(60)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (state != "DONE") {
        std::cerr << "laya async: " << state << std::endl;
        std::exit(1);
    }

    // Under BRONZE_GC_STRESS the launching script is slow enough that the
    // requests reach the scheduler one by one, so batching is not asserted.
    const char* stressEnv = std::getenv("BRONZE_GC_STRESS");
    const bool gcStress = stressEnv && stressEnv[0] && std::string(stressEnv) != "0";
    std::string check = std::string("globalThis.__gcStress = ") + (gcStress ? "true" : "false") + ";\n" + R"JS(
        (function() {
            const laya = globalThis.__laya, st = globalThis.__async, stress = globalThis.__gcStress;
            if (st.loaded !== "billing") throw new Error("loadLayaAsync model answered " + st.loaded);
            if (st.shared === 0 && !stress) throw new Error("no two async requests shared a forward");
            const s = laya.stats();
            if (s.completed < 24 || s.forwards < 1 || (!stress && !(s.meanBatchRequests > 1)) || s.devices.length !== 1)
                throw new Error("stats: " + JSON.stringify(s));
            if (!Array.isArray(s.recentBatches) || s.recentBatches.length !== s.forwards)
                throw new Error("stats.recentBatches");
            laya.dispose();
            let threw = false;
            try { laya.predict("x", { a: { type: "noul", instructions: "?" } }); }
            catch (e) { threw = String(e.message).indexOf("disposed") >= 0; }
            if (!threw) throw new Error("predict after dispose must throw");
            return "SUCCESS " + st.shared + "/24 shared, mean " + s.meanBatchRequests.toFixed(1) + " requests/forward";
        })()
    )JS";
    res = bronze::eval::evalScript(check);
    if (res.thrown) {
        std::cerr << "laya async check threw: " << errorMessage(res.value) << std::endl;
        std::exit(1);
    }
    std::cout << "  async: " << ev::toUtf8(res.value) << std::endl;
}

int main() {
    std::cout << "Running brolm API test..." << std::endl;

    ev::Realm* realm = ev::createRealm();
    {
        ev::RealmScope scope(realm);
        brolm::api::installLM();
        test_mounts();
        test_loader_validation();
        test_script();
        test_async_handle();
        test_grammar();
        test_gemma_tokenizer_binding();
        test_generation_weights();
        test_modernbert();
        test_laya();
        brolm::api::shutdownLM();  // what bro's engine shutdown hook does
    }
    ev::destroyRealm(realm);

    std::cout << "All brolm API tests passed!" << std::endl;
    return 0;
}
