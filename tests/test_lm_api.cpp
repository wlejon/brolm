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
    "loadClip", "loadClipModel", "loadT5", "loadLaya",
};

static const char* kClasses[] = {
    "QwenTokenizer", "MistralTokenizer", "GemmaTokenizer", "Llama3Tokenizer",
    "LMModel", "Qwen35Model", "Qwen3VLModel", "ClipModel", "NllbModel",
    "T5Model", "LayaModel", "AsyncHandle",
};

static Value lmNamespace() {
    ev::GlobalValue broG = ev::globalValue("bro");
    TEST_CHECK(broG.found);
    TEST_CHECK(ev::isObject(broG.value));
    Value lm = ev::getProperty(broG.value, "lm");
    TEST_CHECK(ev::isObject(lm));
    return lm;
}

static void test_mounts() {
    std::cout << "[1/3] mount points..." << std::endl;

    Value lm = lmNamespace();

    for (const char* name : kLoaders) {
        if (!ev::isFunction(ev::getProperty(lm, name))) {
            std::cerr << "missing bro.lm." << name << std::endl;
            std::exit(1);
        }
    }
    TEST_CHECK(ev::isFunction(ev::getProperty(lm, "init")));
    TEST_CHECK(ev::isFunction(ev::getProperty(lm, "generate")));
    TEST_CHECK(ev::isFunction(ev::getProperty(lm, "tick")));

    // Each class constructor sits on bro.lm and on globalThis, and is the
    // same function in both places.
    for (const char* name : kClasses) {
        Value onNs = ev::getProperty(lm, name);
        ev::GlobalValue onGlobal = ev::globalValue(name);
        if (!ev::isFunction(onNs) || !onGlobal.found || !ev::isFunction(onGlobal.value)) {
            std::cerr << "missing constructor " << name << std::endl;
            std::exit(1);
        }
    }

    Value q35Proto = ev::getProperty(ev::globalValue("Qwen35Model").value, "prototype");
    TEST_CHECK(ev::isFunction(ev::getProperty(q35Proto, "generateStream")));

    Value qvlProto = ev::getProperty(ev::globalValue("Qwen3VLModel").value, "prototype");
    TEST_CHECK(ev::isFunction(ev::getProperty(qvlProto, "generateStream")));

    // tick() with nothing pending is a no-op.
    ev::CallResult tick = ev::call(ev::getProperty(lm, "tick"), lm, {});
    TEST_CHECK(!tick.thrown);
}

static void expectThrows(Value lm, const char* fnName, std::span<const Value> args,
                         const char* expectName) {
    Value fn = ev::getProperty(lm, fnName);
    ev::CallResult r = ev::call(fn, lm, args);
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

    Value lm = lmNamespace();

    // No arguments at all: every loader throws a TypeError.
    for (const char* name : kLoaders) {
        expectThrows(lm, name, {}, "TypeError");
    }

    // A number where a path string (or options object) belongs.
    Value num = ev::fromDouble(42.0);
    for (const char* name : kLoaders) {
        expectThrows(lm, name, std::span<const Value>(&num, 1), "TypeError");
    }

    // A path that does not exist: the loader reaches the file and the
    // failure surfaces as a thrown Error carrying the loader's name.
    ev::Persistent missing(ev::fromUtf8("./does-not-exist/model.gguf"));
    {
        Value arg = missing.get();
        Value fn = ev::getProperty(lm, "loadQwen");
        ev::CallResult r = ev::call(fn, lm, std::span<const Value>(&arg, 1));
        TEST_CHECK(r.thrown);
        TEST_CHECK(errorMessage(r.value).find("loadQwen") != std::string::npos);
    }
    {
        // Tokenizer construction from a nonexistent tokenizer.json.
        Value arg = missing.get();
        Value fn = ev::getProperty(lm, "loadTokenizer");
        ev::CallResult r = ev::call(fn, lm, std::span<const Value>(&arg, 1));
        TEST_CHECK(r.thrown);
        TEST_CHECK(errorMessage(r.value).find("loadTokenizer") != std::string::npos);
    }
    {
        Value arg = missing.get();
        Value fn = ev::getProperty(lm, "loadLlama3Tokenizer");
        ev::CallResult r = ev::call(fn, lm, std::span<const Value>(&arg, 1));
        TEST_CHECK(r.thrown);
        TEST_CHECK(errorMessage(r.value).find("loadLlama3Tokenizer") != std::string::npos);
    }
    {
        Value arg = missing.get();
        Value fn = ev::getProperty(lm, "loadLaya");
        ev::CallResult r = ev::call(fn, lm, std::span<const Value>(&arg, 1));
        TEST_CHECK(r.thrown);
        TEST_CHECK(errorMessage(r.value).find("loadLaya") != std::string::npos);
    }
    // loadLayaAsync never throws: a bad path is a rejected promise.
    {
        Value arg = missing.get();
        Value fn = ev::getProperty(lm, "loadLayaAsync");
        TEST_CHECK(ev::isFunction(fn));
        ev::CallResult r = ev::call(fn, lm, std::span<const Value>(&arg, 1));
        TEST_CHECK(!r.thrown);
        TEST_CHECK(ev::isPromise(r.value));
    }

    // loadTokenizer({}) : an options object with none of the doc'd keys.
    {
        Value opts = ev::createObject();
        expectThrows(lm, "loadTokenizer", std::span<const Value>(&opts, 1), "TypeError");
    }

    // generate() without a model, and with something that is not a model.
    expectThrows(lm, "generate", {}, "TypeError");
    {
        ev::Persistent notAModel(ev::createObject());
        Value args[2] = { notAModel.get(), ev::fromUtf8("hello") };
        expectThrows(lm, "generate", std::span<const Value>(args, 2), "TypeError");
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
    Value hVal = brolm::api::makeAsyncHandleValue(handle);
    TEST_CHECK(ev::isObject(hVal));

    // cancel()
    Value cancelFn = ev::getProperty(hVal, "cancel");
    TEST_CHECK(ev::isFunction(cancelFn));
    TEST_CHECK(!handle->cancelled.load());
    ev::call(cancelFn, hVal, {});
    TEST_CHECK(handle->cancelled.load());

    // wait() blocks until finished
    Value waitFn = ev::getProperty(hVal, "wait");
    TEST_CHECK(ev::isFunction(waitFn));
    TEST_CHECK(!handle->finished.load());

    std::thread backgroundWorker([handle]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        handle->finished.store(true, std::memory_order_release);
    });

    ev::call(waitFn, hVal, {});
    TEST_CHECK(handle->finished.load());
    if (backgroundWorker.joinable()) backgroundWorker.join();
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

    std::string check = R"JS(
        (function() {
            const laya = globalThis.__laya, st = globalThis.__async;
            if (st.loaded !== "billing") throw new Error("loadLayaAsync model answered " + st.loaded);
            if (st.shared === 0) throw new Error("no two async requests shared a forward");
            const s = laya.stats();
            if (s.completed < 24 || s.forwards < 1 || !(s.meanBatchRequests > 1) || s.devices.length !== 1)
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
        test_laya();
        brolm::api::shutdownLM();  // what bro's engine shutdown hook does
    }
    ev::destroyRealm(realm);

    std::cout << "All brolm API tests passed!" << std::endl;
    return 0;
}
