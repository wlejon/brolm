// Standalone test for brolm_api, the bronze-runtime JavaScript binding.
// No bro engine, no window, no weights: a fresh bronze realm, installLM(),
// then the mount points and the argument validation of every loader checked
// from both the embed API and a compiled script. A loader given nothing, a
// non-string, or a path that does not exist must throw; never null, never a
// silent success.

#include "api/api.h"
#include "embed/embed.h"
#include "eval/eval.h"

#include <cstdlib>
#include <iostream>
#include <string>

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
    "loadClip", "loadClipModel", "loadT5",
};

static const char* kClasses[] = {
    "QwenTokenizer", "MistralTokenizer", "GemmaTokenizer", "Llama3Tokenizer",
    "LMModel", "Qwen35Model", "Qwen3VLModel", "ClipModel", "NllbModel",
    "T5Model", "AsyncHandle",
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
            expectType(() => lm.generate(), "generate()");
            expectType(() => new QwenTokenizer(), "new QwenTokenizer()");
            expectType(() => new LMModel(), "new LMModel()");

            const e1 = expectThrows(() => lm.loadQwen("./does-not-exist/model.gguf"), "loadQwen(missing)");
            if (e1.message.indexOf("loadQwen") < 0) throw new Error("loadQwen(missing) message: " + e1.message);
            const e2 = expectThrows(() => lm.loadTokenizer("./does-not-exist/tokenizer.json"), "loadTokenizer(missing)");
            if (e2.message.indexOf("loadTokenizer") < 0) throw new Error("loadTokenizer(missing) message: " + e2.message);
            const e3 = expectThrows(() => lm.loadTokenizer({ vocabPath: "./nope/vocab.json", mergesPath: "./nope/merges.txt" }), "loadTokenizer(missing pair)");
            if (e3.message.indexOf("loadTokenizer") < 0) throw new Error("loadTokenizer(missing pair) message: " + e3.message);

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

int main() {
    std::cout << "Running brolm API test..." << std::endl;

    ev::Realm* realm = ev::createRealm();
    {
        ev::RealmScope scope(realm);
        brolm::api::installLM();
        test_mounts();
        test_loader_validation();
        test_script();
    }
    ev::destroyRealm(realm);

    std::cout << "All brolm API tests passed!" << std::endl;
    return 0;
}
