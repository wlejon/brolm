#include "api.h"
#include "host_lm_internal.h"
#include "object_builder.h"

namespace brolm::api {

static std::function<std::string(const std::string&)>& pathResolverSlot() {
    static std::function<std::string(const std::string&)> slot;
    return slot;
}

void setPathResolver(std::function<std::string(const std::string&)> resolver) {
    pathResolverSlot() = std::move(resolver);
}

std::string resolvePath(const std::string& path) {
    auto& r = pathResolverSlot();
    return r ? r(path) : path;
}

void installLM() {
    registerLMTokenizerClasses();
    registerLMModelClasses();
    registerLMVLClasses();
    registerLMClipClasses();
    registerLMLayaClasses();
    registerLMGrammarClass();
    registerLMModernBertClass();

    // Every Value below that outlives an allocating call rides in a
    // Persistent (embed.h GC contract): getProperty, createObject and
    // setProperty may each move everything.
    ev::Persistent globalThisP;
    {
        auto gt = ev::globalValue("globalThis");
        if (gt.found && ev::isObject(gt.value)) globalThisP.set(gt.value);
    }

    ev::Persistent broP;
    {
        auto bg = ev::globalValue("bro");
        if (bg.found && ev::isObject(bg.value)) broP.set(bg.value);
    }
    if (!ev::isObject(broP.get()) && ev::isObject(globalThisP.get())) {
        Value candidate = ev::getProperty(globalThisP.get(), "bro");
        if (ev::isObject(candidate)) broP.set(candidate);
    }
    if (!ev::isObject(broP.get())) {
        broP.set(ev::createObject());
        ev::registerGlobal("bro", broP.get());
        if (ev::isObject(globalThisP.get())) {
            globalThisP.set(ev::setProperty(globalThisP.get(), "bro", broP.get()));
        }
    }

    Value lmVal = ev::getProperty(broP.get(), "lm");
    ObjectBuilder lmObj(ev::isObject(lmVal) ? lmVal : ev::createObject());

    lmObj.def("init", 0, js_init);
    lmObj.def("loadQwen", 1, js_loadQwen);
    lmObj.def("loadMistral", 1, js_loadMistral);
    lmObj.def("loadGemma2", 1, js_loadGemma2);
    lmObj.def("loadModel", 1, js_loadQwen);
    lmObj.def("loadQwen35", 1, js_loadQwen35);
    lmObj.def("loadQwen3VL", 1, js_loadQwen3VL);
    lmObj.def("loadNllb", 1, js_loadNllb);
    lmObj.def("loadTokenizer", 1, js_loadTokenizer);
    lmObj.def("loadLlama3Tokenizer", 1, js_loadLlama3Tokenizer);
    lmObj.def("loadClip", 1, js_loadClip);
    lmObj.def("loadClipModel", 1, js_loadClip);
    lmObj.def("loadT5", 1, js_loadT5);
    lmObj.def("loadModernBert", 1, js_loadModernBert);
    lmObj.def("loadLaya", 1, js_loadLaya);
    lmObj.def("loadLayaAsync", 1, js_loadLayaAsync);
    lmObj.def("generate", 2, js_lm_generate);
    lmObj.def("tick", 0, js_lm_tick);

    lmObj.set("QwenTokenizer", g_qwenTokenizerClass.constructor());
    lmObj.set("MistralTokenizer", g_mistralTokenizerClass.constructor());
    lmObj.set("GemmaTokenizer", g_gemmaTokenizerClass.constructor());
    lmObj.set("Llama3Tokenizer", g_llama3TokenizerClass.constructor());
    lmObj.set("LMModel", g_lmModelClass.constructor());
    lmObj.set("Qwen35Model", g_qwen35ModelClass.constructor());
    lmObj.set("Qwen3VLModel", g_qwen3VLModelClass.constructor());
    lmObj.set("ClipModel", g_clipModelClass.constructor());
    lmObj.set("NllbModel", g_nllbModelClass.constructor());
    lmObj.set("T5Model", g_t5ModelClass.constructor());
    lmObj.set("ModernBertModel", g_modernBertModelClass.constructor());
    lmObj.set("LayaModel", g_layaModelClass.constructor());
    lmObj.set("AsyncHandle", g_asyncHandleClass.constructor());
    lmObj.set("Grammar", g_grammarClass.constructor());

    // HostClass::install already put each constructor on globalThis
    // (setGlobalValue), so the namespace entries above are the only mounts.
    broP.set(ev::setProperty(broP.get(), "lm", lmObj.build()));
}

} // namespace brolm::api
