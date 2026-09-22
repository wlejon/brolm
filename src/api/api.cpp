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

    Value globalThisVal = ev::undefined();
    auto gt = ev::globalValue("globalThis");
    if (gt.found && ev::isObject(gt.value)) {
        globalThisVal = gt.value;
    }

    Value broVal = ev::globalValue("bro").found ? ev::globalValue("bro").value : ev::undefined();
    if (!ev::isObject(broVal)) {
        if (!ev::isUndefined(globalThisVal)) {
            Value candidate = ev::getProperty(globalThisVal, "bro");
            if (ev::isObject(candidate)) {
                broVal = candidate;
            }
        }
    }
    if (!ev::isObject(broVal)) {
        broVal = ev::createObject();
        ev::registerGlobal("bro", broVal);
        if (!ev::isUndefined(globalThisVal)) {
            ev::setProperty(globalThisVal, "bro", broVal);
        }
    }

    ev::Persistent broP(broVal);

    Value lmVal = ev::getProperty(broP.get(), "lm");
    if (!ev::isObject(lmVal)) {
        lmVal = ev::createObject();
    }
    ObjectBuilder lmObj(lmVal);

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
    lmObj.def("loadLaya", 1, js_loadLaya);
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
    lmObj.set("LayaModel", g_layaModelClass.constructor());
    lmObj.set("AsyncHandle", g_asyncHandleClass.constructor());

    if (!ev::isUndefined(globalThisVal)) {
        ev::setProperty(globalThisVal, "QwenTokenizer", g_qwenTokenizerClass.constructor());
        ev::setProperty(globalThisVal, "MistralTokenizer", g_mistralTokenizerClass.constructor());
        ev::setProperty(globalThisVal, "GemmaTokenizer", g_gemmaTokenizerClass.constructor());
        ev::setProperty(globalThisVal, "Llama3Tokenizer", g_llama3TokenizerClass.constructor());
        ev::setProperty(globalThisVal, "LMModel", g_lmModelClass.constructor());
        ev::setProperty(globalThisVal, "Qwen35Model", g_qwen35ModelClass.constructor());
        ev::setProperty(globalThisVal, "Qwen3VLModel", g_qwen3VLModelClass.constructor());
        ev::setProperty(globalThisVal, "ClipModel", g_clipModelClass.constructor());
        ev::setProperty(globalThisVal, "NllbModel", g_nllbModelClass.constructor());
        ev::setProperty(globalThisVal, "T5Model", g_t5ModelClass.constructor());
        ev::setProperty(globalThisVal, "LayaModel", g_layaModelClass.constructor());
        ev::setProperty(globalThisVal, "AsyncHandle", g_asyncHandleClass.constructor());
    }

    broP.set(ev::setProperty(broP.get(), "lm", lmObj.build()));
}

} // namespace brolm::api
