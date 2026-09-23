#include "host_lm_internal.h"

namespace brolm::api {

HostClass g_qwenTokenizerClass;
HostClass g_mistralTokenizerClass;
HostClass g_gemmaTokenizerClass;
HostClass g_llama3TokenizerClass;

// ═══════════════════════════════════════════════════════════════════════════
// Creators and Unwrappers
// ═══════════════════════════════════════════════════════════════════════════

HostQwenTokenizer* hostQwenTokenizerOf(Value v) {
    if (!ev::isObject(v)) return nullptr;
    return static_cast<HostQwenTokenizer*>(g_qwenTokenizerClass.unwrap(v));
}

Value makeQwenTokenizerValue(std::shared_ptr<brolm::qwen::Tokenizer> tok) {
    auto w = std::make_unique<HostQwenTokenizer>();
    w->tok = std::move(tok);
    return g_qwenTokenizerClass.createInstance(std::move(w));
}

HostMistralTokenizer* hostMistralTokenizerOf(Value v) {
    if (!ev::isObject(v)) return nullptr;
    return static_cast<HostMistralTokenizer*>(g_mistralTokenizerClass.unwrap(v));
}

Value makeMistralTokenizerValue(std::shared_ptr<brolm::mistral::Tokenizer> tok) {
    auto w = std::make_unique<HostMistralTokenizer>();
    w->tok = std::move(tok);
    return g_mistralTokenizerClass.createInstance(std::move(w));
}

HostGemmaTokenizer* hostGemmaTokenizerOf(Value v) {
    if (!ev::isObject(v)) return nullptr;
    return static_cast<HostGemmaTokenizer*>(g_gemmaTokenizerClass.unwrap(v));
}

Value makeGemmaTokenizerValue(std::shared_ptr<brolm::gemma::Tokenizer> tok) {
    auto w = std::make_unique<HostGemmaTokenizer>();
    w->tok = std::move(tok);
    return g_gemmaTokenizerClass.createInstance(std::move(w));
}

HostLlama3Tokenizer* hostLlama3TokenizerOf(Value v) {
    if (!ev::isObject(v)) return nullptr;
    return static_cast<HostLlama3Tokenizer*>(g_llama3TokenizerClass.unwrap(v));
}

Value makeLlama3TokenizerValue(std::shared_ptr<brolm::llama3::Tokenizer> tok) {
    auto w = std::make_unique<HostLlama3Tokenizer>();
    w->tok = std::move(tok);
    return g_llama3TokenizerClass.createInstance(std::move(w));
}

// ═══════════════════════════════════════════════════════════════════════════
// QwenTokenizer Decorator
// ═══════════════════════════════════════════════════════════════════════════

static void decorateQwenTokenizer(ObjectBuilder& b) {
    b.accessor("imEndId", [](Value self, std::span<const Value>) {
        auto* w = hostQwenTokenizerOf(self);
        return ev::fromDouble(w && w->tok ? w->tok->im_end_id() : 151645);
    });
    b.accessor("imStartId", [](Value self, std::span<const Value>) {
        auto* w = hostQwenTokenizerOf(self);
        return ev::fromDouble(w && w->tok ? w->tok->im_start_id() : 151644);
    });
    b.accessor("eosId", [](Value self, std::span<const Value>) {
        auto* w = hostQwenTokenizerOf(self);
        return ev::fromDouble(w && w->tok ? w->tok->eos_id() : 151645);
    });
    b.accessor("endoftextId", [](Value self, std::span<const Value>) {
        auto* w = hostQwenTokenizerOf(self);
        return ev::fromDouble(w && w->tok ? w->tok->endoftext_id() : 151643);
    });
    b.accessor("vocabCount", [](Value self, std::span<const Value>) {
        auto* w = hostQwenTokenizerOf(self);
        return ev::fromDouble(w && w->tok ? static_cast<double>(w->tok->vocab_count()) : 0.0);
    });
    b.accessor("mergeCount", [](Value self, std::span<const Value>) {
        auto* w = hostQwenTokenizerOf(self);
        return ev::fromDouble(w && w->tok ? static_cast<double>(w->tok->merge_count()) : 0.0);
    });

    b.def("encode", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostQwenTokenizerOf(self);
        if (!w || !w->tok) return ev::throwTypeError("encode: not a QwenTokenizer");
        if (a.empty() || !ev::isString(a[0]))
            return ev::throwTypeError("encode(text, addSpecial?): text required");
        std::string text = ev::toUtf8(a[0]);
        bool addSpecial = (a.size() >= 2) && ev::toBool(a[1]);
        try {
            auto ids = w->tok->encode(text, addSpecial);
            return makeInt32Array(ids.data(), ids.size());
        } catch (const std::exception& e) {
            return ev::throwError(std::string("encode: ") + e.what());
        }
    });

    b.def("decode", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostQwenTokenizerOf(self);
        if (!w || !w->tok) return ev::throwTypeError("decode: not a QwenTokenizer");
        if (a.empty()) return ev::throwTypeError("decode(ids): ids required");
        std::vector<int32_t> ids = readInt32Array(a[0]);
        try {
            return ev::fromUtf8(w->tok->decode(ids));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("decode: ") + e.what());
        }
    });

    b.def("applyChatTemplate", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostQwenTokenizerOf(self);
        if (!w || !w->tok) return ev::throwTypeError("applyChatTemplate: not a QwenTokenizer");
        if (a.empty() || !ev::isObject(a[0]))
            return ev::throwTypeError("applyChatTemplate(messages, addGenerationPrompt?): messages array required");
        auto msgs = readChatMessages(a[0]);
        bool addGen = (a.size() < 2) ? true : ev::toBool(a[1]);
        try {
            return ev::fromUtf8(w->tok->apply_chat_template(msgs, addGen));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("applyChatTemplate: ") + e.what());
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// MistralTokenizer Decorator
// ═══════════════════════════════════════════════════════════════════════════

static void decorateMistralTokenizer(ObjectBuilder& b) {
    b.accessor("eosId", [](Value self, std::span<const Value>) {
        auto* w = hostMistralTokenizerOf(self);
        return ev::fromDouble(w && w->tok ? w->tok->eos_id() : 2);
    });
    b.accessor("bosId", [](Value self, std::span<const Value>) {
        auto* w = hostMistralTokenizerOf(self);
        return ev::fromDouble(w && w->tok ? w->tok->bos_id() : 1);
    });
    b.accessor("vocabCount", [](Value self, std::span<const Value>) {
        auto* w = hostMistralTokenizerOf(self);
        return ev::fromDouble(w && w->tok ? static_cast<double>(w->tok->vocab_count()) : 32768.0);
    });

    b.def("encode", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostMistralTokenizerOf(self);
        if (!w || !w->tok) return ev::throwTypeError("encode: not a MistralTokenizer");
        if (a.empty() || !ev::isString(a[0]))
            return ev::throwTypeError("encode(text, addSpecial?): text required");
        std::string text = ev::toUtf8(a[0]);
        bool addSpecial = (a.size() >= 2) && ev::toBool(a[1]);
        try {
            auto ids = w->tok->encode(text, addSpecial);
            return makeInt32Array(ids.data(), ids.size());
        } catch (const std::exception& e) {
            return ev::throwError(std::string("encode: ") + e.what());
        }
    });

    b.def("decode", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostMistralTokenizerOf(self);
        if (!w || !w->tok) return ev::throwTypeError("decode: not a MistralTokenizer");
        if (a.empty()) return ev::throwTypeError("decode(ids): ids required");
        std::vector<int32_t> ids = readInt32Array(a[0]);
        try {
            return ev::fromUtf8(w->tok->decode(ids));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("decode: ") + e.what());
        }
    });

    b.def("applyChatTemplate", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostMistralTokenizerOf(self);
        if (!w || !w->tok) return ev::throwTypeError("applyChatTemplate: not a MistralTokenizer");
        if (a.empty() || !ev::isObject(a[0]))
            return ev::throwTypeError("applyChatTemplate(messages, addGenerationPrompt?): messages array required");
        auto msgs = readChatMessages(a[0]);
        bool addGen = (a.size() < 2) ? true : ev::toBool(a[1]);
        try {
            return ev::fromUtf8(w->tok->apply_chat_template(msgs, addGen));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("applyChatTemplate: ") + e.what());
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// GemmaTokenizer Decorator
// ═══════════════════════════════════════════════════════════════════════════

static void decorateGemmaTokenizer(ObjectBuilder& b) {
    b.accessor("eosId", [](Value self, std::span<const Value>) {
        auto* w = hostGemmaTokenizerOf(self);
        return ev::fromDouble(w && w->tok ? w->tok->eos_id() : 1);
    });
    b.accessor("bosId", [](Value self, std::span<const Value>) {
        auto* w = hostGemmaTokenizerOf(self);
        return ev::fromDouble(w && w->tok ? w->tok->bos_id() : 2);
    });
    b.accessor("padId", [](Value self, std::span<const Value>) {
        auto* w = hostGemmaTokenizerOf(self);
        return ev::fromDouble(w && w->tok ? w->tok->pad_id() : 0);
    });
    b.accessor("unkId", [](Value self, std::span<const Value>) {
        auto* w = hostGemmaTokenizerOf(self);
        return ev::fromDouble(w && w->tok ? w->tok->unk_id() : 3);
    });
    b.accessor("vocabCount", [](Value self, std::span<const Value>) {
        auto* w = hostGemmaTokenizerOf(self);
        return ev::fromDouble(w && w->tok ? static_cast<double>(w->tok->vocab_count()) : 256000.0);
    });

    b.def("encode", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostGemmaTokenizerOf(self);
        if (!w || !w->tok) return ev::throwTypeError("encode: not a GemmaTokenizer");
        if (a.empty() || !ev::isString(a[0]))
            return ev::throwTypeError("encode(text, addBos?): text required");
        std::string text = ev::toUtf8(a[0]);
        bool addBos = (a.size() < 2) ? true : ev::toBool(a[1]);
        try {
            auto ids = w->tok->encode(text, addBos, /*add_eos=*/false);
            return makeInt32Array(ids.data(), ids.size());
        } catch (const std::exception& e) {
            return ev::throwError(std::string("encode: ") + e.what());
        }
    });

    b.def("decode", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostGemmaTokenizerOf(self);
        if (!w || !w->tok) return ev::throwTypeError("decode: not a GemmaTokenizer");
        if (a.empty()) return ev::throwTypeError("decode(ids): ids required");
        std::vector<int32_t> ids = readInt32Array(a[0]);
        try {
            return ev::fromUtf8(w->tok->decode(ids));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("decode: ") + e.what());
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Llama3Tokenizer Decorator
// ═══════════════════════════════════════════════════════════════════════════

static void decorateLlama3Tokenizer(ObjectBuilder& b) {
    b.accessor("bosId", [](Value self, std::span<const Value>) {
        auto* w = hostLlama3TokenizerOf(self);
        return ev::fromDouble(w && w->tok ? w->tok->bos_id() : -1);
    });
    b.accessor("eosId", [](Value self, std::span<const Value>) {
        auto* w = hostLlama3TokenizerOf(self);
        return ev::fromDouble(w && w->tok ? w->tok->eos_id() : -1);
    });
    b.accessor("endOfTextId", [](Value self, std::span<const Value>) {
        auto* w = hostLlama3TokenizerOf(self);
        return ev::fromDouble(w && w->tok ? w->tok->end_of_text_id() : -1);
    });
    b.accessor("eotId", [](Value self, std::span<const Value>) {
        auto* w = hostLlama3TokenizerOf(self);
        return ev::fromDouble(w && w->tok ? w->tok->eot_id() : -1);
    });
    b.accessor("startHeaderId", [](Value self, std::span<const Value>) {
        auto* w = hostLlama3TokenizerOf(self);
        return ev::fromDouble(w && w->tok ? w->tok->start_header_id() : -1);
    });
    b.accessor("endHeaderId", [](Value self, std::span<const Value>) {
        auto* w = hostLlama3TokenizerOf(self);
        return ev::fromDouble(w && w->tok ? w->tok->end_header_id() : -1);
    });
    b.accessor("vocabCount", [](Value self, std::span<const Value>) {
        auto* w = hostLlama3TokenizerOf(self);
        return ev::fromDouble(w && w->tok ? static_cast<double>(w->tok->vocab_count()) : 128000.0);
    });

    b.def("encode", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostLlama3TokenizerOf(self);
        if (!w || !w->tok) return ev::throwTypeError("encode: not a Llama3Tokenizer");
        if (a.empty() || !ev::isString(a[0]))
            return ev::throwTypeError("encode(text, addBos?): text required");
        std::string text = ev::toUtf8(a[0]);
        bool addBos = (a.size() < 2) ? true : ev::toBool(a[1]);
        try {
            auto ids = w->tok->encode(text, addBos);
            return makeInt32Array(ids.data(), ids.size());
        } catch (const std::exception& e) {
            return ev::throwError(std::string("encode: ") + e.what());
        }
    });

    b.def("decode", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostLlama3TokenizerOf(self);
        if (!w || !w->tok) return ev::throwTypeError("decode: not a Llama3Tokenizer");
        if (a.empty()) return ev::throwTypeError("decode(ids): ids required");
        std::vector<int32_t> ids = readInt32Array(a[0]);
        try {
            return ev::fromUtf8(w->tok->decode(ids));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("decode: ") + e.what());
        }
    });

    b.def("applyChatTemplate", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostLlama3TokenizerOf(self);
        if (!w || !w->tok) return ev::throwTypeError("applyChatTemplate: not a Llama3Tokenizer");
        if (a.empty() || !ev::isObject(a[0]))
            return ev::throwTypeError("applyChatTemplate(messages, addGenerationPrompt?): messages array required");
        auto msgs = readChatMessages(a[0]);
        bool addGen = (a.size() < 2) ? true : ev::toBool(a[1]);
        try {
            return ev::fromUtf8(w->tok->apply_chat_template(msgs, addGen));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("applyChatTemplate: ") + e.what());
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Registration and Loader Functions
// ═══════════════════════════════════════════════════════════════════════════

void registerLMTokenizerClasses() {
    g_qwenTokenizerClass.install("QwenTokenizer", 0, nullptr, decorateQwenTokenizer);
    g_mistralTokenizerClass.install("MistralTokenizer", 0, nullptr, decorateMistralTokenizer);
    g_gemmaTokenizerClass.install("GemmaTokenizer", 0, nullptr, decorateGemmaTokenizer);
    g_llama3TokenizerClass.install("Llama3Tokenizer", 0, nullptr, decorateLlama3Tokenizer);
}

Value js_loadTokenizer(Value, std::span<const Value> a) {
    if (a.empty() || (!ev::isObject(a[0]) && !ev::isString(a[0])))
        return ev::throwTypeError("loadTokenizer(opts): opts object or path required");

    try {
        if (ev::isString(a[0])) {
            std::string p = resolvePath(ev::toUtf8(a[0]));
            auto tok = std::make_unique<brolm::qwen::Tokenizer>(
                brolm::qwen::Tokenizer::from_tokenizer_json(p));
            return makeQwenTokenizerValue(std::move(tok));
        }

        // a[0] is a rooted argument slot, current across each read.
        std::string vocab, merges, tokJson;
        if (propString(a[0], "vocabPath", vocab) && propString(a[0], "mergesPath", merges)) {
            auto tok = std::make_unique<brolm::qwen::Tokenizer>(
                brolm::qwen::Tokenizer::load(resolvePath(vocab), resolvePath(merges)));
            return makeQwenTokenizerValue(std::move(tok));
        }

        if (propString(a[0], "tokenizerPath", tokJson)) {
            auto tok = std::make_unique<brolm::qwen::Tokenizer>(
                brolm::qwen::Tokenizer::from_tokenizer_json(resolvePath(tokJson)));
            return makeQwenTokenizerValue(std::move(tok));
        }

        return ev::throwTypeError("loadTokenizer: opts.vocabPath + opts.mergesPath or opts.tokenizerPath required");
    } catch (const std::exception& e) {
        return ev::throwError(std::string("loadTokenizer: ") + e.what());
    }
}

Value js_loadLlama3Tokenizer(Value, std::span<const Value> a) {
    if (a.empty())
        return ev::throwTypeError("loadLlama3Tokenizer(path_or_opts): path required");

    std::string path;
    if (ev::isString(a[0])) {
        path = ev::toUtf8(a[0]);
    } else if (ev::isObject(a[0])) {
        propString(a[0], "tokenizerPath", path);
    }
    if (path.empty())
        return ev::throwTypeError("loadLlama3Tokenizer: tokenizerPath required");
    path = resolvePath(path);

    try {
        auto tok = std::make_unique<brolm::llama3::Tokenizer>(
            brolm::llama3::Tokenizer::load(path));
        return makeLlama3TokenizerValue(std::move(tok));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("loadLlama3Tokenizer: ") + e.what());
    }
}

} // namespace brolm::api
