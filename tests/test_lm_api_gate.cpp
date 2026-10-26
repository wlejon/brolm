// bro.lm under the host's access gate (embed/embed_access.h; api.h, "Access"):
// every loader asks the gate about each path it was given before anything is
// opened, a refusal is an Error carrying the gate's sentence, code EACCES and
// the path, and with the gate allowing it the same call gets past the gate to
// its ordinary not-found error. No weights: every path under the refused
// directory is refused whether it exists or not. Run from test_lm_api.cpp's
// main against its realm.

#include "embed/embed.h"
#include "embed/embed_access.h"
#include "eval/eval.h"

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace ev = bronze::embed;

#define GCHECK(cond, msg)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "CHECK failed: " #cond " (" << __FILE__ << ":"        \
                      << __LINE__ << "): " << (msg) << std::endl;              \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

namespace {

std::string utf8(const fs::path& p) {
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
}

// The path as the gate is handed it.
std::string gatePath(const fs::path& p) {
    return utf8(fs::absolute(p).lexically_normal());
}

std::string js(const fs::path& p) {
    std::string out = "'";
    for (char c : utf8(p)) {
        if (c == '\\' || c == '\'') out += '\\';
        out += c;
    }
    return out + "'";
}

std::string evalString(const std::string& src) {
    auto r = bronze::eval::evalScript(src);
    if (r.thrown) return "<thrown>";
    return ev::isString(r.value) ? ev::toUtf8(r.value) : "<not a string>";
}

// Runs `stmt`; "ok", the refusal as "denied|message|code|path", or
// "error|<code>|<message>" for any other throw.
std::string syncJs(const std::string& stmt) {
    return evalString(
        "(function() { try { " + stmt + "; return 'ok'; } catch (e) {\n"
        "  if (e && e.code === 'EACCES') return 'denied|' + e.message + '|' + e.code + '|' + e.path;\n"
        "  return 'error|' + (e && e.code) + '|' + (e && e.message); } })()");
}

// Runs `expr` (a promise) to settlement, in the same shape.
std::string awaitJs(const std::string& expr) {
    bronze::eval::evalScript(
        "globalThis._gateOut = undefined;\n"
        "Promise.resolve().then(() => " + expr + ").then(v => 'ok', e =>\n"
        "  (e && e.code === 'EACCES') ? 'denied|' + e.message + '|' + e.code + '|' + e.path\n"
        "                             : 'error|' + (e && e.code) + '|' + (e && e.message))\n"
        ".then(s => { globalThis._gateOut = s; });");
    for (int i = 0; i < 100; ++i) {
        ev::drainMicrotasks();
        std::string s = evalString("typeof globalThis._gateOut === 'string' ? globalThis._gateOut : ''");
        if (!s.empty()) return s;
    }
    return "<unsettled>";
}

std::string refusal(const fs::path& p) {
    const std::string path = gatePath(p);
    return "denied|no files access: " + path + "|EACCES|" + path;
}

bool isOrdinaryError(const std::string& r) {
    return r.rfind("error|", 0) == 0 && r.find("EACCES") == std::string::npos;
}

struct TestGate {
    std::mutex mu;
    std::vector<std::string> refused;
    std::atomic<int> calls{0};
    std::string ask(const std::string& path) {
        ++calls;
        std::lock_guard lock(mu);
        for (const auto& p : refused) {
            if (path.compare(0, p.size(), p) == 0) return "no files access: " + path;
        }
        return "";
    }
};

} // namespace

void runAccessGateTests() {
    std::cout << "[gate] bro.lm loaders under the access gate..." << std::endl;

    const fs::path root = fs::temp_directory_path() / "brolm_api_gate";
    const fs::path ok = root / "ok";
    const fs::path hidden = root / "hidden";
    fs::create_directories(ok);
    fs::create_directories(hidden);
    {
        // A file that exists under the refused directory: refused all the same.
        std::ofstream(hidden / "tokenizer.json") << "not a tokenizer";
    }

    auto gate = std::make_shared<TestGate>();
    gate->refused.push_back(gatePath(hidden));
    ev::setAccessGate([gate](const ev::AccessRequest& r) -> std::string {
        if (r.kind != ev::AccessKind::Read && r.kind != ev::AccessKind::Write) return "";
        return gate->ask(std::string(r.subject));
    });

    const std::string L = "bro.lm.";
    struct Case { std::string call; fs::path refusedPath; };
    const std::vector<Case> cases = {
        {L + "loadQwen(" + js(hidden / "m.gguf") + ")", hidden / "m.gguf"},
        {L + "loadModel(" + js(hidden / "m.gguf") + ", { onReady() {} })", hidden / "m.gguf"},
        {L + "loadMistral(" + js(hidden / "m.gguf") + ", { tokenizerPath: " + js(ok / "tekken.json") + " })",
         hidden / "m.gguf"},
        {L + "loadMistral(" + js(ok / "m.gguf") + ", { tokenizerPath: " + js(hidden / "tekken.json") + " })",
         hidden / "tekken.json"},
        {L + "loadGemma2(" + js(hidden / "gemma") + ")", hidden / "gemma"},
        {L + "loadQwen35(" + js(hidden / "q35") + ")", hidden / "q35"},
        {L + "loadQwen3VL(" + js(hidden / "qvl") + ")", hidden / "qvl"},
        {L + "loadNllb(" + js(hidden / "nllb") + ")", hidden / "nllb"},
        {L + "loadModernBert(" + js(hidden / "mb") + ")", hidden / "mb"},
        {L + "loadModernBert(" + js(ok / "mb") + ", { weightsPath: " + js(hidden / "w.safetensors") + " })",
         hidden / "w.safetensors"},
        {L + "loadTokenizer(" + js(hidden / "tokenizer.json") + ")", hidden / "tokenizer.json"},
        {L + "loadTokenizer({ vocabPath: " + js(ok / "vocab.json") + ", mergesPath: " + js(hidden / "merges.txt") + " })",
         hidden / "merges.txt"},
        {L + "loadTokenizer({ tokenizerPath: " + js(hidden / "tokenizer.json") + " })", hidden / "tokenizer.json"},
        {L + "loadLlama3Tokenizer(" + js(hidden / "tokenizer.json") + ")", hidden / "tokenizer.json"},
        {L + "loadClip({ vocabPath: " + js(ok / "vocab.json") + ", mergesPath: " + js(ok / "merges.txt") +
             ", weightsPath: " + js(hidden / "clip.safetensors") + " })",
         hidden / "clip.safetensors"},
        {L + "loadClipModel({ vocabPath: " + js(ok / "vocab.json") + ", mergesPath: " + js(ok / "merges.txt") +
             ", weightsPath: " + js(ok / "clip.safetensors") + ", configPath: " + js(hidden / "config.json") + " })",
         hidden / "config.json"},
        {L + "loadT5({ tokenizerPath: " + js(ok / "t.json") + ", shards: [" + js(ok / "a.safetensors") + ", " +
             js(hidden / "b.safetensors") + "] })",
         hidden / "b.safetensors"},
        {L + "loadLaya(" + js(hidden / "laya") + ")", hidden / "laya"},
        {L + "loadLaya({ path: " + js(hidden / "laya") + " })", hidden / "laya"},
    };

    // Refused: the gate's sentence, EACCES and the absolute path, before the
    // loader looks at the disk (a missing path would otherwise be its error).
    for (const auto& c : cases) {
        const std::string r = syncJs(c.call);
        GCHECK(r == refusal(c.refusedPath), c.call + " -> " + r);
    }
    std::string r = awaitJs(L + "loadLayaAsync(" + js(hidden / "laya") + ")");
    GCHECK(r == refusal(hidden / "laya"), "loadLayaAsync -> " + r);
    GCHECK(fs::exists(hidden / "tokenizer.json"), "the refused file went away");

    // The same calls with the gate allowing them get past it, to their own
    // not-found / bad-file errors.
    {
        std::lock_guard lock(gate->mu);
        gate->refused.clear();
    }
    const int callsBefore = gate->calls.load();
    for (const auto& c : cases) {
        r = syncJs(c.call);
        // With opts.onReady the load is queued and fails later, off the call.
        const bool queued = c.call.find("onReady") != std::string::npos;
        GCHECK(queued ? r == "ok" : isOrdinaryError(r), "allowed " + c.call + " -> " + r);
    }
    GCHECK(gate->calls.load() > callsBefore, "the allowing gate was not asked");
    r = awaitJs(L + "loadLayaAsync(" + js(hidden / "laya") + ")");
    GCHECK(isOrdinaryError(r), "allowed loadLayaAsync -> " + r);

    ev::setAccessGate(nullptr);
    std::error_code ec;
    fs::remove_all(root, ec);
    std::cout << "  access gate [PASS]" << std::endl;
}
