// Qwen byte-level BPE vs Hugging Face `tokenizers` on non-ASCII text.
//
// tests/ref/qwen_tokenizer_unicode.json (scripts/gen_qwen_tokenizer_fixture.py)
// holds HF's encodings of a 700-string corpus — English with contractions,
// code, digits, whitespace mixes, CJK, Korean, Cyrillic, Greek, Arabic,
// Hebrew, Indic scripts with combining marks, Thai/Lao/Khmer/Burmese,
// Vietnamese, Turkish, German, emoji + ZWJ sequences, full-width punctuation,
// the seven OmniVoice specials embedded in text, non-verbal tags, and
// decomposed/composed NFC stress strings — for two reference tokenizers:
//   ids      OmniVoice's tokenizer.json (Qwen2/Qwen3 BPE, `tokenizers`)
//   ids_tts  Qwen3-TTS's vocab.json + merges.txt (transformers Qwen2TokenizerFast)
// plus the NFC normalizer's output wherever it changes the input.
//
// Every case must match exactly, decode(encode(s)) must give NFC(s), and the
// standalone NFC normalizer must agree with HF's. The tokenizer files are
// located from BROLM_OMNIVOICE_DIR / BROLM_QWEN_TTS_DIR, falling back to the
// sibling checkouts ../OmniVoice and ../brosoundml/weights/qwen-tts/0.6B-Base;
// with neither present the test skips (exit 0), like the other weight-gated
// tests. The synthetic tokenizer.json check at the end always runs.

#include "brolm/qwen_tokenizer.h"
#include "brolm/detail/json.h"
#include "brolm/detail/unicode.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace qwen = brolm::qwen;
namespace json = brolm::detail::json;
namespace uni = brolm::detail::unicode;
namespace fs = std::filesystem;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

static std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Printable form of arbitrary bytes for failure messages.
static std::string escape(std::string_view s) {
    std::string out;
    char buf[8];
    for (unsigned char c : s) {
        if (c >= 0x20 && c < 0x7F && c != '\\') {
            out += static_cast<char>(c);
        } else {
            std::snprintf(buf, sizeof buf, "\\x%02X", c);
            out += buf;
        }
    }
    return out;
}

static std::string ids_str(const std::vector<int32_t>& ids) {
    std::string out = "[";
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i) out += ",";
        out += std::to_string(ids[i]);
    }
    return out + "]";
}

static std::vector<int32_t> int_array(const json::Value& v) {
    std::vector<int32_t> out;
    for (const auto& e : v.as_array()) out.push_back(static_cast<int32_t>(e.as_number()));
    return out;
}

// First existing path among an env var and the sibling-checkout fallbacks.
static fs::path locate(const char* env, std::initializer_list<fs::path> fallbacks,
                       const char* probe_file) {
    if (const char* d = std::getenv(env)) {
        fs::path p = fs::path(d) / probe_file;
        if (fs::exists(p)) return fs::path(d);
    }
    for (const auto& f : fallbacks) {
        if (fs::exists(f / probe_file)) return f;
    }
    return {};
}

// Compare every fixture case's `key` ids against `encode`, printing the first
// few mismatches with the pre-token-free diagnostic that matters: the text.
template <class Encode>
static int compare_cases(const json::Value& cases, const char* key,
                         const char* label, Encode&& encode) {
    int mismatches = 0;
    int compared = 0;
    for (const auto& c : cases.as_array()) {
        const json::Value* want = c.find(key);
        if (!want) continue;
        const std::string& text = c.at("text").as_string();
        const std::vector<int32_t> expect = int_array(*want);
        const std::vector<int32_t> got = encode(text);
        ++compared;
        if (got != expect) {
            if (++mismatches <= 25) {
                std::fprintf(stderr, "MISMATCH [%s] \"%s\"\n  want %s\n  got  %s\n",
                             label, escape(text).c_str(), ids_str(expect).c_str(),
                             ids_str(got).c_str());
            }
        }
    }
    std::printf("%s: %d/%d cases match\n", label, compared - mismatches, compared);
    return mismatches;
}

int main() {
    const fs::path ref_dir = BROLM_TEST_REF_DIR;
    const fs::path sibling = BROLM_SIBLING_DIR;

    json::Value fixture;
    try {
        fixture = json::parse(slurp(ref_dir / "qwen_tokenizer_unicode.json"));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: fixture: %s\n", e.what());
        return 1;
    }
    const json::Value& cases = fixture.at("cases");
    std::printf("fixture: %zu cases, HF tokenizers %s, Unicode %s (tables: %s)\n",
                cases.as_array().size(),
                fixture.get_string("tokenizers_version", "?").c_str(),
                fixture.get_string("unicode_version", "?").c_str(),
                uni::version());
    CHECK(cases.as_array().size() >= 200);
    CHECK(fixture.get_string("unicode_version", "") == uni::version());

    std::vector<std::string> extras;
    for (const auto& s : fixture.at("extra_special_tokens").as_array()) {
        extras.push_back(s.as_string());
    }
    CHECK(extras.size() == 7);

    // ── Standalone NFC normalizer vs HF's ──────────────────────────────────
    {
        int bad = 0;
        const auto& pairs = fixture.at("nfc").as_array();
        for (const auto& p : pairs) {
            const std::string& in = p.at("in").as_string();
            const std::string& out = p.at("out").as_string();
            const std::string got = uni::nfc(in);
            if (got != out) {
                if (++bad <= 10) {
                    std::fprintf(stderr, "NFC MISMATCH \"%s\"\n  want \"%s\"\n  got  \"%s\"\n",
                                 escape(in).c_str(), escape(out).c_str(),
                                 escape(got).c_str());
                }
            }
            CHECK(!uni::is_nfc(in));          // quick check must not claim NFC
            CHECK(uni::nfc(out) == out);      // idempotent
        }
        std::printf("nfc: %zu/%zu pairs match\n", pairs.size() - bad, pairs.size());
        CHECK(bad == 0);
        // Already-normalized text takes the quick-check fast path.
        CHECK(uni::is_nfc("plain ascii"));
        CHECK(uni::is_nfc("caf\xC3\xA9 \xE4\xB8\xAD\xE6\x96\x87 \xEA\xB0\x81"));
        CHECK(uni::nfc("\xEF\xBF\xBD\xFF\xFE ok") == "\xEF\xBF\xBD\xFF\xFE ok");  // bad bytes pass
    }

    // ── OmniVoice tokenizer.json ───────────────────────────────────────────
    const fs::path omni = locate("BROLM_OMNIVOICE_DIR", {sibling / "OmniVoice"},
                                 "tokenizer.json");
    const fs::path tts = locate(
        "BROLM_QWEN_TTS_DIR",
        {sibling / "brosoundml" / "weights" / "qwen-tts" / "0.6B-Base"}, "vocab.json");

    if (omni.empty()) {
        std::printf("[skip] OmniVoice tokenizer.json not found "
                    "(set BROLM_OMNIVOICE_DIR)\n");
    } else {
        auto tok = qwen::Tokenizer::from_tokenizer_json(
            (omni / "tokenizer.json").string(), extras);
        CHECK(tok.normalizes_nfc());
        CHECK(tok.digit_run_max() == 1);
        CHECK(tok.endoftext_id() == 151643);
        CHECK(tok.im_end_id() == 151645);

        const int bad = compare_cases(cases, "ids", "omnivoice tokenizer.json",
                                      [&](const std::string& s) { return tok.encode(s); });
        CHECK(bad == 0);

        // decode(encode(s)) == NFC(s): lossless for NFC input, and exactly
        // HF's behaviour otherwise.
        int bad_rt = 0;
        for (const auto& c : cases.as_array()) {
            const std::string& text = c.at("text").as_string();
            const std::string back = tok.decode(tok.encode(text));
            if (back != uni::nfc(text)) {
                if (++bad_rt <= 10) {
                    std::fprintf(stderr, "ROUNDTRIP \"%s\" -> \"%s\"\n",
                                 escape(text).c_str(), escape(back).c_str());
                }
            }
        }
        CHECK(bad_rt == 0);

        // Arbitrary bytes, including invalid UTF-8, round-trip untouched.
        const std::string junk =
            "\xFF\xFE bad \xC3 lead \x80 cont \xF4\x90\x80\x80 range \xED\xA0\x80 srg \xC0\xAF";
        CHECK(tok.decode(tok.encode(junk)) == junk);
        CHECK(tok.decode(tok.encode(std::string("nul\0byte", 8))) == std::string("nul\0byte", 8));

        // The seven OmniVoice specials resolve to their added_tokens ids.
        const auto ids = tok.encode("<|denoise|><|lang_start|><|lang_end|><|instruct_start|>"
                                    "<|instruct_end|><|text_start|><|text_end|>");
        CHECK(ids == (std::vector<int32_t>{151669, 151670, 151671, 151672,
                                           151673, 151674, 151675}));
    }

    // ── Qwen3-TTS vocab.json + merges.txt ──────────────────────────────────
    if (tts.empty()) {
        std::printf("[skip] Qwen3-TTS vocab.json + merges.txt not found "
                    "(set BROLM_QWEN_TTS_DIR)\n");
    } else {
        auto tok = qwen::Tokenizer::load((tts / "vocab.json").string(),
                                         (tts / "merges.txt").string());
        CHECK(tok.normalizes_nfc());
        CHECK(tok.digit_run_max() == 1);
        const int bad = compare_cases(cases, "ids_tts", "qwen-tts vocab+merges",
                                      [&](const std::string& s) { return tok.encode(s); });
        CHECK(bad == 0);
    }

    // ── tokenizer.json conventions are read from the file ──────────────────
    // A Llama-3-shaped file: no normalizer, \p{N}{1,3} in the Split regex.
    {
        auto tmp = fs::temp_directory_path() / "brolm_qwen_unicode_llama3_shape.json";
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            f << "{\"normalizer\":null,"
                 "\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":["
                 "{\"type\":\"Split\",\"pattern\":{\"Regex\":"
                 "\"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?\\\\p{L}+|"
                 "\\\\p{N}{1,3}| ?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]+|"
                 "\\\\s+(?!\\\\S)|\\\\s+\"},\"behavior\":\"Isolated\",\"invert\":false},"
                 "{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true,"
                 "\"use_regex\":false}]},"
                 "\"model\":{\"type\":\"BPE\",\"vocab\":{\"1\":17,\"2\":18,\"3\":19,\"123\":20,"
                 "\"4\":21},\"merges\":[\"1 2\",\"12 3\"]},\"added_tokens\":[]}";
        }
        auto tok = qwen::Tokenizer::from_tokenizer_json(tmp.string());
        CHECK(!tok.normalizes_nfc());
        CHECK(tok.digit_run_max() == 3);
        // "1234" -> "123" (one \p{N}{1,3} pre-token, merged) + "4".
        CHECK(tok.encode("1234") == (std::vector<int32_t>{20, 21}));
        std::error_code ec;
        fs::remove(tmp, ec);
    }
    // A Qwen-shaped file with the NFC normalizer and single-digit \p{N}.
    {
        auto tmp = fs::temp_directory_path() / "brolm_qwen_unicode_qwen_shape.json";
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            f << "{\"normalizer\":{\"type\":\"NFC\"},"
                 "\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":["
                 "{\"type\":\"Split\",\"pattern\":{\"Regex\":"
                 "\"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?\\\\p{L}+|"
                 "\\\\p{N}| ?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]+|"
                 "\\\\s+(?!\\\\S)|\\\\s+\"},\"behavior\":\"Isolated\",\"invert\":false},"
                 "{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true,"
                 "\"use_regex\":false}]},"
                 "\"model\":{\"type\":\"BPE\",\"vocab\":{\"1\":17,\"2\":18,\"3\":19,\"123\":20,"
                 "\"4\":21},\"merges\":[\"1 2\",\"12 3\"]},\"added_tokens\":[]}";
        }
        auto tok = qwen::Tokenizer::from_tokenizer_json(tmp.string());
        CHECK(tok.normalizes_nfc());
        CHECK(tok.digit_run_max() == 1);
        CHECK(tok.encode("1234") == (std::vector<int32_t>{17, 18, 19, 21}));
        std::error_code ec;
        fs::remove(tmp, ec);
    }

    if (g_failures == 0) std::printf("qwen_tokenizer_unicode: OK\n");
    else std::fprintf(stderr, "qwen_tokenizer_unicode: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
