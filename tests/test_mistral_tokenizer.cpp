#include "brolm/mistral_tokenizer.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace mistral = brolm::mistral;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

// ─── base64 (encoder, for building the synthetic tekken.json) ───────────────
static std::string b64(const std::string& raw) {
    static const char* T =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    std::size_t i = 0;
    while (i + 3 <= raw.size()) {
        uint32_t n = (static_cast<unsigned char>(raw[i]) << 16) |
                     (static_cast<unsigned char>(raw[i + 1]) << 8) |
                     (static_cast<unsigned char>(raw[i + 2]));
        out += T[(n >> 18) & 63];
        out += T[(n >> 12) & 63];
        out += T[(n >> 6) & 63];
        out += T[n & 63];
        i += 3;
    }
    const std::size_t rem = raw.size() - i;
    if (rem == 1) {
        uint32_t n = static_cast<unsigned char>(raw[i]) << 16;
        out += T[(n >> 18) & 63];
        out += T[(n >> 12) & 63];
        out += "==";
    } else if (rem == 2) {
        uint32_t n = (static_cast<unsigned char>(raw[i]) << 16) |
                     (static_cast<unsigned char>(raw[i + 1]) << 8);
        out += T[(n >> 18) & 63];
        out += T[(n >> 12) & 63];
        out += T[(n >> 6) & 63];
        out += '=';
    }
    return out;
}

// num_special_tokens for the synthetic fixture; regular id = rank + this.
static constexpr int NSPECIAL = 10;
static int byte_id(int b) { return b + NSPECIAL; }

// Build a synthetic tekken.json with a complete 256-byte vocab plus a handful
// of merged tokens, a small reserved special-token block, and a few unknown /
// nested keys to exercise the streaming reader's skip paths. Ranks:
//   single bytes : rank == byte value (0..255)
//   "hi"=256 "ca"=257 "cat"=258 " hi"=259
// Crucially, no "at"/" c"/" h" merge exists, so each cascade is deterministic.
static void write_fixture(const std::filesystem::path& path) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);

    auto vtok = [&](const std::string& raw, int rank, bool& first) {
        if (!first) f << ",";
        first = false;
        f << "{\"rank\":" << rank << ",\"token_bytes\":\"" << b64(raw)
          << "\",\"token_str\":null}";
    };
    auto stok = [&](const std::string& s, int rank, bool& first) {
        if (!first) f << ",";
        first = false;
        f << "{\"rank\":" << rank << ",\"token_str\":\"" << s
          << "\",\"is_control\":true}";
    };

    f << "{";
    // config first — exercises key order independence (vocab parsed after).
    f << "\"type\":\"Tekken\",\"version\":7,";
    f << "\"config\":{\"pattern\":\"[unused approx]\",\"num_vocab_tokens\":270,"
         "\"default_vocab_size\":270,\"default_num_special_tokens\":" << NSPECIAL
      << ",\"version\":\"v7\"},";
    // an unknown nested object to skip.
    f << "\"image\":{\"image_token\":\"[IMG]\",\"sizes\":[14,14,2]},";

    f << "\"vocab\":[";
    {
        bool first = true;
        for (int b = 0; b < 256; ++b) vtok(std::string(1, static_cast<char>(b)), b, first);
        vtok("hi", 256, first);
        vtok("ca", 257, first);
        vtok("cat", 258, first);
        vtok(" hi", 259, first);            // space + h + i
    }
    f << "],";

    f << "\"special_tokens\":[";
    {
        bool first = true;
        stok("<unk>", 0, first);
        stok("<s>", 1, first);
        stok("</s>", 2, first);
        stok("[INST]", 3, first);
        stok("[/INST]", 4, first);
        stok("<pad>", 5, first);
        stok("[SYSTEM_PROMPT]", 6, first);
        stok("[/SYSTEM_PROMPT]", 7, first);
        // ranks 8,9 left unfilled -> <SPECIAL_8>, <SPECIAL_9> fillers.
    }
    f << "]";
    f << "}";
}

static void test_synthetic() {
    auto tmp = std::filesystem::temp_directory_path();
    auto path = tmp / "brolm_tekken.json";
    write_fixture(path);

    auto tok = mistral::Tokenizer::load(path.string());

    // ── id math ──
    CHECK(tok.num_special_tokens() == NSPECIAL);
    CHECK(tok.vocab_size() == 270);
    CHECK(tok.vocab_count() == 260);        // 256 bytes + 4 merged
    CHECK(tok.special_count() == 10);       // 8 explicit + 2 fillers

    // ── named special ids ──
    CHECK(tok.unk_id() == 0);
    CHECK(tok.bos_id() == 1);
    CHECK(tok.eos_id() == 2);
    CHECK(tok.inst_id() == 3);
    CHECK(tok.inst_end_id() == 4);
    CHECK(tok.pad_id() == 5);
    CHECK(tok.special_id("[SYSTEM_PROMPT]") == 6);
    CHECK(tok.special_id("[/SYSTEM_PROMPT]") == 7);
    CHECK(tok.img_id() == -1);              // not declared in this fixture

    // ── fillers ──
    CHECK(tok.special_id("<SPECIAL_8>") == 8);
    CHECK(tok.special_id("<SPECIAL_9>") == 9);
    CHECK(tok.decode({8}) == "<SPECIAL_8>");

    // ── tiktoken raw-byte merges ──
    { auto ids = tok.encode("hi");  CHECK(ids.size() == 1 && ids[0] == 256 + NSPECIAL); }
    { auto ids = tok.encode("cat"); CHECK(ids.size() == 1 && ids[0] == 258 + NSPECIAL); }

    // Leading space folds into a following letter run: " hi" -> single token.
    {
        auto ids = tok.encode("a hi");
        CHECK(ids.size() == 2);
        CHECK(ids[0] == byte_id('a'));
        CHECK(ids[1] == 259 + NSPECIAL);    // " hi"
    }

    // " cat" has no " c" merge, so the space stays a lone byte: [' ', 'cat'].
    {
        auto ids = tok.encode("hi cat");
        CHECK(ids.size() == 3);
        CHECK(ids[0] == 256 + NSPECIAL);    // hi
        CHECK(ids[1] == byte_id(' '));
        CHECK(ids[2] == 258 + NSPECIAL);    // cat
    }

    // Digits split individually (Tekken's \p{N}): "1234" -> four byte tokens.
    {
        auto ids = tok.encode("1234");
        CHECK(ids.size() == 4);
        CHECK(ids[0] == byte_id('1'));
        CHECK(ids[1] == byte_id('2'));
        CHECK(ids[2] == byte_id('3'));
        CHECK(ids[3] == byte_id('4'));
    }

    // Special token embedded in text is emitted atomically.
    {
        auto ids = tok.encode("hi[INST]cat");
        CHECK(ids.size() == 3);
        CHECK(ids[0] == 256 + NSPECIAL);    // hi
        CHECK(ids[1] == 3);                 // [INST]
        CHECK(ids[2] == 258 + NSPECIAL);    // cat
    }

    // add_special prepends BOS.
    {
        auto ids = tok.encode("hi", /*add_special=*/true);
        CHECK(ids.size() == 2);
        CHECK(ids[0] == 1);                 // <s>
        CHECK(ids[1] == 256 + NSPECIAL);
    }

    // ── round-trips (lossless over raw bytes) ──
    {
        const char* samples[] = {
            "hi cat", "cat hi a", "a hi cat!", "Hi! CAT",
            "don't stop", "x = 42 + y;", "  spaced  out  ", "tab\there\n",
            "1234567890", "mixed123text456"
        };
        for (const char* s : samples) {
            auto ids = tok.encode(s);
            CHECK(tok.decode(ids) == s);
        }
    }
    // Round-trip across an embedded special.
    {
        std::string s = "hi[INST]cat[/INST]hi";
        CHECK(tok.decode(tok.encode(s)) == s);
    }

    // ── chat template ──
    {
        std::vector<std::pair<std::string, std::string>> msgs = {
            {"system", "sys"}, {"user", "hi"}
        };
        std::string prompt = tok.apply_chat_template(msgs);
        CHECK(prompt == "<s>[SYSTEM_PROMPT]sys[/SYSTEM_PROMPT][INST]hi[/INST]");
        auto ids = tok.encode(prompt);          // template already carries <s>
        CHECK(ids.front() == 1);                // <s>
        CHECK(ids[1] == 6);                     // [SYSTEM_PROMPT]
        CHECK(tok.decode(ids) == prompt);
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// A tekken.json with NO special_tokens list — as Mistral's real checkpoints
// ship. The loader must fall back to the canonical Tekken v7 table.
static void test_default_specials() {
    constexpr int N2 = 32;
    auto tmp = std::filesystem::temp_directory_path();
    auto path = tmp / "brolm_tekken_nospecials.json";
    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        f << "{\"config\":{\"default_num_special_tokens\":" << N2
          << ",\"default_vocab_size\":" << (N2 + 256) << ",\"version\":\"v7\"},";
        f << "\"vocab\":[";
        bool first = true;
        for (int b = 0; b < 256; ++b) {
            if (!first) f << ",";
            first = false;
            f << "{\"rank\":" << b << ",\"token_bytes\":\""
              << b64(std::string(1, static_cast<char>(b))) << "\"}";
        }
        f << "]}";   // no "special_tokens" key
    }

    auto tok = mistral::Tokenizer::load(path.string());

    CHECK(tok.num_special_tokens() == N2);
    CHECK(tok.special_count() == N2);
    CHECK(tok.vocab_count() == 256);

    // Canonical Tekken v7 ids resolve from the built-in default table.
    CHECK(tok.unk_id() == 0);
    CHECK(tok.bos_id() == 1);
    CHECK(tok.eos_id() == 2);
    CHECK(tok.inst_id() == 3);
    CHECK(tok.inst_end_id() == 4);
    CHECK(tok.special_id("[TOOL_CALLS]") == 9);
    CHECK(tok.img_id() == 10);
    CHECK(tok.pad_id() == 11);
    CHECK(tok.img_break_id() == 12);
    CHECK(tok.img_end_id() == 13);
    CHECK(tok.special_id("[SYSTEM_PROMPT]") == 17);
    CHECK(tok.special_id("[TOOL_CONTENT]") == 19);
    // Slots past the named table are <SPECIAL_n> fillers.
    CHECK(tok.special_id("<SPECIAL_20>") == 20);
    CHECK(tok.special_id("<SPECIAL_31>") == 31);

    CHECK(tok.decode({1}) == "<s>");
    CHECK(tok.decode({3}) == "[INST]");
    { auto ids = tok.encode("[INST]"); CHECK(ids.size() == 1 && ids[0] == 3); }
    { auto ids = tok.encode("hi", true); CHECK(!ids.empty() && ids.front() == 1); }

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// Real-checkpoint smoke test, gated on BROLM_MISTRAL_DIR (skips when absent).
static void test_real_checkpoint() {
    const char* dir = std::getenv("BROLM_MISTRAL_DIR");
    if (!dir) {
        std::printf("mistral_tokenizer: [skip] BROLM_MISTRAL_DIR not set\n");
        return;
    }
    std::filesystem::path tj = std::filesystem::path(dir) / "tekken.json";
    if (!std::filesystem::exists(tj)) {
        std::printf("mistral_tokenizer: [skip] %s not found\n", tj.string().c_str());
        return;
    }

    auto tok = mistral::Tokenizer::load(tj.string());
    CHECK(tok.num_special_tokens() == 1000);
    CHECK(tok.vocab_size() == 131072);
    CHECK(tok.bos_id() >= 0 && tok.bos_id() < 1000);
    CHECK(tok.eos_id() >= 0 && tok.eos_id() < 1000);
    CHECK(tok.inst_id() >= 0 && tok.inst_id() < 1000);
    CHECK(tok.img_id() >= 0);                 // Mistral 3.1 carries [IMG]

    // [INST] matches atomically.
    {
        auto ids = tok.encode("[INST]");
        CHECK(ids.size() == 1 && ids[0] == tok.inst_id());
    }
    // Plain text encodes to regular ids (>= num_special_tokens) and round-trips.
    {
        const std::string s = "The quick brown fox jumps over the lazy dog.";
        auto ids = tok.encode(s);
        CHECK(!ids.empty());
        for (int id : ids) CHECK(id >= tok.num_special_tokens());
        CHECK(tok.decode(ids) == s);
    }
    // BOS prefix.
    {
        auto ids = tok.encode("hello", /*add_special=*/true);
        CHECK(!ids.empty() && ids.front() == tok.bos_id());
    }
    std::printf("mistral_tokenizer: real checkpoint OK (%s)\n", tj.string().c_str());
}

int main() {
    test_synthetic();
    test_default_specials();
    test_real_checkpoint();

    if (g_failures == 0) std::printf("mistral_tokenizer: OK\n");
    else std::fprintf(stderr, "mistral_tokenizer: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
