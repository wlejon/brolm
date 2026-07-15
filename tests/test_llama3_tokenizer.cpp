// Llama-3 tokenizer smoke + correctness test.
//
// Builds a synthetic HF unified tokenizer.json (model.vocab + model.merges +
// added_tokens) shaped exactly like a Llama-3 checkpoint — a full byte-level
// vocab (ids 0..255), a handful of merges, and the control tokens in
// added_tokens at ids 128000+ (which are NOT in model.vocab, mirroring the real
// file). Then verifies:
//   1. from_tokenizer_json loads vocab + merges;
//   2. added_tokens are registered as atomic specials and named ids resolve;
//   3. encode() prepends <|begin_of_text|>;
//   4. byte-level BPE merges cascade correctly (shared with the qwen path);
//   5. a special embedded in text is matched verbatim;
//   6. decode(encode(s)) round-trips ASCII;
//   7. the Llama-3 chat template renders and re-encodes with the header/turn
//      specials as single atomic ids;
//   8. the newer ["a","b"] merges-array format parses equivalently.
//
// Numerical/text parity against the real 128k Llama-3 tokenizer needs the
// shipped tokenizer.json and lives in a future gated test — Llama-3's
// tiktoken-style Unicode-property pre-tokenization is not reproduced here (the
// inherited ASCII GPT-2 approximation matches HF on English/code).

#include "brolm/llama3_tokenizer.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace llama3 = brolm::llama3;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

static std::string cp_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += static_cast<char>(cp);
    } else if (cp < 0x800) {
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        s += static_cast<char>(0xE0 | (cp >> 12));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return s;
}

// GPT-2 byte->unicode form of byte `b` (mirrors the tokenizer's table).
static std::string byte_unicode(int b) {
    auto self = [](int x) {
        return (x >= 33 && x <= 126) || (x >= 161 && x <= 172) ||
               (x >= 174 && x <= 255);
    };
    int next = 256;
    for (int i = 0; i < b; ++i)
        if (!self(i)) ++next;
    return self(b) ? cp_utf8(static_cast<uint32_t>(b))
                   : cp_utf8(static_cast<uint32_t>(next));
}

static void json_escape(std::ofstream& f, const std::string& s) {
    f << '"';
    for (char c : s) {
        if (c == '"' || c == '\\') f << '\\' << c;
        else f << c;
    }
    f << '"';
}

// Build a Llama-3-shaped tokenizer.json. `merges_as_pairs` selects the classic
// "a b" string form (false) vs the newer ["a","b"] array form (true).
static void write_fixture(const std::filesystem::path& path,
                          bool merges_as_pairs) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f << "{\"model\":{\"type\":\"BPE\",\"vocab\":{";
    {
        bool first = true;
        auto add = [&](const std::string& tok, int id) {
            if (!first) f << ",";
            first = false;
            json_escape(f, tok);
            f << ":" << id;
        };
        for (int b = 0; b < 256; ++b) add(byte_unicode(b), b);
        add("hi", 300);                            // h + i
        add("ca", 301);                            // c + a
        add("cat", 302);                           // ca + t
        const std::string G = byte_unicode(' ');   // 'Ġ'
        add(G + "c", 303);                         // Ġ + c
    }
    f << "},\"merges\":[";
    {
        const std::string G = byte_unicode(' ');
        // (a, b) pairs in rank order.
        const std::pair<std::string, std::string> merges[] = {
            {"h", "i"}, {"c", "a"}, {"ca", "t"}, {G, "c"},
        };
        bool first = true;
        for (const auto& [a, b] : merges) {
            if (!first) f << ",";
            first = false;
            if (merges_as_pairs) {
                f << "[";
                json_escape(f, a);
                f << ",";
                json_escape(f, b);
                f << "]";
            } else {
                json_escape(f, a + " " + b);
            }
        }
    }
    f << "]},\"added_tokens\":[";
    {
        // Control tokens at Llama-3 ids — NOT present in model.vocab.
        const std::pair<int, const char*> specials[] = {
            {128000, "<|begin_of_text|>"},
            {128001, "<|end_of_text|>"},
            {128006, "<|start_header_id|>"},
            {128007, "<|end_header_id|>"},
            {128009, "<|eot_id|>"},
        };
        bool first = true;
        for (const auto& [id, content] : specials) {
            if (!first) f << ",";
            first = false;
            f << "{\"id\":" << id << ",\"content\":";
            json_escape(f, content);
            f << ",\"special\":true}";
        }
    }
    f << "]}";
}

int main() {
    auto tmp = std::filesystem::temp_directory_path();
    auto jp = tmp / "brolm_llama3_tokenizer.json";
    write_fixture(jp, /*merges_as_pairs=*/false);

    auto tok = llama3::Tokenizer::load(jp.string());

    // 256 bytes + 4 merged pieces + 5 added specials = 265.
    CHECK(tok.vocab_count() == 265);
    CHECK(tok.merge_count() == 4);

    // Named control ids resolve straight from added_tokens.
    CHECK(tok.bos_id() == 128000);
    CHECK(tok.end_of_text_id() == 128001);
    CHECK(tok.start_header_id() == 128006);
    CHECK(tok.end_header_id() == 128007);
    CHECK(tok.eot_id() == 128009);
    CHECK(tok.eos_id() == 128009);   // eos == eot by convention

    // encode() prepends BOS by default: "hi" -> [BOS, 300].
    {
        auto ids = tok.encode("hi");
        CHECK(ids.size() == 2);
        CHECK(ids[0] == 128000);
        CHECK(ids[1] == 300);
    }

    // add_bos=false omits it. Cascade "cat" -> ['c','a','t'] -> ca -> cat -> 302.
    {
        auto ids = tok.encode("cat", /*add_bos=*/false);
        CHECK(ids.size() == 1);
        CHECK(ids[0] == 302);
    }

    // Special embedded in text: matched verbatim between BPE spans.
    {
        auto ids = tok.encode("hi<|eot_id|>cat", /*add_bos=*/false);
        CHECK(ids.size() == 3);
        CHECK(ids[0] == 300);      // hi
        CHECK(ids[1] == 128009);   // <|eot_id|>
        CHECK(ids[2] == 302);      // cat
    }

    // Round-trip ASCII (decode drops the special ids to their literal form).
    {
        const std::string s = "hi cat hi";
        auto ids = tok.encode(s, /*add_bos=*/false);
        CHECK(tok.decode(ids) == s);
    }

    // Chat template renders Llama-3 headers/turns and re-encodes with the
    // header + eot specials as single atomic ids, BOS from the template.
    {
        std::vector<std::pair<std::string, std::string>> msgs = {
            {"user", "hi"},
        };
        std::string prompt = tok.apply_chat_template(msgs, /*gen_prompt=*/true);
        CHECK(prompt.rfind("<|begin_of_text|>", 0) == 0);
        // Template already carries BOS — encode without adding another.
        auto ids = tok.encode(prompt, /*add_bos=*/false);
        // BOS present exactly once (from the template).
        int bos_count = 0, hdr_open = 0, hdr_close = 0, eot = 0;
        for (int id : ids) {
            if (id == 128000) ++bos_count;
            if (id == 128006) ++hdr_open;
            if (id == 128007) ++hdr_close;
            if (id == 128009) ++eot;
        }
        CHECK(bos_count == 1);
        CHECK(hdr_open == 2);    // user header + assistant gen-prompt header
        CHECK(hdr_close == 2);
        CHECK(eot == 1);         // one closed user turn
    }

    // Newer ["a","b"] merges-array format parses to the same result.
    {
        auto jp2 = tmp / "brolm_llama3_tokenizer_pairs.json";
        write_fixture(jp2, /*merges_as_pairs=*/true);
        auto tok2 = llama3::Tokenizer::load(jp2.string());
        CHECK(tok2.merge_count() == 4);
        auto ids = tok2.encode("cat", /*add_bos=*/false);
        CHECK(ids.size() == 1);
        CHECK(ids[0] == 302);
        std::error_code ec;
        std::filesystem::remove(jp2, ec);
    }

    std::error_code ec;
    std::filesystem::remove(jp, ec);

    if (g_failures == 0) std::printf("llama3_tokenizer: OK\n");
    else std::fprintf(stderr, "llama3_tokenizer: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
