#include "brolm/qwen_tokenizer.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace qwen = brolm::qwen;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

// UTF-8-encode a codepoint (the GPT-2 byte-level mapping uses codepoints up to
// ~U+0143, all in the 2-byte range here aside from ASCII).
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

// The GPT-2 byte->unicode form of byte `b` (mirrors the tokenizer's table).
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

// Build a GPT-2-format vocab.json + merges.txt. The vocab includes a
// byte-encoded entry for every one of the 256 bytes (ids 0..255) — exactly
// like a real GPT-2/Qwen vocab, which guarantees any byte sequence is
// representable and round-trips losslessly. On top of that we add a few merged
// pieces and the three Qwen3 special tokens.
//
// Merged-piece ids start at 300 and special-token ids at 400 — both clear of
// the 0..255 byte-id block, so no id is shared between two vocab strings (a
// real Qwen3 vocab assigns every string a unique id; the tokenizer's id->token
// and special-id maps rely on that).
static void write_fixture(const std::filesystem::path& vocab_path,
                          const std::filesystem::path& merges_path) {
    {
        std::ofstream f(vocab_path, std::ios::binary | std::ios::trunc);
        f << "{";
        bool first = true;
        auto add = [&](const std::string& tok, int id) {
            if (!first) f << ",";
            first = false;
            f << "\"";
            for (char c : tok) {
                if (c == '"' || c == '\\') f << '\\' << c;
                else f << c;
            }
            f << "\":" << id;
        };
        // Every byte 0..255 as its byte-level unicode form: id == byte value.
        for (int b = 0; b < 256; ++b) add(byte_unicode(b), b);
        // Merged pieces (ids picked to be unambiguous).
        add("hi", 300);                                    // h + i
        add("ca", 301);                                    // c + a
        add("cat", 302);                                   // ca + t
        const std::string G = byte_unicode(' ');           // 'Ġ'
        add(G + "h", 303);                                 // Ġ + h
        add(G + "hi", 304);                                // Ġh + i
        // Special tokens — ordinary vocab entries (ids clear of byte block).
        add("<|endoftext|>", 400);
        add("<|im_start|>", 401);
        add("<|im_end|>", 402);
        f << "}";
    }
    {
        std::ofstream f(merges_path);
        const std::string G = byte_unicode(' ');           // 'Ġ'
        f << "#version: test\n";
        // Rank order matters. Ġ-merges rank BEFORE 'h i' so that " hi"
        // forms 'Ġhi' rather than collapsing to 'Ġ' + 'hi'.
        f << G << " h\n";          // rank 0: Ġ + h -> Ġh
        f << G << "h i\n";         // rank 1: Ġh + i -> Ġhi
        f << "h i\n";              // rank 2: h + i -> hi
        f << "c a\n";              // rank 3: c + a -> ca
        f << "ca t\n";             // rank 4: ca + t -> cat
    }
}

int main() {
    auto tmp = std::filesystem::temp_directory_path();
    auto vp = tmp / "brolm_qwen_vocab.json";
    auto mp = tmp / "brolm_qwen_merges.txt";
    write_fixture(vp, mp);

    auto tok = qwen::Tokenizer::load(vp.string(), mp.string());

    // 256 byte entries + 5 merged pieces + 3 specials = 264.
    CHECK(tok.vocab_count() == 264);
    CHECK(tok.merge_count() == 5);

    // Special-token ids resolve. eos_id() is <|im_end|> by convention.
    CHECK(tok.endoftext_id() == 400);
    CHECK(tok.im_start_id() == 401);
    CHECK(tok.im_end_id() == 402);
    CHECK(tok.eos_id() == 402);

    // Plain word at string start: "hi" -> ['h','i'] -> merge -> [300].
    {
        auto ids = tok.encode("hi");
        CHECK(ids.size() == 1);
        CHECK(ids[0] == 300);
    }

    // Multi-step cascade: "cat" -> ['c','a','t'] -> (c,a) -> ['ca','t']
    //                          -> (ca,t) -> ['cat'] -> [302].
    {
        auto ids = tok.encode("cat");
        CHECK(ids.size() == 1);
        CHECK(ids[0] == 302);
    }

    // Leading-space (GPT-2 Ġ) convention: a word following a space carries a
    // folded leading space. "a hi" -> "a" at start, then " hi" (Ġ-prefixed).
    // " hi" byte-encodes to Ġ+h+i; rank 0 merges (Ġ,h)->Ġh, rank 1 merges
    // (Ġh,i)->Ġhi -> [304]. 'a' is byte id 'a' (0x61 == 97).
    {
        auto ids = tok.encode("a hi");
        CHECK(ids.size() == 2);
        CHECK(ids[0] == static_cast<int>('a'));   // 97 — single byte token
        CHECK(ids[1] == 304);                     // ' hi' -> Ġhi
    }

    // "hi cat": first word "hi" at start -> [300]; second word " cat" carries
    // a folded space. " cat" byte-encodes Ġ+c+a+t; no Ġc merge exists so it
    // stays ['Ġ','c','a','t'] -> (c,a) -> (ca,t): ['Ġ','cat'] -> [Ġ-id, 302].
    {
        auto ids = tok.encode("hi cat");
        CHECK(ids.size() == 3);
        CHECK(ids[0] == 300);                     // hi
        CHECK(ids[1] == static_cast<int>(' '));   // 32 — standalone Ġ byte
        CHECK(ids[2] == 302);                     // cat
    }

    // Special token embedded in surrounding text: matched verbatim, emitted as
    // its single id; the text around it is BPE-encoded normally.
    {
        auto ids = tok.encode("hi<|im_end|>cat");
        CHECK(ids.size() == 3);
        CHECK(ids[0] == 300);   // hi
        CHECK(ids[1] == 402);   // <|im_end|>
        CHECK(ids[2] == 302);   // cat
    }

    // add_special appends <|endoftext|> as an EOS hook (Qwen3 has no BOS).
    {
        auto ids = tok.encode("hi", /*add_special=*/true);
        CHECK(ids.size() == 2);
        CHECK(ids[0] == 300);
        CHECK(ids[1] == 400);   // <|endoftext|>
    }

    // Round-trip: decode(encode(s)) == s for ASCII text. With the complete
    // 256-byte vocab this is lossless over every byte.
    {
        const char* samples[] = {
            "hi cat", "cat hi a", "a hi cat!", "Hi! CAT",
            "don't stop", "x = 42 + y;", "  spaced  out  ", "tab\there\n"
        };
        for (const char* s : samples) {
            auto ids = tok.encode(s);
            std::string back = tok.decode(ids);
            CHECK(back == s);
        }
    }

    // Round-trip across an embedded special token.
    {
        std::string s = "hi<|im_start|>cat<|im_end|>hi";
        auto ids = tok.encode(s);
        CHECK(tok.decode(ids) == s);
    }

    // Exact id sequence for a controlled round-trip case. " hi" carries a
    // folded leading space and has the full Ġ-merge cascade, so it collapses
    // to a single 'Ġhi' piece -> [304] (distinct from "hi" at start -> [300]).
    {
        auto ids = tok.encode("cat hi");
        CHECK(ids.size() == 2);
        CHECK(ids[0] == 302);   // cat
        CHECK(ids[1] == 304);   // ' hi' -> Ġhi
        CHECK(tok.decode(ids) == "cat hi");
    }

    // Chat template helper produces ChatML and round-trips through encode.
    {
        std::vector<std::pair<std::string, std::string>> msgs = {
            {"user", "hi cat"}
        };
        std::string prompt =
            tok.apply_chat_template(msgs, /*add_generation_prompt=*/true);
        CHECK(prompt ==
              "<|im_start|>user\nhi cat<|im_end|>\n<|im_start|>assistant\n");
        auto ids = tok.encode(prompt);
        // The two specials at the framing edges resolve to their single ids.
        CHECK(ids.front() == 401);   // <|im_start|>
        CHECK(tok.decode(ids) == prompt);
    }

    std::error_code ec;
    std::filesystem::remove(vp, ec);
    std::filesystem::remove(mp, ec);

    if (g_failures == 0) std::printf("qwen_tokenizer: OK\n");
    else std::fprintf(stderr, "qwen_tokenizer: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
