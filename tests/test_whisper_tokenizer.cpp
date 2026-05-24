#include "brolm/whisper_tokenizer.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace whisper = brolm::whisper;

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

static std::string byte_unicode(int b) {
    auto self = [](int x) {
        return (x >= 33 && x <= 126) || (x >= 161 && x <= 172) ||
               (x >= 174 && x <= 255);
    };
    int next = 256;
    for (int i = 0; i < b; ++i) if (!self(i)) ++next;
    return self(b) ? cp_utf8(static_cast<uint32_t>(b))
                   : cp_utf8(static_cast<uint32_t>(next));
}

// Build a synthetic Whisper-shaped vocab.json + merges.txt.
//
// Whisper's vocab has the byte-level base (0..255), some merged pieces, and a
// block of "<|...|>" specials. We mirror that structure on a tiny scale so we
// can exercise auto-registration, the prompt builder, and the timestamp helpers
// without needing a real 51k-entry vocab.
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
        // 0..255: byte-level base.
        for (int b = 0; b < 256; ++b) add(byte_unicode(b), b);
        // A few merged pieces.
        const std::string G = byte_unicode(' ');
        add("hi", 300);
        add(G + "h", 303);
        add(G + "hi", 304);
        // Whisper specials. Ids picked to be clear of the byte block and of
        // each other; the timestamp block is contiguous (500..504) so we can
        // check first/last detection.
        add("<|endoftext|>", 400);
        add("<|startoftranscript|>", 401);
        add("<|en|>", 402);
        add("<|fr|>", 403);
        add("<|transcribe|>", 404);
        add("<|translate|>", 405);
        add("<|notimestamps|>", 406);
        add("<|nospeech|>", 407);
        // Timestamps at 0.02s granularity: 0.00, 0.02, 0.04, 0.06, 0.08.
        add("<|0.00|>", 500);
        add("<|0.02|>", 501);
        add("<|0.04|>", 502);
        add("<|0.06|>", 503);
        add("<|0.08|>", 504);
        f << "}";
    }
    {
        std::ofstream f(merges_path);
        const std::string G = byte_unicode(' ');
        f << "#version: test\n";
        f << G << " h\n";
        f << G << "h i\n";
        f << "h i\n";
    }
}

int main() {
    auto tmp = std::filesystem::temp_directory_path();
    auto vp = tmp / "brolm_whisper_vocab.json";
    auto mp = tmp / "brolm_whisper_merges.txt";
    write_fixture(vp, mp);

    auto tok = whisper::Tokenizer::load(vp.string(), mp.string());

    // 256 bytes + 3 merged + 13 specials = 272.
    CHECK(tok.vocab_count() == 272);

    // Well-known specials auto-registered.
    CHECK(tok.eos_id()           == 400);
    CHECK(tok.sot_id()           == 401);
    CHECK(tok.transcribe_id()    == 404);
    CHECK(tok.translate_id()     == 405);
    CHECK(tok.no_timestamps_id() == 406);
    CHECK(tok.no_speech_id()     == 407);

    // Timestamp range detection.
    CHECK(tok.first_timestamp_id() == 500);
    CHECK(tok.last_timestamp_id()  == 504);
    CHECK(tok.is_timestamp(500));
    CHECK(tok.is_timestamp(504));
    CHECK(!tok.is_timestamp(499));
    CHECK(!tok.is_timestamp(505));
    CHECK(std::fabs(tok.timestamp_seconds(500) - 0.00f) < 1e-6f);
    CHECK(std::fabs(tok.timestamp_seconds(502) - 0.04f) < 1e-6f);
    CHECK(std::fabs(tok.timestamp_seconds(504) - 0.08f) < 1e-6f);

    // Arbitrary token lookup.
    CHECK(tok.token_to_id("<|en|>") == 402);
    CHECK(tok.token_to_id("<|fr|>") == 403);
    CHECK(tok.token_to_id("<|ja|>") == -1);

    // Encoding of ordinary text reuses the GPT-2 byte-level pre-tokenization.
    {
        auto ids = tok.encode("hi");
        CHECK(ids.size() == 1);
        CHECK(ids[0] == 300);
    }
    {
        // "a hi" -> 'a' + ' hi' (Ġ-folded); ' hi' merges to Ġhi -> [304].
        auto ids = tok.encode("a hi");
        CHECK(ids.size() == 2);
        CHECK(ids[0] == static_cast<int>('a'));
        CHECK(ids[1] == 304);
    }

    // Special-token mid-stream is matched verbatim.
    {
        auto ids = tok.encode("hi<|endoftext|>");
        CHECK(ids.size() == 2);
        CHECK(ids[0] == 300);
        CHECK(ids[1] == 400);
    }

    // add_special appends <|endoftext|>.
    {
        auto ids = tok.encode("hi", /*add_special=*/true);
        CHECK(ids.size() == 2);
        CHECK(ids[1] == 400);
    }

    // Round-trip with and without special-skipping.
    {
        auto ids = tok.encode("hi<|en|>hi");
        CHECK(tok.decode(ids) == "hi<|en|>hi");
        CHECK(tok.decode(ids, /*skip_special=*/true) == "hihi");
    }

    // Prompt builder: <|sot|> <|lang|> <|task|> [<|notimestamps|>]
    {
        auto p = tok.build_prompt("en", "transcribe", /*with_timestamps=*/true);
        CHECK(p.size() == 3);
        CHECK(p[0] == 401);
        CHECK(p[1] == 402);
        CHECK(p[2] == 404);
    }
    {
        auto p = tok.build_prompt("fr", "translate", /*with_timestamps=*/false);
        CHECK(p.size() == 4);
        CHECK(p[0] == 401);
        CHECK(p[1] == 403);
        CHECK(p[2] == 405);
        CHECK(p[3] == 406);
    }

    // build_prompt error paths.
    {
        bool threw = false;
        try { (void)tok.build_prompt("xx", "transcribe"); }
        catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
    }
    {
        bool threw = false;
        try { (void)tok.build_prompt("en", "sing"); }
        catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
    }

    // Round-trip of a realistic mixed-decoded sequence:
    // sot lang task <|0.00|> "hi" <|0.04|> eos
    {
        std::vector<int32_t> ids = {401, 402, 404, 500, 300, 502, 400};
        std::string full = tok.decode(ids);
        CHECK(full == "<|startoftranscript|><|en|><|transcribe|><|0.00|>hi<|0.04|><|endoftext|>");
        std::string text = tok.decode(ids, /*skip_special=*/true);
        CHECK(text == "hi");
    }

    std::error_code ec;
    std::filesystem::remove(vp, ec);
    std::filesystem::remove(mp, ec);

    if (g_failures == 0) std::printf("whisper_tokenizer: OK\n");
    else std::fprintf(stderr, "whisper_tokenizer: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
