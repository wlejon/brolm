// NLLB-200 (M2M-100) SentencePiece-metaspace BPE tokenizer smoke test.
//
// Writes a synthetic HF tokenizer.json with a small BPE vocab + merges plus an
// added_tokens array carrying the specials (<s>/<pad>/</s>/<unk>) and two
// FLORES-200 language codes, then verifies language-code lookup, metaspace BPE
// merging, the source / decoder sequence framing, and skip-special decoding.

#include "brolm/tokenizer_nllb.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace nllb = brolm::nllb;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

// Base BPE vocab (piece -> id): specials at the front (so </s> == 2, as in real
// NLLB), then the metaspace pieces and the single symbols the merges build from.
// Language codes live ONLY in added_tokens at high ids — the case that matters
// for NLLB.
//   0:<s> 1:<pad> 2:</s> 3:<unk> 4:▁ 5:▁a 6:a 7:cat 8:▁cat 9:c 10:at 11:t 12:ca
// merges (ranked): "▁ a"->▁a, "c a"->ca, "ca t"->cat, "▁ cat"->▁cat, so
//   "a cat" -> metaspace ["▁a","▁cat"] -> [5, 8].
// added_tokens: the four specials (echoing their ids) + eng_Latn=100, fra_Latn=101
static void write_fixture(const std::filesystem::path& path) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    const std::string M = "\xE2\x96\x81";  // U+2581 metaspace
    f << "{";
    f << "\"added_tokens\":[";
    f << "{\"id\":0,\"content\":\"<s>\",\"special\":true},";
    f << "{\"id\":1,\"content\":\"<pad>\",\"special\":true},";
    f << "{\"id\":2,\"content\":\"</s>\",\"special\":true},";
    f << "{\"id\":3,\"content\":\"<unk>\",\"special\":true},";
    f << "{\"id\":100,\"content\":\"eng_Latn\",\"special\":true},";
    f << "{\"id\":101,\"content\":\"fra_Latn\",\"special\":true}";
    f << "],";
    f << "\"model\":{";
    f << "\"type\":\"BPE\",";
    f << "\"unk_token\":\"<unk>\",";
    f << "\"vocab\":{";
    f << "\"<s>\":0,\"<pad>\":1,\"</s>\":2,\"<unk>\":3,";
    f << "\"" << M << "\":4,\"" << M << "a\":5,\"a\":6,\"cat\":7,";
    f << "\"" << M << "cat\":8,\"c\":9,\"at\":10,\"t\":11,\"ca\":12";
    f << "},";
    f << "\"merges\":[";
    f << "\"" << M << " a\",";
    f << "\"c a\",";
    f << "\"ca t\",";
    f << "\"" << M << " cat\"";
    f << "]";
    f << "}";
    f << "}";
}

int main() {
    auto tmp = std::filesystem::temp_directory_path();
    auto path = tmp / "brolm_nllb_tokenizer.json";
    write_fixture(path);

    auto tok = nllb::Tokenizer::load(path.string());

    CHECK(tok.bos_id() == 0);
    CHECK(tok.pad_id() == 1);
    CHECK(tok.eos_id() == 2);
    CHECK(tok.unk_id() == 3);

    // Language codes resolve from added_tokens (not present in model.vocab).
    CHECK(tok.language_count() == 2);
    CHECK(tok.has_lang("eng_Latn"));
    CHECK(tok.has_lang("fra_Latn"));
    CHECK(!tok.has_lang("xxx_Yyyy"));
    CHECK(tok.lang_id("eng_Latn") == 100);
    CHECK(tok.lang_id("fra_Latn") == 101);

    // Unknown language code must throw.
    {
        bool threw = false;
        try { (void)tok.lang_id("zzz_Zzzz"); }
        catch (const std::exception&) { threw = true; }
        CHECK(threw);
    }

    // "a cat" -> metaspace "▁a▁cat" -> best [▁a=5][▁cat=8].
    {
        auto ids = tok.tokenize("a cat");
        CHECK(ids.size() == 2);
        if (ids.size() == 2) { CHECK(ids[0] == 5); CHECK(ids[1] == 8); }
    }

    // Source framing: [src_lang] pieces... </s>.
    {
        auto ids = tok.encode_source("a cat", "eng_Latn");
        const std::vector<int32_t> expected = {100, 5, 8, 2};
        CHECK(ids == expected);
    }

    // Decoder prefill: </s> then forced target-language BOS.
    {
        auto ids = tok.decoder_start("fra_Latn");
        const std::vector<int32_t> expected = {2, 101};
        CHECK(ids == expected);
    }

    // skip-special decode drops the language code and </s>, recovering text.
    {
        std::vector<int32_t> ids = {101, 5, 8, 2};
        CHECK(tok.decode(ids) == "a cat");
        // Without skipping, the language-code/eos pieces are not in the base
        // vocab range as text, but the base pieces still join.
        CHECK(tok.decode(ids, /*skip_special=*/false).find("a cat") !=
              std::string::npos);
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);

    if (g_failures == 0) std::printf("tokenizer_nllb: OK\n");
    else std::fprintf(stderr, "tokenizer_nllb: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
