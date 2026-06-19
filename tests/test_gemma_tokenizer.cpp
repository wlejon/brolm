// Gemma / Gemma-2 SentencePiece-BPE tokenizer test.
//
// Writes a synthetic HF tokenizer.json with a small BPE vocab + merges and a
// byte-fallback pair, then verifies: the metaspace BPE merge, "no prefix space"
// (Gemma does not prepend a leading metaspace), byte-fallback on a non-vocab
// UTF-8 char, verbatim special-token matching, BOS/EOS framing, and round-trip
// decode.
//
// A second section, gated on env var BROLM_GEMMA_DIR (a Gemma-2 HF dir), loads
// the real <dir>/tokenizer.json and asserts a known encoding. It skips cleanly
// (prints a line, exits 0) when the var is unset or the file is absent.

#define _CRT_SECURE_NO_WARNINGS   // std::getenv for the gated checkpoint path

#include "brolm/gemma_tokenizer.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace gemma = brolm::gemma;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

// Synthetic Gemma-shaped tokenizer.json. byte_fallback on, no prefix space.
//   vocab: 0:<pad> 1:<eos> 2:<bos> 3:<unk> 4:▁ 5:a 6:c 7:t 8:ca 9:cat
//          10:▁cat 11:<0xC3> 12:<0xA9> 13:<start_of_turn>
//   merges (rank order): "c a"->ca, "ca t"->cat, "▁ cat"->▁cat
// So "a cat" (no prefix space) -> word "a"=[5], word "▁cat"=[10] -> [5,10].
// "é" (U+00E9 = bytes C3 A9, not in vocab) -> byte-fallback [11,12].
static void write_fixture(const std::filesystem::path& path) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    const std::string M = "\xE2\x96\x81";  // U+2581 metaspace
    f << "{";
    f << "\"added_tokens\":[";
    f << "{\"id\":0,\"content\":\"<pad>\",\"special\":true},";
    f << "{\"id\":1,\"content\":\"<eos>\",\"special\":true},";
    f << "{\"id\":2,\"content\":\"<bos>\",\"special\":true},";
    f << "{\"id\":3,\"content\":\"<unk>\",\"special\":true},";
    f << "{\"id\":13,\"content\":\"<start_of_turn>\",\"special\":true}";
    f << "],";
    f << "\"model\":{";
    f << "\"type\":\"BPE\",";
    f << "\"unk_token\":\"<unk>\",";
    f << "\"byte_fallback\":true,";
    f << "\"fuse_unk\":false,";
    f << "\"vocab\":{";
    f << "\"<pad>\":0,\"<eos>\":1,\"<bos>\":2,\"<unk>\":3,";
    f << "\"" << M << "\":4,\"a\":5,\"c\":6,\"t\":7,\"ca\":8,\"cat\":9,";
    f << "\"" << M << "cat\":10,";
    f << "\"<0xC3>\":11,\"<0xA9>\":12,";
    f << "\"<start_of_turn>\":13";
    f << "},";
    f << "\"merges\":[";
    f << "\"c a\",";
    f << "\"ca t\",";
    f << "\"" << M << " cat\"";
    f << "]";
    f << "}";
    f << "}";
}

static void run_unit_tests() {
    auto tmp = std::filesystem::temp_directory_path();
    auto path = tmp / "brolm_gemma_tokenizer.json";
    write_fixture(path);

    auto tok = gemma::Tokenizer::load(path.string());

    CHECK(tok.pad_id() == 0);
    CHECK(tok.eos_id() == 1);
    CHECK(tok.bos_id() == 2);
    CHECK(tok.unk_id() == 3);
    CHECK(tok.vocab_count() == 14);

    // "no prefix space": "a" is a single piece [5], NOT [▁=4, a=5]. With a
    // leading metaspace (T5/NLLB style) it would split into two tokens.
    {
        auto ids = tok.tokenize("a");
        CHECK(ids.size() == 1);
        if (ids.size() == 1) CHECK(ids[0] == 5);
    }

    // "a cat" -> "a" + "▁cat" -> [5, 10] via the merge loop.
    {
        auto ids = tok.tokenize("a cat");
        CHECK(ids.size() == 2);
        if (ids.size() == 2) { CHECK(ids[0] == 5); CHECK(ids[1] == 10); }
    }

    // Round-trip decode, including the leading-space (▁cat) case — no prefix
    // space means a real leading space is preserved, not stripped.
    {
        std::vector<int32_t> a_cat = {5, 10};
        CHECK(tok.decode(a_cat) == "a cat");
        std::vector<int32_t> sp_cat = {10};
        CHECK(tok.decode(sp_cat) == " cat");
        CHECK(tok.decode(tok.tokenize("a cat")) == "a cat");
    }

    // Byte-fallback: "é" (U+00E9, bytes C3 A9) is not a vocab char, so it
    // decomposes to the byte-pieces [<0xC3>=11, <0xA9>=12]; decode rebuilds it.
    {
        auto ids = tok.tokenize("\xC3\xA9");  // "é"
        CHECK(ids.size() == 2);
        if (ids.size() == 2) { CHECK(ids[0] == 11); CHECK(ids[1] == 12); }
        CHECK(tok.decode(ids) == "\xC3\xA9");
    }

    // BOS prepend (default) / EOS append framing.
    {
        auto with_bos = tok.encode("a cat");                 // add_bos=true
        const std::vector<int32_t> e1 = {2, 5, 10};
        CHECK(with_bos == e1);

        auto no_bos = tok.encode("a cat", /*add_bos=*/false);
        const std::vector<int32_t> e2 = {5, 10};
        CHECK(no_bos == e2);

        auto bos_eos = tok.encode("a cat", /*add_bos=*/true, /*add_eos=*/true);
        const std::vector<int32_t> e3 = {2, 5, 10, 1};
        CHECK(bos_eos == e3);
    }

    // Verbatim special-token matching: a control token in the text is emitted
    // atomically as its id, surrounding text BPE-encoded normally.
    {
        auto ids = tok.tokenize("<start_of_turn>a cat");
        const std::vector<int32_t> exp = {13, 5, 10};
        CHECK(ids == exp);
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// Gated on a real Gemma-2 HF directory. Asserts the encoding documented by
// GemmaTokenizerFast: encode("Hello this is a test") == [2,4521,736,603,476,2121].
static void run_checkpoint_tests() {
    const char* dir = std::getenv("BROLM_GEMMA_DIR");
    if (!dir || !*dir) {
        std::printf("gemma_tokenizer: BROLM_GEMMA_DIR unset — skipping "
                    "checkpoint test\n");
        return;
    }
    std::filesystem::path tj = std::filesystem::path(dir) / "tokenizer.json";
    if (!std::filesystem::exists(tj)) {
        std::printf("gemma_tokenizer: %s not found — skipping checkpoint test\n",
                    tj.string().c_str());
        return;
    }

    auto tok = gemma::Tokenizer::load(tj.string());
    std::printf("gemma_tokenizer: loaded %s (vocab=%zu)\n",
                tj.string().c_str(), tok.vocab_count());

    CHECK(tok.pad_id() == 0);
    CHECK(tok.eos_id() == 1);
    CHECK(tok.bos_id() == 2);
    CHECK(tok.unk_id() == 3);
    CHECK(tok.vocab_count() > 200000);

    {
        auto ids = tok.encode("Hello this is a test");  // add_bos=true
        const std::vector<int32_t> exp = {2, 4521, 736, 603, 476, 2121};
        CHECK(ids == exp);
        // Decode of the content (drop bos) round-trips the text.
        std::vector<int32_t> content(ids.begin() + 1, ids.end());
        CHECK(tok.decode(content) == "Hello this is a test");
    }
}

int main() {
    run_unit_tests();
    run_checkpoint_tests();

    if (g_failures == 0) std::printf("gemma_tokenizer: OK\n");
    else std::fprintf(stderr, "gemma_tokenizer: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
