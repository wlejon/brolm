// Tests for brolm::qwen35::Tokenizer.
//
// The BPE machinery is exercised by test_qwen_tokenizer; this file only
// verifies the Qwen3.5-specific surface:
//   * the named accessors (vision_*, image_pad, etc.) resolve when the
//     vocab contains the corresponding token strings,
//   * encode treats the multimodal specials as atomic ids surrounded by
//     ordinary text.
//
// A real checkpoint is required because the special-token ids only exist in
// the Qwen3.5 vocab. The test is gated on BROLM_QWEN35_DIR; without it
// the run reports "skip" and exits 0 so CI without weights stays green.

#include "brolm/qwen35_tokenizer.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// MSVC defines NDEBUG in Release, which compiles assert() to nothing. We
// rely on this test running in any configuration, so use an explicit check
// macro that always aborts on failure.
#define REQUIRE(cond) do {                                              \
    if (!(cond)) {                                                      \
        std::fprintf(stderr, "REQUIRE failed: %s @ %s:%d\n",            \
                     #cond, __FILE__, __LINE__);                        \
        std::abort();                                                   \
    }                                                                   \
} while (0)

int main() {
    const char* dir = std::getenv("BROLM_QWEN35_DIR");
    if (!dir) {
        std::printf("[skip] BROLM_QWEN35_DIR not set\n");
        return 0;
    }
    const std::string vocab  = std::string(dir) + "/vocab.json";
    const std::string merges = std::string(dir) + "/merges.txt";

    brolm::qwen35::Tokenizer tok =
        brolm::qwen35::Tokenizer::load(vocab, merges);

    // Sanity: a non-trivial vocab.
    REQUIRE(tok.vocab_count() > 200000);
    std::printf("[ok] vocab_count=%zu merges=%zu\n",
                tok.vocab_count(), tok.merge_count());

    // Named accessors must resolve to the documented Qwen3.5 IDs.
    REQUIRE(tok.endoftext_id()    == 248044);
    REQUIRE(tok.im_start_id()     == 248045);
    REQUIRE(tok.im_end_id()       == 248046);
    REQUIRE(tok.vision_start_id() == 248053);
    REQUIRE(tok.vision_end_id()   == 248054);
    REQUIRE(tok.vision_pad_id()   == 248055);
    REQUIRE(tok.image_pad_id()    == 248056);
    REQUIRE(tok.video_pad_id()    == 248057);
    REQUIRE(tok.think_open_id()   == 248068);
    REQUIRE(tok.think_close_id()  == 248069);
    std::printf("[ok] named ids resolved\n");

    // <|im_end|> is the conventional EOS for Qwen3-family chat models.
    REQUIRE(tok.eos_id() == tok.im_end_id());

    // Encode a multimodal-flavoured snippet and verify the specials land as
    // single ids in the right spots.
    const std::string snippet =
        "<|vision_start|><|image_pad|>hi<|vision_end|>";
    const std::vector<int32_t> ids = tok.encode(snippet);
    REQUIRE(!ids.empty());
    REQUIRE(ids.front()              == tok.vision_start_id());
    REQUIRE(ids[1]                   == tok.image_pad_id());
    REQUIRE(ids.back()               == tok.vision_end_id());
    // "hi" is the middle: at least one BPE id between image_pad and vision_end.
    REQUIRE(ids.size() >= 4);
    std::printf("[ok] multimodal special tokens encode atomically (%zu ids)\n",
                ids.size());

    // Round-trip: decoding the same ids must recover the original snippet.
    const std::string round = tok.decode(ids);
    REQUIRE(round == snippet);
    std::printf("[ok] decode round-trip\n");

    // apply_chat_template should produce a string that, re-encoded, starts
    // with <|im_start|> and contains <|im_end|>.
    std::vector<std::pair<std::string, std::string>> msgs = {
        {"system", "you are a vision assistant."},
        {"user",   "<|vision_start|><|image_pad|><|vision_end|>describe it."},
    };
    const std::string chat = tok.apply_chat_template(msgs);
    const std::vector<int32_t> chat_ids = tok.encode(chat);
    REQUIRE(!chat_ids.empty() && chat_ids.front() == tok.im_start_id());
    bool seen_im_end = false;
    for (int32_t id : chat_ids) {
        if (id == tok.im_end_id()) { seen_im_end = true; break; }
    }
    REQUIRE(seen_im_end);
    std::printf("[ok] chat template renders with proper turn markers\n");

    return 0;
}
