// Tests for brolm::qwen3vl::Tokenizer.
//
// The BPE machinery is exercised by test_qwen_tokenizer; this file only
// verifies the Qwen3-VL-specific surface:
//   * the named accessors (vision_start/end, image_pad, video_pad) resolve
//     to their documented ids,
//   * encode treats the multimodal specials as atomic ids surrounded by
//     ordinary text.
//
// A real checkpoint is required because the vision special-token ids only
// exist in the Qwen3-VL vocab (they live above vocab.json's own range). The
// test is gated on BROLM_QWEN3VL_DIR; without it the run reports "skip" and
// exits 0 so CI without weights stays green.

#include "brolm/qwen3vl_tokenizer.h"

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
    const char* dir = std::getenv("BROLM_QWEN3VL_DIR");
    if (!dir) {
        std::printf("[skip] BROLM_QWEN3VL_DIR not set\n");
        return 0;
    }
    const std::string vocab  = std::string(dir) + "/vocab.json";
    const std::string merges = std::string(dir) + "/merges.txt";

    brolm::qwen3vl::Tokenizer tok =
        brolm::qwen3vl::Tokenizer::load(vocab, merges);

    // Sanity: a non-trivial vocab.
    REQUIRE(tok.vocab_count() > 100000);
    std::printf("[ok] vocab_count=%zu merges=%zu\n",
                tok.vocab_count(), tok.merge_count());

    // <|endoftext|>/<|im_start|>/<|im_end|> are inside the base vocab.
    REQUIRE(tok.endoftext_id()    == 151643);
    REQUIRE(tok.im_start_id()     == 151644);
    REQUIRE(tok.im_end_id()       == 151645);

    // Vision/video specials live above vocab.json's range.
    REQUIRE(tok.vision_start_id() == 151652);
    REQUIRE(tok.vision_end_id()   == 151653);
    REQUIRE(tok.image_pad_id()    == 151655);
    REQUIRE(tok.video_pad_id()    == 151656);
    std::printf("[ok] named ids resolved\n");

    // <|im_end|> is the conventional EOS for Qwen3-family chat models.
    REQUIRE(tok.eos_id() == tok.im_end_id());

    // Encode a multimodal-flavoured snippet and verify the specials land as
    // single ids in the right spots.
    const std::string snippet =
        "<|vision_start|><|image_pad|>hi<|vision_end|>";
    const std::vector<int32_t> ids = tok.encode(snippet);
    REQUIRE(!ids.empty());
    REQUIRE(ids.front() == tok.vision_start_id());
    REQUIRE(ids[1]      == tok.image_pad_id());
    REQUIRE(ids.back()  == tok.vision_end_id());
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
