// Unit tests for brolm Grammar (constrained decoding & logit masking)
// and Sampler (Min-P, DRY, Frequency/Presence penalties, streaming callbacks).

#include "brolm/grammar.h"
#include "brolm/sampler.h"
#include "brolm/detail/generate.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

// ─── 1. Regex Grammar Tests ──────────────────────────────────────────────────

void test_grammar_regex() {
    std::printf("Running test_grammar_regex...\n");

    // Integer regex: [0-9]+
    {
        auto g = brolm::Grammar::regex("[0-9]+");
        CHECK(!g.is_accepted());  // Starts empty, not accepted yet

        CHECK(g.can_accept("1"));
        CHECK(g.can_accept("12345"));
        CHECK(!g.can_accept("abc"));
        CHECK(!g.can_accept("12a"));

        CHECK(g.accept("12"));
        CHECK(g.is_accepted());  // Now has digits, is valid finished state
        CHECK(g.can_accept("34"));
        CHECK(!g.can_accept("x"));

        CHECK(g.accept("34"));
        CHECK(g.is_accepted());

        g.reset();
        CHECK(!g.is_accepted());
        CHECK(g.can_accept("99"));
    }

    // Signed float regex: [+-]?[0-9]+(\.[0-9]+)?
    {
        auto g = brolm::Grammar::regex("[+-]?[0-9]+(\\.[0-9]+)?");
        CHECK(g.can_accept("+"));
        CHECK(g.can_accept("-12"));
        CHECK(g.can_accept("3.14"));
        CHECK(!g.can_accept("++"));

        CHECK(g.accept("-"));
        CHECK(!g.is_accepted());  // Just "-" is incomplete
        CHECK(g.can_accept("5"));
        CHECK(!g.can_accept("."));

        CHECK(g.accept("5"));
        CHECK(g.is_accepted());  // "-5" is complete
        CHECK(g.can_accept("."));

        CHECK(g.accept("."));
        CHECK(!g.is_accepted());  // "-5." is incomplete

        CHECK(g.accept("25"));
        CHECK(g.is_accepted());  // "-5.25" is complete
    }

    // Alternation and choices: true|false|null
    {
        auto g = brolm::Grammar::regex("true|false|null");
        CHECK(g.can_accept("tr"));
        CHECK(g.can_accept("true"));
        CHECK(g.can_accept("fal"));
        CHECK(g.can_accept("null"));
        CHECK(!g.can_accept("nil"));

        CHECK(g.accept("fa"));
        CHECK(!g.is_accepted());
        CHECK(g.can_accept("lse"));
        CHECK(!g.can_accept("ue"));

        CHECK(g.accept("lse"));
        CHECK(g.is_accepted());
        CHECK(!g.can_accept("x"));  // Finished
    }
}

// ─── 2. BNF Grammar Tests ────────────────────────────────────────────────────

void test_grammar_bnf() {
    std::printf("Running test_grammar_bnf...\n");

    // Key-value list: item ("," item)* where item is [a-z]+ "=" [0-9]+
    const std::string bnf = R"(
root ::= item ("," item)*
item ::= [a-z]+ "=" [0-9]+
)";

    auto g = brolm::Grammar::bnf(bnf);
    CHECK(!g.is_accepted());

    CHECK(g.can_accept("foo=123"));
    CHECK(g.can_accept("a=1,b=2"));
    CHECK(!g.can_accept("123=foo"));

    CHECK(g.accept("x=10"));
    CHECK(g.is_accepted());

    CHECK(g.can_accept(","));
    CHECK(g.accept(",y="));
    CHECK(!g.is_accepted());

    CHECK(g.accept("42"));
    CHECK(g.is_accepted());
}

// ─── 3. JSON & JSON Schema Grammar Tests ─────────────────────────────────────

void test_grammar_json() {
    std::printf("Running test_grammar_json...\n");

    // JSON Object
    {
        auto g = brolm::Grammar::json_object();
        CHECK(!g.is_accepted());

        CHECK(g.can_accept(R"({"name": "BroLM", "version": 1, "active": true})"));
        CHECK(g.can_accept(R"({})"));
        CHECK(!g.can_accept(R"([1, 2, 3])"));

        // Step by step token stream
        CHECK(g.accept(R"({"count": )"));
        CHECK(!g.is_accepted());

        CHECK(g.accept("42"));
        CHECK(!g.is_accepted());

        CHECK(g.accept("}"));
        CHECK(g.is_accepted());
    }

    // JSON Array
    {
        auto g = brolm::Grammar::json_array();
        CHECK(g.can_accept(R"([1, "two", {"three": 3}])"));
        CHECK(g.can_accept(R"([])"));
        CHECK(!g.can_accept(R"({"a": 1})"));
    }

    // JSON Schema
    {
        const std::string schema = R"({
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "role": {"enum": ["admin", "user"]}
            },
            "required": ["id", "role"]
        })";

        auto g = brolm::Grammar::json_schema(schema);
        CHECK(!g.is_accepted());

        CHECK(g.can_accept(R"({"id": 100, "role": "admin"})"));
        CHECK(g.can_accept(R"({"id": -5, "role": "user"})"));

        CHECK(g.accept(R"({"id": 7, "role": ")"));
        CHECK(g.can_accept("admin\"}"));
        CHECK(g.can_accept("user\"}"));
        CHECK(!g.can_accept("guest\"}"));

        CHECK(g.accept("admin\"}"));
        CHECK(g.is_accepted());
    }

    // Choices
    {
        auto g = brolm::Grammar::choice({"red", "green", "blue"});
        CHECK(g.can_accept("red"));
        CHECK(g.can_accept("green"));
        CHECK(g.can_accept("blue"));
        CHECK(!g.can_accept("yellow"));

        CHECK(g.accept("gr"));
        CHECK(!g.is_accepted());
        CHECK(g.accept("een"));
        CHECK(g.is_accepted());
    }
}

// ─── 4. Logit Masking Tests ──────────────────────────────────────────────────

void test_logit_masking() {
    std::printf("Running test_logit_masking...\n");

    auto g = brolm::Grammar::regex("apple|banana");
    std::vector<std::string> vocab = {
        "<eos>",   // id 0
        "ap",      // id 1
        "ple",     // id 2
        "ba",      // id 3
        "nana",    // id 4
        "cherry",  // id 5
        "123"      // id 6
    };
    const int vocab_size = static_cast<int>(vocab.size());
    const int eos_id = 0;

    std::vector<float> logits = {2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 5.0f, 3.0f};

    // Step 1: At start, only "ap" and "ba" are valid. <eos> is invalid (not accepted yet).
    g.mask_logits(logits.data(), vocab_size, vocab, eos_id);

    CHECK(std::isinf(logits[0]) && logits[0] < 0);  // <eos> masked
    CHECK(!std::isinf(logits[1]));                  // "ap" valid
    CHECK(std::isinf(logits[2]) && logits[2] < 0);  // "ple" masked
    CHECK(!std::isinf(logits[3]));                  // "ba" valid
    CHECK(std::isinf(logits[4]) && logits[4] < 0);  // "nana" masked
    CHECK(std::isinf(logits[5]) && logits[5] < 0);  // "cherry" masked
    CHECK(std::isinf(logits[6]) && logits[6] < 0);  // "123" masked

    // Step 2: Accept "ap"
    CHECK(g.accept("ap"));
    logits = {2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 5.0f, 3.0f};
    g.mask_logits(logits.data(), vocab_size, vocab, eos_id);

    CHECK(std::isinf(logits[0]) && logits[0] < 0);  // <eos> still invalid
    CHECK(std::isinf(logits[1]) && logits[1] < 0);  // "ap" invalid
    CHECK(!std::isinf(logits[2]));                  // "ple" valid!
    CHECK(std::isinf(logits[3]) && logits[3] < 0);  // "ba" invalid

    // Step 3: Accept "ple" -> Accepted state!
    CHECK(g.accept("ple"));
    CHECK(g.is_accepted());
    logits = {2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 5.0f, 3.0f};
    g.mask_logits(logits.data(), vocab_size, vocab, eos_id);

    CHECK(!std::isinf(logits[0]));                  // <eos> is now ALLOWED
    CHECK(std::isinf(logits[1]) && logits[1] < 0);  // all tokens masked since match is complete
}

// A token with no text (a special with no literal form) must never pass the
// mask: it would leave the grammar where it was, so a decode could emit it
// forever. The same holds for ids past the token-text table (a model whose
// embedding rows outnumber its tokenizer). EOS keeps its own rule.
static void check_empty_masked(const brolm::Grammar& g, const char* label) {
    std::printf("  empty-text masking: %s\n", label);
    // ids: 0 <eos> (empty text), 1 "", 2 "4", 3 "x", 4 "" ; vocab_size 6 (id 5 past the table)
    const std::vector<std::string> vocab = {"", "", "4", "x", ""};
    const int vocab_size = 6;
    const int eos_id = 0;

    std::vector<float> logits(vocab_size, 1.0f);
    g.mask_logits(logits.data(), vocab_size, vocab, eos_id);
    CHECK(std::isinf(logits[0]) && logits[0] < 0);  // eos: not accepted yet
    CHECK(std::isinf(logits[1]) && logits[1] < 0);  // empty text
    CHECK(!std::isinf(logits[2]));                  // "4" valid
    CHECK(std::isinf(logits[3]) && logits[3] < 0);  // "x" invalid
    CHECK(std::isinf(logits[4]) && logits[4] < 0);  // empty text
    CHECK(std::isinf(logits[5]) && logits[5] < 0);  // past the table

    // The callback overload obeys the same rule.
    std::vector<float> l2(vocab_size, 1.0f);
    g.mask_logits(l2.data(), vocab_size, [&](int id) -> std::string_view {
        return id < static_cast<int>(vocab.size()) ? std::string_view(vocab[static_cast<size_t>(id)])
                                                   : std::string_view();
    }, eos_id);
    for (int i = 0; i < vocab_size; ++i) CHECK(std::isinf(l2[i]) == std::isinf(logits[i]));

    // Once accepted, EOS (empty text or not) is allowed; empties stay masked.
    brolm::Grammar done = g.clone();
    CHECK(done.accept("4"));
    CHECK(done.is_accepted());
    std::vector<float> l3(vocab_size, 1.0f);
    done.mask_logits(l3.data(), vocab_size, vocab, eos_id);
    CHECK(!std::isinf(l3[0]));
    CHECK(std::isinf(l3[1]) && std::isinf(l3[4]) && std::isinf(l3[5]));
    // EOS past the table is allowed too once accepted.
    std::vector<float> l4(vocab_size, 1.0f);
    done.mask_logits(l4.data(), vocab_size, vocab, 5);
    CHECK(!std::isinf(l4[5]));
    CHECK(std::isinf(l4[0]) && l4[0] < 0);  // id 0 is now just an empty token
}

void test_empty_text_masked() {
    std::printf("Running test_empty_text_masked...\n");
    check_empty_masked(brolm::Grammar::regex("[0-9]+"), "regex (DFA/JIT path)");
    check_empty_masked(brolm::Grammar::bnf("root ::= digits\ndigits ::= [0-9] | [0-9] digits"),
                       "bnf");
    check_empty_masked(brolm::Grammar::json_integer(), "json_integer");
}

// ─── 5. Min-P Sampling Tests ─────────────────────────────────────────────────

void test_min_p_sampling() {
    std::printf("Running test_min_p_sampling...\n");

    // Probs: [0.60, 0.30, 0.08, 0.02]
    // Max prob = 0.60
    // min_p = 0.10 => threshold = 0.06 => index 3 (0.02) zeroed out
    // min_p = 0.20 => threshold = 0.12 => indices 2, 3 (0.08, 0.02) zeroed out
    std::vector<float> probs = {0.60f, 0.30f, 0.08f, 0.02f};

    {
        std::vector<float> p = probs;
        brolm::apply_min_p(p, 0.10f);
        CHECK(p[0] == 0.60f);
        CHECK(p[1] == 0.30f);
        CHECK(p[2] == 0.08f);
        CHECK(p[3] == 0.0f);
    }

    {
        std::vector<float> p = probs;
        brolm::apply_min_p(p, 0.20f);
        CHECK(p[0] == 0.60f);
        CHECK(p[1] == 0.30f);
        CHECK(p[2] == 0.0f);
        CHECK(p[3] == 0.0f);
    }

    // Min-P in sample_token
    {
        std::vector<float> logits = {10.0f, 8.0f, 2.0f, 0.0f};
        brolm::SamplingParams sp;
        sp.temperature = 1.0f;
        sp.min_p = 0.2f;
        sp.seed = 42;
        std::mt19937_64 rng(sp.seed);
        int tok = brolm::sample_token(logits.data(), 4, sp, rng);
        CHECK(tok == 0 || tok == 1);  // Lower tokens should be excluded
    }
}

// ─── 6. Repetition, Frequency, Presence Penalty Tests ────────────────────────

void test_penalties() {
    std::printf("Running test_penalties...\n");

    const int vocab = 5;
    // Context: token 1 appears 3 times, token 2 appears 1 time
    std::vector<int32_t> context = {1, 2, 1, 1};

    // Frequency penalty
    {
        std::vector<float> logits = {10.0f, 10.0f, 10.0f, 10.0f, 10.0f};
        brolm::SamplingParams p;
        p.frequency_penalty = 2.0f;
        p.penalty_last_n = 10;

        brolm::apply_repetition_penalties(logits.data(), vocab, context.data(), static_cast<int>(context.size()), p);

        CHECK(logits[0] == 10.0f);
        CHECK(logits[1] == 10.0f - 2.0f * 3.0f);  // 4.0
        CHECK(logits[2] == 10.0f - 2.0f * 1.0f);  // 8.0
        CHECK(logits[3] == 10.0f);
    }

    // Presence penalty
    {
        std::vector<float> logits = {10.0f, 10.0f, 10.0f, 10.0f, 10.0f};
        brolm::SamplingParams p;
        p.presence_penalty = 1.5f;
        p.penalty_last_n = 10;

        brolm::apply_repetition_penalties(logits.data(), vocab, context.data(), static_cast<int>(context.size()), p);

        CHECK(logits[0] == 10.0f);
        CHECK(logits[1] == 10.0f - 1.5f);  // 8.5
        CHECK(logits[2] == 10.0f - 1.5f);  // 8.5
        CHECK(logits[3] == 10.0f);
    }

    // Repetition penalty
    {
        std::vector<float> logits = {5.0f, 10.0f, -4.0f, 0.0f, 2.0f};
        brolm::SamplingParams p;
        p.repetition_penalty = 2.0f;
        p.penalty_last_n = 10;

        brolm::apply_repetition_penalties(logits.data(), vocab, context.data(), static_cast<int>(context.size()), p);

        CHECK(logits[0] == 5.0f);
        CHECK(logits[1] == 5.0f);  // 10.0 / 2.0
        CHECK(logits[2] == -8.0f); // -4.0 * 2.0
    }
}

// ─── 7. DRY (Don't Repeat Yourself) Repetition Penalty Tests ─────────────────

void test_dry_penalty() {
    std::printf("Running test_dry_penalty...\n");

    const int vocab = 10;
    // Context has pattern: [1, 2, 3, 4, 9, 8, 1, 2, 3]
    // Current suffix is [1, 2, 3].
    // Preceding match in context is at [0..2] where [1, 2, 3] was followed by token 4.
    // The match length is 3. With dry_allowed_length = 2, token 4 should be penalized!
    std::vector<int32_t> context = {1, 2, 3, 4, 9, 8, 1, 2, 3};

    std::vector<float> logits(vocab, 10.0f);
    brolm::SamplingParams p;
    p.dry_multiplier = 2.0f;
    p.dry_base = 1.75f;
    p.dry_allowed_length = 2;

    brolm::apply_dry_penalty(logits.data(), vocab, context.data(), static_cast<int>(context.size()), p);

    // match_len = 3 -> exponent = 3 - 2 = 1 -> penalty = 2.0 * (1.75)^1 = 3.5
    CHECK(logits[4] == 10.0f - 3.5f);
    CHECK(logits[0] == 10.0f);
    CHECK(logits[1] == 10.0f);
    CHECK(logits[2] == 10.0f);
    CHECK(logits[5] == 10.0f);

    // With sequence breaker token
    {
        std::vector<float> logits_breaker(vocab, 10.0f);
        p.dry_breaker_tokens = {2};  // token 2 breaks match
        brolm::apply_dry_penalty(logits_breaker.data(), vocab, context.data(), static_cast<int>(context.size()), p);

        // Match stopped at token 2 -> match_len = 1 < dry_allowed_length (2) -> no penalty
        CHECK(logits_breaker[4] == 10.0f);
    }
}

// ─── 8. Stateful Sampler Class Tests ─────────────────────────────────────────

void test_sampler_class() {
    std::printf("Running test_sampler_class...\n");

    brolm::SamplingParams sp;
    sp.temperature = 0.0f;  // Greedy
    brolm::Sampler sampler(sp);

    std::vector<float> logits = {1.0f, 5.0f, 2.0f};
    int tok = sampler.sample(logits.data(), 3);
    CHECK(tok == 1);
    CHECK(sampler.history().size() == 1);
    CHECK(sampler.history()[0] == 1);

    logits = {8.0f, 2.0f, 0.0f};
    tok = sampler.sample(logits.data(), 3);
    CHECK(tok == 0);
    CHECK(sampler.history().size() == 2);
    CHECK(sampler.history()[1] == 0);

    sampler.reset();
    CHECK(sampler.history().empty());
}

// ─── 9. Streaming Generation Callback Tests ──────────────────────────────────

// Mock model for streaming test
struct MockModelConfig {
    int vocab_size = 8;
};

struct MockModel {
    MockModelConfig cfg;
    std::vector<int32_t> forward_history;
    std::atomic<bool> busy{false};

    const MockModelConfig& config() const { return cfg; }
    void allocate_cache(int) {}
    void forward_last(const int32_t* ids, int n, brotensor::Tensor& out) {
        for (int i = 0; i < n; ++i) forward_history.push_back(ids[i]);

        // Output deterministic logits predicting token (history_size % vocab)
        int next_pred = static_cast<int>(forward_history.size()) % cfg.vocab_size;
        std::vector<float> row(cfg.vocab_size, 0.0f);
        row[next_pred] = 10.0f;

        out = brotensor::Tensor::from_host_on(
            brotensor::Device::CPU, row.data(), 1, cfg.vocab_size);
    }
};

// A grammar that is complete with no stop token allowed leaves no token: the
// decode must end there, not sample from an all -inf row (argmax -> id 0).
void test_grammar_exhaustion_stops() {
    std::printf("Running test_grammar_exhaustion_stops...\n");
    MockModel model;
    brolm::detail::GenerateOptions opts;
    opts.max_new_tokens = 10;
    opts.sampling.temperature = 0.0f;
    opts.stop_on_eos = false;
    brolm::Grammar g = brolm::Grammar::exact("tok_5tok_1");
    opts.grammar = &g;
    auto decode_fn = [](int32_t id) -> std::string { return "tok_" + std::to_string(id); };
    std::vector<int32_t> out = brolm::detail::generate(model, {1, 2}, 0, opts, decode_fn);
    CHECK(out.size() == 2);
    if (out.size() == 2) {
        CHECK(out[0] == 5);
        CHECK(out[1] == 1);
    }
    // stop_on_eos with eos 0: once accepted the grammar lets eos through and
    // the decode stops on it.
    opts.stop_on_eos = true;
    MockModel model2;
    out = brolm::detail::generate(model2, {1, 2}, 0, opts, decode_fn);
    CHECK(out.size() == 2);
}

void test_streaming_generation() {
    std::printf("Running test_streaming_generation...\n");

    MockModel model;
    std::vector<int32_t> prompt = {1, 2};

    brolm::detail::GenerateOptions opts;
    opts.max_new_tokens = 10;
    opts.sampling.temperature = 0.0f;  // greedy
    opts.stop_on_eos = false;

    std::vector<int32_t> streamed_tokens;
    std::vector<std::string> streamed_texts;

    auto decode_fn = [](int32_t id) -> std::string {
        return "tok_" + std::to_string(id);
    };

    // Callback that cancels after 4 tokens
    auto callback = [&](int32_t id, const std::string& text) -> bool {
        streamed_tokens.push_back(id);
        streamed_texts.push_back(text);
        return streamed_tokens.size() < 4;  // Return false on 4th token to stop
    };

    std::vector<int32_t> result = brolm::detail::generate(
        model, prompt, -1, opts, decode_fn, callback);

    CHECK(result.size() == 4);
    CHECK(streamed_tokens.size() == 4);
    CHECK(streamed_texts.size() == 4);
    CHECK(streamed_texts[0] == "tok_" + std::to_string(streamed_tokens[0]));
    CHECK(result == streamed_tokens);
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main() {
    try {
        brotensor::init();

        test_grammar_regex();
        test_grammar_bnf();
        test_grammar_json();
        test_logit_masking();
        test_empty_text_masked();
        test_min_p_sampling();
        test_penalties();
        test_dry_penalty();
        test_sampler_class();
        test_streaming_generation();
        test_grammar_exhaustion_stops();

    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_grammar_sampler threw: %s\n", e.what());
        ++g_failures;
    }

    if (g_failures == 0) {
        std::printf("test_grammar_sampler: ALL TESTS PASSED\n");
    } else {
        std::fprintf(stderr, "test_grammar_sampler: %d failure(s)\n", g_failures);
    }

    return g_failures ? 1 : 0;
}
