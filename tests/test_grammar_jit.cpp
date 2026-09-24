#include "brolm/grammar.h"
#include "brolm/grammar_jit.h"
#include "grammar/dfa_compiler.h"
#include "grammar/regex_compiler.h"
#include "grammar/bnf_compiler.h"
#include "grammar/vocab_indexer.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace brolm;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

#define CHECK_EQ(a, b) do { \
    if ((a) != (b)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s == %s\n", __FILE__, __LINE__, #a, #b); \
        ++g_failures; \
    } \
} while (0)

// ─── 1. DFA Compilation Tests ────────────────────────────────────────────────

void test_dfa_compilation_regex() {
    std::printf("Running test_dfa_compilation_regex...\n");

    // Integer regex: [0-9]+
    {
        GrammarGraph graph;
        int rule_id = graph.add_rule("root");
        graph.root_rule_id = rule_id;
        RegexCompiler compiler("[0-9]+", graph, rule_id);
        auto [entry, exit] = compiler.compile();
        graph.rules[rule_id].entry_state = entry;
        graph.rules[rule_id].exit_state = exit;

        CompiledDfa dfa = DfaCompiler::compile(graph);
        CHECK(dfa.is_valid());
        CHECK_EQ(dfa.start_state, 0);
        CHECK(dfa.num_states >= 2);
        CHECK(!dfa.is_accept_state[0]); // Start state is not accept yet

        // Step digits
        int32_t s1 = dfa.step(0, '5');
        CHECK(s1 >= 0);
        CHECK(dfa.is_accept_state[s1]); // Reached accepting state

        int32_t s2 = dfa.step(s1, '9');
        CHECK(s2 >= 0);
        CHECK(dfa.is_accept_state[s2]);

        // Step non-digit
        int32_t s_bad = dfa.step(0, 'a');
        CHECK_EQ(s_bad, -1);
    }

    // Keyword alternation: true|false
    {
        GrammarGraph graph;
        int rule_id = graph.add_rule("root");
        graph.root_rule_id = rule_id;
        RegexCompiler compiler("true|false", graph, rule_id);
        auto [entry, exit] = compiler.compile();
        graph.rules[rule_id].entry_state = entry;
        graph.rules[rule_id].exit_state = exit;

        CompiledDfa dfa = DfaCompiler::compile(graph);
        CHECK(dfa.is_valid());

        // Test "true"
        int32_t s = 0;
        for (char c : std::string("true")) {
            s = dfa.step(s, static_cast<uint8_t>(c));
            CHECK(s >= 0);
        }
        CHECK(dfa.is_accept_state[s]);

        // Test "false"
        s = 0;
        for (char c : std::string("false")) {
            s = dfa.step(s, static_cast<uint8_t>(c));
            CHECK(s >= 0);
        }
        CHECK(dfa.is_accept_state[s]);

        // Test "null" -> dead
        s = 0;
        s = dfa.step(s, 'n');
        CHECK_EQ(s, -1);
    }
}

void test_dfa_compilation_bnf() {
    std::printf("Running test_dfa_compilation_bnf...\n");

    std::string bnf =
        "root ::= key \"=\" val\n"
        "key  ::= [a-z]+\n"
        "val  ::= [0-9]+\n";

    GrammarGraph graph;
    BNFCompiler compiler(bnf, graph);
    compiler.compile();

    CompiledDfa dfa = DfaCompiler::compile(graph);
    CHECK(dfa.is_valid());
    CHECK_EQ(dfa.start_state, 0);

    // Scan "foo=123"
    int32_t s = 0;
    for (char c : std::string("foo=123")) {
        s = dfa.step(s, static_cast<uint8_t>(c));
        CHECK(s >= 0);
    }
    CHECK(dfa.is_accept_state[s]);

    // Incomplete "foo="
    s = 0;
    for (char c : std::string("foo=")) {
        s = dfa.step(s, static_cast<uint8_t>(c));
        CHECK(s >= 0);
    }
    CHECK(!dfa.is_accept_state[s]);

    // Invalid "123=foo"
    s = dfa.step(0, '1');
    CHECK_EQ(s, -1);
}

void test_dfa_compilation_json_schema() {
    std::printf("Running test_dfa_compilation_json_schema...\n");

    // JSON boolean schema
    {
        auto g = brolm::Grammar::json_boolean();
        CHECK(g.can_accept("true"));
        CHECK(g.can_accept("false"));
        CHECK(!g.can_accept("null"));
        CHECK(!g.can_accept("123"));

        CHECK(g.accept("tr"));
        CHECK(!g.is_accepted());
        CHECK(g.accept("ue"));
        CHECK(g.is_accepted());
    }

    // JSON schema object with string and integer
    {
        std::string schema = R"({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age":  {"type": "integer"}
            },
            "required": ["name", "age"]
        })";
        auto g = brolm::Grammar::json_schema(schema);
        CHECK(!g.is_accepted());
        CHECK(g.can_accept("{"));
        CHECK(g.accept("{\"name\": \"Alice\", \"age\": 30}"));
        CHECK(g.is_accepted());
    }
}

// ─── 2. JitGrammar Lifecycle & Transitions ───────────────────────────────────

void test_jit_grammar_lifecycle() {
    std::printf("Running test_jit_grammar_lifecycle...\n");

    GrammarGraph graph;
    int rule_id = graph.add_rule("root");
    graph.root_rule_id = rule_id;
    RegexCompiler compiler("[a-zA-Z_][a-zA-Z0-9_]*", graph, rule_id);
    auto [entry, exit] = compiler.compile();
    graph.rules[rule_id].entry_state = entry;
    graph.rules[rule_id].exit_state = exit;

    CompiledDfa dfa = DfaCompiler::compile(graph);
    CHECK(dfa.is_valid());

    brolm::JitGrammar jit(dfa);
    CHECK_EQ(jit.current_state(), dfa.start_state);
    CHECK(jit.is_jit_compiled());
    CHECK(jit.mask_fn != nullptr);
    CHECK(jit.scan_fn != nullptr);

    // Initial state: not accepted (identifier requires >= 1 char)
    CHECK(!jit.is_accepted());
    CHECK(jit.can_accept("hello"));
    CHECK(jit.can_accept("_var123"));
    CHECK(!jit.can_accept("123var"));

    // Accept valid token
    CHECK(jit.accept("hello"));
    CHECK(jit.is_accepted());
    CHECK(jit.can_accept("_world"));
    CHECK(jit.accept("_world"));
    CHECK(jit.is_accepted());

    // Copy constructor test
    brolm::JitGrammar copy = jit;
    CHECK_EQ(copy.current_state(), jit.current_state());
    CHECK(copy.is_accepted());

    // Advance copy with invalid char -> enters dead state
    CHECK(!copy.accept("!"));
    CHECK_EQ(copy.current_state(), -1);
    CHECK(!copy.is_accepted());
    // Original should remain valid
    CHECK(jit.is_accepted());

    // Reset original
    jit.reset();
    CHECK_EQ(jit.current_state(), dfa.start_state);
    CHECK(!jit.is_accepted());
    CHECK(jit.can_accept("alpha"));
}

// ─── 3. Logit Masking Correctness across Vocabularies ─────────────────────────

void test_logit_masking_correctness() {
    std::printf("Running test_logit_masking_correctness...\n");

    auto run_vocab_test = [](size_t vocab_size, int eos_id) {
        // Grammar: alphanumeric tokens
        auto grammar = brolm::Grammar::regex("[0-9]+");

        std::mt19937 rng(42 + static_cast<unsigned>(vocab_size));
        std::uniform_real_distribution<float> logit_dist(-10.0f, 10.0f);

        // Build vocabulary of varying tokens
        std::vector<std::string> vocab(vocab_size);
        std::vector<float> orig_logits(vocab_size);
        std::vector<float> ref_logits(vocab_size);
        std::vector<float> jit_logits(vocab_size);

        for (size_t i = 0; i < vocab_size; ++i) {
            float val = logit_dist(rng);
            orig_logits[i] = val;
            ref_logits[i] = val;
            jit_logits[i] = val;

            if (static_cast<int>(i) == eos_id) {
                vocab[i] = "<|eos|>";
            } else if (i % 4 == 0) {
                vocab[i] = std::to_string(i); // valid digits
            } else if (i % 4 == 1) {
                vocab[i] = "token_" + std::to_string(i); // invalid
            } else if (i % 4 == 2) {
                vocab[i] = std::to_string(i % 10); // single digit
            } else {
                vocab[i] = "+" + std::to_string(i); // invalid symbol
            }
        }

        const float kNegInf = -std::numeric_limits<float>::infinity();

        // 1. Test at start state (not accepted yet, EOS should be masked)
        CHECK(!grammar.is_accepted());

        // Compute scalar reference
        for (size_t i = 0; i < vocab_size; ++i) {
            if (static_cast<int>(i) == eos_id) {
                if (!grammar.is_accepted()) ref_logits[i] = kNegInf;
                continue;
            }
            // Empty text is masked (it never advances the grammar).
            if (vocab[i].empty() || !grammar.can_accept(vocab[i])) {
                ref_logits[i] = kNegInf;
            }
        }

        // Compute JIT masking
        grammar.mask_logits(jit_logits.data(), static_cast<int>(vocab_size), vocab, eos_id);

        // Verify exact match
        for (size_t i = 0; i < vocab_size; ++i) {
            if (std::isinf(ref_logits[i])) {
                CHECK(std::isinf(jit_logits[i]));
                CHECK(jit_logits[i] < 0.0f); // -inf
            } else {
                CHECK_EQ(jit_logits[i], orig_logits[i]);
            }
        }

        // 2. Advance grammar to accepted state ("42") and re-test
        grammar.accept("42");
        CHECK(grammar.is_accepted());

        for (size_t i = 0; i < vocab_size; ++i) {
            ref_logits[i] = orig_logits[i];
            jit_logits[i] = orig_logits[i];
        }

        for (size_t i = 0; i < vocab_size; ++i) {
            if (static_cast<int>(i) == eos_id) {
                if (!grammar.is_accepted()) ref_logits[i] = kNegInf;
                continue; // accepted -> EOS is preserved!
            }
            if (vocab[i].empty() || !grammar.can_accept(vocab[i])) {
                ref_logits[i] = kNegInf;
            }
        }

        grammar.mask_logits(jit_logits.data(), static_cast<int>(vocab_size), vocab, eos_id);

        for (size_t i = 0; i < vocab_size; ++i) {
            if (std::isinf(ref_logits[i])) {
                CHECK(std::isinf(jit_logits[i]));
                CHECK(jit_logits[i] < 0.0f);
            } else {
                CHECK_EQ(jit_logits[i], orig_logits[i]);
            }
        }

        // Verify EOS was preserved when accepted
        if (eos_id >= 0 && static_cast<size_t>(eos_id) < vocab_size) {
            CHECK_EQ(jit_logits[eos_id], orig_logits[eos_id]);
        }
    };

    // Small Vocab: 1,024
    run_vocab_test(1024, 0);

    // Large Vocab: 32,768 (512 words)
    run_vocab_test(32768, 128);

    // Very Large Vocab: 128,000 (2,000 words)
    run_vocab_test(128000, 2);

    // Non-multiple of 8 tail: 128,003
    run_vocab_test(128003, 128001);
}

// ─── 4. Latency Benchmark: 128k Vocab Masking ────────────────────────────────

void test_latency_benchmark() {
    std::printf("Running test_latency_benchmark (128k vocabulary)...\n");

    constexpr size_t kVocabSize = 128000;
    std::vector<std::string> vocab(kVocabSize);
    std::vector<float> logits_nfa(kVocabSize, 0.0f);
    std::vector<float> logits_jit(kVocabSize, 0.0f);

    for (size_t i = 0; i < kVocabSize; ++i) {
        if (i % 3 == 0) {
            vocab[i] = std::to_string(i);
        } else if (i % 3 == 1) {
            vocab[i] = "word_" + std::to_string(i);
        } else {
            vocab[i] = " " + std::to_string(i % 100);
        }
    }

    auto grammar = brolm::Grammar::regex("[0-9]+");
    const int eos_id = 1;

    // Warm-up JIT cache
    grammar.mask_logits(logits_jit.data(), static_cast<int>(kVocabSize), vocab, eos_id);

    // Benchmark JIT Masking
    constexpr int kIterations = 100;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < kIterations; ++it) {
        grammar.mask_logits(logits_jit.data(), static_cast<int>(kVocabSize), vocab, eos_id);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double jit_total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double jit_per_step_ms = jit_total_ms / kIterations;
    double jit_per_step_us = jit_per_step_ms * 1000.0;

    std::printf("  [BENCHMARK] JIT 128k Logit Masking: %.3f ms per step (%.1f us) [%d iterations]\n",
                jit_per_step_ms, jit_per_step_us, kIterations);

    // Benchmark NFA Masking (small sample to estimate without stalling test suite)
    constexpr int kNfaIterations = 2;
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < kNfaIterations; ++it) {
        const float kNegInf = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < kVocabSize; ++i) {
            if (static_cast<int>(i) == eos_id) {
                if (!grammar.is_accepted()) logits_nfa[i] = kNegInf;
                continue;
            }
            // Direct NFA traversal per token:
            if (!vocab[i].empty() && !grammar.can_accept(vocab[i])) {
                logits_nfa[i] = kNegInf;
            }
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double nfa_per_step_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / kNfaIterations;
    std::printf("  [BENCHMARK] NFA 128k Scalar Checking: %.2f ms per step\n", nfa_per_step_ms);

    double speedup = nfa_per_step_ms / jit_per_step_ms;
    std::printf("  [BENCHMARK] Speedup: %.1fx faster with JIT masking\n", speedup);

    // The timed JIT path masks exactly what the per-token NFA walk does.
    size_t mismatches = 0;
    for (size_t i = 0; i < kVocabSize; ++i) {
        if (logits_jit[i] != logits_nfa[i]) ++mismatches;
    }
    CHECK_EQ(mismatches, size_t{0});

    // Requirement: sub-millisecond masking across 128k tokens. A wall-clock
    // bound only means something in an optimized, uninstrumented build; the
    // coverage build is -O0 with a counter on every edge, and Debug is -O0.
#if defined(NDEBUG) && !defined(BROLM_COVERAGE_BUILD)
    CHECK(jit_per_step_ms < 1.0);
#else
    std::printf("  [BENCHMARK] sub-ms bound not enforced (unoptimized or instrumented build)\n");
#endif
}

// ─── 5. Same-size vocabularies do not share masks ────────────────────────────

void test_vocab_cache_keyed_on_content() {
    std::printf("Running test_vocab_cache_keyed_on_content...\n");

    // Two tokenizers with the same vocabulary size and different tokens: the
    // grammar's cached state masks must follow the vocabulary handed in, not
    // the size, or the second tokenizer is masked with the first one's bits.
    const std::vector<std::string> va = {"1", "a", "2", "b", "", "3"};
    const std::vector<std::string> vb = {"x", "7", "y", "8", "", "z"};
    const int n = static_cast<int>(va.size());
    const float neg_inf = -std::numeric_limits<float>::infinity();

    auto grammar = brolm::Grammar::regex("[0-9]+");
    std::vector<float> la(n, 0.0f), lb(n, 0.0f), la2(n, 0.0f);
    grammar.mask_logits(la.data(), n, va, /*eos_id=*/-1);
    grammar.mask_logits(lb.data(), n, vb, /*eos_id=*/-1);
    grammar.mask_logits(la2.data(), n, va, /*eos_id=*/-1);
    for (int i = 0; i < n; ++i) {
        const bool a_ok = !va[i].empty() && va[i][0] >= '0' && va[i][0] <= '9';
        const bool b_ok = !vb[i].empty() && vb[i][0] >= '0' && vb[i][0] <= '9';
        CHECK_EQ(la[i] == neg_inf, !a_ok);
        CHECK_EQ(lb[i] == neg_inf, !b_ok);
        CHECK_EQ(la2[i] == neg_inf, !a_ok);
    }

    // The same vocabulary content in a different vector reuses the cache.
    VocabIndexer idx(va);
    const std::vector<std::string> va_copy = va;
    CHECK(idx.same_vocab(va_copy));
    CHECK(!idx.same_vocab(vb));
    std::vector<std::string> va_short(va.begin(), va.end() - 1);
    CHECK(!idx.same_vocab(va_short));
}

// ─── Main Driver ─────────────────────────────────────────────────────────────

int main() {
    std::printf("=== brolm JIT Grammar Constrained Decoding Test Suite ===\n");

    test_dfa_compilation_regex();
    test_dfa_compilation_bnf();
    test_dfa_compilation_json_schema();
    test_jit_grammar_lifecycle();
    test_logit_masking_correctness();
    test_vocab_cache_keyed_on_content();
    test_latency_benchmark();

    if (g_failures > 0) {
        std::fprintf(stderr, "test_grammar_jit: %d TEST(S) FAILED!\n", g_failures);
        return 1;
    }

    std::printf("test_grammar_jit: ALL TESTS PASSED\n");
    return 0;
}
