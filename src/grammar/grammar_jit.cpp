#include "brolm/grammar_jit.h"
#include "grammar/vocab_indexer.h"

#include <brass/codegen/grammar_builder.hpp>
#include <brass/codegen/kernel_jit.hpp>
#include <brass/target/target.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>

namespace brolm {

namespace {

#if BROLM_HAS_BRASS_JIT
struct CompiledKernels {
    brass::codegen::KernelFunction mask_kfn;
    brass::codegen::KernelFunction scan_kfn;
    brass::codegen::KernelFunction filter_kfn;

    brass::codegen::LogitMaskFn mask_fn = nullptr;
    brass::codegen::DfaScanStringFn scan_fn = nullptr;
    brass::codegen::DfaFilterTokensFn filter_fn = nullptr;

    bool initialized = false;
    std::mutex mtx;

    void ensure_initialized() {
        std::lock_guard<std::mutex> lock(mtx);
        if (initialized) return;

        if (!brass::Target::host().is_x64() && !brass::Target::host().is_aarch64()) {
            initialized = true;
            return;
        }

        try {
            brass::codegen::KernelOptions opts;
            opts.enable_avx2 = true;
            opts.enable_optimizations = true;

            // 1. Logit Mask Kernel
            {
                brass::Module mod("brolm_logit_mask_mod");
                brass::codegen::GrammarBuilder gb(mod);
                brass::Function* fn = gb.build_logit_mask_function("brolm_logit_mask");
                if (fn) {
                    brass::codegen::KernelJit jit(opts);
                    mask_kfn = jit.compile(*fn);
                    if (mask_kfn.is_valid()) {
                        mask_fn = mask_kfn.as<brass::codegen::LogitMaskFn>();
                    }
                }
            }

            // 2. DFA Scan String Kernel
            {
                brass::Module mod("brolm_dfa_scan_mod");
                brass::codegen::GrammarBuilder gb(mod);
                brass::Function* fn = gb.build_dfa_scan_string_function("brolm_dfa_scan", 4);
                if (fn) {
                    brass::codegen::KernelJit jit(opts);
                    scan_kfn = jit.compile(*fn);
                    if (scan_kfn.is_valid()) {
                        scan_fn = scan_kfn.as<brass::codegen::DfaScanStringFn>();
                    }
                }
            }

            // 3. Batched DFA Filter Tokens Kernel
            {
                brass::Module mod("brolm_dfa_filter_mod");
                brass::codegen::GrammarBuilder gb(mod);
                brass::Function* fn = gb.build_dfa_filter_tokens_function("brolm_dfa_filter");
                if (fn) {
                    brass::codegen::KernelJit jit(opts);
                    filter_kfn = jit.compile(*fn);
                    if (filter_kfn.is_valid()) {
                        filter_fn = filter_kfn.as<brass::codegen::DfaFilterTokensFn>();
                    }
                }
            }
        } catch (...) {
            mask_fn = nullptr;
            scan_fn = nullptr;
            filter_fn = nullptr;
        }

        initialized = true;
    }
};

static CompiledKernels g_kernels;
#endif

}  // namespace

void JitGrammar::compile_kernels() {
#if BROLM_HAS_BRASS_JIT
    g_kernels.ensure_initialized();
    mask_kfn_ = g_kernels.mask_kfn;
    scan_kfn_ = g_kernels.scan_kfn;
    filter_kfn_ = g_kernels.filter_kfn;
    mask_fn = g_kernels.mask_fn;
    scan_fn = g_kernels.scan_fn;
    filter_fn_ = g_kernels.filter_fn;
#endif
}

JitGrammar::JitGrammar(CompiledDfa dfa)
    : dfa_(std::move(dfa)),
      vocab_indexer_(std::make_unique<VocabIndexer>()) {
    current_state_ = dfa_.start_state;
    compile_kernels();
}

JitGrammar::JitGrammar(CompiledDfa dfa, const std::vector<std::string>& vocab)
    : dfa_(std::move(dfa)) {
    current_state_ = dfa_.start_state;
    compile_kernels();
    vocab_indexer_ = std::make_unique<VocabIndexer>(vocab, filter_fn_);
}

JitGrammar::JitGrammar(const JitGrammar& other)
    : mask_fn(other.mask_fn),
      scan_fn(other.scan_fn),
      current_state_(other.current_state_),
      dfa_(other.dfa_),
      mask_kfn_(other.mask_kfn_),
      scan_kfn_(other.scan_kfn_),
      filter_kfn_(other.filter_kfn_),
      filter_fn_(other.filter_fn_) {
    if (other.vocab_indexer_) {
        vocab_indexer_ = std::make_unique<VocabIndexer>(*other.vocab_indexer_);
    } else {
        vocab_indexer_ = std::make_unique<VocabIndexer>();
    }
}

JitGrammar& JitGrammar::operator=(const JitGrammar& other) {
    if (this != &other) {
        mask_fn = other.mask_fn;
        scan_fn = other.scan_fn;
        current_state_ = other.current_state_;
        dfa_ = other.dfa_;
        mask_kfn_ = other.mask_kfn_;
        scan_kfn_ = other.scan_kfn_;
        filter_kfn_ = other.filter_kfn_;
        filter_fn_ = other.filter_fn_;
        scratch_mask_.clear();
        if (other.vocab_indexer_) {
            vocab_indexer_ = std::make_unique<VocabIndexer>(*other.vocab_indexer_);
        } else {
            vocab_indexer_ = std::make_unique<VocabIndexer>();
        }
    }
    return *this;
}

JitGrammar::JitGrammar(JitGrammar&& other) noexcept = default;
JitGrammar& JitGrammar::operator=(JitGrammar&& other) noexcept = default;
JitGrammar::~JitGrammar() = default;

bool JitGrammar::accept(std::string_view text) {
    if (current_state_ < 0) return false;
    int32_t next = -1;
    if (scan_fn) {
        next = scan_fn(
            dfa_.transition_table.data(),
            current_state_,
            reinterpret_cast<const uint8_t*>(text.data()),
            text.size()
        );
    } else {
        next = current_state_;
        for (char c : text) {
            next = dfa_.step(next, static_cast<uint8_t>(c));
            if (next < 0) break;
        }
    }

    if (next < 0) {
        current_state_ = -1;
        return false;
    }
    current_state_ = next;
    return true;
}

bool JitGrammar::can_accept(std::string_view text) const {
    if (current_state_ < 0) return false;
    if (scan_fn) {
        int32_t next = scan_fn(
            dfa_.transition_table.data(),
            current_state_,
            reinterpret_cast<const uint8_t*>(text.data()),
            text.size()
        );
        return next >= 0;
    } else {
        int32_t next = current_state_;
        for (char c : text) {
            next = dfa_.step(next, static_cast<uint8_t>(c));
            if (next < 0) return false;
        }
        return true;
    }
}

bool JitGrammar::is_accepted() const {
    if (current_state_ < 0 || current_state_ >= dfa_.num_states) return false;
    return dfa_.is_accept_state[current_state_];
}

void JitGrammar::reset() {
    current_state_ = dfa_.start_state;
}

void JitGrammar::precompute_vocab(const std::vector<std::string>& vocab_tokens) {
    if (!vocab_indexer_) {
        vocab_indexer_ = std::make_unique<VocabIndexer>(vocab_tokens, filter_fn_);
    } else if (!vocab_indexer_->same_vocab(vocab_tokens)) {
        vocab_indexer_->set_vocab(vocab_tokens, filter_fn_);
    }
    vocab_indexer_->precompute_all(dfa_);
}

void JitGrammar::mask_logits(float* logits, int vocab_size,
                             const std::vector<std::string>& vocab_tokens,
                             int eos_id) const {
    if (!logits || vocab_size <= 0) return;

    if (!vocab_indexer_) {
        vocab_indexer_ = std::make_unique<VocabIndexer>();
    }

    if (!vocab_indexer_->same_vocab(vocab_tokens)) {
        vocab_indexer_->set_vocab(vocab_tokens, filter_fn_);
    }

    const std::vector<uint64_t>& valid_mask = vocab_indexer_->get_valid_mask(current_state_, dfa_);
    size_t words = (static_cast<size_t>(vocab_size) + 63) / 64;

    scratch_mask_.resize(words);
    size_t copy_words = std::min(words, valid_mask.size());
    if (copy_words > 0) {
        std::memcpy(scratch_mask_.data(), valid_mask.data(), copy_words * sizeof(uint64_t));
    }
    if (words > copy_words) {
        std::memset(scratch_mask_.data() + copy_words, 0, (words - copy_words) * sizeof(uint64_t));
    }

    // Set/clear EOS bit according to whether current state is accepted
    if (eos_id >= 0 && eos_id < vocab_size) {
        size_t w = static_cast<size_t>(eos_id) / 64;
        uint64_t bit = 1ULL << (static_cast<size_t>(eos_id) % 64);
        if (is_accepted()) {
            scratch_mask_[w] |= bit;
        } else {
            scratch_mask_[w] &= ~bit;
        }
    }

    const float kMaskVal = -std::numeric_limits<float>::infinity();

    if (mask_fn) {
        mask_fn(logits, scratch_mask_.data(), static_cast<uint64_t>(vocab_size), kMaskVal);
    } else {
        // Scalar fallback loop
        for (int i = 0; i < vocab_size; ++i) {
            size_t w = static_cast<size_t>(i) / 64;
            uint64_t bit = 1ULL << (static_cast<size_t>(i) % 64);
            if ((scratch_mask_[w] & bit) == 0) {
                logits[i] = kMaskVal;
            }
        }
    }
}

}  // namespace brolm
