#include "brolm/grammar_jit.h"
#include "grammar/vocab_indexer.h"

#include <brass/cache/artifact_cache.hpp>
#include <brass/cache/native_module_cache.hpp>
#include <brass/codegen/grammar_builder.hpp>
#include <brass/codegen/kernel_jit.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/codegen/peephole.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/smt_guard_elim.hpp>
#include <brass/mir/superopt.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/target/target.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <brass/verify/smt_equiv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace brolm {

namespace {

#if BROLM_HAS_BRASS_JIT

constexpr uint32_t kGrammarKernelSchemaVersion = 3;
constexpr uint64_t kGrammarCacheMaxBytes = 64ULL * 1024 * 1024;

std::filesystem::path resolve_cache_directory() {
    if (const char* env_dir = std::getenv("BRO_LM_CACHE_DIR"); env_dir && env_dir[0] != '\0') {
        return std::filesystem::path(env_dir);
    }
#if defined(_WIN32)
    if (const char* local_app = std::getenv("LOCALAPPDATA"); local_app && local_app[0] != '\0') {
        return std::filesystem::path(local_app) / "brolm" / "kernels";
    }
    if (const char* user_prof = std::getenv("USERPROFILE"); user_prof && user_prof[0] != '\0') {
        return std::filesystem::path(user_prof) / ".cache" / "brolm" / "kernels";
    }
#else
    if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && xdg[0] != '\0') {
        return std::filesystem::path(xdg) / "brolm" / "kernels";
    }
    if (const char* home = std::getenv("HOME"); home && home[0] != '\0') {
        return std::filesystem::path(home) / ".cache" / "brolm" / "kernels";
    }
#endif
    std::error_code ec;
    auto tmp = std::filesystem::temp_directory_path(ec);
    if (!ec && !tmp.empty()) {
        return tmp / "brolm" / "kernels";
    }
    return std::filesystem::path(".cache") / "brolm" / "kernels";
}

brass::cache::CacheKey build_grammar_client_key() {
    static constexpr std::string_view kSchemaSig =
        "brolm_logit_mask:f32x8:v3;brolm_dfa_scan:u4:v3;brolm_dfa_filter:v3";
    return brass::cache::KeyBuilder("brolm.grammar_kernels/3")
        .add_u64("schema_version", kGrammarKernelSchemaVersion)
        .add("schema", kSchemaSig)
        .finish();
}

std::filesystem::path resolve_kernel_cache_path() {
    brass::cache::ArtifactCache store(resolve_cache_directory(), kGrammarCacheMaxBytes);
    brass::cache::NativeModuleCache cache(store);
    const brass::cache::NativeOptions native_opts;
    return store.path_for(cache.storage_key(build_grammar_client_key(), native_opts));
}

// Prepares a verification clone of a MIR function for fast, bit-exact QF_BV
// equivalence checking:
// Lowers power-of-two `umod` to bitwise mask and each `load` / `load_indexed`
// site to a site-tagged uninterpreted call so the QF_BV equivalence checker
// proves index/address/store equivalence directly without O(N^2) cross-array
// byte aliasing or 64x64 multiplier bit-blasting on `scale = 4` / `scale = 8`.
void prepare_function_for_smt_verification(brass::Function& fn, brass::Module& mod) {
    size_t load_site_id = 0;
    for (brass::BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (brass::Instruction* inst = bb->head(); inst != nullptr; inst = inst->next()) {
            if (inst->opcode() == brass::Opcode::umod && inst->operand_count() >= 2) {
                brass::Value* rhs = inst->operand(1);
                if (rhs && rhs->is_instruction() && rhs->defining_instruction() &&
                    (rhs->defining_instruction()->opcode() == brass::Opcode::iconst_i64 ||
                     rhs->defining_instruction()->opcode() == brass::Opcode::iconst_i32)) {
                    uint64_t c = static_cast<uint64_t>(rhs->defining_instruction()->imm_i64());
                    if (c > 0 && (c & (c - 1)) == 0) {
                        auto* mask_inst = mod.arena().make<brass::Instruction>(
                            rhs->defining_instruction()->opcode(), inst->type());
                        mask_inst->set_imm_i64(static_cast<int64_t>(c - 1));
                        auto* mask_val = mod.arena().make<brass::Value>(
                            fn.next_value_id(), inst->type(), brass::ValueKind::InstructionResult);
                        mask_inst->set_result(mask_val);
                        mask_val->set_defining_instruction(mask_inst);
                        bb->insert_before(mask_inst, inst);
                        inst->set_opcode(brass::Opcode::and_);
                        inst->set_operand(1, mask_val);
                    }
                }
            } else if (inst->opcode() == brass::Opcode::load ||
                       inst->opcode() == brass::Opcode::load_indexed) {
                std::string sym_str = "__verify_load_" + std::to_string(load_site_id++);
                std::string_view sym = mod.string_pool().intern(sym_str);
                inst->set_opcode(brass::Opcode::call);
                inst->set_symbol(sym);
            }
        }
    }
}

struct CompiledKernels {
    brass::codegen::KernelFunction mask_kfn;
    brass::codegen::KernelFunction scan_kfn;
    brass::codegen::KernelFunction filter_kfn;
    std::shared_ptr<void> aot_lib;
    std::unique_ptr<brass::cache::ArtifactCache> disk_store;
    std::unique_ptr<brass::cache::NativeModuleCache> native_cache;
    // Keep retired libraries/engines alive across test cache resets so any
    // previously constructed JitGrammar instances never hold dangling pointers.
    std::vector<std::shared_ptr<void>> retired_libs;
    std::vector<brass::codegen::KernelFunction> retired_kfns;

    brass::codegen::LogitMaskFn mask_fn = nullptr;
    brass::codegen::DfaScanStringFn scan_fn = nullptr;
    brass::codegen::DfaFilterTokensFn filter_fn = nullptr;

    GrammarKernelCacheStats stats;
    bool initialized = false;
    std::mutex mtx;

    void reset_for_testing() {
        std::lock_guard<std::mutex> lock(mtx);
        if (aot_lib) retired_libs.push_back(std::move(aot_lib));
        if (mask_kfn.is_valid()) retired_kfns.push_back(std::move(mask_kfn));
        if (scan_kfn.is_valid()) retired_kfns.push_back(std::move(scan_kfn));
        if (filter_kfn.is_valid()) retired_kfns.push_back(std::move(filter_kfn));
        native_cache.reset();
        disk_store.reset();
        mask_fn = nullptr;
        scan_fn = nullptr;
        filter_fn = nullptr;
        stats = GrammarKernelCacheStats{};
        initialized = false;
    }

    GrammarKernelCacheStats get_stats() {
        std::lock_guard<std::mutex> lock(mtx);
        return stats;
    }

    bool build_grammar_module(const brass::Target& host,
                              brass::Module& mod,
                              brass::object::ObjectFile* out_obj) {
        {
            brass::codegen::GrammarBuilder gb(mod);
            gb.build_logit_mask_function("brolm_logit_mask");
            gb.build_dfa_scan_string_function("brolm_dfa_scan", 4);
            gb.build_dfa_filter_tokens_function("brolm_dfa_filter");
        }

        // Clone unoptimized module for SMT formal equivalence verification
        std::unique_ptr<brass::Module> unopt_mod = brass::clone_module(mod);

        for (brass::Function* fn : mod.functions()) {
            if (!fn || fn->blocks().empty()) continue;

            // 2a. SMT relational guard & bounds-check elimination
            brass::SmtGuardElimStats guard_stats;
            brass::SmtGuardElimOptions guard_opts;
            guard_opts.max_sat_conflicts = 1000;
            guard_opts.stats = &guard_stats;
            brass::run_smt_guard_elimination(*fn, mod, guard_opts);
            stats.smt_guards_eliminated +=
                guard_stats.guards_proven_eliminated +
                guard_stats.comparisons_proven_true +
                guard_stats.comparisons_proven_false +
                guard_stats.branches_folded;
            stats.smt_queries_solved += guard_stats.queries_solved;

            // 2b. Solver-verified MIR superoptimization (mask elision, ext elision, DAG synthesis)
            brass::SuperoptStats sopt_stats;
            brass::SuperoptOptions sopt_opts;
            sopt_opts.max_sat_conflicts = 1000;
            sopt_opts.stats = &sopt_stats;
            brass::run_mir_superopt(*fn, mod, sopt_opts);
            stats.superopt_masks_elided += sopt_stats.masks_elided;
            stats.superopt_extensions_elided += sopt_stats.extensions_elided;
            stats.superopt_dags_synthesized += sopt_stats.dags_synthesized;
            stats.superopt_constants_synthesized += sopt_stats.constants_synthesized;
        }

        // 2c. Formal equivalence verification: Unopt MIR == Superopt MIR (with native SIMD & memory checks)
        if (unopt_mod) {
            for (const brass::Function* opt_fn : mod.functions()) {
                if (!opt_fn || opt_fn->blocks().empty()) continue;
                const brass::Function* unopt_fn = unopt_mod->get_function(opt_fn->name());
                if (!unopt_fn) continue;

                brass::Module v_before_mod("verify_before");
                brass::Module v_after_mod("verify_after");
                brass::Function* vb = brass::clone_function(*unopt_fn, v_before_mod);
                brass::Function* va = brass::clone_function(*opt_fn, v_after_mod);
                prepare_function_for_smt_verification(*vb, v_before_mod);
                prepare_function_for_smt_verification(*va, v_after_mod);

                brass::verify::EquivOptions fn_eq_opts;
                fn_eq_opts.loop_unroll_bound = 1;
                fn_eq_opts.max_paths = 16;
                fn_eq_opts.max_sat_conflicts = 4000;
                // prepare_function_for_smt_verification replaces read-only table loads with synthetic __verify_load_* calls.
                fn_eq_opts.assume_calls_pure = true;
                fn_eq_opts.check_memory = true;

                brass::verify::EquivResult eq =
                    brass::verify::verify_mir_equivalence(*vb, *va, fn_eq_opts);

                if (eq.is_counterexample()) {
                    throw std::runtime_error(
                        "Grammar kernel MIR equivalence counterexample in " +
                        std::string(opt_fn->name()) + ": " + eq.diagnostic);
                }
                if (eq.is_equivalent()) {
                    stats.mir_functions_verified_equivalent++;
                }
            }
        }

        // 2d. Standard MIR scalar & loop optimizations (GVN, SCCP, LICM, IVSR, DCE) + verification
        for (brass::Function* fn : mod.functions()) {
            if (!fn || fn->blocks().empty()) continue;
            brass::cfg_simplify_function(*fn);
            fn->rebuild_cfg_predecessors();
        }
        brass::codegen::KernelOptions kopts;
        kopts.opt_level = 2;
        kopts.enable_optimizations = true;
        kopts.enable_sroa = true;
        kopts.enable_bce = true;
        kopts.enable_smt_guard_elim = false; // already run with stats in Stage 2a
        kopts.enable_superopt = false;       // already run with stats in Stage 2b
        kopts.enable_avx2 = true;
        kopts.enable_fma = true;
        kopts.enable_fp_reassociation = false;
        kopts.enable_vectorize = false; // GrammarBuilder already emits f32x8 vector loop
        kopts.enable_slp = false;
        kopts.enable_unroll = false;    // GrammarBuilder already unrolls scan 4x
        kopts.enable_parallel = false;
        brass::codegen::KernelJit::prepare_module(mod, kopts);
        for (brass::Function* fn : mod.functions()) {
            if (!fn || fn->blocks().empty()) continue;
            fn->sort_blocks_rpo();
            fn->rebuild_cfg_predecessors();
        }

        // 2e. x64 LIR peephole & window superoptimization verification
        if (host.is_x64()) {
            brass::CallingConvention cc = brass::CallingConvention::for_target(host);
            for (const brass::Function* fn : mod.functions()) {
                if (!fn || fn->blocks().empty()) continue;
                brass::x64::X64ISel isel(host, cc);
                isel.set_callee_module(&mod);
                auto lir = isel.lower(*fn);
                if (!lir) continue;
                lir->sort_blocks_rpo();
                brass::codegen::LivenessAnalysis liveness(*lir);
                liveness.run();
                brass::codegen::LinearScanAllocator regalloc(*lir, liveness, cc);
                regalloc.allocate();
                brass::codegen::run_lir_peephole_optimizations(*lir);

                brass::SuperoptStats lir_stats;
                brass::SuperoptOptions lir_opts;
                lir_opts.max_sat_conflicts = 250;
                lir_opts.stats = &lir_stats;
                for (auto& blk : lir->blocks) {
                    if (blk && blk->instructions.size() <= 16) {
                        brass::run_x64_lir_superopt(*blk, lir_opts);
                    }
                }
                stats.superopt_lir_windows_optimized += lir_stats.lir_windows_optimized;
            }
        }

        mod.set_has_loop_optimizations(true);
        if (out_obj) {
            brass::object::ModuleCompiler compiler(host);
            compiler.set_enable_mir_opts(false);
            compiler.set_emit_stack_map_symbol(false);
            compiler.set_x64_avx(host.is_x64() && brass::Target::host_has_avx());
            *out_obj = compiler.compile(mod);
        }
        return true;
    }

    void ensure_initialized() {
        std::lock_guard<std::mutex> lock(mtx);
        if (initialized) return;

        const brass::Target host = brass::Target::host();
        if (!host.is_x64() && !host.is_aarch64()) {
            initialized = true;
            return;
        }

        const std::filesystem::path cache_dir = resolve_cache_directory();
        if (!disk_store || disk_store->dir() != cache_dir) {
            native_cache.reset();
            disk_store = std::make_unique<brass::cache::ArtifactCache>(cache_dir, kGrammarCacheMaxBytes);
            native_cache = std::make_unique<brass::cache::NativeModuleCache>(*disk_store);
        }

        const brass::cache::CacheKey client_key = build_grammar_client_key();
        const brass::cache::NativeOptions native_opts;
        stats.cache_file_path =
            disk_store->path_for(native_cache->storage_key(client_key, native_opts)).string();

        try {
            std::string cache_err;
            std::shared_ptr<const brass::cache::NativeModule> native_mod =
                native_cache->get_or_build_object(
                    client_key,
                    [&](brass::object::ObjectFile& obj, std::string& err) -> bool {
                        brass::Module mod("brolm_grammar_kernels_mod");
                        if (!build_grammar_module(host, mod, &obj)) {
                            err = "Grammar kernel build failed";
                            return false;
                        }
                        return true;
                    },
                    native_opts,
                    &cache_err);

            if (native_mod) {
                auto m_fn = native_mod->function<brass::codegen::LogitMaskFn>("brolm_logit_mask");
                auto s_fn = native_mod->function<brass::codegen::DfaScanStringFn>("brolm_dfa_scan");
                auto f_fn = native_mod->function<brass::codegen::DfaFilterTokensFn>("brolm_dfa_filter");
                if (m_fn && s_fn && f_fn) {
                    aot_lib = std::const_pointer_cast<void>(
                        std::static_pointer_cast<const void>(native_mod));
                    mask_kfn = brass::codegen::KernelFunction::from_owner(
                        "brolm_logit_mask", reinterpret_cast<void*>(m_fn), 0, native_mod);
                    scan_kfn = brass::codegen::KernelFunction::from_owner(
                        "brolm_dfa_scan", reinterpret_cast<void*>(s_fn), 0, native_mod);
                    filter_kfn = brass::codegen::KernelFunction::from_owner(
                        "brolm_dfa_filter", reinterpret_cast<void*>(f_fn), 0, native_mod);
                    mask_fn = m_fn;
                    scan_fn = s_fn;
                    filter_fn = f_fn;
                    if (native_mod->from_cache()) {
                        stats.cache_hits++;
                        stats.loaded_from_disk_cache = true;
                    } else {
                        stats.cache_misses++;
                        stats.loaded_from_disk_cache = false;
                    }
                    stats.cache_file_path = disk_store->path_for(native_mod->storage_key()).string();
                    initialized = true;
                    return;
                }
            }

            // Fallback to in-process JIT execution engine if disk cache/AOT link failed
            stats.cache_misses++;
            stats.loaded_from_disk_cache = false;
            brass::Module mod("brolm_grammar_kernels_mod");
            brass::object::ObjectFile obj;
            build_grammar_module(host, mod, &obj);

            auto engine = std::make_shared<brass::codegen::JitExecutionEngine>(host);
            if (!engine->load_object(obj)) {
                throw std::runtime_error("Grammar kernel JIT load_object failed");
            }

            void* mask_addr = engine->get_symbol_address("brolm_logit_mask");
            void* scan_addr = engine->get_symbol_address("brolm_dfa_scan");
            void* filter_addr = engine->get_symbol_address("brolm_dfa_filter");

            if (mask_addr) {
                mask_kfn = brass::codegen::KernelFunction("brolm_logit_mask", mask_addr, 0, engine);
                mask_fn = mask_kfn.as<brass::codegen::LogitMaskFn>();
            }
            if (scan_addr) {
                scan_kfn = brass::codegen::KernelFunction("brolm_dfa_scan", scan_addr, 0, engine);
                scan_fn = scan_kfn.as<brass::codegen::DfaScanStringFn>();
            }
            if (filter_addr) {
                filter_kfn = brass::codegen::KernelFunction("brolm_dfa_filter", filter_addr, 0, engine);
                filter_fn = filter_kfn.as<brass::codegen::DfaFilterTokensFn>();
            }
        } catch (...) {
            mask_fn = nullptr;
            scan_fn = nullptr;
            filter_fn = nullptr;
        }

        initialized = true;
    }
};

static CompiledKernels& kernels() {
    static auto* k = new CompiledKernels();
    return *k;
}
#endif

}  // namespace

GrammarKernelCacheStats JitGrammar::kernel_cache_stats() {
#if BROLM_HAS_BRASS_JIT
    return kernels().get_stats();
#else
    return GrammarKernelCacheStats{};
#endif
}

std::string JitGrammar::kernel_cache_path() {
#if BROLM_HAS_BRASS_JIT
    return resolve_kernel_cache_path().string();
#else
    return {};
#endif
}

void JitGrammar::reset_kernel_cache_for_testing() {
#if BROLM_HAS_BRASS_JIT
    kernels().reset_for_testing();
#endif
}

void JitGrammar::compile_kernels() {
#if BROLM_HAS_BRASS_JIT
    auto& k = kernels();
    k.ensure_initialized();
    mask_kfn_ = k.mask_kfn;
    scan_kfn_ = k.scan_kfn;
    filter_kfn_ = k.filter_kfn;
    aot_lib_ = k.aot_lib;
    mask_fn = k.mask_fn;
    scan_fn = k.scan_fn;
    filter_fn_ = k.filter_fn;
    loaded_from_disk_cache_ = k.stats.loaded_from_disk_cache;
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
      aot_lib_(other.aot_lib_),
      filter_fn_(other.filter_fn_),
      loaded_from_disk_cache_(other.loaded_from_disk_cache_) {
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
        aot_lib_ = other.aot_lib_;
        filter_fn_ = other.filter_fn_;
        loaded_from_disk_cache_ = other.loaded_from_disk_cache_;
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
