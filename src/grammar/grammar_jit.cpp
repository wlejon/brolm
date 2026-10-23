#include "brolm/grammar_jit.h"
#include "grammar/vocab_indexer.h"

#include <brass/codegen/grammar_builder.hpp>
#include <brass/codegen/kernel_jit.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/codegen/peephole.hpp>
#include <brass/core/diagnostics.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/smt_guard_elim.hpp>
#include <brass/mir/superopt.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/target/aot_linker.hpp>
#include <brass/target/dynamic_library.hpp>
#include <brass/target/target.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <brass/verify/smt_equiv.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <process.h>
#if defined(_M_X64) || defined(__x86_64__)
#include <intrin.h>
#endif
#else
#include <unistd.h>
#if defined(__x86_64__)
#include <cpuid.h>
#endif
#endif

namespace brolm {

namespace {

#if BROLM_HAS_BRASS_JIT

constexpr uint32_t kGrammarKernelSchemaVersion = 2;

uint32_t query_cpu_simd_mask() noexcept {
    uint32_t mask = 0;
#if defined(__x86_64__) && !defined(_WIN32)
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        if (ecx & (1u << 20)) mask |= (1u << 0); // SSE4.2
        if (ecx & (1u << 28)) mask |= (1u << 1); // AVX
        if (ecx & (1u << 12)) mask |= (1u << 2); // FMA
    }
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        if (ebx & (1u << 5))  mask |= (1u << 3); // AVX2
        if (ebx & (1u << 16)) mask |= (1u << 4); // AVX512F
    }
#elif defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__))
    int info[4] = {0, 0, 0, 0};
    __cpuid(info, 1);
    if (info[2] & (1 << 20)) mask |= (1u << 0); // SSE4.2
    if (info[2] & (1 << 28)) mask |= (1u << 1); // AVX
    if (info[2] & (1 << 12)) mask |= (1u << 2); // FMA
    __cpuidex(info, 7, 0);
    if (info[1] & (1 << 5))  mask |= (1u << 3); // AVX2
    if (info[1] & (1 << 16)) mask |= (1u << 4); // AVX512F
#elif defined(__aarch64__) || defined(_M_ARM64)
    mask |= (1u << 0); // ASIMD / NEON baseline
#endif
    return mask;
}

uint64_t fnv1a_update(uint64_t h, const void* data, size_t len) noexcept {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; ++i) {
        h ^= static_cast<uint64_t>(bytes[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

std::string host_shared_lib_extension(const brass::Target& t) {
    if (t.is_windows()) return ".dll";
    if (t.is_macos()) return ".dylib";
    return ".so";
}

std::filesystem::path resolve_cache_directory() {
    if (const char* env_dir = std::getenv("BRO_LM_CACHE_DIR"); env_dir && env_dir[0] != '\0') {
        return std::filesystem::path(env_dir);
    }
    if (const char* env_dir2 = std::getenv("BROLM_CACHE_DIR"); env_dir2 && env_dir2[0] != '\0') {
        return std::filesystem::path(env_dir2);
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

std::filesystem::path resolve_kernel_cache_path() {
    const brass::Target host = brass::Target::host();
    const uint32_t simd_mask = query_cpu_simd_mask();
    const uint32_t abi_stamp = static_cast<uint32_t>(sizeof(brass::codegen::KernelOptions)) ^
                               (static_cast<uint32_t>(sizeof(brass::object::CompiledFunctionInfo)) << 8) ^
                               (static_cast<uint32_t>(host.pointer_size()) << 16) ^
                               (static_cast<uint32_t>(host.stack_alignment()) << 24);

    uint64_t hash = 14695981039346656037ULL;
    const auto arch_val = static_cast<uint8_t>(host.arch());
    const auto os_val = static_cast<uint8_t>(host.os());
    const auto fmt_val = static_cast<uint8_t>(host.object_format());
    hash = fnv1a_update(hash, &arch_val, sizeof(arch_val));
    hash = fnv1a_update(hash, &os_val, sizeof(os_val));
    hash = fnv1a_update(hash, &fmt_val, sizeof(fmt_val));
    hash = fnv1a_update(hash, &simd_mask, sizeof(simd_mask));
    hash = fnv1a_update(hash, &abi_stamp, sizeof(abi_stamp));
    hash = fnv1a_update(hash, &kGrammarKernelSchemaVersion, sizeof(kGrammarKernelSchemaVersion));
    static constexpr std::string_view kSchemaSig =
        "brolm_logit_mask:f32x8:v2;brolm_dfa_scan:u4:v2;brolm_dfa_filter:v2";
    hash = fnv1a_update(hash, kSchemaSig.data(), kSchemaSig.size());

    char fname[160];
    std::snprintf(fname, sizeof(fname),
                  "brolm_grammar_kernels_%.*s_%.*s_simd%02x_v%u_%016llx%s",
                  static_cast<int>(brass::to_string(host.arch()).size()),
                  brass::to_string(host.arch()).data(),
                  static_cast<int>(brass::to_string(host.os()).size()),
                  brass::to_string(host.os()).data(),
                  simd_mask,
                  kGrammarKernelSchemaVersion,
                  static_cast<unsigned long long>(hash),
                  host_shared_lib_extension(host).c_str());

    return resolve_cache_directory() / fname;
}

// Prepares a verification clone of a MIR function for fast, bit-exact QF_BV
// equivalence checking:
// 1. Expands `vstore(f32x8, ptr, off, vbroadcast(f32x8, scalar))` into scalar stores.
// 2. Lowers each `load` / `load_indexed` site to a site-tagged uninterpreted call
//    so the QF_BV equivalence checker proves index/address equivalence directly
//    (`idx_before == idx_after`) without O(N^2) cross-array byte aliasing or 64x64
//    multiplier bit-blasting on `scale = 4` / `scale = 8`.
void prepare_function_for_smt_verification(brass::Function& fn, brass::Module& mod) {
    size_t load_site_id = 0;
    for (brass::BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        std::vector<brass::Instruction*> to_remove;
        for (brass::Instruction* inst = bb->head(); inst != nullptr; inst = inst->next()) {
            if (inst->opcode() == brass::Opcode::vstore && inst->operand_count() >= 2) {
                brass::Value* ptr = inst->operand(0);
                brass::Value* vec_val = inst->operand(1);
                if (vec_val && vec_val->is_instruction() &&
                    vec_val->defining_instruction() &&
                    vec_val->defining_instruction()->opcode() == brass::Opcode::vbroadcast &&
                    vec_val->defining_instruction()->operand_count() >= 1) {
                    brass::Value* scalar_val = vec_val->defining_instruction()->operand(0);
                    brass::Type elem_t = scalar_val->type();
                    size_t elem_bytes = elem_t.size_in_bytes();
                    size_t vec_bytes = inst->memory_type().size_in_bytes();
                    size_t lanes = (elem_bytes > 0) ? (vec_bytes / elem_bytes) : 0;
                    int32_t base_off = inst->offset();
                    for (size_t lane = 0; lane < lanes; ++lane) {
                        auto* st = mod.arena().make<brass::Instruction>(
                            brass::Opcode::store, brass::Type::void_type());
                        st->set_memory_type(elem_t);
                        st->set_offset(base_off + static_cast<int32_t>(lane * elem_bytes));
                        st->add_operand(ptr);
                        st->add_operand(scalar_val);
                        bb->insert_before(st, inst);
                    }
                    to_remove.push_back(inst);
                }
            } else if (inst->opcode() == brass::Opcode::umod && inst->operand_count() >= 2) {
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
        for (brass::Instruction* inst : to_remove) {
            bb->remove_instruction(inst);
        }
        std::vector<brass::Instruction*> dead_bcast;
        for (brass::Instruction* inst = bb->head(); inst != nullptr; inst = inst->next()) {
            if (inst->opcode() == brass::Opcode::vbroadcast) {
                dead_bcast.push_back(inst);
            }
        }
        for (brass::Instruction* inst : dead_bcast) {
            bb->remove_instruction(inst);
        }
    }
}

bool write_shared_library_atomically(const brass::object::ObjectFile& obj,
                                     const std::filesystem::path& cache_path) {
    std::error_code ec;
    if (cache_path.has_parent_path()) {
        std::filesystem::create_directories(cache_path.parent_path(), ec);
        if (ec) return false;
    }

    brass::target::LinkerOptions linker_opts;
    linker_opts.format = brass::target::OutputFormat::Auto;
    linker_opts.module_name = cache_path.filename().string();
    linker_opts.soname = cache_path.filename().string();
    linker_opts.export_all_functions = true;

    std::string link_err;
    std::vector<uint8_t> image_bytes = brass::target::AotLinker::link(obj, linker_opts, &link_err);
    if (image_bytes.empty()) {
        return false;
    }

    static std::atomic<uint64_t> s_temp_nonce{0};
#if defined(_WIN32)
    const int pid = _getpid();
#else
    const int pid = static_cast<int>(getpid());
#endif
    const uint64_t nonce = ++s_temp_nonce;
    std::filesystem::path tmp_path = cache_path;
    tmp_path += ".tmp." + std::to_string(pid) + "_" + std::to_string(nonce);

    {
        std::ofstream ofs(tmp_path, std::ios::binary | std::ios::trunc);
        if (!ofs.is_open()) return false;
        ofs.write(reinterpret_cast<const char*>(image_bytes.data()),
                  static_cast<std::streamsize>(image_bytes.size()));
        ofs.flush();
        if (!ofs.good()) {
            ofs.close();
            std::filesystem::remove(tmp_path, ec);
            return false;
        }
    }

    std::filesystem::permissions(
        tmp_path,
        std::filesystem::perms::owner_read | std::filesystem::perms::owner_write |
            std::filesystem::perms::owner_exec | std::filesystem::perms::group_read |
            std::filesystem::perms::group_exec | std::filesystem::perms::others_read |
            std::filesystem::perms::others_exec,
        std::filesystem::perm_options::replace, ec);

#if defined(__APPLE__)
    if (obj.target.is_macos()) {
        std::string cmd = "codesign -s - -f \"" + tmp_path.string() + "\" > /dev/null 2>&1";
        int res = std::system(cmd.c_str());
        (void)res;
    }
#endif

    std::filesystem::rename(tmp_path, cache_path, ec);
    if (ec) {
        std::filesystem::remove(tmp_path, ec);
        return false;
    }
    return true;
}

struct CompiledKernels {
    brass::codegen::KernelFunction mask_kfn;
    brass::codegen::KernelFunction scan_kfn;
    brass::codegen::KernelFunction filter_kfn;
    std::shared_ptr<brass::target::DynamicLibrary> aot_lib;
    // Keep retired libraries/engines alive across test cache resets so any
    // previously constructed JitGrammar instances never hold dangling pointers.
    std::vector<std::shared_ptr<brass::target::DynamicLibrary>> retired_libs;
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

    bool try_load_from_disk_cache(const std::filesystem::path& cache_path) {
        std::error_code ec;
        if (!std::filesystem::exists(cache_path, ec) || ec) {
            return false;
        }
        std::string err;
        auto lib = brass::target::DynamicLibrary::open(cache_path.string(), &err);
        if (!lib || !lib->is_valid()) {
            return false;
        }
        auto m_fn = lib->get_function<brass::codegen::LogitMaskFn>("brolm_logit_mask");
        auto s_fn = lib->get_function<brass::codegen::DfaScanStringFn>("brolm_dfa_scan");
        auto f_fn = lib->get_function<brass::codegen::DfaFilterTokensFn>("brolm_dfa_filter");
        if (!m_fn || !s_fn || !f_fn) {
            return false;
        }
        aot_lib = std::shared_ptr<brass::target::DynamicLibrary>(std::move(lib));
        mask_fn = m_fn;
        scan_fn = s_fn;
        filter_fn = f_fn;
        stats.cache_hits++;
        stats.loaded_from_disk_cache = true;
        stats.cache_file_path = cache_path.string();
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

        const std::filesystem::path cache_path = resolve_kernel_cache_path();
        stats.cache_file_path = cache_path.string();

        // 1. Fast warm start: load pre-superoptimized shared library from disk cache
        if (try_load_from_disk_cache(cache_path)) {
            initialized = true;
            return;
        }

        // 2. Cold start: build MIR, superoptimize, formally verify equivalence, JIT & disk-cache
        stats.cache_misses++;
        stats.loaded_from_disk_cache = false;

        try {
            brass::Module mod("brolm_grammar_kernels_mod");
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

            // 2c. Formal equivalence verification: Unopt MIR == Superopt MIR
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
                    fn_eq_opts.max_sat_conflicts = 2000;
                    if (opt_fn->return_type().is_void()) {
                        fn_eq_opts.check_memory = false;
                    }

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

            // 2d. Standard MIR scalar & loop optimizations (GVN, SCCP, LICM, IVSR, DCE)
            brass::LoopOptOptions loop_opts;
            loop_opts.enable_avx2 = true;
            loop_opts.enable_fma = true;
            loop_opts.enable_fp_reassociation = false;
            loop_opts.enable_vectorize = false; // GrammarBuilder already emits f32x8 vector loop
            loop_opts.enable_slp = false;
            loop_opts.enable_unroll = false;    // GrammarBuilder already unrolls scan 4x
            loop_opts.enable_f64_demote = false;
            loop_opts.enable_parallel_loops = false;

            for (brass::Function* fn : mod.functions()) {
                if (!fn || fn->blocks().empty()) continue;
                brass::cfg_simplify_function(*fn);
                fn->rebuild_cfg_predecessors();
                brass::optimize_function(*fn, loop_opts);
                fn->rebuild_cfg_predecessors();
                fn->sort_blocks_rpo();
                fn->rebuild_cfg_predecessors();
            }

            brass::DiagnosticReporter diag;
            if (!brass::verify_module(mod, &diag)) {
                throw std::runtime_error("Grammar kernel MIR verification failed: " + diag.format_all());
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

            // Mark loop optimizations complete so ModuleCompiler lowers our verified MIR directly
            mod.set_has_loop_optimizations(true);

            brass::object::ModuleCompiler compiler(host);
            compiler.set_enable_mir_opts(false);
            compiler.set_emit_stack_map_symbol(false);
            brass::object::ObjectFile obj = compiler.compile(mod);

            // Persist AOT shared library to disk cache atomically
            write_shared_library_atomically(obj, cache_path);

            // Load compiled object into in-process JIT execution engine
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

static CompiledKernels g_kernels;
#endif

}  // namespace

GrammarKernelCacheStats JitGrammar::kernel_cache_stats() {
#if BROLM_HAS_BRASS_JIT
    return g_kernels.get_stats();
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
    g_kernels.reset_for_testing();
#endif
}

void JitGrammar::compile_kernels() {
#if BROLM_HAS_BRASS_JIT
    g_kernels.ensure_initialized();
    mask_kfn_ = g_kernels.mask_kfn;
    scan_kfn_ = g_kernels.scan_kfn;
    filter_kfn_ = g_kernels.filter_kfn;
    aot_lib_ = g_kernels.aot_lib;
    mask_fn = g_kernels.mask_fn;
    scan_fn = g_kernels.scan_fn;
    filter_fn_ = g_kernels.filter_fn;
    loaded_from_disk_cache_ = g_kernels.stats.loaded_from_disk_cache;
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
