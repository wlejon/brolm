#ifdef _MSC_VER
// std::getenv is fine here: read-once, no concurrent setenv in this process.
#pragma warning(disable : 4996)
#endif

#include "brolm/detail/profile.h"

#include "brotensor/runtime.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace brolm::detail::profile {

namespace {

constexpr int kStageCount = static_cast<int>(Stage::COUNT);

constexpr const char* kStageNames[kStageCount] = {
    "embed",
    "idx_upload",
    "rms_norm",
    "qkv_proj",
    "qk_norm",
    "rope",
    "kv_append",
    "attention",
    "o_proj",
    "mlp_proj",
    "swiglu",
    "residual_add",
    "final_norm",
    "lm_head",
    "logits_download",
    "sample",
    "other",
};

struct Accum {
    std::uint64_t ns    = 0;
    std::uint64_t calls = 0;
};

std::array<Accum, kStageCount> g_accum;

std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

}  // namespace

bool enabled() {
    static const bool on = [] {
        const char* v = std::getenv("BROLM_PROFILE");
        return v != nullptr && *v != '\0' && !(v[0] == '0' && v[1] == '\0');
    }();
    return on;
}

void add(Stage s, std::uint64_t ns) {
    Accum& a = g_accum[static_cast<std::size_t>(static_cast<int>(s))];
    a.ns += ns;
    a.calls += 1;
}

void reset() { g_accum.fill(Accum{}); }

void report() {
    std::uint64_t total_ns = 0;
    for (const Accum& a : g_accum) total_ns += a.ns;
    if (total_ns == 0) return;

    std::array<int, kStageCount> order{};
    for (int i = 0; i < kStageCount; ++i) order[static_cast<std::size_t>(i)] = i;
    std::sort(order.begin(), order.end(), [](int a, int b) {
        return g_accum[static_cast<std::size_t>(a)].ns >
               g_accum[static_cast<std::size_t>(b)].ns;
    });

    std::fprintf(stderr,
                 "[brolm profile] %-16s %10s %10s %12s %6s\n",
                 "stage", "total ms", "calls", "us/call", "%");
    for (int idx : order) {
        const Accum& a = g_accum[static_cast<std::size_t>(idx)];
        if (a.calls == 0) continue;
        std::fprintf(stderr,
                     "[brolm profile] %-16s %10.2f %10llu %12.2f %5.1f%%\n",
                     kStageNames[idx],
                     static_cast<double>(a.ns) / 1e6,
                     static_cast<unsigned long long>(a.calls),
                     static_cast<double>(a.ns) / 1e3 /
                         static_cast<double>(a.calls),
                     100.0 * static_cast<double>(a.ns) /
                         static_cast<double>(total_ns));
    }
    std::fprintf(stderr, "[brolm profile] %-16s %10.2f\n", "TOTAL",
                 static_cast<double>(total_ns) / 1e6);
}

ScopedStage::ScopedStage(Stage s) : stage_(s) {
    if (!enabled()) return;
    brotensor::sync_all();
    t0_ = now_ns();
}

ScopedStage::~ScopedStage() {
    if (t0_ == 0) return;
    brotensor::sync_all();
    add(stage_, now_ns() - t0_);
}

}  // namespace brolm::detail::profile
