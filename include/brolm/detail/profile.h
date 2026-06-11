#pragma once

// Env-gated per-stage timing for the decoder hot paths.
//
// Off by default with near-zero cost (one cached bool test per scope). Set
// BROLM_PROFILE=1 to accumulate wall time per pipeline stage; each scope then
// brackets its ops with brotensor::sync_all() so GPU work is attributed to the
// stage that issued it (this serialises the stream — profile numbers measure
// per-stage cost, not end-to-end throughput; bench throughput with the gate
// off). profile_report() prints the accumulated table and call counts.
//
// Single-threaded by design, like the decoders it instruments: plain
// accumulators, no synchronisation.

#include <cstdint>

namespace brolm::detail::profile {

enum class Stage : int {
    embed = 0,
    idx_upload,
    rms_norm,
    qkv_proj,
    qk_norm,
    rope,
    kv_append,
    attention,
    o_proj,
    mlp_proj,       // gate/up/down projections
    swiglu,
    residual_add,
    final_norm,
    lm_head,
    logits_download,
    sample,
    other,
    COUNT,
};

// True when BROLM_PROFILE is set to a non-empty, non-"0" value. Evaluated
// once on first call.
bool enabled();

// Add `ns` nanoseconds (and one call) to `s`. No-op gating is the caller's
// job — ScopedStage below is the normal entry point.
void add(Stage s, std::uint64_t ns);

// Zero all accumulators.
void reset();

// Print the per-stage table (total ms, calls, ms/call) to stderr, sorted by
// total time. No-op when nothing was recorded.
void report();

// RAII stage scope. When profiling is enabled, syncs the device before and
// after the bracketed ops and accumulates the elapsed wall time; when
// disabled, both constructor and destructor reduce to one bool test.
class ScopedStage {
public:
    explicit ScopedStage(Stage s);
    ~ScopedStage();

    ScopedStage(const ScopedStage&)            = delete;
    ScopedStage& operator=(const ScopedStage&) = delete;

private:
    Stage         stage_;
    std::uint64_t t0_ = 0;   // steady_clock ns at entry; 0 when disabled
};

}  // namespace brolm::detail::profile
