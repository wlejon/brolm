// brolm_bench_laya — latency / throughput scoreboard for the Laya decision model.
//
// For every (question count x state length) cell it runs predict() end to
// end — tokenize, encoder, head, scorer, act head, host readback, calibration
// — and reports latency percentiles and throughput; then, with per-stage
// device syncs switched on, the mean stage breakdown and the encoder split by
// op family. Unprofiled and profiled runs are separate: the stage syncs
// serialise the stream, so only the unprofiled numbers are latency.
//
// Then the open-loop cells (bench_laya_load.cpp): the request scheduler on 1
// and on every GPU, clients submitting 5-question requests as Poisson
// arrivals at fixed offered rates, reporting submit -> result percentiles,
// the achieved rate and how full the forwards ran. --sweep instead raises
// the rate until p99 passes --sweep-p99 ms and prints the sustained rate.
//
// Usage:
//   brolm_bench_laya [model_dir] [--iters N] [--warmup N] [--questions 1,5,10,50]
//                    [--states short,medium,max|none] [--packed 8x5,16x5,32x5|none] [--profile-iters N]
//                    [--json out.json] [--compare baseline.json] [--label text]
//                    [--checkpoints english,multilingual,typed-decisions]
//                    [--open short,medium|none] [--open-q 5] [--open-replicas 1,2]
//                    [--open-rates 75,150] [--open-clients 8] [--open-seconds 3] [--sweep] [--sweep-p99 30]
//                    [--tail]
//
// model_dir defaults to $LAYA_MODEL_DIR or ../laya beside the brolm checkout.
// --json writes the scoreboard; --compare prints each cell's p50 / p95 /
// throughput against a previously written scoreboard (ratio < 1 = faster).

#include "bench_laya_load.h"

#include "brolm/laya.h"
#include "brolm/laya_scheduler.h"
#include "brolm/detail/json.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace j = brolm::detail::json;
using Clock = std::chrono::steady_clock;

namespace {

struct Args {
    std::string model_dir;
    int iters = 30;
    int warmup = 3;
    int profile_iters = 5;
    std::vector<int> questions = {1, 5, 10, 50};
    std::vector<std::string> states = {"short", "medium", "max"};
    // Packed multi-request cells "RxQ" (R requests x Q questions), run over
    // the short and medium states that are listed in `states`.
    std::vector<std::string> packed = {"8x5", "16x5", "32x5"};
    std::string json_out, compare, label;
    // Family members to run, under model_dir (english = model_dir itself).
    std::vector<std::string> checkpoints;
    // Open-loop cells: the request scheduler under Poisson arrivals.
    std::vector<std::string> open_states = {"short", "medium"};
    int open_q = 5;
    std::vector<int> open_replicas = {1, 2};  // replica counts (clamped to the GPUs present)
    std::vector<double> open_rates = {75, 150};  // per replica, short state (medium: half)
    int open_clients = 8;
    double open_seconds = 3.0;
    bool sweep = false;
    double sweep_p99 = 30.0;
    bool tail = false;  // take apart each open cell's slowest 1 %
};

[[noreturn]] void usage() {
    std::fprintf(stderr,
                 "usage: brolm_bench_laya [model_dir] [--iters N] [--warmup N] [--questions 1,5,10,50]\n"
                 "                        [--states short,medium,max] [--packed 8x5,16x5,32x5] [--profile-iters N]\n"
                 "                        [--json out.json] [--compare baseline.json] [--label text]\n"
                 "                        [--checkpoints english,multilingual,typed-decisions]\n"
                 "                        [--open short,medium|none] [--open-q 5] [--open-replicas 1,2]\n"
                 "                        [--open-rates 75,150] [--open-clients 8] [--open-seconds 3]\n"
                 "                        [--sweep] [--sweep-p99 30] [--tail]\n");
    std::exit(2);
}

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, ',');) {
        if (!t.empty()) out.push_back(t);
    }
    return out;
}

Args parse_args(int argc, char** argv) {
    Args a;
    const char* env = std::getenv("LAYA_MODEL_DIR");
    a.model_dir = (env && env[0]) ? env : "../laya";
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) usage();
            return argv[++i];
        };
        if (k == "--iters") a.iters = std::stoi(val());
        else if (k == "--warmup") a.warmup = std::stoi(val());
        else if (k == "--profile-iters") a.profile_iters = std::stoi(val());
        else if (k == "--questions") {
            a.questions.clear();
            for (const auto& t : split(val())) a.questions.push_back(std::stoi(t));
        } else if (k == "--states") {
            const std::string v = val();
            a.states = v == "none" ? std::vector<std::string>{} : split(v);
        } else if (k == "--packed") {
            const std::string v = val();
            a.packed = v == "none" ? std::vector<std::string>{} : split(v);
        }
        else if (k == "--json") a.json_out = val();
        else if (k == "--compare") a.compare = val();
        else if (k == "--label") a.label = val();
        else if (k == "--checkpoints") a.checkpoints = split(val());
        else if (k == "--open") {
            const std::string v = val();
            a.open_states = v == "none" ? std::vector<std::string>{} : split(v);
        } else if (k == "--open-q") a.open_q = std::stoi(val());
        else if (k == "--open-replicas") {
            a.open_replicas.clear();
            for (const auto& t : split(val())) a.open_replicas.push_back(std::stoi(t));
        } else if (k == "--open-rates") {
            a.open_rates.clear();
            for (const auto& t : split(val())) a.open_rates.push_back(std::stod(t));
        } else if (k == "--open-clients") a.open_clients = std::stoi(val());
        else if (k == "--open-seconds") a.open_seconds = std::stod(val());
        else if (k == "--sweep") a.sweep = true;
        else if (k == "--tail") a.tail = true;
        else if (k == "--sweep-p99") a.sweep_p99 = std::stod(val());
        else if (k.rfind("--", 0) == 0) usage();
        else a.model_dir = k;
    }
    return a;
}

// ─── Workload ──────────────────────────────────────────────────────────────

std::string make_state(const std::string& kind) {
    const std::string email =
        "{\"from\": \"user@acme.com\", \"subject\": \"Duplicate charge on invoice #4411\", "
        "\"body\": \"Hi, we were billed twice for March. Please refund the duplicate today or we "
        "will cancel our plan.\"}";
    if (kind == "short") return email;  // ~50 state tokens
    const std::string turn =
        "Customer: my dashboard has been blank since the 3.2 upgrade and the export button "
        "returns a 502. Agent: thanks, could you share your browser version and a screenshot? ";
    std::string conv = "{\"channel\": \"chat\", \"tier\": \"gold\", \"conversation\": \"";
    const int turns = kind == "medium" ? 4 : 40;  // medium ~200 tokens, max > 512 (truncated)
    for (int i = 0; i < turns; ++i) conv += turn;
    return conv + "\"}";
}

// n varied questions cycling choice(4) / score(3) / noul, distinct wording.
std::vector<brolm::LayaQuestion> make_questions(int n) {
    std::vector<brolm::LayaQuestion> qs;
    for (int i = 0; i < n; ++i) {
        brolm::LayaQuestion q;
        q.id = "q" + std::to_string(i);
        switch (i % 3) {
            case 0:
                q.type = "choice";
                q.instructions = "Which team should handle this (routing rule " + std::to_string(i) + ")?";
                q.criteria_choice = {{"billing", "invoices, payments, refunds"},
                                     {"technical", "bugs, outages, errors"},
                                     {"sales", "pricing, new contracts"},
                                     {"other", "everything else"}};
                break;
            case 1:
                q.type = "score";
                q.instructions = "How urgent is this request (scale " + std::to_string(i) + ")?";
                q.criteria_score = {"not urgent", "soon", "critical deadline or blocking issue"};
                break;
            default:
                q.type = "noul";
                q.instructions = "Does the user mention policy item " + std::to_string(i) + " or threaten to leave?";
                break;
        }
        qs.push_back(std::move(q));
    }
    return qs;
}

// ─── Stats ─────────────────────────────────────────────────────────────────

double pct(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    if (v.empty()) return 0;
    const double idx = p / 100.0 * static_cast<double>(v.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(idx);
    const std::size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (idx - static_cast<double>(lo));
}

struct Cell {
    std::string key;
    std::string checkpoint;  // the variant it ran on
    int questions = 0;
    std::string state;
    int tokens = 0;
    int uploads = 0, downloads = 0;
    double p50 = 0, p95 = 0, p99 = 0, mean = 0, min = 0;
    double qps = 0, tok_s = 0;
    brolm::laya::LayaTimings stages;  // mean over profile iters
    // Open-loop cells only (replicas > 0).
    int replicas = 0, clients = 0;
    double offered_rps = 0, achieved_rps = 0, max = 0, missed_pct = 0;
    double batch_items = 0, batch_requests = 0, occupancy = 0;
    double fwd_ms = 0, busy = 0;  // mean device ms per forward; mean device busy fraction
    int budget = 0;
};

void accumulate(brolm::laya::LayaTimings& acc, const brolm::laya::LayaTimings& t, double w) {
    acc.tokenize_ms += t.tokenize_ms * w;
    acc.encoder_ms += t.encoder_ms * w;
    acc.head_ms += t.head_ms * w;
    acc.scorer_ms += t.scorer_ms * w;
    acc.act_ms += t.act_ms * w;
    acc.download_ms += t.download_ms * w;
    acc.calibrate_ms += t.calibrate_ms * w;
    acc.total_ms += t.total_ms * w;
    auto& e = acc.encoder;
    const auto& f = t.encoder;
    e.embed_ms += f.embed_ms * w;
    e.norm_ms += f.norm_ms * w;
    e.qkv_ms += f.qkv_ms * w;
    e.rope_ms += f.rope_ms * w;
    e.attn_full_ms += f.attn_full_ms * w;
    e.attn_local_ms += f.attn_local_ms * w;
    e.wo_ms += f.wo_ms * w;
    e.mlp_in_ms += f.mlp_in_ms * w;
    e.geglu_ms += f.geglu_ms * w;
    e.mlp_out_ms += f.mlp_out_ms * w;
    e.residual_ms += f.residual_ms * w;
}

Cell run_cell(brolm::LayaModel& model, int nq, const std::string& state_kind, const Args& a) {
    Cell c;
    c.questions = nq;
    c.state = state_kind;
    c.key = "q" + std::to_string(nq) + "_" + state_kind;
    const std::string state = make_state(state_kind);
    const auto qs = make_questions(nq);

    model.set_profiling(false);
    for (int i = 0; i < a.warmup; ++i) model.predict(state, qs);
    std::vector<double> lat;
    for (int i = 0; i < a.iters; ++i) {
        const auto t0 = Clock::now();
        const brolm::LayaResult r = model.predict(state, qs);
        lat.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        c.tokens = r.input_tokens;
    }
    c.uploads = model.last_timings().uploads;
    c.downloads = model.last_timings().downloads;
    c.p50 = pct(lat, 50);
    c.p95 = pct(lat, 95);
    c.p99 = pct(lat, 99);
    c.min = *std::min_element(lat.begin(), lat.end());
    c.mean = std::accumulate(lat.begin(), lat.end(), 0.0) / static_cast<double>(lat.size());
    c.qps = 1000.0 * nq / c.mean;
    c.tok_s = 1000.0 * c.tokens / c.mean;

    if (a.profile_iters > 0) {
        model.set_profiling(true);
        model.predict(state, qs);  // settle
        for (int i = 0; i < a.profile_iters; ++i) {
            model.predict(state, qs);
            accumulate(c.stages, model.last_timings(), 1.0 / a.profile_iters);
        }
        model.set_profiling(false);
    }
    return c;
}

// Packed multi-request cell: R concurrent requests, each its own state (the
// request index is written into it, so no two tokenize alike) with Q
// questions, served as one forward_items() call — what a request scheduler
// does with whatever arrived together. Latency is arrival to the last
// decision: every request's tokenize + one packed forward + calibration.
Cell run_packed_cell(brolm::LayaModel& model, int R, int nq, const std::string& state_kind, const Args& a) {
    Cell c;
    c.questions = R * nq;
    c.state = state_kind;
    c.key = "r" + std::to_string(R) + "x" + std::to_string(nq) + "_" + state_kind;
    std::vector<std::string> states;
    for (int r = 0; r < R; ++r) {
        std::string s = make_state(state_kind);
        s.insert(1, "\"request\": \"req-" + std::to_string(r * 7919) + "\", ");
        states.push_back(std::move(s));
    }
    const auto qs = make_questions(nq);

    auto serve = [&]() -> int {
        std::vector<std::vector<brolm::laya::SequenceResult>> seqs(static_cast<std::size_t>(R));
        std::vector<brolm::laya::LayaItem> items;
        int tokens = 0;
        for (int r = 0; r < R; ++r) {
            seqs[static_cast<std::size_t>(r)] = model.build_sequences(states[static_cast<std::size_t>(r)], qs);
            for (int i = 0; i < nq; ++i) {
                const auto& sq = seqs[static_cast<std::size_t>(r)][static_cast<std::size_t>(i)];
                items.push_back(brolm::laya::LayaItem::of(sq, qs[static_cast<std::size_t>(i)].qtype_index()));
                tokens += static_cast<int>(sq.input_ids.size());
            }
        }
        const auto outs = model.forward_items(items);
        std::vector<brolm::LayaAnswer> answers;
        answers.reserve(outs.size());
        for (std::size_t k = 0; k < outs.size(); ++k)
            answers.push_back(model.answer_from_logits(qs[k % static_cast<std::size_t>(nq)], outs[k]));
        return tokens;
    };

    model.set_profiling(false);
    for (int i = 0; i < a.warmup; ++i) serve();
    std::vector<double> lat;
    for (int i = 0; i < a.iters; ++i) {
        const auto t0 = Clock::now();
        c.tokens = serve();
        lat.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
    c.uploads = model.last_timings().uploads;
    c.downloads = model.last_timings().downloads;
    c.p50 = pct(lat, 50);
    c.p95 = pct(lat, 95);
    c.p99 = pct(lat, 99);
    c.min = *std::min_element(lat.begin(), lat.end());
    c.mean = std::accumulate(lat.begin(), lat.end(), 0.0) / static_cast<double>(lat.size());
    c.qps = 1000.0 * c.questions / c.mean;
    c.tok_s = 1000.0 * c.tokens / c.mean;
    return c;
}

// Open-loop cells: a Scheduler over `replicas` devices, `open_clients`
// Poisson clients each submitting a distinct `open_q`-question request. With
// --sweep, the offered rate climbs until p99 passes --sweep-p99 or the
// scheduler stops keeping up; otherwise the fixed --open-rates run.
Cell open_cell(const bench_laya::OpenLoopResult& r, int replicas, int nq, const std::string& state, int clients) {
    Cell c;
    c.replicas = replicas;
    c.clients = clients;
    c.questions = nq;
    c.state = state;
    char key[96];
    std::snprintf(key, sizeof key, "open_d%d_q%d_%s_%.0frps", replicas, nq, state.c_str(), r.offered_rps);
    c.key = key;
    c.tokens = r.tokens;
    c.p50 = r.p50;
    c.p95 = r.p95;
    c.p99 = r.p99;
    c.mean = r.mean;
    c.min = r.min;
    c.max = r.max;
    c.offered_rps = r.offered_rps;
    c.achieved_rps = r.achieved_rps;
    c.qps = r.achieved_rps * nq;
    c.tok_s = r.achieved_rps * r.tokens;
    c.missed_pct = r.deadline_missed_pct;
    c.batch_items = r.stats.mean_batch_items;
    c.batch_requests = r.stats.mean_batch_requests;
    c.occupancy = r.stats.mean_occupancy;
    c.budget = r.stats.token_budget;
    double busy = 0, frac = 0;
    uint64_t fwd = 0;
    for (const auto& d : r.stats.devices) {
        busy += d.busy_ms;
        frac += d.busy_fraction;
        fwd += d.forwards;
    }
    c.fwd_ms = fwd ? busy / static_cast<double>(fwd) : 0;
    c.busy = r.stats.devices.empty() ? 0 : frac / static_cast<double>(r.stats.devices.size());
    return c;
}

void run_open_cells(const Args& a, const std::string& model_dir, std::vector<Cell>& cells) {
    const std::vector<int> all = brolm::laya::Scheduler::all_devices();
    std::vector<int> counts;
    for (int n : a.open_replicas) {
        const int k = std::clamp(n, 1, static_cast<int>(all.size()));
        if (std::find(counts.begin(), counts.end(), k) == counts.end()) counts.push_back(k);
    }
    const auto qs = make_questions(a.open_q);
    for (int n : counts) {
        brolm::laya::SchedulerOptions so;
        so.devices.assign(all.begin(), all.begin() + n);
        const auto t0 = Clock::now();
        brolm::laya::Scheduler sched(model_dir, so);
        sched.wait_ready();
        std::fprintf(stderr, "  scheduler: %d replica(s) ready in %.0f ms, token budget %d\n", n,
                     std::chrono::duration<double, std::milli>(Clock::now() - t0).count(), sched.token_budget());
        for (const std::string& s : a.open_states) {
            const std::string base = make_state(s);
            auto state_of = [&base](int i) {
                std::string st = base;
                st.insert(1, "\"request\": \"req-" + std::to_string(i * 7919) + "\", ");
                return st;
            };
            bench_laya::OpenLoopConfig cfg;
            cfg.clients = a.open_clients;
            cfg.seconds = a.open_seconds;
            if (a.sweep) {
                const double start = 50.0 * n, step = 25.0 * n;
                const auto tried = bench_laya::sweep_open_loop(sched, state_of, qs, cfg, start, step, a.sweep_p99, 4000);
                const bench_laya::OpenLoopResult* best = nullptr;
                for (const auto& r : tried) {
                    cells.push_back(open_cell(r, n, a.open_q, s, cfg.clients));
                    std::fprintf(stderr, "  %s p99 %.1f ms, achieved %.0f rps\n", cells.back().key.c_str(), r.p99,
                                 r.achieved_rps);
                    if (r.p99 <= a.sweep_p99 && r.achieved_rps >= 0.95 * r.offered_rps) best = &r;
                }
                std::printf("sustained (p99 <= %.0f ms): %d replica(s), q%d %s: %s\n", a.sweep_p99, n, a.open_q,
                            s.c_str(),
                            best ? (std::to_string(static_cast<int>(best->offered_rps)) + " req/s").c_str()
                                 : "below the first step");
            } else {
                for (double rate : a.open_rates) {
                    // Rates are per replica, for the short state; the medium
                    // state (~2.2x the tokens) runs at half.
                    cfg.rate_rps = rate * n * (s == "short" ? 1.0 : 0.5);
                    const auto r = bench_laya::run_open_loop(sched, state_of, qs, cfg);
                    cells.push_back(open_cell(r, n, a.open_q, s, cfg.clients));
                    std::fprintf(stderr, "  %s done\n", cells.back().key.c_str());
                    if (a.tail) bench_laya::print_tail(r, cells.back().key.c_str());
                }
            }
        }
    }
}

// ─── Output ────────────────────────────────────────────────────────────────

void print_open(const std::vector<Cell>& cells) {
    bool any = false;
    for (const Cell& c : cells) any = any || c.replicas > 0;
    if (!any) return;
    std::printf("\nopen loop (Poisson arrivals, latency = submit -> result incl. tokenize, ms)\n");
    std::printf("%-44s %7s %7s %7s %7s %7s %7s %6s %6s %6s %6s %5s %5s %6s\n", "cell", "offered", "achvd", "p50",
                "p95", "p99", "max", "items", "reqs", "occ", "fwdms", "busy", "miss%", "budget");
    for (const Cell& c : cells) {
        if (c.replicas == 0) continue;
        std::printf("%-44s %7.0f %7.0f %7.2f %7.2f %7.2f %7.2f %6.1f %6.1f %6.2f %6.2f %5.2f %5.1f %6d\n",
                    c.key.c_str(), c.offered_rps, c.achieved_rps, c.p50, c.p95, c.p99, c.max, c.batch_items,
                    c.batch_requests, c.occupancy, c.fwd_ms, c.busy, c.missed_pct, c.budget);
    }
    std::printf("(items / reqs = mean per forward; occ = mean forward rows / token budget; fwdms = device ms per "
                "forward; busy = device busy fraction; miss = past the 30 ms default deadline)\n");
}

void print_latency(const std::vector<Cell>& cells) {
    std::printf("\n%-24s %6s %8s %8s %8s %8s %8s %9s %10s %6s\n", "cell", "tokens", "p50", "p95", "p99",
                "mean", "min", "q/s", "tok/s", "xfers");
    for (const Cell& c : cells) {
        if (c.replicas > 0) continue;  // print_open
        std::printf("%-24s %6d %8.2f %8.2f %8.2f %8.2f %8.2f %9.1f %10.0f %3d/%-3d\n", c.key.c_str(), c.tokens,
                    c.p50, c.p95, c.p99, c.mean, c.min, c.qps, c.tok_s, c.uploads, c.downloads);
    }
    std::printf("(latency in ms per predict() call; xfers = host->device uploads / device->host downloads)\n");
}

void print_stages(const std::vector<Cell>& cells) {
    std::printf("\nstage breakdown, ms per call (profiled: device sync at every stage boundary)\n");
    std::printf("%-24s %8s %8s %8s %8s %8s %8s %8s %8s\n", "cell", "tokenize", "encoder", "head", "scorer",
                "act", "download", "calib", "total");
    for (const Cell& c : cells) {
        const auto& s = c.stages;
        if (s.total_ms <= 0) continue;  // packed cells are not profiled
        std::printf("%-24s %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f\n", c.key.c_str(), s.tokenize_ms,
                    s.encoder_ms, s.head_ms, s.scorer_ms, s.act_ms, s.download_ms, s.calibrate_ms, s.total_ms);
    }
    std::printf("\nencoder split by op family, ms per call (profiled per op family)\n");
    std::printf("%-24s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s\n", "cell", "embed", "norm", "qkv",
                "rope", "attnG", "attnL", "Wo", "Wi", "geglu", "mlpWo", "resid");
    for (const Cell& c : cells) {
        const auto& e = c.stages.encoder;
        if (c.stages.total_ms <= 0) continue;
        std::printf("%-24s %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f\n", c.key.c_str(),
                    e.embed_ms, e.norm_ms, e.qkv_ms, e.rope_ms, e.attn_full_ms, e.attn_local_ms, e.wo_ms,
                    e.mlp_in_ms, e.geglu_ms, e.mlp_out_ms, e.residual_ms);
    }
}

std::string to_json(const std::vector<Cell>& cells, const Args& a, const std::string& gpu,
                    const std::vector<std::pair<std::string, double>>& loads) {
    std::ostringstream o;
    o.precision(6);
    o << "{\n  \"tool\": \"brolm_bench_laya\",\n  \"label\": \"" << a.label << "\",\n  \"device\": \"" << gpu
      << "\",\n  \"iters\": " << a.iters << ",\n  \"load_ms\": " << (loads.empty() ? 0.0 : loads.front().second)
      << ",\n  \"checkpoints\": {";
    for (std::size_t i = 0; i < loads.size(); ++i) {
        o << (i ? ", " : "") << "\"" << loads[i].first << "\": {\"load_ms\": " << loads[i].second << "}";
    }
    o << "},\n  \"cells\": {\n";
    for (std::size_t i = 0; i < cells.size(); ++i) {
        const Cell& c = cells[i];
        const auto& s = c.stages;
        const auto& e = s.encoder;
        o << "    \"" << c.key << "\": {\"checkpoint\": \"" << c.checkpoint << "\", \"questions\": " << c.questions
          << ", \"state\": \"" << c.state
          << "\", \"tokens\": " << c.tokens << ", \"p50\": " << c.p50 << ", \"p95\": " << c.p95
          << ", \"p99\": " << c.p99 << ", \"mean\": " << c.mean << ", \"min\": " << c.min
          << ", \"qps\": " << c.qps << ", \"tok_s\": " << c.tok_s << ", \"uploads\": " << c.uploads
          << ", \"downloads\": " << c.downloads;
        if (c.replicas > 0) {
            o << ",\n      \"open\": {\"replicas\": " << c.replicas << ", \"clients\": " << c.clients
              << ", \"offered_rps\": " << c.offered_rps << ", \"achieved_rps\": " << c.achieved_rps
              << ", \"max\": " << c.max << ", \"missed_pct\": " << c.missed_pct << ", \"batch_items\": "
              << c.batch_items << ", \"batch_requests\": " << c.batch_requests << ", \"occupancy\": " << c.occupancy
              << ", \"fwd_ms\": " << c.fwd_ms << ", \"busy\": " << c.busy << ", \"budget\": " << c.budget << "}}" << (i + 1 < cells.size() ? "," : "") << "\n";
            continue;
        }
        o << ",\n      \"stages\": {\"tokenize\": " << s.tokenize_ms
          << ", \"encoder\": " << s.encoder_ms << ", \"head\": " << s.head_ms << ", \"scorer\": " << s.scorer_ms
          << ", \"act\": " << s.act_ms << ", \"download\": " << s.download_ms << ", \"calibrate\": "
          << s.calibrate_ms << ", \"total\": " << s.total_ms << "},\n      \"encoder\": {\"embed\": " << e.embed_ms
          << ", \"norm\": " << e.norm_ms << ", \"qkv\": " << e.qkv_ms << ", \"rope\": " << e.rope_ms
          << ", \"attn_full\": " << e.attn_full_ms << ", \"attn_local\": " << e.attn_local_ms
          << ", \"wo\": " << e.wo_ms << ", \"mlp_in\": " << e.mlp_in_ms << ", \"geglu\": " << e.geglu_ms
          << ", \"mlp_out\": " << e.mlp_out_ms << ", \"residual\": " << e.residual_ms << "}}"
          << (i + 1 < cells.size() ? "," : "") << "\n";
    }
    o << "  }\n}\n";
    return o.str();
}

void print_compare(const std::vector<Cell>& cells, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "compare: cannot open %s\n", path.c_str());
        return;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    const j::Value base = j::parse(ss.str());
    const j::Value* bc = base.find("cells");
    std::printf("\nvs %s (%s): ratio = now / baseline, < 1 is faster\n", path.c_str(),
                base.get_string("label", "").c_str());
    std::printf("%-24s %9s %9s %7s %9s %9s %7s %9s %7s\n", "cell", "p50 base", "p50 now", "ratio", "p95 base",
                "p95 now", "ratio", "q/s now", "x");
    for (const Cell& c : cells) {
        const j::Value* b = bc ? bc->find(c.key) : nullptr;
        if (!b) {
            std::printf("%-24s (not in baseline)\n", c.key.c_str());
            continue;
        }
        const double b50 = b->get_float("p50", 0), b95 = b->get_float("p95", 0), bq = b->get_float("qps", 0);
        std::printf("%-24s %9.2f %9.2f %7.3f %9.2f %9.2f %7.3f %9.1f %7.2f\n", c.key.c_str(), b50, c.p50,
                    b50 > 0 ? c.p50 / b50 : 0, b95, c.p95, b95 > 0 ? c.p95 / b95 : 0, c.qps, bq > 0 ? c.qps / bq : 0);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const Args a = parse_args(argc, argv);
    // The checkpoints to run: model_dir itself, or with --checkpoints the
    // named members of the family under it (english = the root).
    std::vector<std::string> dirs;
    if (a.checkpoints.empty()) {
        dirs.push_back(a.model_dir);
    } else {
        for (const std::string& c : a.checkpoints) dirs.push_back(c == "english" ? a.model_dir : a.model_dir + "/" + c);
    }
    for (const std::string& d : dirs) {
        if (!fs::exists(d + "/model.safetensors")) {
            std::fprintf(stderr, "no Laya checkpoint at %s (pass model_dir or set LAYA_MODEL_DIR)\n", d.c_str());
            return 1;
        }
    }
    brotensor::init();
    const std::string gpu = brotensor::device_product_name(brotensor::default_device());
    std::printf("brolm_bench_laya  device=%s (%s)  iters=%d warmup=%d%s%s\n",
                brotensor::device_name(brotensor::default_device()), gpu.c_str(), a.iters, a.warmup,
                a.label.empty() ? "" : "  label=", a.label.c_str());

    std::vector<Cell> cells;
    std::vector<std::pair<std::string, double>> loads;
    for (const std::string& dir : dirs) {
        const std::size_t first = cells.size();
        std::string variant;
        {
            const auto t0 = Clock::now();
            brolm::LayaModel model;
            model.load_model(dir);
            const double load_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            const auto& cfg = model.config();
            const auto& enc = model.encoder().config();
            variant = cfg.variant;
            loads.emplace_back(variant, load_ms);
            std::printf("%s: %s, %d layers x %d, vocab %d (%s), max_len %d, head_max_len %d, load %.0f ms\n",
                        variant.c_str(), cfg.encoder.c_str(), enc.num_hidden_layers, enc.hidden_size,
                        enc.vocab_size, model.tokenizer().kind_name(), cfg.max_len, cfg.head_max_len, load_ms);

            for (const std::string& s : a.states) {
                for (int nq : a.questions) {
                    cells.push_back(run_cell(model, nq, s, a));
                    std::fprintf(stderr, "  %s %s done\n", variant.c_str(), cells.back().key.c_str());
                }
            }
            for (const std::string& s : a.states) {
                if (s == "max") continue;
                for (const std::string& p : a.packed) {
                    const std::size_t x = p.find('x');
                    if (x == std::string::npos) usage();
                    cells.push_back(
                        run_packed_cell(model, std::stoi(p.substr(0, x)), std::stoi(p.substr(x + 1)), s, a));
                    std::fprintf(stderr, "  %s %s done\n", variant.c_str(), cells.back().key.c_str());
                }
            }
        }  // free this replica before the scheduler loads its own
        if (!a.open_states.empty()) run_open_cells(a, dir, cells);
        // The English checkpoint keeps the historical cell names; the others
        // are prefixed with their variant ("multilingual/q5_short").
        for (std::size_t i = first; i < cells.size(); ++i) {
            cells[i].checkpoint = variant;
            if (variant != "english") cells[i].key = variant + "/" + cells[i].key;
        }
    }
    print_latency(cells);
    if (a.profile_iters > 0) print_stages(cells);
    print_open(cells);
    if (!a.compare.empty()) print_compare(cells, a.compare);
    if (!a.json_out.empty()) {
        std::ofstream(a.json_out, std::ios::binary) << to_json(cells, a, gpu, loads);
        std::printf("\nwrote %s\n", a.json_out.c_str());
    }
    return 0;
}
