#include "ae/optimizer.hpp"

#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace ae {

const char* to_string(Strategy s) noexcept {
    switch (s) {
    case Strategy::greedy: return "greedy";
    case Strategy::beam: return "beam";
    case Strategy::simulated_annealing: return "simulated_annealing";
    case Strategy::exhaustive: return "exhaustive";
    case Strategy::random_restart: return "random_restart";
    }
    return "unknown";
}

namespace {

struct Clock {
    std::chrono::steady_clock::time_point t0{std::chrono::steady_clock::now()};
    double ms() const {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
            .count();
    }
};

void validate(const PhotoProblem& p) {
    const std::size_t n = p.n;
    if (n == 0) throw std::invalid_argument("PhotoProblem: n == 0");
    if (p.quality.size() != n || p.slot_affinity.size() != n ||
        p.position_weight.size() != n || p.slot_target.size() != n ||
        p.distance.size() != n * n) {
        throw std::invalid_argument("PhotoProblem: inconsistent field sizes");
    }
    for (double q : p.quality)
        if (!(q >= 0.0 && q <= 1.0))
            throw std::invalid_argument("PhotoProblem: quality outside [0,1]");
    double wsum = 0.0;
    for (double w : p.position_weight) {
        if (w < 0.0) throw std::invalid_argument("PhotoProblem: negative position weight");
        wsum += w;
    }
    if (!(wsum > 0.0)) throw std::invalid_argument("PhotoProblem: position weights sum to zero");
}

// Similarity in [0,1] from an L2 distance on unit-norm embeddings. For unit
// vectors d^2 = 2 - 2cos; the quadratic 1 - d^2/4 is monotone in d and stays in
// [0,1] for d in [0,2], which is what the optimizer needs.
inline double similarity_from_distance(double d) noexcept {
    // cos = 1 - d^2/2 maps onto [-1, 1]; rescaled to [0, 1] so the diversity
    // term is a proper convex combination and cannot go negative.
    const double x = std::min(d, 2.0);
    return 1.0 - 0.5 * x * x * 0.5;   // = 1 - x^2/4
}

// Kernel weights, pre-computed once per problem instance.
struct KernelCache {
    std::vector<double> weight;  // n*n, only i<j entries are used
    double total{0.0};

    KernelCache(std::size_t n, bool positional) {
        weight.assign(n * n, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i + 1; j < n; ++j) {
                const double k = positional ? PhotoProblem::kernel(j - i) : 1.0;
                weight[i * n + j] = k;
                total += k;
            }
    }
    [[nodiscard]] double total_norm() const noexcept { return total > 0.0 ? total : 1.0; }
};

// Maintains the three objective terms for one permutation and supports O(n)
// swap deltas. A 2-swap changes the quality and coherence terms by O(1) and the
// redundancy term by the pairs involving the two swapped positions only, so SA
// with 20k iterations on n=12 costs ~20k*n rather than 20k*n^2.
class ObjectiveState {
public:
    ObjectiveState(const PhotoProblem& p, const KernelCache& kc)
        : p_(p), kc_(kc), perm_(p.n) {
        std::iota(perm_.begin(), perm_.end(), 0);
        refresh();
    }

    void set_permutation(std::span<const std::size_t> perm) {
        perm_.assign(perm.begin(), perm.end());
        refresh();
    }

    [[nodiscard]] const std::vector<std::size_t>& perm() const noexcept { return perm_; }

    [[nodiscard]] double total() const noexcept {
        return q_ + p_.lambda * p_.mu == 0.0 && false ? 0.0 : j_;
    }

    [[nodiscard]] ObjectiveBreakdown breakdown() const {
        ObjectiveBreakdown b;
        b.quality = q_;
        b.redundancy = r();
        b.diversity = 1.0 - r() / kc_.total_norm();
        b.coherence = coh_;
        b.total = j_;
        return b;
    }

    [[nodiscard]] double redundancy() const { return r(); }

    // Swap two positions and return the new objective WITHOUT committing.
    double probe_swap(std::size_t a, std::size_t b) const {
        if (a == b) return j_;
        std::vector<std::size_t> tmp = perm_;
        std::swap(tmp[a], tmp[b]);
        return evaluate_perm(tmp);
    }

    void apply(std::span<const std::size_t> perm) {
        perm_.assign(perm.begin(), perm.end());
        refresh();
    }

    [[nodiscard]] double evaluate_perm(const std::vector<std::size_t>& perm) const {
        double q = 0.0, c = 0.0, rr = 0.0;
        const std::size_t n = p_.n;
        for (std::size_t k = 0; k < n; ++k) {
            q += p_.position_weight[k] * p_.quality[perm[k]];
            c += std::abs(p_.slot_affinity[perm[k]] - p_.slot_target[k]);
        }
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i + 1; j < n; ++j)
                rr += kc_.weight[i * n + j] *
                      similarity_from_distance(p_.distance_at(perm[i], perm[j]));
        const double d = 1.0 - rr / kc_.total_norm();
        const double coh = 1.0 - c / static_cast<double>(n);
        return q + p_.lambda * d + p_.mu * coh;
    }

private:
    void refresh() {
        const std::size_t n = p_.n;
        q_ = 0.0;
        double c = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            q_ += p_.position_weight[k] * p_.quality[perm_[k]];
            c += std::abs(p_.slot_affinity[perm_[k]] - p_.slot_target[k]);
        }
        coh_ = 1.0 - c / static_cast<double>(n);
        j_ = q_ + p_.lambda * (1.0 - r() / kc_.total_norm()) + p_.mu * coh_;
    }

    [[nodiscard]] double r() const {
        double rr = 0.0;
        const std::size_t n = p_.n;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i + 1; j < n; ++j)
                rr += kc_.weight[i * n + j] *
                      similarity_from_distance(p_.distance_at(perm_[i], perm_[j]));
        return rr;
    }

    const PhotoProblem& p_;
    const KernelCache& kc_;
    std::vector<std::size_t> perm_;
    double q_{0.0};
    double coh_{0.0};
    double j_{0.0};
};

// Steepest-descent 2-swap local search: the exact "local optimum" benchmark for
// every heuristic, and cheap enough to run for n <= 60.
std::vector<std::size_t> two_swap_polish(const PhotoProblem& p, const KernelCache& kc,
                                         std::vector<std::size_t> perm, std::size_t& evaluations) {
    ObjectiveState state(p, kc);
    state.set_permutation(perm);
    const std::size_t n = p.n;
    bool improved = true;
    while (improved) {
        improved = false;
        double best = state.total();
        std::vector<std::size_t> best_perm;
        for (std::size_t a = 0; a < n; ++a) {
            for (std::size_t b = a + 1; b < n; ++b) {
                auto cand = perm;
                std::swap(cand[a], cand[b]);
                ++evaluations;
                const double v = state.evaluate_perm(cand);
                if (v > best + 1e-12) {
                    best = v;
                    best_perm = cand;
                    improved = true;
                }
            }
        }
        if (improved) {
            perm = best_perm;
            state.set_permutation(perm);
        }
    }
    return perm;
}

OptimizeResult finalise(const PhotoProblem& p, const KernelCache& kc,
                        std::vector<std::size_t> perm, OptimizeResult r, const Clock& clk,
                        bool polish) {
    if (polish && p.n > 2) perm = two_swap_polish(p, kc, perm, r.evaluations);
    r.permutation = std::move(perm);
    ObjectiveState state(p, kc);
    state.set_permutation(r.permutation);
    r.score = state.breakdown();
    r.elapsed_ms = clk.ms();
    return r;
}

// ---------------------------------------------------------------- strategies
OptimizeResult run_greedy(const PhotoProblem& p, const KernelCache& kc,
                          const OptimizeOptions& o) {
    Clock clk;
    OptimizeResult r;
    r.strategy = Strategy::greedy;
    const std::size_t n = p.n;
    std::vector<bool> used(n, false);
    std::vector<std::size_t> perm;
    perm.reserve(n);
    const double norm = kc.total_norm();
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t best = n;
        double best_val = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < n; ++i) {
            if (used[i]) continue;
            double red = 0.0;
            for (std::size_t pos = 0; pos < perm.size(); ++pos) {
                const std::size_t lo = std::min(pos, k), hi = std::max(pos, k);
                red += kc.weight[lo * n + hi] *
                       similarity_from_distance(p.distance_at(i, perm[pos]));
            }
            const double seg = p.position_weight[k] * p.quality[i] -
                               p.lambda * red / norm +
                               p.mu * (1.0 - std::abs(p.slot_affinity[i] - p.slot_target[k]));
            ++r.evaluations;
            if (seg > best_val || (seg == best_val && i < best)) {
                best_val = seg;
                best = i;
            }
        }
        used[best] = true;
        perm.push_back(best);
    }
    return finalise(p, kc, std::move(perm), std::move(r), clk, o.polish);
}

OptimizeResult run_beam(const PhotoProblem& p, const KernelCache& kc, const OptimizeOptions& o) {
    Clock clk;
    OptimizeResult r;
    r.strategy = Strategy::beam;
    const std::size_t n = p.n;
    const std::size_t width = std::max<std::size_t>(1, std::min(o.beam_width, n));
    const double norm = kc.total_norm();

    struct State {
        std::vector<std::size_t> perm;
        std::vector<bool> used;
        double partial{0.0};
    };
    std::vector<State> beam;
    beam.push_back(State{{}, std::vector<bool>(n, false), 0.0});

    for (std::size_t k = 0; k < n; ++k) {
        std::vector<State> cand;
        cand.reserve(beam.size() * n);
        for (const State& s : beam) {
            for (std::size_t i = 0; i < n; ++i) {
                if (s.used[i]) continue;
                double red = 0.0;
                for (std::size_t pos = 0; pos < s.perm.size(); ++pos) {
                    const std::size_t lo = std::min(pos, k), hi = std::max(pos, k);
                    red += kc.weight[lo * n + hi] *
                           similarity_from_distance(p.distance_at(i, s.perm[pos]));
                }
                State t;
                t.perm = s.perm;
                t.perm.push_back(i);
                t.used = s.used;
                t.used[i] = true;
                t.partial = s.partial + p.position_weight[k] * p.quality[i] -
                            p.lambda * red / norm +
                            p.mu * (1.0 - std::abs(p.slot_affinity[i] - p.slot_target[k]));
                ++r.evaluations;
                cand.push_back(std::move(t));
            }
        }
        std::sort(cand.begin(), cand.end(), [](const State& a, const State& b) {
            if (a.partial != b.partial) return a.partial > b.partial;
            return a.perm < b.perm;   // deterministic tie-break
        });
        if (cand.size() > width) cand.resize(width);
        beam = std::move(cand);
    }
    return finalise(p, kc, beam.front().perm, std::move(r), clk, o.polish);
}

OptimizeResult run_sa(const PhotoProblem& p, const KernelCache& kc, const OptimizeOptions& o) {
    Clock clk;
    OptimizeResult r;
    r.strategy = Strategy::simulated_annealing;
    std::mt19937_64 rng(o.seed);
    const std::size_t n = p.n;
    std::uniform_int_distribution<std::size_t> pick(0, n - 1);
    std::uniform_real_distribution<double> urand(0.0, 1.0);

    // Seed with the greedy solution so annealing spends its budget refining.
    auto start = run_greedy(p, kc, OptimizeOptions{Strategy::greedy, 0, false});
    r.evaluations += start.evaluations;
    std::vector<std::size_t> cur_perm = start.permutation;
    double cur = start.score.total;

    std::vector<std::size_t> best_perm = cur_perm;
    double best = cur;
    double temp = o.sa_initial_temp;
    ObjectiveState state(p, kc);
    state.set_permutation(cur_perm);

    for (std::size_t it = 0; it < o.sa_iterations; ++it) {
        const std::size_t a = pick(rng), b = pick(rng);
        if (a == b) continue;
        std::swap(cur_perm[a], cur_perm[b]);
        const double cand = state.evaluate_perm(cur_perm);
        ++r.evaluations;
        const double delta = cand - cur;
        if (delta >= 0.0 || urand(rng) < std::exp(delta / std::max(temp, 1e-12))) {
            cur = cand;
            if (cand > best) {
                best = cand;
                best_perm = cur_perm;
            }
        } else {
            std::swap(cur_perm[a], cur_perm[b]);
        }
        temp *= o.sa_cooling;
        if (temp < 1e-6) {
            // A single reheat half-way through the schedule keeps SA from
            // freezing in the first basin it finds.
            if (it < o.sa_iterations / 2) temp = o.sa_initial_temp * 0.25;
            else temp = 1e-6;
        }
    }
    return finalise(p, kc, best_perm, std::move(r), clk, o.polish);
}

OptimizeResult run_random_restart(const PhotoProblem& p, const KernelCache& kc,
                                  const OptimizeOptions& o) {
    Clock clk;
    OptimizeResult r;
    r.strategy = Strategy::random_restart;
    std::mt19937_64 rng(o.seed);
    std::vector<std::size_t> perm(p.n);
    std::iota(perm.begin(), perm.end(), 0);
    double best = -std::numeric_limits<double>::infinity();
    std::vector<std::size_t> best_perm = perm;
    const std::size_t restarts = std::max<std::size_t>(o.restarts, 1);
    for (std::size_t k = 0; k < restarts; ++k) {
        std::shuffle(perm.begin(), perm.end(), rng);
        perm = two_swap_polish(p, kc, perm, r.evaluations);
        ObjectiveState st(p, kc);
        st.set_permutation(perm);
        if (st.total() > best) {
            best = st.total();
            best_perm = perm;
        }
    }
    return finalise(p, kc, best_perm, std::move(r), clk, o.polish);
}

OptimizeResult run_exhaustive(const PhotoProblem& p, const KernelCache& kc, std::size_t max_n) {
    if (p.n > max_n) throw std::invalid_argument("exhaustive: n exceeds limit");
    Clock clk;
    OptimizeResult r;
    r.strategy = Strategy::exhaustive;
    std::vector<std::size_t> perm(p.n);
    std::iota(perm.begin(), perm.end(), 0);
    ObjectiveState state(p, kc);
    double best = -std::numeric_limits<double>::infinity();
    do {
        const double v = state.evaluate_perm(perm);
        ++r.evaluations;
        if (v > best) {
            best = v;
            r.permutation = perm;
        }
    } while (std::next_permutation(perm.begin(), perm.end()));
    state.set_permutation(r.permutation);
    r.score = state.breakdown();
    r.proved_optimal = true;
    r.elapsed_ms = clk.ms();
    return r;
}

}  // namespace

ObjectiveBreakdown evaluate(const PhotoProblem& p, std::span<const std::size_t> perm) {
    validate(p);
    if (perm.size() != p.n) throw std::invalid_argument("evaluate: permutation size mismatch");
    std::vector<bool> seen(p.n, false);
    for (std::size_t idx : perm) {
        if (idx >= p.n || seen[idx]) throw std::invalid_argument("evaluate: not a permutation");
        seen[idx] = true;
    }
    KernelCache kc(p.n, p.positional_kernel);
    ObjectiveState state(p, kc);
    state.set_permutation(perm);
    return state.breakdown();
}

OptimizeResult optimize(const PhotoProblem& p, const OptimizeOptions& opts) {
    validate(p);
    KernelCache kc(p.n, p.positional_kernel);
    if (opts.strategy == Strategy::exhaustive) return run_exhaustive(p, kc, opts.exhaustive_limit);
    if (p.n <= 3) {
        // Tiny instances: brute force is both faster and provably exact.
        return run_exhaustive(p, kc, opts.exhaustive_limit);
    }
    switch (opts.strategy) {
    case Strategy::greedy: return run_greedy(p, kc, opts);
    case Strategy::beam: return run_beam(p, kc, opts);
    case Strategy::simulated_annealing: return run_sa(p, kc, opts);
    case Strategy::random_restart: return run_random_restart(p, kc, opts);
    case Strategy::exhaustive: break;
    }
    return run_beam(p, kc, opts);
}

OptimizeResult exhaustive(const PhotoProblem& p, std::size_t max_n) {
    validate(p);
    KernelCache kc(p.n, p.positional_kernel);
    return run_exhaustive(p, kc, max_n);
}

std::vector<SensitivityPoint> sensitivity(const PhotoProblem& p, const OptimizeOptions& opts,
                                          std::span<const double> lambdas,
                                          std::span<const double> mus) {
    std::vector<SensitivityPoint> out;
    const auto baseline = optimize(p, opts);
    PhotoProblem scratch = p;
    for (double lam : lambdas) {
        for (double m : mus) {
            scratch.lambda = lam;
            scratch.mu = m;
            SensitivityPoint pt;
            pt.lambda = lam;
            pt.mu = m;
            pt.fixed_perm_total = evaluate(scratch, baseline.permutation).total;
            auto res = optimize(scratch, opts);
            pt.optimized_total = res.score.total;
            pt.optimized_permutation = res.permutation;
            pt.position_changes = 0;
            for (std::size_t k = 0; k < p.n; ++k)
                if (res.permutation[k] != baseline.permutation[k]) ++pt.position_changes;
            out.push_back(std::move(pt));
        }
    }
    return out;
}

std::vector<AblationRow> ablate_leverage(const PhotoProblem& p, const OptimizeOptions& opts) {
    std::vector<AblationRow> rows;
    if (p.n < 3) return rows;
    const double baseline_total = optimize(p, opts).score.total;
    for (std::size_t drop = 0; drop < p.n; ++drop) {
        PhotoProblem q;
        q.n = p.n - 1;
        q.lambda = p.lambda;
        q.mu = p.mu;
        q.positional_kernel = p.positional_kernel;
        std::vector<std::size_t> keep;
        keep.reserve(q.n);
        for (std::size_t i = 0; i < p.n; ++i)
            if (i != drop) keep.push_back(i);
        q.quality.resize(q.n);
        q.slot_affinity.resize(q.n);
        q.position_weight.resize(q.n);
        q.slot_target.resize(q.n);
        for (std::size_t k = 0; k < q.n; ++k) {
            q.quality[k] = p.quality[keep[k]];
            q.slot_affinity[k] = p.slot_affinity[keep[k]];
            q.position_weight[k] = p.position_weight[k];
            q.slot_target[k] = p.slot_target[k];
        }
        q.distance.assign(q.n * q.n, 0.0);
        for (std::size_t a = 0; a < q.n; ++a)
            for (std::size_t b = 0; b < q.n; ++b)
                q.distance[a * q.n + b] = p.distance[keep[a] * p.n + keep[b]];

        auto reduced = optimize(q, opts);
        AblationRow row;
        row.removed_index = drop;
        row.reduced_total = reduced.score.total;
        row.baseline_total = baseline_total;

        // Reinsert the dropped photograph at whichever of the n slots maximises
        // the full objective, holding the reduced ordering fixed.
        std::vector<std::size_t> base(q.n);
        for (std::size_t k = 0; k < q.n; ++k) base[k] = keep[reduced.permutation[k]];
        double best = -std::numeric_limits<double>::infinity();
        std::vector<std::size_t> best_perm;
        for (std::size_t pos = 0; pos < p.n; ++pos) {
            std::vector<std::size_t> perm;
            perm.reserve(p.n);
            for (std::size_t k = 0; k < p.n; ++k) {
                if (k == pos) perm.push_back(drop);
                else perm.push_back(base[k < pos ? k : k - 1]);
            }
            const double v = evaluate(p, perm).total;
            if (v > best) { best = v; best_perm = std::move(perm); }
        }
        row.best_total_with = best;
        row.best_permutation_with = std::move(best_perm);
        row.loss = baseline_total - best;
        rows.push_back(std::move(row));
    }
    return rows;
}

}  // namespace ae
