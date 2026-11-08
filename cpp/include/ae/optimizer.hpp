// optimizer.hpp — photograph ordering optimization over the symmetric group.
//
// Objective
// ---------
//   J(pi) = sum_{k=1..n} w_k * q_{pi(k)}  +  lambda * D(pi)  +  mu * C(pi)
//
//   q_i        estimated quality of photograph i in [0,1]
//   w_k        position weight, non-increasing, sum(w) = 1
//   D(pi)      normalised visual diversity in [0,1] (order-SENSITIVE):
//
//                R(pi) = sum_{i<j} kappa(j-i) * s_{pi(i),pi(j)}      redundancy
//                D(pi) = 1 - R(pi) / sum_{i<j} kappa(j-i)            diversity
//
//              with kappa(Delta) = 1/Delta (adjacent positions weigh most) and
//              s in [0,1] the pairwise visual similarity. See the note below on
//              why the literal spec formula needs this positional kernel.
//   C(pi)      1 - sum_{k} |s_{pi(k)} - t_k| / n where s_i is a narrative "slot
//              affinity" in [0,1] and t_k the ideal affinity for position k.
//              Rewards placing storyteller shots (full-body, activity) later.
//
// IMPORTANT DEVIATION, DELIBERATE
// ------------------------------
// The specification defines D(pi) = sum_{i<j} a_ij d(z_pi(i), z_pi(j)). If a_ij
// does not depend on position, that sum ranges over *the same set of pairs* for
// every permutation, so D is permutation-invariant and the term cannot change
// which ordering is optimal — it merely shifts J by a constant. We therefore
// weight each pair by kappa(j-i) = 1/(j-i), which makes "two similar photographs
// in adjacent slots" the worst case. This is the only assumption added, it is
// reported in the sensitivity UI, and the invariant case is covered by a test
// (Optimizer.DiversityTermIsPermutationInvariantWithoutKernel) so the behaviour
// is explicit rather than accidental.
//
// Special cases exercised by the benchmarks:
//   lambda = mu = 0        -> assignment problem, solved exactly by sorting q
//   lambda > 0, mu = 0     -> QAP-like; NP-hard in general
//   n <= exhaustive_limit  -> exhaustive enumeration is tractable
#pragma once

#include "ae/compat.hpp"
#include "ae/similarity.hpp"

#include <functional>

namespace ae {

struct PhotoProblem {
    std::size_t n{0};                 // number of photographs
    std::vector<double> quality;      // n values in [0,1]
    std::vector<double> slot_affinity;// n values in [0,1]
    std::vector<double> position_weight;  // n weights, non-increasing
    std::vector<double> slot_target;      // n ideal affinities for each position
    std::vector<double> distance;     // n*n row-major pairwise L2 distances
    double lambda{0.35};
    double mu{0.20};
    bool positional_kernel{true};     // false reproduces the invariant spec form

    // kappa(j-i) = 1/(j-i); exposed so the Python layer documents the exact
    // weight it is optimising against.
    [[nodiscard]] static double kernel(std::size_t delta) noexcept {
        return 1.0 / static_cast<double>(delta);
    }

    [[nodiscard]] double distance_at(std::size_t i, std::size_t j) const {
        return distance[i * n + j];
    }
};

struct ObjectiveBreakdown {
    double quality{0};    // sum w_k q_{pi(k)}
    double redundancy{0}; // R(pi), the kernel-weighted similarity sum
    double diversity{0};  // D(pi) = 1 - R(pi)/sum(kappa), in [0,1], higher is better
    double coherence{0};  // C(pi) in [0,1]
    double total{0};
};

[[nodiscard]] ObjectiveBreakdown evaluate(const PhotoProblem& p, std::span<const std::size_t> perm);

enum class Strategy : std::uint8_t {
    greedy = 0,
    beam = 1,
    simulated_annealing = 2,
    exhaustive = 3,
    random_restart = 4,
};

struct OptimizeOptions {
    Strategy strategy{Strategy::beam};
    std::size_t beam_width{8};
    bool polish{true};                  // steepest-descent 2-swap polish at the end
    std::size_t exhaustive_limit{9};    // n! evaluations must stay tractable
    double sa_initial_temp{1.0};
    double sa_cooling{0.995};
    std::size_t sa_iterations{20000};
    std::size_t sa_reheats{2};
    std::size_t restarts{16};
    std::uint64_t seed{42};
};

struct OptimizeResult {
    std::vector<std::size_t> permutation;  // pi[position] = photograph index
    ObjectiveBreakdown score;
    std::size_t evaluations{0};
    double elapsed_ms{0.0};
    Strategy strategy{Strategy::greedy};
    bool proved_optimal{false};            // true only for completed exhaustive search
};

[[nodiscard]] const char* to_string(Strategy s) noexcept;

// Stateless, deterministic given (problem, options.seed).
[[nodiscard]] OptimizeResult optimize(const PhotoProblem& p, const OptimizeOptions& opts);

// Exhaustive reference used by the test-suite to prove the heuristics agree on
// small instances. Returns the true optimum; throws when n! is too large.
[[nodiscard]] OptimizeResult exhaustive(const PhotoProblem& p, std::size_t max_n = 9);

// Sensitivity: recompute J for a grid of (lambda, mu) holding the ordering fixed
// and, separately, re-optimise at each grid point. Used by the frontend's
// sensitivity panel so the numbers shown are computed, not decorative.
struct SensitivityPoint {
    double lambda;
    double mu;
    double fixed_perm_total;    // J of the baseline permutation at this (lambda, mu)
    double optimized_total;     // J of the best permutation found at this (lambda, mu)
    std::vector<std::size_t> optimized_permutation;
    std::size_t position_changes;  // Kendall-style count of images that moved
};
[[nodiscard]] std::vector<SensitivityPoint> sensitivity(const PhotoProblem& p,
                                                        const OptimizeOptions& opts,
                                                        std::span<const double> lambdas,
                                                        std::span<const double> mus);

// Leave-one-image-out ablation.
//
// For each image i we (a) re-optimise the ordering of the remaining n-1 photos,
// (b) put i back into whichever single slot maximises J, and (c) report the loss
// relative to the full-problem baseline optimum. All three full-scale numbers
// live on the same objective scale; `reduced_total` is reported separately and
// is only meaningful on the (n-1)-photo problem, which is why the header says so.
struct AblationRow {
    std::size_t removed_index{0};
    double reduced_total{0};    // optimum of the (n-1)-photo problem (different scale)
    double baseline_total{0};   // full-problem optimum J(P)
    double best_total_with{0};  // J(P) after remove + best single-slot reinsertion
    double loss{0};             // baseline_total - best_total_with (>= ~0)
    std::vector<std::size_t> best_permutation_with;
};
[[nodiscard]] std::vector<AblationRow> ablate_leverage(const PhotoProblem& p,
                                                       const OptimizeOptions& opts);

}  // namespace ae
