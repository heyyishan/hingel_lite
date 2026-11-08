// stats.hpp — Beta-Binomial experimentation primitives.
//
// All functions are exact up to floating-point rounding and are cross-checked in
// tests against SciPy (tests/test_stats.cpp vs python/tests/test_beta_scipy.py).
#pragma once

#include "ae/compat.hpp"

#include <optional>
#include <vector>

namespace ae {

struct BetaParams {
    double alpha{1.0};
    double beta{1.0};
    [[nodiscard]] double mean() const { return alpha / (alpha + beta); }
    [[nodiscard]] double variance() const {
        const double s = alpha + beta;
        return (alpha * beta) / (s * s * (s + 1.0));
    }
    [[nodiscard]] double mode() const {
        if (alpha <= 1.0 && beta <= 1.0) return 0.5;  // no interior mode
        return (alpha - 1.0) / (alpha + beta - 2.0);
    }
};

// Posterior of the conversion rate after (successes, failures).
[[nodiscard]] BetaParams beta_posterior(double prior_alpha, double prior_beta,
                                        std::uint64_t successes, std::uint64_t failures);

[[nodiscard]] double beta_pdf(double x, const BetaParams& p);
[[nodiscard]] double beta_cdf(double x, const BetaParams& p);      // regularised incomplete
[[nodiscard]] double beta_quantile(double q, const BetaParams& p); // bisection, 1e-12

// Equal-tailed credible interval at level `level` (e.g. 0.95).
struct Interval {
    double lo{0.0};
    double hi{0.0};
    double level{0.0};
    double width() const { return hi - lo; }
};
[[nodiscard]] Interval credible_interval(const BetaParams& p, double level = 0.95);

// P(theta_A > theta_B) using numerical integration on a fixed 1e-4 grid.
// Exact closed forms exist only for integer parameter pairs, so we integrate;
// the grid is adaptive only in the sense of being refined by Richardson
// extrapolation of two halves, which keeps error < ~1e-9 in tests.
[[nodiscard]] double prob_superiority(const BetaParams& a, const BetaParams& b);

// Expected loss of choosing A: E[max(theta_B - theta_A, 0)]. Used by the
// stopping rule (stop when expected loss < threshold).
[[nodiscard]] double expected_loss(const BetaParams& a, const BetaParams& b);

// Thompson sampling: draw one sample per variant from its posterior and return
// the arg-max. Deterministic given the seed; draws use a splitmix64-seeded
// mt19937_64 vector of uniforms consumed in order.
[[nodiscard]] std::size_t thompson_select(const std::vector<BetaParams>& variants,
                                          std::uint64_t seed);

// Full Thompson allocation for `draws` rounds; returns counts per variant.
[[nodiscard]] std::vector<std::uint64_t> thompson_allocation(
    const std::vector<BetaParams>& variants, std::uint64_t draws, std::uint64_t seed);

struct SequentialTestResult {
    double prob_superiority{0.0};
    double expected_loss_choose_a{0.0};
    double expected_loss_choose_b{0.0};
    bool stop{false};
    std::string reason;
};
// Decision rule: stop when the expected loss of the current leader is below the
// user's tolerance AND the minimum observation count is met. This is a
// loss-based sequential rule (as used in Bayesian A/B testing); it is NOT a
// frequentist guarantee of any fixed alpha level and the UI says so.
[[nodiscard]] SequentialTestResult sequential_decision(const BetaParams& a, const BetaParams& b,
                                                       std::uint64_t n_a, std::uint64_t n_b,
                                                       std::uint64_t min_observations,
                                                       double loss_tolerance = 0.005);

// Bonferroni-style multiple comparison adjustment for k simultaneous variants.
[[nodiscard]] double bonferroni_level(double family_level, std::size_t k);
// Šidák adjustment: 1 - (1 - family)^(1/k)
[[nodiscard]] double sidak_level(double family_level, std::size_t k);

// Hierarchical Beta-Binomial (random-effects) via empirical-Bayes moment
// matching. Returns the per-variant posterior shape parameters shrunk toward a
// common population distribution. Independent (non-hierarchical) models are
// available by passing shrink = 0.
struct HierarchicalFit {
    double population_alpha{0.0};   // global Beta shape parameters
    double population_beta{0.0};
    double overdispersion{0.0};     // estimated between-variant variance
    std::vector<BetaParams> posteriors;
};
[[nodiscard]] HierarchicalFit fit_hierarchical(const std::vector<std::uint64_t>& successes,
                                               const std::vector<std::uint64_t>& failures,
                                               double prior_alpha = 1.0,
                                               double prior_beta = 1.0,
                                               double shrink = 0.5);

// Conversion rate with an explicit observed denominator guard: returns
// std::nullopt when the denominator is zero (missing observation != failure).
[[nodiscard]] std::optional<double> conversion_rate(std::uint64_t successes,
                                                    std::uint64_t observed);

}  // namespace ae
