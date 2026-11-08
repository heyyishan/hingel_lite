// capacity.hpp — conservative experiment planning.
//
// Observed per-variant "capacity" (impressions, likes received, ...) supplies an
// upper bound on how many observations an experiment can accumulate over its
// planned duration. The planner answers: given the current posterior, the
// observed daily rate and the remaining days, is the planned sample size able
// to separate the variants at all? This is what keeps the UI honest instead of
// showing a "significant" result derived from four impressions.
#pragma once

#include "ae/compat.hpp"
#include "ae/stats.hpp"

namespace ae {

struct CapacityPlan {
    double observed_rate_per_day{0.0};   // from the user's own observation log
    std::uint64_t planned_days{0};
    std::uint64_t current_n{0};
    std::uint64_t projected_n{0};
    std::uint64_t observations_needed{0};  // crude power calculation, see note
    bool achievable{false};
    std::string note;
};

// Sample-size heuristic for a two-proportion comparison. It is the standard
// normal-approximation formula with a continuity-free simplification:
//   n_per_arm = (z_{1-a/2} + z_{power})^2 * (p1(1-p1) + p2(1-p2)) / (p1-p2)^2
// It assumes independent Bernoulli observations and a FIXED horizon; the
// sequential monitoring elsewhere in the product does not inherit this
// guarantee, and the UI labels the number as "planning estimate".
[[nodiscard]] std::uint64_t required_sample_size(double p1, double p2, double alpha = 0.05,
                                                double power = 0.8);

[[nodiscard]] CapacityPlan plan_capacity(double observed_rate_per_day, std::uint64_t planned_days,
                                         std::uint64_t current_observations,
                                         std::uint64_t observations_needed);

// z-quantile of the standard normal (Acklam's rational approximation, |err|<1e-9)
[[nodiscard]] double normal_quantile(double p);

}  // namespace ae
