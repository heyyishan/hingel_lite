#include "ae/capacity.hpp"

#include <algorithm>
#include <cmath>

namespace ae {

double normal_quantile(double p) {
    if (!(p > 0.0 && p < 1.0)) throw std::invalid_argument("normal_quantile: p outside (0,1)");
    // Acklam (2003) rational approximation, max relative error ~1.15e-9.
    static const double a[6] = {-3.969683028665376e+01,  2.209460984245205e+02,
                                -2.759285104469687e+02,  1.383577518672690e+02,
                                -3.066479806614716e+01,  2.506628277459239e+00};
    static const double b[5] = {-5.447609879822406e+01,  1.615858368580409e+02,
                                -1.556989798598866e+02,  6.680131188771972e+01,
                                -1.328068155288572e+01};
    static const double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
                                -2.400758277161838e+00, -2.549732539343734e+00,
                                 4.374664141464968e+00,  2.938163982698783e+00};
    static const double d[4] = {7.784695709041462e-03, 3.224671290700398e-01,
                                2.445134137142996e+00, 3.754408661907416e+00};
    constexpr double p_low = 0.02425, p_high = 1.0 - p_low;
    double q, r;
    if (p < p_low) {
        q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if (p <= p_high) {
        q = p - 0.5;
        r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    }
    q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
}

std::uint64_t required_sample_size(double p1, double p2, double alpha, double power) {
    if (!(p1 > 0.0 && p1 < 1.0) || !(p2 > 0.0 && p2 < 1.0))
        throw std::invalid_argument("required_sample_size: rates must be in (0,1)");
    if (p1 == p2) return 0;
    const double z_a = normal_quantile(1.0 - alpha / 2.0);
    const double z_b = normal_quantile(power);
    const double num = (z_a + z_b) * (z_a + z_b) * (p1 * (1.0 - p1) + p2 * (1.0 - p2));
    const double den = (p1 - p2) * (p1 - p2);
    const double n = num / den;
    if (!(n > 0.0) || !std::isfinite(n)) return 0;
    return static_cast<std::uint64_t>(std::ceil(n));
}

CapacityPlan plan_capacity(double observed_rate_per_day, std::uint64_t planned_days,
                           std::uint64_t current_observations,
                           std::uint64_t observations_needed) {
    CapacityPlan plan;
    plan.observed_rate_per_day = std::max(0.0, observed_rate_per_day);
    plan.planned_days = planned_days;
    plan.current_n = current_observations;
    plan.observations_needed = observations_needed;
    const double projected = static_cast<double>(current_observations) +
                             plan.observed_rate_per_day * static_cast<double>(planned_days);
    plan.projected_n = static_cast<std::uint64_t>(std::max(0.0, std::floor(projected)));
    plan.achievable = plan.projected_n >= observations_needed;
    const double factor = plan.observed_rate_per_day > 0.0
                              ? static_cast<double>(observations_needed) /
                                    plan.observed_rate_per_day
                              : -1.0;
    if (plan.achievable) {
        plan.note = "projected observations cover the planning estimate";
    } else if (factor > 0.0) {
        plan.note = "at the observed rate this design needs about " +
                    std::to_string(static_cast<std::uint64_t>(std::ceil(factor))) + " days";
    } else {
        plan.note = "no observations logged yet; the rate is unknown";
    }
    return plan;
}

}  // namespace ae
