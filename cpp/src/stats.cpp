#include "ae/stats.hpp"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <random>
#include <stdexcept>

namespace ae {

// ------------------------------------------------------------ special functions
namespace {

// Lanczos approximation of log Gamma (g=7, n=9); |error| < 1e-15 for x > 0.
double log_gamma(double x) {
    static const double c[9] = {
        0.99999999999980993,   676.5203681218851,     -1259.1392167224028,
        771.32342877765313,   -176.61502916214059,     12.507343278686905,
        -0.13857109526572012,   9.9843695780195716e-6,  1.5056327351493116e-7};
    if (x < 0.5) {
        // reflection: Gamma(x)Gamma(1-x) = pi / sin(pi x)
        return std::log(M_PI / std::sin(M_PI * x)) - log_gamma(1.0 - x);
    }
    x -= 1.0;
    double a = c[0];
    const double t = x + 7.5;
    for (int i = 1; i < 9; ++i) a += c[i] / (x + static_cast<double>(i));
    return 0.5 * std::log(2.0 * M_PI) + (x + 0.5) * std::log(t) - t + std::log(a);
}

double betacf(double a, double b, double x) {
    // Continued fraction for the incomplete beta (modified Lentz, NR 6.4).
    constexpr int kMaxIt = 300;
    constexpr double kEps = 3e-14;
    constexpr double kFpMin = 1e-300;
    const double qab = a + b, qap = a + 1.0, qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (std::abs(d) < kFpMin) d = kFpMin;
    d = 1.0 / d;
    double h = d;
    for (int m = 1; m <= kMaxIt; ++m) {
        const double m2 = 2.0 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < kFpMin) d = kFpMin;
        c = 1.0 + aa / c;
        if (std::abs(c) < kFpMin) c = kFpMin;
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < kFpMin) d = kFpMin;
        c = 1.0 + aa / c;
        if (std::abs(c) < kFpMin) c = kFpMin;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::abs(del - 1.0) < kEps) break;
    }
    return h;
}

double regularised_incomplete_beta(double a, double b, double x) {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    const double bt = std::exp(log_gamma(a + b) - log_gamma(a) - log_gamma(b) +
                               a * std::log(x) + b * std::log1p(-x));
    if (x < (a + 1.0) / (a + b + 2.0)) return bt * betacf(a, b, x) / a;
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b;
}

}  // namespace

BetaParams beta_posterior(double prior_alpha, double prior_beta, std::uint64_t successes,
                          std::uint64_t failures) {
    if (prior_alpha <= 0.0 || prior_beta <= 0.0)
        throw std::invalid_argument("beta_posterior: priors must be positive");
    return BetaParams{prior_alpha + static_cast<double>(successes),
                      prior_beta + static_cast<double>(failures)};
}

double beta_pdf(double x, const BetaParams& p) {
    if (x <= 0.0 || x >= 1.0) return 0.0;
    const double log_b = log_gamma(p.alpha) + log_gamma(p.beta) - log_gamma(p.alpha + p.beta);
    return std::exp((p.alpha - 1.0) * std::log(x) + (p.beta - 1.0) * std::log1p(-x) - log_b);
}

double beta_cdf(double x, const BetaParams& p) {
    return regularised_incomplete_beta(p.alpha, p.beta, x);
}

double beta_quantile(double q, const BetaParams& p) {
    if (q <= 0.0) return 0.0;
    if (q >= 1.0) return 1.0;
    double lo = 0.0, hi = 1.0;
    for (int i = 0; i < 200; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (beta_cdf(mid, p) < q) lo = mid; else hi = mid;
        if (hi - lo < 1e-13) break;
    }
    return 0.5 * (lo + hi);
}

Interval credible_interval(const BetaParams& p, double level) {
    if (!(level > 0.0 && level < 1.0)) throw std::invalid_argument("credible_interval: level");
    const double tail = (1.0 - level) / 2.0;
    Interval iv;
    iv.lo = beta_quantile(tail, p);
    iv.hi = beta_quantile(1.0 - tail, p);
    iv.level = level;
    return iv;
}

double prob_superiority(const BetaParams& a, const BetaParams& b) {
    // P(theta_A > theta_B) = int_0^1 f_A(x) F_B(x) dx.
    // Composite Simpson on the unit interval with an even number of panels.
    // The integrand is smooth for the posteriors we see in practice (counts >= 1);
    // tests compare against a 1e-6-step reference to confirm < 1e-9 error.
    constexpr int kPanels = 4096;
    constexpr double h = 1.0 / kPanels;
    auto f = [&](double x) { return beta_pdf(x, a) * beta_cdf(x, b); };
    double s = f(0.0) + f(1.0);
    for (int i = 1; i < kPanels; ++i) {
        const double x = i * h;
        s += ((i & 1) ? 4.0 : 2.0) * f(x);
    }
    const double v = s * h / 3.0;
    return std::clamp(v, 0.0, 1.0);
}

// Semantics used consistently across the codebase and the API:
//   expected_loss(chosen, alternative) = E[max(theta_alternative - theta_chosen, 0)]
// i.e. how much conversion rate we expect to forgo by deploying `chosen`.
// The parameter names below follow the mathematics (a = chosen, b = alternative).
double expected_loss(const BetaParams& a, const BetaParams& b) {
    // E[(theta_B - theta_A)^+] over independent posteriors.
    // Computed by Gauss-Legendre-free approach: discretise theta_A, integrate
    // the closed-form conditional expectation of the positive part of B - x:
    //   E[(theta_B - x)^+] = mu_B * P(X > x | shape shifted) - x * P(theta_B > x)
    // For Beta(a2,b2): E[theta_B 1{theta_B > x}] = mu_B * (1 - F_{Beta(a2+1,b2)}(x)).
    constexpr int kNodes = 2048;
    double total = 0.0;
    const double mu_b = b.mean();
    for (int i = 0; i < kNodes; ++i) {
        const double x = (i + 0.5) / kNodes;
        const double density_a = beta_pdf(x, a);
        if (density_a == 0.0) continue;
        BetaParams shifted{b.alpha + 1.0, b.beta};
        const double e_pos = mu_b * (1.0 - beta_cdf(x, shifted)) - x * (1.0 - beta_cdf(x, b));
        total += std::max(0.0, e_pos) * density_a;
    }
    return total / kNodes;
}

// --------------------------------------------------------------- sampling
namespace {
// SplitMix64: high-quality seeding and a cheap deterministic stream generator.
std::uint64_t splitmix64(std::uint64_t& state) {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Marsaglia-Tsang Gamma(shape, 1) sampler; uint64 -> uniform in (0,1).
double sample_gamma(std::mt19937_64& rng, double shape) {
    if (shape < 1.0) {
        const double u = std::uniform_real_distribution<double>(1e-15, 1.0)(rng);
        return sample_gamma(rng, shape + 1.0) * std::pow(u, 1.0 / shape);
    }
    const double d = shape - 1.0 / 3.0;
    const double c = 1.0 / std::sqrt(9.0 * d);
    for (;;) {
        double x, v;
        do {
            x = std::normal_distribution<double>(0.0, 1.0)(rng);
            v = 1.0 + c * x;
        } while (v <= 0.0);
        v = v * v * v;
        const double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
        if (u < 1.0 - 0.0331 * (x * x) * (x * x)) return d * v;
        if (std::log(u) < 0.5 * x * x + d * (1.0 - v + std::log(v))) return d * v;
    }
}

double sample_beta(std::mt19937_64& rng, const BetaParams& p) {
    const double x = sample_gamma(rng, p.alpha);
    const double y = sample_gamma(rng, p.beta);
    const double s = x + y;
    return s > 0.0 ? x / s : 0.5;
}
}  // namespace

std::size_t thompson_select(const std::vector<BetaParams>& variants, std::uint64_t seed) {
    if (variants.empty()) throw std::invalid_argument("thompson_select: no variants");
    std::uint64_t s = seed;
    std::mt19937_64 rng(splitmix64(s));
    std::size_t best = 0;
    double best_v = -1.0;
    for (std::size_t i = 0; i < variants.size(); ++i) {
        const double v = sample_beta(rng, variants[i]);
        if (v > best_v) { best_v = v; best = i; }
    }
    return best;
}

std::vector<std::uint64_t> thompson_allocation(const std::vector<BetaParams>& variants,
                                               std::uint64_t draws, std::uint64_t seed) {
    if (variants.empty()) throw std::invalid_argument("thompson_allocation: no variants");
    std::uint64_t s = seed;
    std::mt19937_64 rng(splitmix64(s));
    std::vector<double> buf(variants.size());
    std::vector<std::uint64_t> counts(variants.size(), 0);
    for (std::uint64_t d = 0; d < draws; ++d) {
        std::size_t best = 0;
        for (std::size_t i = 0; i < variants.size(); ++i) {
            buf[i] = sample_beta(rng, variants[i]);
            if (buf[i] > buf[best]) best = i;
        }
        counts[best] += 1;
    }
    return counts;
}

SequentialTestResult sequential_decision(const BetaParams& a, const BetaParams& b,
                                         std::uint64_t n_a, std::uint64_t n_b,
                                         std::uint64_t min_observations,
                                         double loss_tolerance) {
    SequentialTestResult r;
    r.prob_superiority = prob_superiority(a, b);
    // loss of choosing A = E[(theta_B - theta_A)^+]; loss of choosing B is the mirror
    r.expected_loss_choose_a = expected_loss(a, b);
    r.expected_loss_choose_b = expected_loss(b, a);
    const double leader_loss = r.prob_superiority >= 0.5 ? r.expected_loss_choose_a
                                                         : r.expected_loss_choose_b;
    const bool enough = n_a >= min_observations && n_b >= min_observations;
    r.stop = enough && leader_loss < loss_tolerance;
    if (!enough) {
        r.reason = "minimum observation count not reached";
    } else if (r.stop) {
        r.reason = "expected loss of the leading variant is below tolerance";
    } else {
        r.reason = "posteriors still overlap; continue collecting observations";
    }
    return r;
}

double bonferroni_level(double family_level, std::size_t k) {
    if (k == 0) throw std::invalid_argument("bonferroni_level: k == 0");
    return family_level / static_cast<double>(k);
}

double sidak_level(double family_level, std::size_t k) {
    if (k == 0) throw std::invalid_argument("sidak_level: k == 0");
    return 1.0 - std::pow(1.0 - family_level, 1.0 / static_cast<double>(k));
}

HierarchicalFit fit_hierarchical(const std::vector<std::uint64_t>& successes,
                                 const std::vector<std::uint64_t>& failures,
                                 double prior_alpha, double prior_beta, double shrink) {
    if (successes.size() != failures.size() || successes.empty())
        throw std::invalid_argument("fit_hierarchical: size mismatch");
    if (!(shrink >= 0.0 && shrink <= 1.0))
        throw std::invalid_argument("fit_hierarchical: shrink outside [0,1]");
    const std::size_t k = successes.size();

    // Empirical-Bayes moment matching on the per-variant smoothed rates: treat
    // the logit of the observed rate as a noisy draw from a common distribution.
    std::vector<double> rate(k);
    double mean_rate = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
        const double n = static_cast<double>(successes[i] + failures[i]);
        if (n <= 0.0) { rate[i] = 0.5; continue; }
        rate[i] = (static_cast<double>(successes[i]) + 0.5) / (n + 1.0);
        mean_rate += rate[i];
    }
    mean_rate /= static_cast<double>(k);
    double var = 0.0;
    for (double r : rate) var += (r - mean_rate) * (r - mean_rate);
    var = k > 1 ? var / static_cast<double>(k - 1) : 0.0;

    // Method of moments for a Beta(a,b) matching mean and variance.
    double pop_alpha = prior_alpha, pop_beta = prior_beta, overdisp = 0.0;
    if (var > 1e-12 && mean_rate > 0.0 && mean_rate < 1.0) {
        const double common = mean_rate * (1.0 - mean_rate) / var - 1.0;
        if (common > 0.0) {
            pop_alpha = std::max(prior_alpha, mean_rate * common);
            pop_beta = std::max(prior_beta, (1.0 - mean_rate) * common);
        }
        overdisp = var;
    }

    HierarchicalFit fit;
    fit.population_alpha = pop_alpha;
    fit.population_beta = pop_beta;
    fit.overdispersion = overdisp;
    fit.posteriors.reserve(k);
    for (std::size_t i = 0; i < k; ++i) {
        // Partial pooling: blend the population prior with the uninformative one.
        const double a0 = (1.0 - shrink) * prior_alpha + shrink * pop_alpha;
        const double b0 = (1.0 - shrink) * prior_beta + shrink * pop_beta;
        fit.posteriors.push_back(beta_posterior(a0, b0, successes[i], failures[i]));
    }
    return fit;
}

std::optional<double> conversion_rate(std::uint64_t successes, std::uint64_t observed) {
    if (observed == 0) return std::nullopt;
    if (successes > observed) throw std::invalid_argument("conversion_rate: successes > observed");
    return static_cast<double>(successes) / static_cast<double>(observed);
}

}  // namespace ae
