#include <gtest/gtest.h>
#include "ae/stats.hpp"

#include <cmath>

using namespace ae;

TEST(Beta, PosteriorShapesAddCounts) {
    auto p = beta_posterior(1.0, 1.0, 7, 3);
    EXPECT_DOUBLE_EQ(p.alpha, 8.0);
    EXPECT_DOUBLE_EQ(p.beta, 4.0);
    EXPECT_NEAR(p.mean(), 8.0 / 12.0, 1e-15);
    EXPECT_NEAR(p.variance(), (8.0 * 4.0) / (144.0 * 13.0), 1e-15);
}

TEST(Beta, PdfIntegratesToOne) {
    BetaParams p{3.0, 7.0};
    double s = 0.0;
    const int n = 100000;
    for (int i = 0; i < n; ++i) {
        const double x = (i + 0.5) / n;
        s += beta_pdf(x, p);
    }
    EXPECT_NEAR(s / n, 1.0, 1e-6);
}

TEST(Beta, CdfKnownClosedForms) {
    // Beta(1,1) is uniform
    BetaParams u{1.0, 1.0};
    for (double x : {0.0, 0.1, 0.5, 0.9, 1.0}) EXPECT_NEAR(beta_cdf(x, u), x, 1e-12);
    // Beta(2,1) has CDF x^2
    BetaParams b21{2.0, 1.0};
    for (double x : {0.1, 0.3, 0.7}) EXPECT_NEAR(beta_cdf(x, b21), x * x, 1e-11);
    // Beta(1,2) has CDF 1-(1-x)^2
    BetaParams b12{1.0, 2.0};
    for (double x : {0.1, 0.3, 0.7}) EXPECT_NEAR(beta_cdf(x, b12), 1.0 - (1 - x) * (1 - x), 1e-11);
}

TEST(Beta, QuantileInvertsCdf) {
    BetaParams p{5.0, 9.0};
    for (double q : {0.025, 0.25, 0.5, 0.75, 0.975}) {
        const double x = beta_quantile(q, p);
        EXPECT_NEAR(beta_cdf(x, p), q, 1e-9);
    }
}

TEST(Beta, CredibleIntervalCoversMode) {
    BetaParams p{20.0, 30.0};
    auto iv = credible_interval(p, 0.95);
    EXPECT_LT(iv.lo, p.mean());
    EXPECT_GT(iv.hi, p.mean());
    EXPECT_LT(iv.lo, p.mode());
    EXPECT_GT(iv.hi, p.mode());
    EXPECT_NEAR(iv.width(), iv.hi - iv.lo, 0.0);
    auto narrow = credible_interval(p, 0.5);
    EXPECT_LT(narrow.width(), iv.width());   // 50% interval must be narrower
}

TEST(Beta, SuperiorshipIsComplementary) {
    BetaParams a{10.0, 5.0}, b{5.0, 10.0};
    const double p_ab = prob_superiority(a, b);
    const double p_ba = prob_superiority(b, a);
    EXPECT_NEAR(p_ab + p_ba, 1.0, 1e-6);
    EXPECT_GT(p_ab, 0.9);
}

TEST(Beta, SuperiorshipIsHalfForIdenticalPosteriors) {
    BetaParams a{7.0, 3.0};
    EXPECT_NEAR(prob_superiority(a, a), 0.5, 1e-3);
}

TEST(Beta, ExpectedLossSemantics) {
    // expected_loss(chosen, alternative) = E[(theta_alt - theta_chosen)^+]
    BetaParams strong{1000.0, 1.0}, weak{1.0, 1000.0};
    EXPECT_LT(expected_loss(strong, weak), 1e-3);   // picking strong forgoes nothing
    EXPECT_GT(expected_loss(weak, strong), 0.9);    // picking weak forgoes ~1.0
    // Closed forms for Beta(2,1) vs Uniform:
    //   E[(U - X)^+] = 1/12   and   E[(X - U)^+] = 1/4
    // (E[X] - E[U] = 1/6 = E[(X-U)^+] - E[(U-X)^+].)
    BetaParams x{2.0, 1.0}, u{1.0, 1.0};
    EXPECT_NEAR(expected_loss(x, u), 1.0 / 12.0, 2e-3);
    EXPECT_NEAR(expected_loss(u, x), 1.0 / 4.0, 2e-3);
}

TEST(Thompson, SelectsBetterVariantMostOfTheTime) {
    std::vector<BetaParams> v{{50.0, 50.0}, {80.0, 20.0}, {10.0, 90.0}};
    auto alloc = thompson_allocation(v, 3000, 11);
    EXPECT_GT(alloc[1], alloc[0]);      // clearly the strongest
    EXPECT_GT(alloc[1], alloc[2]);
    EXPECT_EQ(alloc[0] + alloc[1] + alloc[2], 3000u);
}

TEST(Thompson, IsReproducibleForAGivenSeed) {
    std::vector<BetaParams> v{{10.0, 10.0}, {12.0, 8.0}};
    auto a = thompson_allocation(v, 500, 99);
    auto b = thompson_allocation(v, 500, 99);
    EXPECT_EQ(a, b);
    auto c = thompson_allocation(v, 500, 100);
    EXPECT_NE(a, c);
}

TEST(Thompson, ExploresUntriedArms) {
    std::vector<BetaParams> v{{100.0, 100.0}, {2.0, 2.0}};
    auto alloc = thompson_allocation(v, 1000, 3);
    EXPECT_GT(alloc[1], 150u);   // wide posterior must still get traffic
}

TEST(Sequential, WaitsForMinimumObservations) {
    auto r = sequential_decision(beta_posterior(1, 1, 90, 10),
                                 beta_posterior(1, 1, 10, 90), 5, 5, 100);
    EXPECT_FALSE(r.stop);
    EXPECT_NE(r.reason.find("minimum"), std::string::npos);
}

TEST(Sequential, StopsWhenClear) {
    auto r = sequential_decision(beta_posterior(1, 1, 900, 100),
                                 beta_posterior(1, 1, 100, 900), 1000, 1000, 100, 0.005);
    EXPECT_TRUE(r.stop);
    EXPECT_GT(r.prob_superiority, 0.99);
}

TEST(Sequential, ContinuesWhenPosteriorsOverlap) {
    auto r = sequential_decision(beta_posterior(1, 1, 50, 50),
                                 beta_posterior(1, 1, 52, 48), 100, 100, 50, 0.005);
    EXPECT_FALSE(r.stop);
}

TEST(Multiplicity, AdjustmentsAreConservative) {
    EXPECT_NEAR(bonferroni_level(0.05, 5), 0.01, 1e-15);
    const double sidak = sidak_level(0.05, 5);
    EXPECT_GT(sidak, bonferroni_level(0.05, 5));
    EXPECT_LT(sidak, 0.05);
}

TEST(Hierarchical, ShrinkagePullsExtremesTowardPopulation) {
    std::vector<std::uint64_t> s{100, 3, 51}, f{100, 3, 49};
    auto fit = fit_hierarchical(s, f, 1.0, 1.0, 1.0);
    ASSERT_EQ(fit.posteriors.size(), 3u);
    EXPECT_GT(fit.overdispersion, 0.0);
    // the extreme low arm must be pulled above its empirical rate 0.5
    const double empirical = 3.0 / 6.0;
    EXPECT_GT(fit.posteriors[1].mean(), empirical);
    // with zero shrinkage the posterior equals the plain Beta-Binomial result
    auto flat = fit_hierarchical(s, f, 1.0, 1.0, 0.0);
    EXPECT_NEAR(flat.posteriors[0].mean(), 101.0 / 202.0, 1e-12);
}

TEST(Conversion, MissingDenominatorIsNotAFailure) {
    EXPECT_FALSE(conversion_rate(0, 0).has_value());
    EXPECT_FALSE(conversion_rate(0, 0).has_value());
    ASSERT_TRUE(conversion_rate(3, 10).has_value());
    EXPECT_NEAR(*conversion_rate(3, 10), 0.3, 1e-15);
    EXPECT_THROW(conversion_rate(11, 10), std::invalid_argument);
}
