#include <gtest/gtest.h>
#include "ae/compat_matcher.hpp"
#include "test_helpers.hpp"

#include <cmath>

using namespace ae;

namespace {
CandidateProfile make_profile(std::size_t idx, std::uint64_t seed, std::size_t d = 16) {
    CandidateProfile p;
    p.index = idx;
    p.visual = test::random_unit_vector(d, seed * 2);
    p.textual = test::random_unit_vector(d, seed * 2 + 1);
    p.attributes.resize(6);
    p.interests.resize(10);
    for (std::size_t i = 0; i < p.attributes.size(); ++i)
        p.attributes[i] = static_cast<double>((seed * 7 + i * 13) % 100) / 100.0;
    for (std::size_t i = 0; i < p.interests.size(); ++i)
        p.interests[i] = ((seed + i) % 4 == 0) ? 1.0 : 0.0;
    return p;
}
}

TEST(Compat, WeightsAreNormalised) {
    CompatTerms t;
    CandidateProfile a = make_profile(0, 1), b = make_profile(1, 1);  // identical
    CompatWeights w;
    t = score_pair(a, b, w);
    // identical profiles: every term is 1.0, so S must be exactly 1.0 regardless
    // of how the raw weights are scaled.
    EXPECT_NEAR(t.visual, 1.0, 1e-12);
    EXPECT_NEAR(t.textual, 1.0, 1e-12);
    EXPECT_NEAR(t.attributes, 1.0, 1e-12);
    EXPECT_NEAR(t.interests, 1.0, 1e-12);
    EXPECT_NEAR(t.score, 1.0, 1e-12);

    CompatWeights doubled{0.5, 0.6, 0.5, 0.4};
    EXPECT_NEAR(score_pair(a, b, doubled).score, 1.0, 1e-12);
}

TEST(Compat, ScoresAreBounded) {
    CompatWeights w;
    auto a = make_profile(0, 3);
    for (std::uint64_t s = 1; s < 30; ++s) {
        auto b = make_profile(static_cast<std::size_t>(s), s);
        auto t = score_pair(a, b, w);
        EXPECT_GE(t.score, 0.0);
        EXPECT_LE(t.score, 1.0);
        EXPECT_GE(t.visual, 0.0);
        EXPECT_LE(t.visual, 1.0);
    }
}

TEST(Compat, EmptyModalitiesDegradeGracefully) {
    CompatWeights w;
    auto a = make_profile(0, 4);
    CandidateProfile empty = a;
    empty.visual.clear();
    empty.textual.clear();
    auto t = score_pair(a, empty, w);
    // to_unit(0) == 0.5: a missing modality is treated as "neutral", not "bad",
    // and the confidence field is what flags it as unreliable.
    EXPECT_DOUBLE_EQ(t.visual, 0.5);
    EXPECT_DOUBLE_EQ(t.textual, 0.5);
    // attribute and interest terms still contribute
    EXPECT_GT(t.score, 0.0);
}

TEST(Compat, ReciprocalRankingIsOrdered) {
    auto viewer = make_profile(0, 9);
    std::vector<CandidateProfile> cands;
    for (std::size_t i = 1; i <= 20; ++i) cands.push_back(make_profile(i, i));
    auto ranked = rank_candidates(viewer, cands, CompatWeights{}, 5);
    ASSERT_EQ(ranked.size(), 5u);
    for (std::size_t i = 1; i < ranked.size(); ++i)
        EXPECT_GE(ranked[i - 1].harmonic, ranked[i].harmonic);
    for (const auto& r : ranked) {
        EXPECT_GE(r.confidence, 0.0);
        EXPECT_LE(r.confidence, 1.0);
    }
}

TEST(Compat, IdenticalProfileScoresHighest) {
    auto viewer = make_profile(0, 42);
    std::vector<CandidateProfile> cands{make_profile(1, 1), make_profile(2, 2), viewer};
    auto ranked = rank_candidates(viewer, cands, CompatWeights{}, 3);
    EXPECT_EQ(ranked.front().index, 0u);      // index of the clone is 0
    EXPECT_NEAR(ranked.front().harmonic, 1.0, 1e-9);
}

TEST(Compat, ColdStartReducesConfidence) {
    auto viewer = make_profile(0, 5);
    viewer.cold_start = true;
    std::vector<CandidateProfile> cands{make_profile(1, 6)};
    auto warm = rank_candidates(make_profile(0, 5), cands, CompatWeights{}, 1);
    auto cold = rank_candidates(viewer, cands, CompatWeights{}, 1);
    EXPECT_LT(cold.front().confidence, warm.front().confidence);
}

TEST(Compat, DiversityLambdaReshufflesNearDuplicates) {
    CandidateProfile viewer = make_profile(0, 1);
    // two nearly identical candidates plus a distinct one, all similar to viewer
    CandidateProfile c1 = make_profile(1, 2);
    CandidateProfile c2 = c1; c2.index = 2;
    c2.visual[0] += 1e-6;
    normalize_inplace(c2.visual);
    CandidateProfile c3 = make_profile(3, 4);
    std::vector<CandidateProfile> cands{c1, c2, c3};
    auto plain = rank_candidates(viewer, cands, CompatWeights{}, 3, 0.0);
    auto diverse = rank_candidates(viewer, cands, CompatWeights{}, 3, 0.6);
    EXPECT_EQ(plain.size(), 3u);
    ASSERT_EQ(diverse.size(), 3u);
    // with diversity pressure the duplicate pair should not be the first two
    const bool dup_first_two = (diverse[0].index == 1 && diverse[1].index == 2) ||
                              (diverse[0].index == 2 && diverse[1].index == 1);
    EXPECT_FALSE(dup_first_two);
}

TEST(PairRanker, UntrainedReturnsNeutralAndFlagsItself) {
    PairRanker r;
    EXPECT_FALSE(r.is_trained());
    std::vector<double> a(8, 0.5), b(8, 0.25);
    EXPECT_DOUBLE_EQ(r.predict(a, b), 0.5);
}

TEST(PairRanker, FeatureVectorLayout) {
    std::vector<double> a{1, 2, 3}, b{4, 6, 3};
    auto f = PairRanker::pair_features(a, b);
    ASSERT_EQ(f.size(), 12u);
    EXPECT_EQ(f[0], 1.0);      // u_i
    EXPECT_EQ(f[3], 4.0);      // u_j
    EXPECT_EQ(f[6], 3.0);      // |1-4|
    EXPECT_EQ(f[8], 0.0);      // |2-6|
    EXPECT_EQ(f[9], 4.0);      // 1*4
    EXPECT_EQ(f[11], 9.0);     // 3*3
}

TEST(PairRanker, SerialisationRoundTrip) {
    std::vector<PairRanker::Layer> layers;
    layers.push_back({4, 3, {}, {}});
    layers[0].weight.resize(12);
    layers[0].bias.resize(3);
    for (std::size_t i = 0; i < 12; ++i) layers[0].weight[i] = 0.1 * static_cast<double>(i);
    layers[0].bias = {0.1, -0.2, 0.3};
    layers.push_back({3, 1, {0.5, -0.5, 0.25}, {0.01}});
    PairRanker r(layers);
    const std::string text = r.serialise();
    PairRanker back = PairRanker::deserialise(text);
    EXPECT_TRUE(back.is_trained());
    EXPECT_EQ(back.feature_dim(), 4u);
    std::vector<double> a{0.2, 0.4, 0.6, 0.8}, b{0.1, 0.3, 0.5, 0.7};
    EXPECT_NEAR(r.predict(a, b), back.predict(a, b), 1e-15);
}

TEST(PairRanker, SymmetricPredictionAveragesOrderings) {
    std::vector<PairRanker::Layer> layers;
    layers.push_back({4, 2, {0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4}, {0.0, 0.1}});
    layers.push_back({2, 1, {0.7, -0.3}, {0.05}});
    PairRanker r(layers);
    std::vector<double> a{0.9, 0.1, 0.5, 0.2}, b{0.2, 0.8, 0.1, 0.4};
    const double sym = r.predict_symmetric(a, b);
    const double avg = 0.5 * (r.predict(a, b) + r.predict(b, a));
    EXPECT_NEAR(sym, avg, 1e-15);
    EXPECT_GE(sym, 0.0);
    EXPECT_LE(sym, 1.0);
}

TEST(PairRanker, MalformedLayersRejected) {
    std::vector<PairRanker::Layer> bad;
    bad.push_back({4, 2, {1.0, 2.0}, {0.0, 0.0}});   // wrong weight size
    EXPECT_THROW(PairRanker{bad}, std::invalid_argument);
    std::vector<PairRanker::Layer> chain;
    chain.push_back({4, 2, std::vector<double>(8, 0.1), std::vector<double>(2, 0.0)});
    chain.push_back({3, 1, std::vector<double>(3, 0.1), {0.0}});   // broken chain
    EXPECT_THROW(PairRanker{chain}, std::invalid_argument);
    EXPECT_THROW(PairRanker::deserialise("garbage"), std::invalid_argument);
}
