#include <gtest/gtest.h>
#include "ae/ranking.hpp"
#include "test_helpers.hpp"

#include <random>

using namespace ae;

namespace {
std::vector<double> make_scores(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<double> s(n);
    for (auto& v : s) v = u(rng);
    return s;
}
}

TEST(TopK, AllStrategiesAgreeWithBruteForce) {
    const std::size_t n = 200;
    auto scores = make_scores(n, 5);
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (scores[a] != scores[b]) return scores[a] > scores[b];
        return a < b;
    });
    for (std::size_t k : {1u, 5u, 17u, 200u}) {
        for (auto strat : {TopKStrategy::sort_full, TopKStrategy::partial_sort,
                           TopKStrategy::nth_element, TopKStrategy::heap_bounded}) {
            auto got = top_k(scores, k, strat);
            ASSERT_EQ(got.size(), k) << to_string(strat);
            for (std::size_t i = 0; i < k; ++i) {
                EXPECT_EQ(got[i].index, order[i]) << to_string(strat) << " pos " << i;
                EXPECT_DOUBLE_EQ(got[i].score, scores[order[i]]);
            }
        }
    }
}

TEST(TopK, TieBreaksAreDeterministic) {
    std::vector<double> s(10, 0.5);
    auto a = top_k(s, 3, TopKStrategy::heap_bounded);
    ASSERT_EQ(a.size(), 3u);
    EXPECT_EQ(a[0].index, 0u);
    EXPECT_EQ(a[1].index, 1u);
    EXPECT_EQ(a[2].index, 2u);
}

TEST(TopK, InvalidArgumentsThrow) {
    std::vector<double> s{1.0, 2.0};
    EXPECT_THROW(top_k(s, 0), std::invalid_argument);
    EXPECT_THROW(top_k(s, 3), std::invalid_argument);
}

TEST(TopK, HandlesNegativesAndNaNFreeInput) {
    std::vector<double> s{-5.0, -1.0, -3.0, -2.0};
    auto r = top_k(s, 2, TopKStrategy::heap_bounded);
    EXPECT_EQ(r[0].index, 1u);
    EXPECT_EQ(r[1].index, 3u);
}

TEST(TopK, RowsParallelEqualsSequential) {
    SimilarityMatrix m{40, 100, std::vector<double>(40 * 100)};
    std::mt19937_64 rng(9);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (auto& v : m.value) v = u(rng);
    auto seq = top_k_rows(m, 7, TopKStrategy::heap_bounded, 1);
    auto par = top_k_rows(m, 7, TopKStrategy::heap_bounded, 4);
    ASSERT_EQ(seq.size(), par.size());
    for (std::size_t r = 0; r < seq.size(); ++r) {
        ASSERT_EQ(seq[r].size(), par[r].size());
        for (std::size_t i = 0; i < seq[r].size(); ++i)
            EXPECT_EQ(seq[r][i].index, par[r][i].index);
    }
}

TEST(Mmr, SeedIsMostRelevant) {
    EmbeddingMatrix cand(6, 8);
    for (std::size_t i = 0; i < 6; ++i) {
        auto v = test::random_unit_vector(8, 1000 + i);
        std::copy(v.begin(), v.end(), cand.mutable_row(i).begin());
    }
    std::vector<double> rel{0.1, 0.2, 0.95, 0.4, 0.3, 0.15};
    auto sel = mmr_select(cand, rel, MmrParams{0.7, 3});
    ASSERT_EQ(sel.size(), 3u);
    EXPECT_EQ(sel[0].index, 2u);
}

TEST(Mmr, PureRelevanceEqualsTopK) {
    EmbeddingMatrix cand(8, 4);
    for (std::size_t i = 0; i < 8; ++i) {
        auto v = test::random_unit_vector(4, 2000 + i);
        std::copy(v.begin(), v.end(), cand.mutable_row(i).begin());
    }
    std::vector<double> rel{0.5, 0.9, 0.1, 0.7, 0.3, 0.8, 0.2, 0.6};
    auto sel = mmr_select(cand, rel, MmrParams{1.0, 4});
    std::vector<double> got;
    for (auto& s : sel) got.push_back(s.score);
    EXPECT_EQ(got[0], 0.9);
    EXPECT_EQ(got[1], 0.8);
    EXPECT_EQ(got[2], 0.7);
    EXPECT_EQ(got[3], 0.6);
}

TEST(Mmr, DiversityPullsInDissimilarItem) {
    // candidates 0 and 1 are near-identical, 2 is orthogonal to both
    EmbeddingMatrix cand(3, 2);
    cand.mutable_row(0)[0] = 1.0; cand.mutable_row(0)[1] = 0.0;
    cand.mutable_row(1)[0] = 0.999999; cand.mutable_row(1)[1] = 0.0014142;
    normalize_inplace(cand.mutable_row(1));
    cand.mutable_row(2)[0] = 0.0; cand.mutable_row(2)[1] = 1.0;
    std::vector<double> rel{1.0, 0.99, 0.30};
    auto diverse = mmr_select(cand, rel, MmrParams{0.5, 2});
    ASSERT_EQ(diverse.size(), 2u);
    EXPECT_EQ(diverse[0].index, 0u);
    EXPECT_EQ(diverse[1].index, 2u);   // 1 is too similar to 0 to be worth it
}

TEST(Capped, RespectsCategoryCap) {
    std::vector<double> scores{0.9, 0.8, 0.7, 0.6, 0.5};
    std::vector<std::int32_t> cat{0, 0, 0, 1, 1};
    auto r = top_k_capped(scores, cat, 4, 2);
    ASSERT_EQ(r.size(), 4u);
    int cat0 = 0, cat1 = 0;
    for (auto& item : r) (cat[item.index] == 0 ? cat0 : cat1)++;
    EXPECT_LE(cat0, 2);
    EXPECT_EQ(r[0].index, 0u);
    EXPECT_EQ(r[2].index, 3u);   // third category-0 item skipped
}

TEST(Ndcg, PerfectOrderingIsOne) {
    std::vector<double> rel{3, 2, 1, 0};
    EXPECT_NEAR(ndcg_at_k(rel, rel, 4), 1.0, 1e-12);
}

TEST(Ndcg, ReverseOrderingIsLower) {
    std::vector<double> rel{3, 2, 1, 0};
    std::vector<double> rev{0, 1, 2, 3};
    const double v = ndcg_at_k(rev, rel, 4);
    EXPECT_LT(v, 1.0);
    EXPECT_GE(v, 0.0);
}

TEST(Kendall, IdenticalAndReversed) {
    std::vector<std::size_t> a{0, 1, 2, 3, 4};
    std::vector<std::size_t> b{4, 3, 2, 1, 0};
    EXPECT_DOUBLE_EQ(kendall_tau_b(a, a), 1.0);
    EXPECT_DOUBLE_EQ(kendall_tau_b(a, b), -1.0);
}
