#include <gtest/gtest.h>
#include "ae/ranking.hpp"
#include "ae/similarity.hpp"
#include "ae/optimizer.hpp"
#include "ae/compat_matcher.hpp"
#include "test_helpers.hpp"

#include <atomic>
#include <thread>
#include <vector>

using namespace ae;

TEST(Concurrency, ParallelBatchMatchesSingleThreaded) {
    EmbeddingMatrix q(30, 64), x(200, 64);
    for (std::size_t i = 0; i < q.rows(); ++i) {
        auto v = test::random_unit_vector(64, 7000 + i);
        std::copy(v.begin(), v.end(), q.mutable_row(i).begin());
    }
    for (std::size_t i = 0; i < x.rows(); ++i) {
        auto v = test::random_unit_vector(64, 9000 + i);
        std::copy(v.begin(), v.end(), x.mutable_row(i).begin());
    }
    auto s1 = batch_cosine_unit(q, x, 1);
    auto s4 = batch_cosine_unit(q, x, 4);
    ASSERT_EQ(s1.value.size(), s4.value.size());
    for (std::size_t i = 0; i < s1.value.size(); ++i) EXPECT_DOUBLE_EQ(s1.value[i], s4.value[i]);
}

TEST(Concurrency, RepeatedParallelCallsAreStable) {
    EmbeddingMatrix x(120, 32);
    for (std::size_t i = 0; i < x.rows(); ++i) {
        auto v = test::random_unit_vector(32, 50 + i);
        std::copy(v.begin(), v.end(), x.mutable_row(i).begin());
    }
    const auto ref = within_set_similarity(x, 4);
    for (int rep = 0; rep < 5; ++rep) {
        auto cur = within_set_similarity(x, 4);
        for (std::size_t k = 0; k < ref.value.size(); ++k)
            ASSERT_DOUBLE_EQ(cur.value[k], ref.value[k]);
    }
}

TEST(Concurrency, RankerBackgroundThreadsSeeConsistentState) {
    // Concurrent read-only ranking from several threads must not race or throw.
    auto viewer = test::random_unit_vector(24, 1);
    std::vector<CandidateProfile> cands;
    for (std::size_t i = 0; i < 64; ++i) {
        CandidateProfile c;
        c.index = i;
        c.visual = test::random_unit_vector(24, 100 + i);
        c.textual = test::random_unit_vector(24, 200 + i);
        c.attributes = {0.5, 0.5};
        c.interests = {1.0, 0.0};
        cands.push_back(std::move(c));
    }
    CandidateProfile v;
    v.index = 999;
    v.visual = viewer;
    v.textual = viewer;

    std::atomic<int> failures{0};
    std::vector<std::thread> pool;
    const auto reference = rank_candidates(v, cands, CompatWeights{}, 5);
    for (int t = 0; t < 4; ++t) {
        pool.emplace_back([&] {
            for (int rep = 0; rep < 20; ++rep) {
                try {
                    auto r = rank_candidates(v, cands, CompatWeights{}, 5);
                    if (r.size() != reference.size()) { failures++; continue; }
                    for (std::size_t i = 0; i < r.size(); ++i)
                        if (r[i].index != reference[i].index) failures++;
                } catch (...) {
                    failures++;
                }
            }
        });
    }
    for (auto& t : pool) t.join();
    EXPECT_EQ(failures.load(), 0);
}

TEST(Concurrency, OptimizerIsThreadSafeOnSeparateProblems) {
    std::atomic<int> errors{0};
    std::vector<std::thread> pool;
    for (std::uint64_t seed = 0; seed < 4; ++seed) {
        pool.emplace_back([&, seed] {
            try {
                auto p = test::make_problem(9, 0.35, 0.2, seed + 1);
                OptimizeOptions o;
                o.strategy = Strategy::beam;
                o.beam_width = 6;
                auto a = optimize(p, o);
                auto b = optimize(p, o);
                if (a.permutation != b.permutation) errors++;
                if (!(a.score.total >= evaluate(p, a.permutation).total - 1e-12)) errors++;
            } catch (...) {
                errors++;
            }
        });
    }
    for (auto& t : pool) t.join();
    EXPECT_EQ(errors.load(), 0);
}

TEST(ExceptionSafety, ExceptionsLeaveObjectsUsable) {
    EmbeddingMatrix m(3, 4);
    for (auto& v : m.mutable_row(0)) v = 1.0;
    EXPECT_THROW(m.normalize_rows(), std::domain_error);
    // object still usable after the throw
    EXPECT_NO_THROW(m.mutable_row(0)[0] = 2.0);
    EXPECT_EQ(m.rows(), 3u);
    EmbeddingMatrix copy = m;   // still copyable
    EXPECT_EQ(copy.rows(), 3u);
    std::vector<double> s{1.0};
    EXPECT_THROW(top_k(s, 5), std::invalid_argument);
    auto ok = top_k(std::vector<double>{1.0, 2.0, 3.0}, 2);
    EXPECT_EQ(ok.size(), 2u);
}
