#include <gtest/gtest.h>
#include "ae/similarity.hpp"
#include "test_helpers.hpp"

#include <cmath>

using namespace ae;

TEST(EmbeddingMatrix, RejectsEmptyDimensions) {
    EXPECT_THROW(EmbeddingMatrix(0, 8), std::invalid_argument);
    EXPECT_THROW(EmbeddingMatrix(4, 0), std::invalid_argument);
}

TEST(EmbeddingMatrix, RowsAreAligned) {
    EmbeddingMatrix m(7, 13);
    for (std::size_t r = 0; r < m.rows(); ++r)
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(m.row(r).data()) % kSimdAlign, 0u);
}

TEST(EmbeddingMatrix, CopyIsDeepAndIndependent) {
    EmbeddingMatrix a(4, 8);
    for (auto& v : a.mutable_row(1)) v = 1.0;
    EmbeddingMatrix b = a;
    b.mutable_row(1)[0] = 42.0;
    EXPECT_EQ(a.row(1)[0], 1.0);   // value semantics: a is untouched
    EmbeddingMatrix c(2, 2);
    c = a;                          // copy assignment
    EXPECT_EQ(c.row(1)[0], 1.0);
    EXPECT_EQ(c.rows(), 4u);
}

TEST(EmbeddingMatrix, MoveTransfersBuffer) {
    EmbeddingMatrix a(3, 5);
    a.mutable_row(0)[0] = 9.0;
    EmbeddingMatrix b = std::move(a);
    EXPECT_EQ(b.row(0)[0], 9.0);
    EXPECT_EQ(b.rows(), 3u);
}

TEST(EmbeddingMatrix, NormalisationRejectsZeroRow) {
    EmbeddingMatrix m(2, 4);
    for (auto& v : m.mutable_row(0)) v = 1.0;
    // row 1 stays all-zero
    EXPECT_THROW(m.normalize_rows(), std::domain_error);
    EmbeddingMatrix ok(2, 4);
    for (auto& v : ok.mutable_row(0)) v = 2.0;
    for (auto& v : ok.mutable_row(1)) v = -3.0;
    ok.normalize_rows();
    EXPECT_TRUE(ok.is_row_normalized(0, 1e-12));
    EXPECT_TRUE(ok.is_row_normalized(1, 1e-12));
    EXPECT_NEAR(ok.row(0)[0], 0.5, 1e-12);
}

TEST(Similarity, CosineInvariants) {
    auto a = test::random_unit_vector(64, 1);
    auto b = test::random_unit_vector(64, 2);
    EXPECT_NEAR(cosine_unit(a, b), cosine_raw(a, b), 1e-12);
    EXPECT_NEAR(cosine_unit(a, a), 1.0, 1e-12);
    const double ab = cosine_unit(a, b);
    const double ba = cosine_unit(b, a);
    EXPECT_DOUBLE_EQ(ab, ba);                // exact symmetry
    EXPECT_GE(ab, -1.0);
    EXPECT_LE(ab, 1.0);
    EXPECT_THROW(cosine_unit(a, std::span<const double>(b).first(32)), std::invalid_argument);
}

TEST(Similarity, CosineMatchesNaiveReference) {
    const std::size_t d = 97;
    auto a = test::random_unit_vector(d, 11);
    auto b = test::random_unit_vector(d, 12);
    double ref = 0.0;
    for (std::size_t i = 0; i < d; ++i) ref += a[i] * b[i];
    EXPECT_NEAR(cosine_unit(a, b), ref, 1e-14);
}

TEST(Similarity, BatchMatchesPairwise) {
    EmbeddingMatrix q(5, 32), x(9, 32);
    for (std::size_t i = 0; i < q.rows(); ++i) {
        auto v = test::random_unit_vector(32, 100 + i);
        std::copy(v.begin(), v.end(), q.mutable_row(i).begin());
    }
    for (std::size_t i = 0; i < x.rows(); ++i) {
        auto v = test::random_unit_vector(32, 200 + i);
        std::copy(v.begin(), v.end(), x.mutable_row(i).begin());
    }
    auto m = batch_cosine_unit(q, x, 1);
    for (std::size_t i = 0; i < q.rows(); ++i)
        for (std::size_t j = 0; j < x.rows(); ++j)
            EXPECT_NEAR(m.at(i, j), cosine_unit(q.row(i), x.row(j)), 1e-12);

    auto mt = batch_cosine_unit(q, x, 2);
    for (std::size_t k = 0; k < m.value.size(); ++k)
        EXPECT_DOUBLE_EQ(m.value[k], mt.value[k]);  // parallel result identical
}

TEST(Similarity, WithinSetIsSymmetricWithUnitDiagonal) {
    EmbeddingMatrix x(6, 16);
    for (std::size_t i = 0; i < x.rows(); ++i) {
        auto v = test::random_unit_vector(16, 300 + i);
        std::copy(v.begin(), v.end(), x.mutable_row(i).begin());
    }
    auto s = within_set_similarity(x);
    for (std::size_t i = 0; i < 6; ++i) {
        EXPECT_DOUBLE_EQ(s.at(i, i), 1.0);
        for (std::size_t j = 0; j < 6; ++j) EXPECT_DOUBLE_EQ(s.at(i, j), s.at(j, i));
    }
}

TEST(Similarity, PairwiseUpperTriangleIndexing) {
    const std::size_t n = 4, d = 3;
    std::vector<double> flat(n * d);
    for (std::size_t i = 0; i < n * d; ++i) flat[i] = static_cast<double>(i);
    auto dist = pairwise_upper(flat, n, d);
    ASSERT_EQ(dist.size(), n * (n - 1) / 2);
    // index 0 is the (0,1) pair; the reference distance is sqrt of the squared L2
    EXPECT_NEAR(dist[0], std::sqrt(l2sq(std::span(flat).first(d), std::span(flat).subspan(d, d))),
                1e-12);
}

TEST(Similarity, NormalizeInplaceHandlesZero) {
    std::vector<double> z(4, 0.0);
    EXPECT_EQ(normalize_inplace(z), Status::numerical);
    std::vector<double> v{3.0, 4.0};
    EXPECT_EQ(normalize_inplace(v), Status::ok);
    EXPECT_NEAR(v[0], 0.6, 1e-15);
    EXPECT_NEAR(v[1], 0.8, 1e-15);
}
