// similarity.hpp — vector representation and similarity engine.
//
// Layout choice: embeddings are stored row-major and contiguous in a single
// allocation (EmbeddingMatrix). Each row is 64-byte aligned so AVX-512 loads do
// not straddle cache lines. The layout is benchmarked against a vector-of-vectors
// in benchmarks/core_bench.cpp (Results: see docs/benchmarks.md).
#pragma once

#include "ae/compat.hpp"

#include <memory>
#include <span>

namespace ae {

// Owned, aligned, immutable-after-construction matrix of row vectors.
class EmbeddingMatrix {
public:
    EmbeddingMatrix(std::size_t rows, std::size_t dim);

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t dim() const noexcept { return dim_; }
    [[nodiscard]] std::size_t stride() const noexcept { return stride_; }
    [[nodiscard]] std::span<const double> row(std::size_t i) const;
    [[nodiscard]] std::span<double> mutable_row(std::size_t i);
    [[nodiscard]] std::span<const double> flat() const noexcept {
        return {data_.get(), rows_ * stride_};
    }

    // Copy: deep copy, value semantics. Move: transfers the buffer.
    EmbeddingMatrix(const EmbeddingMatrix& other);
    EmbeddingMatrix& operator=(const EmbeddingMatrix& other);
    EmbeddingMatrix(EmbeddingMatrix&&) noexcept = default;
    EmbeddingMatrix& operator=(EmbeddingMatrix&&) noexcept = default;
    ~EmbeddingMatrix() = default;

    void fill_from_row_major(std::span<const double> src);  // rows*dim doubles
    void normalize_rows();
    [[nodiscard]] bool is_row_normalized(std::size_t i, double tol = 1e-9) const;

private:
    struct AlignedDeleter {
        void operator()(double* p) const noexcept;
    };
    std::size_t rows_{0};
    std::size_t dim_{0};
    std::size_t stride_{0};
    std::unique_ptr<double[], AlignedDeleter> data_{nullptr};
};

// ------------------------------------------------------------ similarity
// Unit-vector inner product. Prefers the C23 library (SIMD dispatched) when
// linked; the header-only fallback keeps the C++ core self-contained for tests.
[[nodiscard]] double cosine_unit(std::span<const double> a, std::span<const double> b);
[[nodiscard]] double cosine_raw(std::span<const double> a, std::span<const double> b);
[[nodiscard]] double l2sq(std::span<const double> a, std::span<const double> b);

struct SimilarityMatrix {
    std::size_t rows{0};
    std::size_t cols{0};
    std::vector<double> value;  // row-major rows x cols

    [[nodiscard]] double at(std::size_t i, std::size_t j) const { return value[i * cols + j]; }
};

// Full cross-similarity of Q (m x d) against X (n x d), rows assumed normalised.
[[nodiscard]] SimilarityMatrix batch_cosine_unit(const EmbeddingMatrix& q,
                                                 const EmbeddingMatrix& x,
                                                 unsigned threads = 0);

// Symmetric within-set similarity with a zero diagonal (used for diversity).
[[nodiscard]] SimilarityMatrix within_set_similarity(const EmbeddingMatrix& x,
                                                     unsigned threads = 0);

// Pairwise embedding distances as a flat upper triangle:
// element index of (i<j) is i*n - i(i+1)/2 + (j-i-1).
[[nodiscard]] std::vector<double> pairwise_upper(std::span<const double> flat_rows,
                                                 std::size_t n, std::size_t d);

// Normalise in place with numerically safe handling of the zero vector
// (returns Status::numerical instead of producing NaNs).
Status normalize_inplace(std::span<double> v);

// Default thread count: hardware concurrency, capped to avoid oversubscription
// on the 2-vCPU reference machine (measured in benchmarks/scaling.csv).
[[nodiscard]] unsigned recommended_threads(std::size_t work_items) noexcept;

}  // namespace ae
