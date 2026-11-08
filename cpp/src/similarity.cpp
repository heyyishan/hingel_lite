#include "ae/similarity.hpp"
#include "../include/ae/aenum_bridge.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace ae {

// ------------------------------------------------------------ EmbeddingMatrix
void EmbeddingMatrix::AlignedDeleter::operator()(double* p) const noexcept {
    ae_aligned_free(p);
}

EmbeddingMatrix::EmbeddingMatrix(std::size_t rows, std::size_t dim)
    : rows_(rows), dim_(dim), stride_(round_up(dim, kSimdAlign / sizeof(double))) {
    if (rows == 0 || dim == 0) throw std::invalid_argument("EmbeddingMatrix: empty dimensions");
    const std::size_t bytes = rows_ * stride_ * sizeof(double);
    double* p = static_cast<double*>(ae_aligned_alloc(bytes, kSimdAlign));
    if (p == nullptr) throw std::bad_alloc();
    std::memset(p, 0, bytes);
    data_.reset(p);
}

EmbeddingMatrix::EmbeddingMatrix(const EmbeddingMatrix& other)
    : rows_(other.rows_), dim_(other.dim_), stride_(other.stride_) {
    const std::size_t bytes = rows_ * stride_ * sizeof(double);
    double* p = static_cast<double*>(ae_aligned_alloc(bytes, kSimdAlign));
    if (p == nullptr) throw std::bad_alloc();
    std::memcpy(p, other.data_.get(), bytes);
    data_.reset(p);
}

EmbeddingMatrix& EmbeddingMatrix::operator=(const EmbeddingMatrix& other) {
    if (this == &other) return *this;
    EmbeddingMatrix tmp(other);       // strong exception guarantee
    *this = std::move(tmp);
    return *this;
}

std::span<const double> EmbeddingMatrix::row(std::size_t i) const {
    if (i >= rows_) throw std::out_of_range("EmbeddingMatrix::row");
    return {data_.get() + i * stride_, dim_};
}

std::span<double> EmbeddingMatrix::mutable_row(std::size_t i) {
    if (i >= rows_) throw std::out_of_range("EmbeddingMatrix::mutable_row");
    return {data_.get() + i * stride_, dim_};
}

void EmbeddingMatrix::fill_from_row_major(std::span<const double> src) {
    if (src.size() < rows_ * dim_) throw std::invalid_argument("fill_from_row_major: short input");
    for (std::size_t r = 0; r < rows_; ++r) {
        std::memcpy(data_.get() + r * stride_, src.data() + r * dim_, dim_ * sizeof(double));
    }
}

Status normalize_inplace(std::span<double> v) {
    double nrm = 0.0;
    if (ae_vec_norm(v.data(), v.size(), &nrm) != AE_OK) return Status::invalid_argument;
    if (!(nrm > 0.0) || !std::isfinite(nrm)) return Status::numerical;
    const double inv = 1.0 / nrm;
    for (double& x : v) x *= inv;
    return Status::ok;
}

void EmbeddingMatrix::normalize_rows() {
    for (std::size_t r = 0; r < rows_; ++r) {
        if (normalize_inplace(mutable_row(r)) != Status::ok) {
            throw std::domain_error("EmbeddingMatrix: zero-norm row " + std::to_string(r));
        }
    }
}

bool EmbeddingMatrix::is_row_normalized(std::size_t i, double tol) const {
    double nrm = 0.0;
    ae_vec_norm(row(i).data(), dim_, &nrm);
    return std::abs(nrm - 1.0) <= tol;
}

// -------------------------------------------------------------- similarity
double cosine(std::span<const double> a, std::span<const double> b) {
    if (a.size() != b.size()) throw std::invalid_argument("cosine: dimension mismatch");
    if (a.empty()) return std::numeric_limits<double>::quiet_NaN();
    double out = 0.0;
    if (ae_cosine_raw(a.data(), b.data(), a.size(), &out) != AE_OK)
        return std::numeric_limits<double>::quiet_NaN();  // zero vector -> NaN, not 1.0
    return out;
}

double cosine_unit(std::span<const double> a, std::span<const double> b) {
    if (a.size() != b.size()) throw std::invalid_argument("cosine_unit: dimension mismatch");
    double out = 0.0;
    if (ae_cosine_unit(a.data(), b.data(), a.size(), &out) != AE_OK) return std::numeric_limits<double>::quiet_NaN();
    return out;
}

double cosine_raw(std::span<const double> a, std::span<const double> b) {
    if (a.size() != b.size()) throw std::invalid_argument("cosine_raw: dimension mismatch");
    double out = 0.0;
    if (ae_cosine_raw(a.data(), b.data(), a.size(), &out) != AE_OK) return std::numeric_limits<double>::quiet_NaN();
    return out;
}

double l2sq(std::span<const double> a, std::span<const double> b) {
    if (a.size() != b.size()) throw std::invalid_argument("l2sq: dimension mismatch");
    double out = 0.0;
    ae_l2sq(a.data(), b.data(), a.size(), &out);
    return out;
}

unsigned recommended_threads(std::size_t work_items) noexcept {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    // Do not spawn more threads than there are outer-loop iterations; below
    // ~64 rows the synchronisation cost dominates (measured in scaling.csv).
    if (work_items < 64) return 1;
    return std::min<unsigned>(hw, 8u);
}

namespace {

// Deterministic static partition of [0, n) across `threads` workers.
template <class Fn>
void parallel_for_rows(std::size_t n, unsigned threads, Fn&& fn) {
    if (threads <= 1 || n < 2) {
        for (std::size_t i = 0; i < n; ++i) fn(i);
        return;
    }
    threads = static_cast<unsigned>(std::min<std::size_t>(threads, n));
#if defined(AE_HAS_OPENMP)
    #pragma omp parallel for schedule(static) num_threads(static_cast<int>(threads))
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
        fn(static_cast<std::size_t>(i));
    }
#else
    std::vector<std::thread> pool;
    pool.reserve(threads);
    const std::size_t chunk = (n + threads - 1) / threads;
    for (unsigned t = 0; t < threads; ++t) {
        const std::size_t begin = t * chunk;
        const std::size_t end = std::min(n, begin + chunk);
        if (begin >= end) break;
        pool.emplace_back([&fn, begin, end] {
            for (std::size_t i = begin; i < end; ++i) fn(i);
        });
    }
    for (auto& th : pool) th.join();
#endif
}

}  // namespace

SimilarityMatrix batch_cosine_unit(const EmbeddingMatrix& q, const EmbeddingMatrix& x,
                                   unsigned threads) {
    if (q.dim() != x.dim()) throw std::invalid_argument("batch_cosine_unit: dim mismatch");
    SimilarityMatrix out{q.rows(), x.rows(), std::vector<double>(q.rows() * x.rows(), 0.0)};
    const std::size_t m = q.rows(), n = x.rows(), d = q.dim();
    if (threads == 0) threads = recommended_threads(m * n / 4);

    parallel_for_rows(m, threads, [&](std::size_t i) {
        const double* qi = q.row(i).data();
        double* dst = out.value.data() + i * n;
        // Rows of X are contiguous with stride x.stride(); the C23 kernel
        // assumes contiguity, so we call the dot product per row.
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            ae_cosine_unit(qi, x.row(j).data(), d, &s);
            dst[j] = s;
        }
    });
    return out;
}

SimilarityMatrix within_set_similarity(const EmbeddingMatrix& x, unsigned threads) {
    const std::size_t m = x.rows(), d = x.dim();
    SimilarityMatrix out{m, m, std::vector<double>(m * m, 0.0)};
    if (threads == 0) threads = recommended_threads(m);
    parallel_for_rows(m, threads, [&](std::size_t i) {
        const double* xi = x.row(i).data();
        for (std::size_t j = i + 1; j < m; ++j) {
            double s = 0.0;
            ae_cosine_unit(xi, x.row(j).data(), d, &s);
            out.value[i * m + j] = s;
            out.value[j * m + i] = s;
        }
        out.value[i * m + i] = 1.0;
    });
    return out;
}

std::vector<double> pairwise_upper(std::span<const double> flat_rows, std::size_t n, std::size_t d) {
    if (flat_rows.size() < n * d) throw std::invalid_argument("pairwise_upper: short buffer");
    const std::size_t m = n * (n - 1) / 2;
    std::vector<double> dist(m, 0.0);
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            double s = 0.0;
            ae_l2sq(flat_rows.data() + i * d, flat_rows.data() + j * d, d, &s);
            dist[k++] = std::sqrt(s);
        }
    }
    return dist;
}

}  // namespace ae
