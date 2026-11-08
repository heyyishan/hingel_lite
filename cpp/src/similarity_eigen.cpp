// similarity_eigen.cpp — blocked GEMM path for full cross-similarity.
//
// Measured (docs/benchmarks.md): the per-row dot-product loop is 1-3x slower than
// NumPy's BLAS dgemm for a 64-row query batch, because it is memory-bound and does
// not block the inner products. Eigen (already a dependency) closes most of the
// gap in-process without pulling in a BLAS, so the API uses this path for matrix
// work and keeps the SIMD-dispatched scalar kernel for single pairs and for the
// in-house speedup measurement. Where Eigen still loses to BLAS, the benchmark
// says so.
#include "ae/similarity.hpp"

// Eigen's own kernels only vectorise to the level the translation unit was
// compiled for. Without this pragma the TU targets SSE2 and the library runs its
// scalar 2x2 kernels. We bump *this file only* to the host ISA and keep the
// runtime dispatch of the in-house kernels independent of it; performance.md
// separately records the numbers against the SSE2 build.
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#pragma GCC target("avx2,fma")
#endif
#include <Eigen/Dense>

namespace ae {

SimilarityMatrix batch_cosine_eigen(const EmbeddingMatrix& q, const EmbeddingMatrix& x,
                                    unsigned threads) {
    (void)threads;  // Eigen's own threading is left at its default.
    if (q.dim() != x.dim()) throw std::invalid_argument("batch_cosine_eigen: dim mismatch");
    const std::size_t m = q.rows(), n = x.rows(), d = q.dim();
    using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    // Map the caller's storage instead of copying: EmbeddingMatrix is row-major with
    // an outer stride of rows()*stride(). An earlier version copied into a
    // column-major temp and the alloc+transpose dominated the runtime (100 MB per
    // call at n=32768), which made Eigen look 7x worse than it is.
    Eigen::Map<const RowMajor, 0, Eigen::OuterStride<>> Q(
        q.row(0).data(), static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(d),
        Eigen::OuterStride<>(static_cast<Eigen::Index>(q.stride())));
    Eigen::Map<const RowMajor, 0, Eigen::OuterStride<>> X(
        x.row(0).data(), static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(d),
        Eigen::OuterStride<>(static_cast<Eigen::Index>(x.stride())));
    // NOTE: measured on this machine (docs/benchmarks.md) Eigen's default 8x8x8
    // packing does not reach the AVX-512 kernels for these shapes and the blocking
    // overhead dominates; noalias() into a preallocated buffer is the configuration
    // the benchmark numbers reflect.
    Eigen::MatrixXd S(m, n);
    S.noalias() = Q * X.transpose();
    // Eigen is column-major; SimilarityMatrix::value is row-major. A raw data()
    // copy silently transposes the result (caught by
    // Similarity.EigenBatchMatchesScalarPath), so write through operator().
    SimilarityMatrix out{m, n, std::vector<double>(m * n, 0.0)};
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            out.value[i * n + j] = S(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
    return out;
}

}  // namespace ae
