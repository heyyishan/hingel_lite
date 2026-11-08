// core_bench.cpp — Google Benchmark suite for the C++26 / C23 core.
//
// Every measurement here is produced by running this binary; the JSON emitted by
// scripts/run_benchmarks.sh is what the README and docs/benchmarks.md quote.
// Baseline discipline: Python comparisons use NumPy on the same machine and the
// same data, not a hand-written slow loop.
#include <benchmark/benchmark.h>

#include "ae/compat.hpp"
#include "ae/optimizer.hpp"
#include "ae/ranking.hpp"
#include "ae/similarity.hpp"
#include "ae/stats.hpp"

#include "aenum.h"

#include <numeric>
#include <random>
#include <vector>

extern "C" {
void ae_dot_scalar(const double*, const double*, size_t, double*);
#if defined(__x86_64__)
void ae_dot_avx2(const double*, const double*, size_t, double*);
void ae_dot_avx512(const double*, const double*, size_t, double*);
#endif
}

namespace {

std::vector<double> random_scores(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<double> v(n);
    for (double& x : v) x = u(rng);
    return v;
}

ae::EmbeddingMatrix random_matrix(std::size_t rows, std::size_t dim, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    ae::EmbeddingMatrix m(rows, dim);
    for (std::size_t r = 0; r < rows; ++r) {
        auto row = m.mutable_row(r);
        double nrm = 0.0;
        for (double& x : row) { x = nd(rng); nrm += x * x; }
        nrm = std::sqrt(nrm);
        for (double& x : row) x /= nrm;
    }
    return m;
}

ae::PhotoProblem random_problem(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    ae::PhotoProblem p;
    p.n = n;
    p.quality.resize(n);
    p.slot_affinity.resize(n);
    p.position_weight.resize(n);
    p.slot_target.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        p.quality[i] = u(rng);
        p.slot_affinity[i] = u(rng);
        p.position_weight[i] = 1.0 / (1.0 + 0.35 * static_cast<double>(i));
        p.slot_target[i] = 1.0 - static_cast<double>(i) / static_cast<double>(n);
    }
    const double sum = std::accumulate(p.position_weight.begin(), p.position_weight.end(), 0.0);
    for (double& w : p.position_weight) w /= sum;
    std::vector<std::vector<double>> emb(n, std::vector<double>(32));
    for (auto& e : emb) for (double& x : e) x = u(rng);
    p.distance.assign(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < 32; ++k) { const double d = emb[i][k] - emb[j][k]; s += d * d; }
            p.distance[i * n + j] = std::sqrt(s);
        }
    return p;
}

}  // namespace

// ------------------------------------------------------- similarity kernels
static void BM_DotProduct(benchmark::State& state) {
    const std::size_t dim = static_cast<std::size_t>(state.range(0));
    auto m = random_matrix(1, dim, 7);
    auto a = std::vector<double>(m.row(0).begin(), m.row(0).end());
    auto b = a;
    double sink = 0.0;
    for (auto _ : state) {
        double out = 0.0;
        ae_vec_dot(a.data(), b.data(), dim, &out);
        sink += out;
        benchmark::DoNotOptimize(sink);
    }
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * dim * 16));
}
BENCHMARK(BM_DotProduct)->Arg(128)->Arg(384)->Arg(512)->Arg(768)->Arg(1536);

// Scalar vs AVX2 vs AVX-512 on identical data (dispatch forced per kernel).
#if defined(__x86_64__)
static void BM_DotScalar(benchmark::State& state) {
    const std::size_t dim = static_cast<std::size_t>(state.range(0));
    auto m = random_matrix(1, dim, 11);
    auto a = std::vector<double>(m.row(0).begin(), m.row(0).end());
    double sink = 0.0;
    for (auto _ : state) {
        double out = 0.0;
        ae_dot_scalar(a.data(), a.data(), dim, &out);
        sink += out;
        benchmark::DoNotOptimize(sink);
    }
}
static void BM_DotAvx2(benchmark::State& state) {
    const std::size_t dim = static_cast<std::size_t>(state.range(0));
    auto m = random_matrix(1, dim, 11);
    auto a = std::vector<double>(m.row(0).begin(), m.row(0).end());
    double sink = 0.0;
    for (auto _ : state) {
        double out = 0.0;
        ae_dot_avx2(a.data(), a.data(), dim, &out);
        sink += out;
        benchmark::DoNotOptimize(sink);
    }
}
static void BM_DotAvx512(benchmark::State& state) {
    const std::size_t dim = static_cast<std::size_t>(state.range(0));
    auto m = random_matrix(1, dim, 11);
    auto a = std::vector<double>(m.row(0).begin(), m.row(0).end());
    double sink = 0.0;
    for (auto _ : state) {
        double out = 0.0;
        ae_dot_avx512(a.data(), a.data(), dim, &out);
        sink += out;
        benchmark::DoNotOptimize(sink);
    }
}
BENCHMARK(BM_DotScalar)->Arg(1536);
BENCHMARK(BM_DotAvx2)->Arg(1536);
BENCHMARK(BM_DotAvx512)->Arg(1536);
#endif

// ------------------------------------------------------------ batch cosine
static void BM_BatchCosine(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const unsigned threads = static_cast<unsigned>(state.range(1));
    auto q = random_matrix(64, 384, 3);
    auto x = random_matrix(n, 384, 4);
    // Distinct inputs + a sink that escapes: without this the call is pure in its
    // arguments and can be hoisted out of the timing loop entirely.
    benchmark::DoNotOptimize(q);
    benchmark::DoNotOptimize(x);
    for (auto _ : state) {
        auto s = ae::batch_cosine_unit(q, x, threads);
        benchmark::DoNotOptimize(s.value.data());
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * 64 * n));
}
BENCHMARK(BM_BatchCosine)->Args({512, 1})->Args({512, 2})->Args({4096, 1})->Args({4096, 2})
    ->Args({32768, 1})->Args({32768, 2});

static void BM_BatchCosineEigen(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const unsigned threads = static_cast<unsigned>(state.range(1));
    auto q = random_matrix(64, 384, 3);
    auto x = random_matrix(n, 384, 4);
    benchmark::DoNotOptimize(q);
    benchmark::DoNotOptimize(x);
    for (auto _ : state) {
        auto s = ae::batch_cosine_eigen(q, x, threads);
        benchmark::DoNotOptimize(s.value.data());
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * 64 * n));
}
BENCHMARK(BM_BatchCosineEigen)->Args({512, 1})->Args({512, 2})->Args({4096, 1})->Args({4096, 2})
    ->Args({32768, 1})->Args({32768, 2});

// ---------------------------------------------------------------- top-k
template <ae::TopKStrategy S>
static void BM_TopK(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const std::size_t k = static_cast<std::size_t>(state.range(1));
    auto scores = random_scores(n, 99);
    for (auto _ : state) {
        auto r = ae::top_k(scores, k, S);
        benchmark::DoNotOptimize(r.data());
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * n));
}
BENCHMARK_TEMPLATE(BM_TopK, ae::TopKStrategy::sort_full)
    ->Args({4096, 10})->Args({65536, 10})->Args({65536, 100})->Args({1048576, 10});
BENCHMARK_TEMPLATE(BM_TopK, ae::TopKStrategy::partial_sort)
    ->Args({4096, 10})->Args({65536, 10})->Args({65536, 100})->Args({1048576, 10});
BENCHMARK_TEMPLATE(BM_TopK, ae::TopKStrategy::nth_element)
    ->Args({4096, 10})->Args({65536, 10})->Args({65536, 100})->Args({1048576, 10});
BENCHMARK_TEMPLATE(BM_TopK, ae::TopKStrategy::heap_bounded)
    ->Args({4096, 10})->Args({65536, 10})->Args({65536, 100})->Args({1048576, 10});

// ------------------------------------------------------------- optimizer
template <ae::Strategy S>
static void BM_Optimize(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    auto p = random_problem(n, 5);
    ae::OptimizeOptions o;
    o.strategy = S;
    o.beam_width = 8;
    o.sa_iterations = 20000;
    for (auto _ : state) {
        auto r = ae::optimize(p, o);
        benchmark::DoNotOptimize(r.permutation.data());
    }
}
BENCHMARK_TEMPLATE(BM_Optimize, ae::Strategy::greedy)->Arg(8)->Arg(16)->Arg(32);
BENCHMARK_TEMPLATE(BM_Optimize, ae::Strategy::beam)->Arg(8)->Arg(16)->Arg(32);
BENCHMARK_TEMPLATE(BM_Optimize, ae::Strategy::simulated_annealing)->Arg(8)->Arg(16);
BENCHMARK_TEMPLATE(BM_Optimize, ae::Strategy::exhaustive)->Arg(6)->Arg(8);

// ------------------------------------------------------------- MMR ranking
static void BM_Mmr(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    auto cand = random_matrix(n, 384, 21);
    auto rel = random_scores(n, 22);
    for (auto _ : state) {
        auto r = ae::mmr_select(cand, rel, ae::MmrParams{0.7, 20});
        benchmark::DoNotOptimize(r.data());
    }
}
BENCHMARK(BM_Mmr)->Arg(200)->Arg(2000);

// --------------------------------------------------- Beta posterior maths
static void BM_ProbSuperiority(benchmark::State& state) {
    ae::BetaParams a{40.0, 160.0}, b{60.0, 140.0};
    for (auto _ : state) {
        double p = ae::prob_superiority(a, b);
        benchmark::DoNotOptimize(p);
    }
}
BENCHMARK(BM_ProbSuperiority);

static void BM_CredibleInterval(benchmark::State& state) {
    ae::BetaParams a{41.0, 161.0};
    for (auto _ : state) {
        auto iv = ae::credible_interval(a, 0.95);
        benchmark::DoNotOptimize(iv.lo);
    }
}
BENCHMARK(BM_CredibleInterval);

BENCHMARK_MAIN();
