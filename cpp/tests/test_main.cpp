// test_main.cpp — shared helpers for the C++ test suite.
#include "test_helpers.hpp"

#include <random>

namespace ae::test {

std::vector<double> random_unit_vector(std::size_t d, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> v(d);
    double nrm = 0.0;
    for (double& x : v) { x = nd(rng); nrm += x * x; }
    nrm = std::sqrt(nrm);
    for (double& x : v) x /= nrm;
    return v;
}

PhotoProblem make_problem(std::size_t n, double lambda, double mu, std::uint64_t seed) {
    PhotoProblem p;
    p.n = n;
    p.lambda = lambda;
    p.mu = mu;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    p.quality.resize(n);
    p.slot_affinity.resize(n);
    p.position_weight.resize(n);
    p.slot_target.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        p.quality[i] = u(rng);
        p.slot_affinity[i] = u(rng);
        p.position_weight[i] = 1.0 / (1.0 + static_cast<double>(i) * 0.35);
        p.slot_target[i] = 1.0 - static_cast<double>(i) / static_cast<double>(n);
    }
    const double sumw = std::accumulate(p.position_weight.begin(), p.position_weight.end(), 0.0);
    for (double& w : p.position_weight) w /= sumw;

    // distances between synthetic 16-d embeddings
    std::vector<std::vector<double>> emb(n, std::vector<double>(16));
    for (auto& e : emb) for (double& x : e) x = u(rng);
    p.distance.assign(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < 16; ++k) { const double d = emb[i][k] - emb[j][k]; s += d * d; }
            p.distance[i * n + j] = std::sqrt(s);
        }
    return p;
}

}  // namespace ae::test
