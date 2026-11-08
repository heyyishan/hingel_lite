#pragma once
#include "ae/optimizer.hpp"
#include "ae/similarity.hpp"

#include <cstdint>
#include <vector>

namespace ae::test {
std::vector<double> random_unit_vector(std::size_t d, std::uint64_t seed);
PhotoProblem make_problem(std::size_t n, double lambda = 0.35, double mu = 0.20,
                          std::uint64_t seed = 7);
}
