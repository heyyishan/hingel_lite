// ranking.hpp — top-k retrieval, MMR diversity ranking, and the pairwise
// neural ranker's inference path.
//
// Four top-k strategies are implemented so the benchmark suite can compare them
// on equal footing (see benchmarks/core_bench.cpp):
//   1. sort_full        std::sort on (index, score)
//   2. partial_sort     std::partial_sort
//   3. nth_element      std::nth_element + sort of the head
//   4. heap_bounded     bounded min-heap, O(n log k), allocation-free
// Each returns indices with strictly decreasing scores; ties break by ascending
// index so results are deterministic and comparable across strategies.
#pragma once

#include "ae/compat.hpp"
#include "ae/similarity.hpp"

#include <utility>

namespace ae {

enum class TopKStrategy : std::uint8_t {
    sort_full = 0,
    partial_sort = 1,
    nth_element = 2,
    heap_bounded = 3,
};

struct RankedItem {
    std::size_t index{0};
    double score{0.0};
};

[[nodiscard]] const char* to_string(TopKStrategy s) noexcept;

// Deterministic top-k. Throws std::invalid_argument when k > scores.size().
[[nodiscard]] std::vector<RankedItem> top_k(std::span<const double> scores, std::size_t k,
                                            TopKStrategy strategy = TopKStrategy::heap_bounded);

// Row-parallel top-k over a score matrix (queries x candidates); each row is
// ranked independently. Used by the ANN candidate stage and by compat reranking.
[[nodiscard]] std::vector<std::vector<RankedItem>> top_k_rows(
    const SimilarityMatrix& scores, std::size_t k,
    TopKStrategy strategy = TopKStrategy::heap_bounded, unsigned threads = 0);

// ------------------------------------------------------- MMR diversification
struct MmrParams {
    double lambda{0.7};      // relevance weight; (1-lambda) is the diversity weight
    std::size_t top_k{10};
};

// Maximal Marginal Relevance (Carbonell & Goldstein, 1998):
//   MMR(d) = lambda * sim(d, q) - (1 - lambda) * max_{s in S} sim(d, s)
// where S is the set already selected. Greedy, O(k * n * d).
[[nodiscard]] std::vector<RankedItem> mmr_select(const EmbeddingMatrix& candidates,
                                                 std::span<const double> relevance,
                                                 const MmrParams& params);

// ------------------------------------------------- rare-category constraints
// Deterministic constrained selection: at most `cap` items per category, still
// ordering by score. Used for "don't show five nearly identical scenes".
[[nodiscard]] std::vector<RankedItem> top_k_capped(
    std::span<const double> scores, std::span<const std::int32_t> category,
    std::size_t k, std::size_t cap_per_category);

// ---------------------------------------------------------- fairness/utility
// NDCG@k against binary or graded relevance. Both arrays must be aligned and
// indexed by position; returns the discounted-gain ratio in [0, 1].
[[nodiscard]] double ndcg_at_k(std::span<const double> ranked_relevance,
                               std::span<const double> ideal_relevance, std::size_t k,
                               double gain_floor = 1.0);

// Kendall tau-b between two consecutive-quality permutations, used to check how
// stable the optimizer's output is under small objective perturbations.
[[nodiscard]] double kendall_tau_b(std::span<const std::size_t> a,
                                   std::span<const std::size_t> b);

}  // namespace ae
