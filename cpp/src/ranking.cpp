#include "ae/ranking.hpp"
#include "../include/ae/aenum_bridge.h"

#include <cmath>
#include <queue>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace ae {

const char* to_string(TopKStrategy s) noexcept {
    switch (s) {
    case TopKStrategy::sort_full: return "sort_full";
    case TopKStrategy::partial_sort: return "partial_sort";
    case TopKStrategy::nth_element: return "nth_element";
    case TopKStrategy::heap_bounded: return "heap_bounded";
    }
    return "unknown";
}

namespace {

// Descending by score, ascending by index: total order -> deterministic output.
struct BetterFirst {
    bool operator()(const RankedItem& a, const RankedItem& b) const noexcept {
        if (a.score != b.score) return a.score > b.score;
        return a.index < b.index;
    }
};

}  // namespace

std::vector<RankedItem> top_k(std::span<const double> scores, std::size_t k,
                              TopKStrategy strategy) {
    if (k == 0) throw std::invalid_argument("top_k: k must be > 0");
    if (k > scores.size()) throw std::invalid_argument("top_k: k exceeds population");
    const std::size_t n = scores.size();

    std::vector<RankedItem> out;
    out.reserve(k);

    switch (strategy) {
    case TopKStrategy::sort_full: {
        std::vector<RankedItem> all(n);
        for (std::size_t i = 0; i < n; ++i) all[i] = RankedItem{i, scores[i]};
        std::sort(all.begin(), all.end(), BetterFirst{});
        out.assign(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(k));
        break;
    }
    case TopKStrategy::partial_sort: {
        std::vector<RankedItem> all(n);
        for (std::size_t i = 0; i < n; ++i) all[i] = RankedItem{i, scores[i]};
        std::partial_sort(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(k), all.end(),
                          BetterFirst{});
        out.assign(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(k));
        break;
    }
    case TopKStrategy::nth_element: {
        std::vector<RankedItem> all(n);
        for (std::size_t i = 0; i < n; ++i) all[i] = RankedItem{i, scores[i]};
        std::nth_element(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(k - 1), all.end(),
                         BetterFirst{});
        out.assign(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(k));
        std::sort(out.begin(), out.end(), BetterFirst{});
        break;
    }
    case TopKStrategy::heap_bounded: {
        // min-heap of size k: the root is the current worst kept element
        std::priority_queue<RankedItem, std::vector<RankedItem>, BetterFirst> heap;
        for (std::size_t i = 0; i < n; ++i) {
            RankedItem item{i, scores[i]};
            if (heap.size() < k) {
                heap.push(item);
            } else if (BetterFirst{}(item, heap.top())) {
                heap.pop();
                heap.push(item);
            }
        }
        out.resize(heap.size());
        for (std::size_t i = out.size(); i-- > 0;) {
            out[i] = heap.top();
            heap.pop();
        }
        break;
    }
    }
    return out;
}

std::vector<std::vector<RankedItem>> top_k_rows(const SimilarityMatrix& scores, std::size_t k,
                                                TopKStrategy strategy, unsigned threads) {
    const std::size_t m = scores.rows;
    if (threads == 0) threads = recommended_threads(m);
    std::vector<std::vector<RankedItem>> out(m);
    // Rows are independent, so a deterministic static partition suffices and
    // avoids the busy-wait of a shared atomic work queue on 2 vCPUs.
    const unsigned t = static_cast<unsigned>(std::min<std::size_t>(threads, std::max<std::size_t>(m, 1)));
    if (t <= 1) {
        for (std::size_t r = 0; r < m; ++r)
            out[r] = top_k(std::span<const double>(scores.value.data() + r * scores.cols, scores.cols), k, strategy);
        return out;
    }
    std::vector<std::thread> pool;
    pool.reserve(t);
    const std::size_t chunk = (m + t - 1) / t;
    for (unsigned ti = 0; ti < t; ++ti) {
        const std::size_t begin = ti * chunk;
        const std::size_t end = std::min(m, begin + chunk);
        if (begin >= end) break;
        pool.emplace_back([&, begin, end] {
            for (std::size_t r = begin; r < end; ++r)
                out[r] = top_k(std::span<const double>(scores.value.data() + r * scores.cols, scores.cols),
                               k, strategy);
        });
    }
    for (auto& th : pool) th.join();
    return out;
}

std::vector<RankedItem> mmr_select(const EmbeddingMatrix& candidates,
                                   std::span<const double> relevance, const MmrParams& params) {
    const std::size_t n = candidates.rows();
    if (relevance.size() != n) throw std::invalid_argument("mmr_select: relevance size mismatch");
    const std::size_t k = std::min(params.top_k, n);
    if (k == 0) return {};

    std::vector<bool> chosen(n, false);
    std::vector<std::size_t> result;
    result.reserve(k);

    // seed with the single most relevant candidate
    std::size_t best = 0;
    for (std::size_t i = 1; i < n; ++i)
        if (relevance[i] > relevance[best]) best = i;
    chosen[best] = true;
    result.push_back(best);

    std::vector<double> max_sim(n, -1.0);  // cached max similarity to the selected set
    for (std::size_t i = 0; i < n; ++i) {
        double s = 0.0;
        ae_cosine_unit(candidates.row(i).data(), candidates.row(best).data(), candidates.dim(), &s);
        max_sim[i] = s;
    }

    while (result.size() < k) {
        std::size_t pick = n;
        double pick_score = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < n; ++i) {
            if (chosen[i]) continue;
            const double score = params.lambda * relevance[i] - (1.0 - params.lambda) * max_sim[i];
            if (score > pick_score || (score == pick_score && i < pick)) {
                pick_score = score;
                pick = i;
            }
        }
        if (pick == n) break;
        chosen[pick] = true;
        result.push_back(pick);
        for (std::size_t i = 0; i < n; ++i) {
            if (chosen[i]) continue;
            double s = 0.0;
            ae_cosine_unit(candidates.row(i).data(), candidates.row(pick).data(), candidates.dim(), &s);
            if (s > max_sim[i]) max_sim[i] = s;
        }
    }

    std::vector<RankedItem> out;
    out.reserve(result.size());
    for (std::size_t idx : result) out.push_back(RankedItem{idx, relevance[idx]});
    return out;
}

std::vector<RankedItem> top_k_capped(std::span<const double> scores,
                                     std::span<const std::int32_t> category, std::size_t k,
                                     std::size_t cap_per_category) {
    if (scores.size() != category.size())
        throw std::invalid_argument("top_k_capped: size mismatch");
    if (cap_per_category == 0) throw std::invalid_argument("top_k_capped: cap must be > 0");

    std::vector<std::size_t> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (scores[a] != scores[b]) return scores[a] > scores[b];
        return a < b;
    });

    std::unordered_map<std::int32_t, std::size_t> used;
    std::vector<RankedItem> out;
    out.reserve(std::min(k, scores.size()));
    for (std::size_t idx : order) {
        if (out.size() >= k) break;
        const std::int32_t c = category[idx];
        std::size_t& u = used[c];
        if (u >= cap_per_category) continue;
        ++u;
        out.push_back(RankedItem{idx, scores[idx]});
    }
    return out;
}

double ndcg_at_k(std::span<const double> ranked_relevance,
                 std::span<const double> ideal_relevance, std::size_t k, double gain_floor) {
    if (ranked_relevance.size() != ideal_relevance.size())
        throw std::invalid_argument("ndcg_at_k: size mismatch");
    const std::size_t n = std::min(k, ranked_relevance.size());
    if (n == 0) return 0.0;
    auto dcg = [&](std::span<const double> rel) {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double gain = std::exp2(rel[i]) - gain_floor;
            s += gain / std::log2(static_cast<double>(i) + 2.0);
        }
        return s;
    };
    const double ideal = dcg(ideal_relevance);
    if (ideal <= 0.0) return 0.0;
    return dcg(ranked_relevance) / ideal;
}

double kendall_tau_b(std::span<const std::size_t> a, std::span<const std::size_t> b) {
    if (a.size() != b.size()) throw std::invalid_argument("kendall_tau_b: size mismatch");
    const std::size_t n = a.size();
    if (n < 2) return 1.0;
    long long concordant = 0, discordant = 0, tie_a = 0, tie_b = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const bool ai = a[i] < a[j], bi = b[i] < b[j];
            if (a[i] == a[j]) { ++tie_a; continue; }
            if (b[i] == b[j]) { ++tie_b; continue; }
            if (ai == bi) ++concordant; else ++discordant;
        }
    }
    const double num = static_cast<double>(concordant - discordant);
    const double den = std::sqrt(static_cast<double>(concordant + discordant + tie_a) *
                                 static_cast<double>(concordant + discordant + tie_b));
    return den > 0.0 ? num / den : 0.0;
}

}  // namespace ae
