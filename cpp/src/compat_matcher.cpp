#include "ae/compat_matcher.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace ae {

namespace {

double cosine_or_zero(std::span<const double> a, std::span<const double> b) {
    if (a.empty() || b.empty() || a.size() != b.size()) return 0.0;
    const double c = cosine_raw(a, b);
    return std::isfinite(c) ? c : 0.0;
}

// Map cosine [-1,1] to a similarity in [0,1]; a negative correlation is no
// worse than "unrelated" for scoring purposes, so it clamps to zero.
double to_unit(double cosine) { return std::clamp((cosine + 1.0) * 0.5, 0.0, 1.0); }

// Agreement over structured attribute vectors: mean of per-dimension
// 1 - |a-b| (raised-cosine-like trapezoid for ordinal encodings).
double attribute_agreement(std::span<const double> a, std::span<const double> b) {
    const std::size_t n = std::min(a.size(), b.size());
    if (n == 0) return 0.0;
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += 1.0 - std::min(std::abs(a[i] - b[i]), 1.0);
    return s / static_cast<double>(n);
}

// Interest agreement: cosine on non-negative sparse interest vectors, which is
// equivalent to the Ochiai coefficient for binary vectors.
double interest_overlap(std::span<const double> a, std::span<const double> b) {
    if (a.empty() || b.empty() || a.size() != b.size()) return 0.0;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na <= 0.0 || nb <= 0.0) return 0.0;
    return std::clamp(dot / std::sqrt(na * nb), 0.0, 1.0);
}

double gelu(double x) {
    // tanh approximation of the GELU activation (Hendrycks & Gimpel, 2016)
    constexpr double kC = 0.7978845608028654;  // sqrt(2/pi)
    return 0.5 * x * (1.0 + std::tanh(kC * (x + 0.044715 * x * x * x)));
}

}  // namespace

CompatTerms score_pair(const CandidateProfile& viewer, const CandidateProfile& candidate,
                       const CompatWeights& w) {
    CompatTerms t;
    t.visual = to_unit(cosine_or_zero(viewer.visual, candidate.visual));
    t.textual = to_unit(cosine_or_zero(viewer.textual, candidate.textual));
    t.attributes = attribute_agreement(viewer.attributes, candidate.attributes);
    t.interests = interest_overlap(viewer.interests, candidate.interests);
    const double z = w.normalised_total();
    t.score = (w.visual * t.visual + w.textual * t.textual +
               w.attributes * t.attributes + w.interests * t.interests) / z;
    return t;
}

std::vector<ReciprocalScore> rank_candidates(const CandidateProfile& viewer,
                                             std::span<const CandidateProfile> candidates,
                                             const CompatWeights& w, std::size_t top_k,
                                             double lambda_diversity, unsigned threads) {
    std::vector<ReciprocalScore> out(candidates.size());
    const std::size_t n = candidates.size();

    auto work = [&](std::size_t i) {
        const CandidateProfile& c = candidates[i];
        ReciprocalScore r;
        r.index = c.index;
        r.terms_forward = score_pair(viewer, c, w);
        r.forward = r.terms_forward.score;
        // Reverse direction uses the candidate's preference weights, which for
        // this build are the same structure but evaluated with roles swapped.
        r.backward = score_pair(c, viewer, w).score;
        r.harmonic = (r.forward + r.backward) > 0.0
                         ? 2.0 * r.forward * r.backward / (r.forward + r.backward)
                         : 0.0;
        double conf = viewer.cold_start || c.cold_start ? 0.55 : 0.95;
        if (c.visual.empty()) conf *= 0.8;
        if (c.textual.empty()) conf *= 0.8;
        r.confidence = conf;
        out[i] = r;
    };

    if (threads <= 1 || n < 64) {
        for (std::size_t i = 0; i < n; ++i) work(i);
    } else {
        threads = static_cast<unsigned>(std::min<std::size_t>(threads, n));
        std::vector<std::thread> pool;
        pool.reserve(threads);
        const std::size_t chunk = (n + threads - 1) / threads;
        for (unsigned t = 0; t < threads; ++t) {
            const std::size_t begin = t * chunk, end = std::min(n, begin + chunk);
            if (begin >= end) break;
            pool.emplace_back([&work, begin, end] {
                for (std::size_t i = begin; i < end; ++i) work(i);
            });
        }
        for (auto& th : pool) th.join();
    }

    std::sort(out.begin(), out.end(), [](const ReciprocalScore& a, const ReciprocalScore& b) {
        if (a.harmonic != b.harmonic) return a.harmonic > b.harmonic;
        return a.index < b.index;
    });

    if (lambda_diversity > 0.0 && out.size() > 1) {
        // Greedy MMR over the candidates' visual embeddings, reusing the
        // harmonic score as relevance:
        //   MMR(c) = lambda*rel(c) - (1-lambda)*max_{s in S} cos(c, s)
        // `out` is already sorted by relevance, so the first pick is the leader.
        std::unordered_map<std::size_t, std::size_t> pos_by_index;
        for (std::size_t i = 0; i < candidates.size(); ++i) pos_by_index[candidates[i].index] = i;

        std::vector<ReciprocalScore> pool(out.begin(), out.end());
        std::vector<ReciprocalScore> picked;
        const std::size_t want = std::min(top_k, out.size());
        picked.reserve(want);
        picked.push_back(pool.front());
        pool.erase(pool.begin());

        std::vector<double> max_sim(pool.size(), -1.0);
        for (std::size_t i = 0; i < pool.size(); ++i) {
            const auto& ci = candidates[pos_by_index[pool[i].index]].visual;
            const auto& cs = candidates[pos_by_index[picked.front().index]].visual;
            max_sim[i] = to_unit(cosine_or_zero(ci, cs));
        }
        while (picked.size() < want && !pool.empty()) {
            std::size_t best = pool.size();
            double best_v = -std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < pool.size(); ++i) {
                const double v = lambda_diversity * pool[i].harmonic -
                                 (1.0 - lambda_diversity) * max_sim[i];
                if (v > best_v) { best_v = v; best = i; }
            }
            if (best == pool.size()) break;
            const auto chosen = pool[best];
            pool.erase(pool.begin() + static_cast<std::ptrdiff_t>(best));
            max_sim.erase(max_sim.begin() + static_cast<std::ptrdiff_t>(best));
            picked.push_back(chosen);
            const auto& cv = candidates[pos_by_index[chosen.index]].visual;
            for (std::size_t i = 0; i < pool.size(); ++i) {
                const auto& ci = candidates[pos_by_index[pool[i].index]].visual;
                const double sim = to_unit(cosine_or_zero(ci, cv));
                if (sim > max_sim[i]) max_sim[i] = sim;
            }
        }
        out = std::move(picked);
    }

    if (out.size() > top_k) out.resize(top_k);
    return out;
}

// ------------------------------------------------------------- PairRanker
PairRanker::PairRanker(std::vector<Layer> layers) : layers_(std::move(layers)) {
    if (!layers_.empty()) {
        input_dim_ = layers_.front().in;
        for (std::size_t i = 0; i < layers_.size(); ++i) {
            const Layer& l = layers_[i];
            if (l.weight.size() != l.in * l.out || l.bias.size() != l.out)
                throw std::invalid_argument("PairRanker: malformed layer");
            if (i > 0 && layers_[i - 1].out != l.in)
                throw std::invalid_argument("PairRanker: layer dimension chain broken");
        }
        if (layers_.back().out != 1)
            throw std::invalid_argument("PairRanker: output layer must be scalar");
    }
}

std::vector<double> PairRanker::pair_features(std::span<const double> ui,
                                              std::span<const double> uj) {
    if (ui.size() != uj.size() || ui.empty())
        throw std::invalid_argument("pair_features: dimension mismatch");
    const std::size_t d = ui.size();
    std::vector<double> f(4 * d);
    for (std::size_t k = 0; k < d; ++k) {
        f[k] = ui[k];
        f[d + k] = uj[k];
        f[2 * d + k] = std::abs(ui[k] - uj[k]);
        f[3 * d + k] = ui[k] * uj[k];
    }
    return f;
}

double PairRanker::forward(std::vector<double> x) const {
    for (const Layer& l : layers_) {
        std::vector<double> y(l.out, 0.0);
        for (std::size_t o = 0; o < l.out; ++o) {
            double acc = l.bias[o];
            const double* w = l.weight.data() + o * l.in;
            for (std::size_t i = 0; i < l.in; ++i) acc += w[i] * x[i];
            y[o] = acc;
        }
        // hidden layers use GELU, the final scalar uses the logistic link
        const bool last = (&l == &layers_.back());
        for (double& v : y) v = last ? 1.0 / (1.0 + std::exp(-v)) : gelu(v);
        x = std::move(y);
    }
    return x.empty() ? 0.5 : x[0];
}

double PairRanker::predict(std::span<const double> ui, std::span<const double> uj) const {
    if (!is_trained()) return 0.5;
    return forward(pair_features(ui, uj));
}

double PairRanker::predict_symmetric(std::span<const double> ui,
                                     std::span<const double> uj) const {
    if (!is_trained()) return 0.5;
    const double a = forward(pair_features(ui, uj));
    const double b = forward(pair_features(uj, ui));
    return 0.5 * (a + b);
}

std::string PairRanker::serialise() const {
    std::ostringstream os;
    os << "AE_PAIRRANKER v1\n";
    os << "layers " << layers_.size() << "\n";
    for (const Layer& l : layers_) {
        os << "layer " << l.in << " " << l.out << "\n";
        for (double v : l.weight) os << v << "\n";
        for (double v : l.bias) os << v << "\n";
    }
    return os.str();
}

PairRanker PairRanker::deserialise(std::string_view text) {
    std::istringstream is{std::string(text)};
    std::string magic, version;
    is >> magic >> version;
    if (magic != "AE_PAIRRANKER" || version != "v1")
        throw std::invalid_argument("PairRanker::deserialise: bad header");
    std::size_t n_layers = 0;
    is >> magic >> n_layers;
    if (magic != "layers") throw std::invalid_argument("PairRanker::deserialise: bad layers");
    std::vector<Layer> layers;
    layers.reserve(n_layers);
    for (std::size_t i = 0; i < n_layers; ++i) {
        std::size_t in = 0, out = 0;
        is >> magic >> in >> out;
        if (magic != "layer") throw std::invalid_argument("PairRanker::deserialise: bad layer");
        Layer l{in, out, std::vector<double>(in * out), std::vector<double>(out)};
        for (double& v : l.weight) { if (!(is >> v)) throw std::invalid_argument("short weights"); }
        for (double& v : l.bias) { if (!(is >> v)) throw std::invalid_argument("short bias"); }
        layers.push_back(std::move(l));
    }
    return PairRanker(std::move(layers));
}

}  // namespace ae
