// compat_matcher.hpp — pairwise compatibility scoring and reciprocal ranking.
//
// Baseline (transparent, interpretable):
//   S(i) = alpha * V(i) + beta * T(i) + gamma * A(i) + delta * P(i)
// where V is visual similarity, T textual similarity, A structured-preference
// agreement (Jaccard/cosine over declared attributes) and P shared declared
// interests. All four terms live in [0,1] and the weights are normalised to sum
// to 1 so S is comparable between users.
//
// Neural ranker (inference only — training lives in Python/torch):
//   P(Y_ij = 1) = sigmoid( MLP([u_i, u_j, |u_i - u_j|, u_i (.) u_j]) )
// The network is trained with a *symmetric* head: the pair features are made
// order-invariant (both orderings are averaged) because compatibility is a
// reciprocal notion. The Python trainer enforces the same symmetry by
// construction, and the C++ path reproduces it exactly for the tests.
#pragma once

#include "ae/compat.hpp"
#include "ae/similarity.hpp"

namespace ae {

struct CompatWeights {
    double visual{0.25};      // alpha
    double textual{0.30};     // beta
    double attributes{0.25};  // gamma
    double interests{0.20};   // delta

    [[nodiscard]] double normalised_total() const {
        const double s = visual + textual + attributes + interests;
        return s > 0.0 ? s : 1.0;
    }
};

struct CompatTerms {
    double visual{0};      // V(i)
    double textual{0};     // T(i)
    double attributes{0};  // A(i)
    double interests{0};   // P(i)
    double score{0};       // S(i)
};

struct CandidateProfile {
    std::size_t index{0};
    std::vector<double> visual;      // embedding (unit norm)
    std::vector<double> textual;     // embedding (unit norm)
    std::vector<double> attributes;  // structured prefs in [0,1]
    std::vector<double> interests;   // interest vector in [0,1]
    bool cold_start{false};          // true when either modality is missing
};

[[nodiscard]] CompatTerms score_pair(const CandidateProfile& viewer,
                                     const CandidateProfile& candidate,
                                     const CompatWeights& w);

// Reciprocal compatibility: visibility of the candidate as seen by the viewer
// combined with how the viewer looks from the candidate's side. With a
// symmetric model the two coincide; the API keeps the asymmetry explicit
// because user-supplied preference weights need not be symmetric.
struct ReciprocalScore {
    std::size_t index{0};
    double forward{0};    // S(viewer -> candidate)
    double backward{0};   // S(candidate -> viewer)
    double harmonic{0};   // 2ab/(a+b), which penalises one-sided matches
    double confidence{0}; // reduced when either side is cold-start
    CompatTerms terms_forward;
};

[[nodiscard]] std::vector<ReciprocalScore> rank_candidates(
    const CandidateProfile& viewer, std::span<const CandidateProfile> candidates,
    const CompatWeights& w, std::size_t top_k, double lambda_diversity = 0.0,
    unsigned threads = 0);

// ----------------------------------------------------------- neural ranker
// A fixed-topology MLP with GELU activations. Weights are loaded from an
// exported JSON produced by python/attraction_engine/ml/pair_ranker.py; when no
// weights are supplied `is_trained()` is false and callers MUST fall back to the
// transparent baseline (the API labels such scores as untrained_model).
class PairRanker {
public:
    // feature_dim: dimension of the concatenated pair feature vector
    //   [u_i (d) , u_j (d) , |u_i-u_j| (d) , u_i (.) u_j (d)]  -> 4d
    struct Layer {
        std::size_t in{0};
        std::size_t out{0};
        std::vector<double> weight;  // out x in, row-major
        std::vector<double> bias;    // out
    };

    PairRanker() = default;
    explicit PairRanker(std::vector<Layer> layers);

    [[nodiscard]] bool is_trained() const noexcept { return !layers_.empty(); }
    [[nodiscard]] std::size_t feature_dim() const noexcept { return input_dim_; }

    // Build the order-invariant pair feature vector described in the header.
    [[nodiscard]] static std::vector<double> pair_features(std::span<const double> ui,
                                                           std::span<const double> uj);

    [[nodiscard]] double predict(std::span<const double> ui,
                                 std::span<const double> uj) const;

    // Symmetric prediction: average of both orderings. For a network trained
    // with the symmetric head this equals predict(ui,uj); the average guards
    // against asymmetry introduced by a hand-written weight export.
    [[nodiscard]] double predict_symmetric(std::span<const double> ui,
                                           std::span<const double> uj) const;

    // Serialisation: simple deterministic text format (one layer per block) so
    // the Python exporter and this loader cannot silently disagree.
    [[nodiscard]] std::string serialise() const;
    static PairRanker deserialise(std::string_view text);  // throws on bad input

private:
    [[nodiscard]] double forward(std::vector<double> x) const;
    std::vector<Layer> layers_;
    std::size_t input_dim_{0};
};

}  // namespace ae
