// ae_bindings.cpp — pybind11 surface for the C++26 core and the C23 library.
//
// Contract: numpy arrays are accepted as float64 C-contiguous 2-D/1-D arrays;
// every returned container is a plain Python object (list/dict/float) so the
// API layer never depends on pybind11's ndarray semantics.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "aenum.h"

#include "ae/capacity.hpp"
#include "ae/compat.hpp"
#include "ae/compat_matcher.hpp"
#include "ae/optimizer.hpp"
#include "ae/ranking.hpp"
#include "ae/similarity.hpp"
#include "ae/stats.hpp"

#include <chrono>
#include <sstream>

namespace py = pybind11;
using namespace ae;

namespace {

using Matrix = py::array_t<double, py::array::c_style | py::array::forcecast>;

EmbeddingMatrix to_embedding_matrix(const Matrix& a) {
    if (a.ndim() != 2) throw std::invalid_argument("expected a 2-D array");
    const ssize_t rows = a.shape(0), dim = a.shape(1);
    EmbeddingMatrix m(static_cast<std::size_t>(rows), static_cast<std::size_t>(dim));
    m.fill_from_row_major({a.data(), static_cast<std::size_t>(rows * dim)});
    return m;
}

std::vector<double> to_vector(const py::array_t<double, py::array::c_style | py::array::forcecast>& a) {
    auto buf = a.request();
    const double* p = static_cast<const double*>(buf.ptr);
    return std::vector<double>(p, p + buf.size);
}

py::array_t<double> to_array(const std::vector<double>& v) {
    return py::array_t<double>(v.size(), v.data());
}

// ------------------------------------------------------------------ module
void bind_similarity(py::module_& m) {
    py::class_<EmbeddingMatrix>(m, "EmbeddingMatrix")
        .def(py::init<std::size_t, std::size_t>())
        .def_property_readonly("rows", &EmbeddingMatrix::rows)
        .def_property_readonly("dim", &EmbeddingMatrix::dim)
        .def_property_readonly("stride", &EmbeddingMatrix::stride)
        .def("set_row", [](EmbeddingMatrix& self, std::size_t i, const py::array_t<double>& v) {
            auto vals = to_vector(v);
            auto row = self.mutable_row(i);
            if (vals.size() != row.size()) throw std::invalid_argument("row size mismatch");
            std::copy(vals.begin(), vals.end(), row.begin());
        })
        .def("normalize_rows", &EmbeddingMatrix::normalize_rows)
        .def("to_numpy", [](const EmbeddingMatrix& self) {
            py::array_t<double> out({self.rows(), self.dim()});
            auto buf = out.request();
            double* dst = static_cast<double*>(buf.ptr);
            for (std::size_t r = 0; r < self.rows(); ++r)
                std::copy(self.row(r).begin(), self.row(r).end(), dst + r * self.dim());
            return out;
        });

    m.def("cosine", [](const Matrix& a, const Matrix& b) {
        const auto va = to_vector(a), vb = to_vector(b);
        return cosine(va, vb);
    }, py::arg("a"), py::arg("b"),
       "Cosine similarity of two arbitrary vectors (normalises internally). "
       "Returns NaN for a zero vector.");
    m.def("cosine_unit", [](const Matrix& a, const Matrix& b) {
        const auto va = to_vector(a), vb = to_vector(b);
        return cosine_unit(va, vb);
    }, py::arg("a"), py::arg("b"),
       "Inner product of two UNIT vectors. Do not pass raw vectors: use cosine().");
    m.def("cosine_raw", [](const py::array_t<double>& a, const py::array_t<double>& b) {
        const auto va = to_vector(a), vb = to_vector(b);
        return cosine_raw(va, vb);
    });
    m.def("batch_cosine_unit", [](const Matrix& q, const Matrix& x, unsigned threads) {
        auto mq = to_embedding_matrix(q), mx = to_embedding_matrix(x);
        auto s = batch_cosine_unit(mq, mx, threads);
        return py::array_t<double>({s.rows, s.cols}, s.value.data());
    }, py::arg("q"), py::arg("x"), py::arg("threads") = 0,
       "Cross similarity for UNIT-normalised rows. Rows of q and x must have unit "
       "L2 norm; batch_cosine_eigen normalises nothing either.");
    m.def("batch_cosine", [](const Matrix& q, const Matrix& x, unsigned threads) {
        auto mq = to_embedding_matrix(q), mx = to_embedding_matrix(x);
        auto s = batch_cosine_eigen(mq, mx, threads);
        return py::array_t<double>({s.rows, s.cols}, s.value.data());
    }, py::arg("q"), py::arg("x"), py::arg("threads") = 0);
    m.def("batch_cosine_eigen", [](const Matrix& q, const Matrix& x, unsigned threads) {
        auto mq = to_embedding_matrix(q), mx = to_embedding_matrix(x);
        auto s = batch_cosine_eigen(mq, mx, threads);
        return py::array_t<double>({s.rows, s.cols}, s.value.data());
    }, py::arg("q"), py::arg("x"), py::arg("threads") = 0);
    m.def("within_set_similarity", [](const Matrix& x, unsigned threads) {
        auto mx = to_embedding_matrix(x);
        auto s = within_set_similarity(mx, threads);
        return py::array_t<double>({s.rows, s.cols}, s.value.data());
    }, py::arg("x"), py::arg("threads") = 0);
    m.def("pairwise_upper", [](const Matrix& x) {
        auto mx = to_embedding_matrix(x);
        std::vector<double> flat(mx.rows() * mx.dim());
        for (std::size_t r = 0; r < mx.rows(); ++r)
            std::copy(mx.row(r).begin(), mx.row(r).end(), flat.begin() + r * mx.dim());
        return to_array(pairwise_upper(flat, mx.rows(), mx.dim()));
    });
    m.def("normalize_rows_numpy", [](py::array_t<double> x) {
        auto buf = x.request();
        const ssize_t rows = buf.shape[0], dim = buf.shape[1];
        EmbeddingMatrix em(static_cast<std::size_t>(rows), static_cast<std::size_t>(dim));
        em.fill_from_row_major({static_cast<double*>(buf.ptr), static_cast<std::size_t>(rows * dim)});
        em.normalize_rows();
        py::array_t<double> out({rows, dim});
        auto ob = out.request();
        double* dst = static_cast<double*>(ob.ptr);
        for (ssize_t r = 0; r < rows; ++r)
            std::copy(em.row(r).begin(), em.row(r).end(), dst + r * dim);
        return out;
    });
}

void bind_ranking(py::module_& m) {
    py::enum_<TopKStrategy>(m, "TopKStrategy")
        .value("sort_full", TopKStrategy::sort_full)
        .value("partial_sort", TopKStrategy::partial_sort)
        .value("nth_element", TopKStrategy::nth_element)
        .value("heap_bounded", TopKStrategy::heap_bounded);

    py::class_<RankedItem>(m, "RankedItem")
        .def_readonly("index", &RankedItem::index)
        .def_readonly("score", &RankedItem::score)
        .def("__repr__", [](const RankedItem& r) {
            std::ostringstream os;
            os << "RankedItem(index=" << r.index << ", score=" << r.score << ")";
            return os.str();
        });

    m.def("top_k", [](const py::array_t<double>& scores, std::size_t k, TopKStrategy strat) {
        auto v = to_vector(scores);
        return top_k(v, k, strat);
    }, py::arg("scores"), py::arg("k"), py::arg("strategy") = TopKStrategy::heap_bounded);

    m.def("mmr_select", [](const Matrix& cand, const py::array_t<double>& relevance,
                           double lambda, std::size_t k) {
        auto mx = to_embedding_matrix(cand);
        auto rel = to_vector(relevance);
        return mmr_select(mx, rel, MmrParams{lambda, k});
    }, py::arg("candidates"), py::arg("relevance"), py::arg("lambda") = 0.7, py::arg("k") = 10);

    m.def("top_k_capped", [](const py::array_t<double>& scores, const py::array_t<std::int32_t>& cat,
                             std::size_t k, std::size_t cap) {
        auto s = to_vector(scores);
        auto buf = cat.request();
        const std::int32_t* cp = static_cast<const std::int32_t*>(buf.ptr);
        std::vector<std::int32_t> c(cp, cp + buf.size);
        return top_k_capped(s, c, k, cap);
    });

    m.def("ndcg_at_k", [](const py::array_t<double>& ranked, const py::array_t<double>& ideal,
                          std::size_t k) {
        auto r = to_vector(ranked), i = to_vector(ideal);
        return ndcg_at_k(r, i, k);
    });
    m.def("kendall_tau_b", [](const std::vector<std::size_t>& a, const std::vector<std::size_t>& b) {
        return kendall_tau_b(a, b);
    });
}

void bind_optimizer(py::module_& m) {
    py::enum_<Strategy>(m, "Strategy")
        .value("greedy", Strategy::greedy)
        .value("beam", Strategy::beam)
        .value("simulated_annealing", Strategy::simulated_annealing)
        .value("exhaustive", Strategy::exhaustive)
        .value("random_restart", Strategy::random_restart);

    py::class_<PhotoProblem>(m, "PhotoProblem")
        .def(py::init<>())
        .def_readwrite("quality", &PhotoProblem::quality)
        .def_readwrite("slot_affinity", &PhotoProblem::slot_affinity)
        .def_readwrite("position_weight", &PhotoProblem::position_weight)
        .def_readwrite("slot_target", &PhotoProblem::slot_target)
        .def_readwrite("distance", &PhotoProblem::distance)
        .def_readwrite("lambda_", &PhotoProblem::lambda)
        .def_readwrite("mu", &PhotoProblem::mu)
        .def_readwrite("positional_kernel", &PhotoProblem::positional_kernel)
        // `n` is settable so callers can size the problem before assigning the
        // per-photo vectors; every list is resized to match.
        .def_property("n",
                      [](const PhotoProblem& p) { return p.n; },
                      [](PhotoProblem& p, std::size_t n) {
                          p.n = n;
                          p.quality.resize(n, 0.0);
                          p.slot_affinity.resize(n, 0.0);
                          p.position_weight.resize(n, n ? 1.0 / static_cast<double>(n) : 0.0);
                          p.slot_target.resize(n, 0.0);
                          p.distance.assign(n * n, 0.0);
                      })
        .def(py::init([](std::size_t n) {
                 PhotoProblem p;
                 p.n = n;
                 p.quality.resize(n, 0.0);
                 p.slot_affinity.resize(n, 0.0);
                 p.position_weight.assign(n, n ? 1.0 / static_cast<double>(n) : 0.0);
                 p.slot_target.resize(n, 0.0);
                 p.distance.assign(n * n, 0.0);
                 return p;
             }), py::arg("n"))
        .def("set_distance_matrix", [](PhotoProblem& p, const Matrix& d) {
            auto buf = d.request();
            if (buf.ndim != 2 || buf.shape[0] != buf.shape[1])
                throw std::invalid_argument("distance matrix must be square");
            p.n = static_cast<std::size_t>(buf.shape[0]);
            const double* ptr = static_cast<const double*>(buf.ptr);
            p.distance.assign(ptr, ptr + p.n * p.n);
        });

    py::class_<ObjectiveBreakdown>(m, "ObjectiveBreakdown")
        .def_readonly("quality", &ObjectiveBreakdown::quality)
        .def_readonly("redundancy", &ObjectiveBreakdown::redundancy)
        .def_readonly("diversity", &ObjectiveBreakdown::diversity)
        .def_readonly("coherence", &ObjectiveBreakdown::coherence)
        .def_readonly("total", &ObjectiveBreakdown::total);

    py::class_<OptimizeOptions>(m, "OptimizeOptions")
        .def(py::init<>())
        .def_readwrite("strategy", &OptimizeOptions::strategy)
        .def_readwrite("beam_width", &OptimizeOptions::beam_width)
        .def_readwrite("polish", &OptimizeOptions::polish)
        .def_readwrite("exhaustive_limit", &OptimizeOptions::exhaustive_limit)
        .def_readwrite("sa_initial_temp", &OptimizeOptions::sa_initial_temp)
        .def_readwrite("sa_cooling", &OptimizeOptions::sa_cooling)
        .def_readwrite("sa_iterations", &OptimizeOptions::sa_iterations)
        .def_readwrite("restarts", &OptimizeOptions::restarts)
        .def_readwrite("seed", &OptimizeOptions::seed);

    py::class_<OptimizeResult>(m, "OptimizeResult")
        .def_readonly("permutation", &OptimizeResult::permutation)
        .def_readonly("score", &OptimizeResult::score)
        .def_readonly("evaluations", &OptimizeResult::evaluations)
        .def_readonly("elapsed_ms", &OptimizeResult::elapsed_ms)
        .def_readonly("strategy", &OptimizeResult::strategy)
        .def_readonly("proved_optimal", &OptimizeResult::proved_optimal);

    py::class_<SensitivityPoint>(m, "SensitivityPoint")
        .def_readonly("lambda_", &SensitivityPoint::lambda)
        .def_readonly("mu", &SensitivityPoint::mu)
        .def_readonly("fixed_perm_total", &SensitivityPoint::fixed_perm_total)
        .def_readonly("optimized_total", &SensitivityPoint::optimized_total)
        .def_readonly("optimized_permutation", &SensitivityPoint::optimized_permutation)
        .def_readonly("position_changes", &SensitivityPoint::position_changes);

    py::class_<AblationRow>(m, "AblationRow")
        .def_readonly("removed_index", &AblationRow::removed_index)
        .def_readonly("reduced_total", &AblationRow::reduced_total)
        .def_readonly("baseline_total", &AblationRow::baseline_total)
        .def_readonly("best_total_with", &AblationRow::best_total_with)
        .def_readonly("loss", &AblationRow::loss)
        .def_readonly("best_permutation_with", &AblationRow::best_permutation_with);

    m.def("evaluate_ordering", [](const PhotoProblem& p, const std::vector<std::size_t>& perm) {
        return evaluate(p, perm);
    });
    m.def("optimize_ordering", &optimize);
    m.def("exhaustive_search", &exhaustive, py::arg("problem"), py::arg("max_n") = 9);
    m.def("sensitivity", [](const PhotoProblem& p, const OptimizeOptions& o,
                            const std::vector<double>& lams, const std::vector<double>& mus) {
        return sensitivity(p, o, lams, mus);
    });
    m.def("ablate_leverage", &ablate_leverage);
}

void bind_stats(py::module_& m) {
    py::class_<BetaParams>(m, "BetaParams")
        .def(py::init<>())
        .def(py::init<double, double>(), py::arg("alpha"), py::arg("beta"))
        .def_readwrite("alpha", &BetaParams::alpha)
        .def_readwrite("beta", &BetaParams::beta)
        .def_property_readonly("mean", &BetaParams::mean)
        .def_property_readonly("variance", &BetaParams::variance)
        .def_property_readonly("mode", &BetaParams::mode);

    py::class_<Interval>(m, "Interval")
        .def_readonly("lo", &Interval::lo)
        .def_readonly("hi", &Interval::hi)
        .def_readonly("level", &Interval::level)
        .def_property_readonly("width", &Interval::width);

    m.def("beta_posterior", &beta_posterior, py::arg("prior_alpha") = 1.0,
          py::arg("prior_beta") = 1.0, py::arg("successes") = 0, py::arg("failures") = 0);
    m.def("beta_pdf", &beta_pdf);
    m.def("beta_cdf", &beta_cdf);
    m.def("beta_quantile", &beta_quantile);
    m.def("credible_interval", &credible_interval, py::arg("params"), py::arg("level") = 0.95);
    m.def("prob_superiority", &prob_superiority);
    m.def("expected_loss", &expected_loss);
    m.def("thompson_select", &thompson_select);
    m.def("thompson_allocation", &thompson_allocation);
    m.def("sequential_decision", &sequential_decision, py::arg("a"), py::arg("b"),
          py::arg("n_a"), py::arg("n_b"), py::arg("min_observations"),
          py::arg("loss_tolerance") = 0.005);
    m.def("bonferroni_level", &bonferroni_level);
    m.def("sidak_level", &sidak_level);

    py::class_<SequentialTestResult>(m, "SequentialTestResult")
        .def_readonly("prob_superiority", &SequentialTestResult::prob_superiority)
        .def_readonly("expected_loss_choose_a", &SequentialTestResult::expected_loss_choose_a)
        .def_readonly("expected_loss_choose_b", &SequentialTestResult::expected_loss_choose_b)
        .def_readonly("stop", &SequentialTestResult::stop)
        .def_readonly("reason", &SequentialTestResult::reason);

    py::class_<HierarchicalFit>(m, "HierarchicalFit")
        .def_readonly("population_alpha", &HierarchicalFit::population_alpha)
        .def_readonly("population_beta", &HierarchicalFit::population_beta)
        .def_readonly("overdispersion", &HierarchicalFit::overdispersion)
        .def_readonly("posteriors", &HierarchicalFit::posteriors);
    m.def("fit_hierarchical", &fit_hierarchical, py::arg("successes"), py::arg("failures"),
          py::arg("prior_alpha") = 1.0, py::arg("prior_beta") = 1.0, py::arg("shrink") = 0.5);

    m.def("conversion_rate", [](std::uint64_t s, std::uint64_t n) {
        auto v = conversion_rate(s, n);
        if (!v) return py::object(py::none());
        return py::object(py::float_(*v));
    }, "Returns None when the denominator was never observed.");

    py::class_<CapacityPlan>(m, "CapacityPlan")
        .def_readonly("observed_rate_per_day", &CapacityPlan::observed_rate_per_day)
        .def_readonly("planned_days", &CapacityPlan::planned_days)
        .def_readonly("current_n", &CapacityPlan::current_n)
        .def_readonly("projected_n", &CapacityPlan::projected_n)
        .def_readonly("observations_needed", &CapacityPlan::observations_needed)
        .def_readonly("achievable", &CapacityPlan::achievable)
        .def_readonly("note", &CapacityPlan::note);
    m.def("required_sample_size", &required_sample_size, py::arg("p1"), py::arg("p2"),
          py::arg("alpha") = 0.05, py::arg("power") = 0.8);
    m.def("plan_capacity", &plan_capacity);
    m.def("normal_quantile", &normal_quantile);
}

void bind_compat(py::module_& m) {
    py::class_<CompatWeights>(m, "CompatWeights")
        .def(py::init<>())
        .def_readwrite("visual", &CompatWeights::visual)
        .def_readwrite("textual", &CompatWeights::textual)
        .def_readwrite("attributes", &CompatWeights::attributes)
        .def_readwrite("interests", &CompatWeights::interests);

    py::class_<CompatTerms>(m, "CompatTerms")
        .def_readonly("visual", &CompatTerms::visual)
        .def_readonly("textual", &CompatTerms::textual)
        .def_readonly("attributes", &CompatTerms::attributes)
        .def_readonly("interests", &CompatTerms::interests)
        .def_readonly("score", &CompatTerms::score);

    py::class_<CandidateProfile>(m, "CandidateProfile")
        .def(py::init<>())
        .def_readwrite("index", &CandidateProfile::index)
        .def_readwrite("visual", &CandidateProfile::visual)
        .def_readwrite("textual", &CandidateProfile::textual)
        .def_readwrite("attributes", &CandidateProfile::attributes)
        .def_readwrite("interests", &CandidateProfile::interests)
        .def_readwrite("cold_start", &CandidateProfile::cold_start);

    py::class_<ReciprocalScore>(m, "ReciprocalScore")
        .def_readonly("index", &ReciprocalScore::index)
        .def_readonly("forward", &ReciprocalScore::forward)
        .def_readonly("backward", &ReciprocalScore::backward)
        .def_readonly("harmonic", &ReciprocalScore::harmonic)
        .def_readonly("confidence", &ReciprocalScore::confidence)
        .def_readonly("terms_forward", &ReciprocalScore::terms_forward);

    m.def("score_pair", &score_pair);
    m.def("rank_candidates", &rank_candidates, py::arg("viewer"), py::arg("candidates"),
          py::arg("weights"), py::arg("top_k"), py::arg("lambda_diversity") = 0.0,
          py::arg("threads") = 0);

    py::class_<PairRanker> ranker(m, "PairRanker");
    ranker.def(py::init<>())
        .def_static("from_serialised", &PairRanker::deserialise)
        .def_property_readonly("is_trained", &PairRanker::is_trained)
        .def_property_readonly("feature_dim", &PairRanker::feature_dim)
        .def("predict", [](const PairRanker& r, const py::array_t<double>& a,
                           const py::array_t<double>& b) {
            auto va = to_vector(a), vb = to_vector(b);
            return r.predict(va, vb);
        })
        .def("predict_symmetric", [](const PairRanker& r, const py::array_t<double>& a,
                                     const py::array_t<double>& b) {
            auto va = to_vector(a), vb = to_vector(b);
            return r.predict_symmetric(va, vb);
        })
        .def("serialise", &PairRanker::serialise);
    ranker.def_static("pair_features", [](const py::array_t<double>& a, const py::array_t<double>& b) {
        auto va = to_vector(a), vb = to_vector(b);
        return to_array(PairRanker::pair_features(va, vb));
    });
}

void bind_system(py::module_& m) {
    m.def("build_info", [] {
        auto b = build_info();
        py::dict d;
        d["cxx_dialect"] = b.cxx_dialect;
        d["has_expected"] = b.has_expected;
        d["has_print"] = b.has_print;
        d["has_format"] = b.has_format;
        d["has_openmp"] = b.has_openmp;
        d["compiler"] = b.compiler;
        d["compiler_version"] = b.compiler_version;
        return d;
    });
    m.def("cpu_info", [] {
        py::dict d;
        d["target"] = ae_cpu_target_name();
        d["mask"] = ae_cpu_features();
        return d;
    });
    m.def("set_dispatch_scalar", [](bool scalar) {
        ae_cpu_set_policy(scalar ? AE_DISPATCH_SCALAR : AE_DISPATCH_AUTO);
    });
}

}  // namespace

PYBIND11_MODULE(ae_core, m) {
    m.doc() = "ATTRACTION ENGINE computational core (C++26 + C23)";
    m.attr("__version__") = "1.0.0";
    bind_system(m);
    bind_similarity(m);
    bind_ranking(m);
    bind_optimizer(m);
    bind_stats(m);
    bind_compat(m);
}
