"""Contract tests for the pybind11 surface.

These are the checks that the Python layer relies on; several of them existed only
as assumptions before this file was written (PhotoProblem/Strategy/OptimizeOptions,
the whole stats.bayesian namespace, mmr_select/ndcg_at_k).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.native


@pytest.fixture(scope="module")
def core(native_core):
    return native_core


def test_build_info_reports_real_values(core):
    info = core.build_info()
    assert info["compiler"] in {"gcc", "clang", "msvc"}
    assert int(info["compiler_version"].split(".")[0]) >= 11
    assert info["has_openmp"] in (True, False)
    cpu = core.cpu_info()
    assert cpu["target"] in {"scalar", "avx2", "avx512f"}
    assert 0 < cpu["mask"] < 256


def test_cosine_matches_definition(core):
    a = np.array([1.0, 2.0, 2.0])
    b = np.array([2.0, 1.0, 0.0])
    want = float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert core.cosine(a, b) == pytest.approx(want, rel=1e-12)
    # cosine_raw is the same quantity via the C23 raw path.
    assert core.cosine_raw(a, b) == pytest.approx(want, rel=1e-12)
    # cosine_unit is a pure inner product: identical only for unit input.
    an = a / np.linalg.norm(a)
    bn = b / np.linalg.norm(b)
    assert core.cosine_unit(an, bn) == pytest.approx(want, rel=1e-12)
    assert core.cosine_unit(a, b) != pytest.approx(want, rel=1e-6)


def test_cosine_zero_vector_is_nan_not_one(core):
    # Documented contract: an undefined cosine is NaN. Returning 1.0 here (the
    # pre-fix behaviour) would have silently scored unrelated photos as identical.
    assert math.isnan(core.cosine(np.zeros(8), np.ones(8)))
    # The C23 raw path returns the same NaN rather than an error code; only the
    # normalise helper rejects a zero vector outright.
    assert math.isnan(core.cosine_raw(np.zeros(8), np.ones(8)))


def test_batch_cosine_matches_eigen_and_scalar(core):
    rng = np.random.default_rng(5)
    q = rng.normal(size=(3, 16))
    x = rng.normal(size=(11, 16))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    ref = q @ x.T
    for fn in ("batch_cosine_unit", "batch_cosine_eigen"):
        got = np.asarray(getattr(core, fn)(np.ascontiguousarray(q), np.ascontiguousarray(x), 1))
        assert got.shape == (3, 11)
        np.testing.assert_allclose(got, ref, atol=1e-12, err_msg=fn)


def test_batch_cosine_rejects_dim_mismatch(core):
    with pytest.raises(Exception):
        core.batch_cosine_eigen(np.zeros((2, 8)), np.zeros((2, 9)), 0)


@pytest.mark.parametrize("strategy", ["sort_full", "partial_sort", "nth_element", "heap_bounded"])
def test_top_k_agrees_with_numpy(core, strategy):
    rng = np.random.default_rng(7)
    scores = rng.random(5000)
    ranked = list(core.top_k(scores, 12, getattr(core.TopKStrategy, strategy)))
    got = [r.index for r in ranked]
    want = list(np.argsort(-scores)[:12])
    np.testing.assert_allclose(scores[got], scores[want])
    np.testing.assert_allclose([r.score for r in ranked], scores[want])


def test_top_k_returns_indices_sorted_by_score(core):
    rng = np.random.default_rng(8)
    scores = rng.random(1000)
    ranked = list(core.top_k(scores, 20, core.TopKStrategy.heap_bounded))
    idx = np.asarray([r.index for r in ranked])
    assert np.all(np.diff(scores[idx]) <= 0)
    assert len(set(idx.tolist())) == 20


def test_ndcg_at_k_hand_values(core):
    rel = [3.0, 2.0, 1.0, 0.0]
    # DCG  = 3/log2(2) + 2/log2(3) + 1/log2(4) = 3 + 1.26186 + 0.5
    dcg = 3.0 + 2.0 / math.log2(3) + 1.0 / math.log2(4)
    ideal = dcg
    assert core.ndcg_at_k(rel, rel, 4) == pytest.approx(dcg / ideal, rel=1e-12)
    assert core.ndcg_at_k([0.0, 0.0, 0.0, 0.0], rel, 4) == 0.0
    assert core.ndcg_at_k([3.0, 2.0, 1.0, 0.0], rel, 0) == 0.0


def test_ndcg_perfect_ranking_is_one(core):
    rel = [2.0, 1.5, 1.0, 0.5, 0.1]
    assert core.ndcg_at_k(rel, rel, 5) == pytest.approx(1.0, rel=1e-12)


def test_mmr_select_balances_relevance_and_diversity(core):
    # Two clusters: items 0..4 near e1, items 5..9 near e2. Relevance favours the
    # first cluster only; MMR with a small lambda must still cover both.
    rng = np.random.default_rng(9)
    base = np.zeros((10, 16))
    base[:5] = np.eye(16)[0]
    base[5:] = np.eye(16)[1]
    cand = base + 0.01 * rng.normal(size=base.shape)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    rel = np.array([1.0] * 5 + [0.6] * 5)
    def ids(lam):
        return np.asarray([r.index for r in
                           core.mmr_select(np.ascontiguousarray(cand), rel, lam, 4)])

    chosen = ids(0.3)
    assert len(chosen) == 4 and len(set(chosen.tolist())) == 4
    assert (chosen >= 5).sum() >= 1, "diversity term never selected the second cluster"
    pure_relevance = ids(0.999)
    assert (pure_relevance >= 5).sum() <= (chosen >= 5).sum()
    assert set(pure_relevance.tolist()) <= {0, 1, 2, 3, 4}


def test_prob_superiority_matches_numeric_integral(core):
    from scipy import stats

    a = core.BetaParams(14.0, 9.0)
    b = core.BetaParams(9.0, 14.0)
    xs = np.linspace(1e-6, 1 - 1e-6, 200001)
    ref = float(np.trapezoid(stats.beta.pdf(xs, 14, 9) * stats.beta.cdf(xs, 9, 14), xs))
    assert core.prob_superiority(a, b) == pytest.approx(ref, abs=2e-4)
    assert core.prob_superiority(a, a) == pytest.approx(0.5, abs=1e-9)


def test_credible_interval_matches_scipy(core):
    from scipy import stats

    p = core.BetaParams(41.0, 161.0)
    iv = core.credible_interval(p, 0.95)
    assert iv.lo == pytest.approx(stats.beta.ppf(0.025, 41, 161), rel=1e-6)
    assert iv.hi == pytest.approx(stats.beta.ppf(0.975, 41, 161), rel=1e-6)
    narrower = core.credible_interval(p, 0.5)
    assert (narrower.hi - narrower.lo) < (iv.hi - iv.lo)


def test_posterior_from_counts(core):
    # Uniform prior (1, 1) + 40 successes / 60 failures -> Beta(41, 61).
    p = core.beta_posterior(1.0, 1.0, 40, 60)
    assert p.alpha == pytest.approx(41.0)
    assert p.beta == pytest.approx(61.0)
    # A non-uniform prior must be used, not ignored.
    q = core.beta_posterior(2.0, 5.0, 40, 60)
    assert q.alpha == pytest.approx(42.0) and q.beta == pytest.approx(65.0)


def test_expected_loss_prefers_better_arm(core):
    good = core.BetaParams(alpha=40.0, beta=10.0)
    bad = core.BetaParams(alpha=10.0, beta=40.0)
    assert core.expected_loss(good, bad) < core.expected_loss(bad, good)


def test_required_sample_size_monotone(core):
    small = core.required_sample_size(0.40, 0.45)
    large = core.required_sample_size(0.40, 0.42)
    assert 0 < small < large
    assert small == pytest.approx(1531, rel=0.02)


def test_thompson_allocation_is_normalised(core):
    arms = [core.BetaParams(10.0, 10.0), core.BetaParams(20.0, 5.0)]
    alloc = np.asarray(core.thompson_allocation(arms, 2000, 123))
    assert alloc.shape == (2,)
    assert alloc.sum() == 2000
    assert alloc[1] > alloc[0], "the arm with the higher posterior mean should win"
    # Determinism: the seed fixes the allocation.
    again = np.asarray(core.thompson_allocation(arms, 2000, 123))
    np.testing.assert_array_equal(alloc, again)


def test_photo_optimizer_binding_end_to_end(core):
    rng = np.random.default_rng(11)
    n = 5
    problem = core.PhotoProblem(n)
    problem.quality = list(rng.random(n))
    problem.slot_affinity = list(rng.random(n))
    w = [1.0 / (i + 1) for i in range(n)]
    problem.position_weight = [x / sum(w) for x in w]
    problem.slot_target = list(np.linspace(1.0, 0.0, n))
    problem.distance = list(rng.random(n * n))
    opts = core.OptimizeOptions()
    opts.strategy = core.Strategy.exhaustive
    best = core.optimize_ordering(problem, opts)
    assert sorted(best.permutation) == list(range(n))
    breakdown = core.evaluate_ordering(problem, best.permutation)
    assert breakdown.total == pytest.approx(best.score.total, abs=1e-9)
    # The reported breakdown must reconstruct the scored objective
    # J = quality + lambda*D + mu*C with D = 1 - R/sum(kappa) already normalised.
    manual = (breakdown.quality + problem.lambda_ * breakdown.diversity
              + problem.mu * breakdown.coherence)
    assert manual == pytest.approx(breakdown.total, abs=1e-9)
    assert 0.0 <= breakdown.diversity <= 1.0 + 1e-12
    assert 0.0 <= breakdown.coherence <= 1.0 + 1e-12
    # Nothing may beat the exhaustive optimum.
    for perm in (list(reversed(best.permutation)), list(np.roll(best.permutation, 1))):
        assert core.evaluate_ordering(problem, perm).total <= best.score.total + 1e-9


def test_behavioural_sensitive_similarity_and_ablation(core):
    rng = np.random.default_rng(12)
    n = 4
    problem = core.PhotoProblem(n)
    problem.quality = list(rng.random(n))
    problem.slot_affinity = list(rng.random(n))
    problem.position_weight = [0.25] * n
    problem.slot_target = [1.0, 0.7, 0.4, 0.1]
    problem.distance = list(rng.random(n * n))
    opts = core.OptimizeOptions()
    opts.strategy = core.Strategy.exhaustive
    lambdas, mus = [0.0, 0.2, 0.4], [0.0, 0.1]
    sens = core.sensitivity(problem, opts, lambdas, mus)
    # One row per (lambda, mu) grid cell.
    assert len(sens) == len(lambdas) * len(mus)
    for row in sens:
        assert math.isfinite(row.optimized_total)
        assert math.isfinite(row.fixed_perm_total)
    # Re-weighting can only ever change the ranking of permutations, so the
    # optimum under each grid point must be >= the fixed permutation's score.
    for row in sens:
        assert row.optimized_total >= row.fixed_perm_total - 1e-9
    abl = core.ablate_leverage(problem, opts)
    assert len(abl) >= 1
    assert all(math.isfinite(a.loss) for a in abl)
