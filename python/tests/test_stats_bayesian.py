"""Module D: Beta-Binomial experimentation maths.

Assertions are either algebraic identities that must hold exactly (conjugacy,
credible-interval mass) or properties checked against an independent
implementation (SciPy) or Monte-Carlo with a stated tolerance.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from attraction_engine.stats import bayesian

PRIOR = (2.0, 8.0)   # the "weak_centred" prior the API defaults to


def arms(*pairs):
    return [bayesian.Variant(id=f"v{i}", label=f"V{i}", successes=s, failures=f,
                             impressions_missing=0) for i, (s, f) in enumerate(pairs)]


def test_posterior_is_the_conjugate_update():
    v = arms((40, 60))[0]
    post = bayesian.posterior(v, PRIOR)
    assert post.alpha == pytest.approx(PRIOR[0] + 40)
    assert post.beta == pytest.approx(PRIOR[1] + 60)
    assert post.mean == pytest.approx((PRIOR[0] + 40) / (PRIOR[0] + PRIOR[1] + 100))


def test_sequential_updates_equal_the_batch_update():
    big = bayesian.posterior(arms((50, 65))[0], PRIOR)
    step = bayesian.posterior(arms((10, 5))[0], (big.alpha, big.beta))
    # Feeding the first batch as a prior then the rest must reach the same place.
    first = bayesian.posterior(arms((40, 60))[0], PRIOR)
    second = bayesian.posterior(arms((10, 5))[0], (first.alpha, first.beta))
    assert second.alpha == pytest.approx(big.alpha)
    assert second.beta == pytest.approx(big.beta)
    assert step.alpha > PRIOR[0]


def test_credible_interval_brackets_the_requested_mass():
    out = bayesian.analyse(arms((40, 60), (60, 140)), prior_name="weak_centred")
    for v in out["variants"]:
        p = v["posterior"]
        ci = v["credible_interval"]
        mass = stats.beta.cdf(ci["hi"], p["alpha"], p["beta"]) - stats.beta.cdf(
            ci["lo"], p["alpha"], p["beta"])
        assert mass == pytest.approx(0.95, abs=1e-9)
        assert ci["lo"] < p["mean"] < ci["hi"]
        assert ci["width"] == pytest.approx(ci["hi"] - ci["lo"], rel=1e-12)


def test_prob_superiority_matches_monte_carlo():
    out = bayesian.analyse(arms((40, 60), (60, 140)), prior_name="weak_centred")
    pair = out["pairwise"][0]
    a, b = out["variants"][0]["posterior"], out["variants"][1]["posterior"]
    rng = np.random.default_rng(0)
    mc = np.mean(rng.beta(a["alpha"], a["beta"], 400_000)
                 > rng.beta(b["alpha"], b["beta"], 400_000))
    assert pair["prob_a_better"] == pytest.approx(mc, abs=5e-3)
    assert pair["prob_a_better"] > 0.5, "arm A observed 40% vs arm B 30%"


def test_expected_loss_is_zero_for_identical_arms():
    out = bayesian.analyse(arms((30, 70), (30, 70)), prior_name="weak_centred")
    pair = out["pairwise"][0]
    assert pair["prob_a_better"] == pytest.approx(0.5, abs=1e-4)
    assert pair["expected_loss_if_a"] == pytest.approx(pair["expected_loss_if_b"], rel=1e-6)
    # Expected loss shrinks as evidence accumulates (same rates, 10x the data).
    more = bayesian.analyse(arms((300, 700), (300, 700)), prior_name="weak_centred")
    assert more["pairwise"][0]["expected_loss_if_a"] < pair["expected_loss_if_a"] / 2


def test_expected_loss_direction():
    out = bayesian.analyse(arms((80, 20), (20, 80)), prior_name="weak_centred")
    pair = out["pairwise"][0]
    assert pair["expected_loss_if_b"] > pair["expected_loss_if_a"]


def test_required_n_matches_the_two_proportion_formula():
    out = bayesian.required_n(0.40, 0.45)
    p1, p2 = 0.40, 0.45
    pbar = (p1 + p2) / 2
    z_a, z_b = 1.959963985, 0.841621234     # two-sided 5%, power 80%
    ref = ((z_a * math.sqrt(2 * pbar * (1 - pbar))
            + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) / (p2 - p1)) ** 2
    assert out["per_arm"] == pytest.approx(ref, rel=0.02)
    assert out["total"] == 2 * out["per_arm"]
    assert bayesian.required_n(0.40, 0.42)["per_arm"] > out["per_arm"]


def test_missing_impressions_are_not_failures():
    with_missing = arms((10, 5))[0]
    with_missing.impressions_missing = 500
    out = bayesian.analyse([with_missing], prior_name="weak_centred")
    v = out["variants"][0]
    assert v["missing"] == 500
    assert v["observed"] == 15, "missing impressions must not enter the denominator"
    assert v["observed_rate"] == pytest.approx(10 / 15)


def test_zero_successes_is_not_reported_as_zero_percent():
    out = bayesian.analyse(arms((0, 40)), prior_name="weak_centred")
    ci = out["variants"][0]["credible_interval"]
    assert ci["lo"] > 0.0, "a Beta posterior never claims the true rate is 0%"
    assert out["variants"][0]["observed_rate"] == 0.0


def test_thompson_allocation_concentrates_on_the_better_arm():
    out = bayesian.analyse(arms((10, 10), (30, 5)), prior_name="weak_centred")
    share = out["thompson"]["share"]
    assert sum(out["thompson"]["allocation"]) == out["thompson"]["draws"]
    assert share[1] > share[0]
    assert share[1] > 0.6
    again = bayesian.analyse(arms((10, 10), (30, 5)), prior_name="weak_centred", seed=42)
    assert again["thompson"]["allocation"] == out["thompson"]["allocation"]


def test_stopping_rule_requires_minimum_observations():
    out = bayesian.analyse(arms((2, 2), (2, 2)), min_observations=100)
    assert out["decision"]["stop"] is False


def test_stopping_rule_fires_on_a_clear_winner():
    out = bayesian.analyse(arms((700, 300), (300, 700)), min_observations=200,
                           loss_tolerance=0.005)
    assert out["decision"]["stop"] is True
    assert out["decision"]["expected_loss_choose_a"] < 0.005


def test_multiplicity_correction():
    out = bayesian.analyse(arms((40, 60), (60, 140), (50, 50)), family_level=0.05)
    m = out["multiplicity"]
    assert m["bonferroni_alpha"] == pytest.approx(0.05 / 3)
    assert m["sidak_alpha"] == pytest.approx(1 - (1 - 0.05) ** (1 / 3))


def test_hierarchical_shrinkage_pulls_toward_the_population():
    out = bayesian.hierarchical(arms((10, 90), (12, 88), (8, 92), (200, 200)), shrink=0.5)
    rates = [10 / 100, 12 / 100, 8 / 100, 200 / 400]
    means = [p["mean"] for p in out["posteriors"]]
    pop = out["population"]["mean"]
    for rate, mean in zip(rates, means):
        lo, hi = sorted((rate, pop))
        assert lo - 1e-9 <= mean <= hi + 1e-9
    # The 100-observation arms move further than the 400-observation arm.
    assert abs(means[0] - rates[0]) > abs(means[3] - rates[3])


def test_prior_is_explicit_and_reported():
    out = bayesian.analyse(arms((10, 10)))
    assert out["prior"]["name"] == "weak_centred"
    assert out["prior"]["alpha"] > 0 and out["prior"]["beta"] > 0
    hist = bayesian.prior_from_history(60, 240)
    assert hist[0] > 1 and hist[1] > 1
