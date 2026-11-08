"""Module D: Bayesian experimentation lab.

Posteriors, credible intervals, Thompson sampling and the stopping rule all come
from the C++ core (single implementation, tested against SciPy). This module adds
orchestration: priors, missing-observation handling, multiplicity and the
hierarchical (partial-pooling) model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..native import require as require_native

# Prior catalogue. Every prior is defensible a priori from platform-agnostic
# reasoning; none is fitted to private data.
PRIORS: dict[str, tuple[float, float, str]] = {
    "uniform": (1.0, 1.0, "Beta(1,1): no information; the posterior is the likelihood."),
    "jeffreys": (0.5, 0.5, "Jeffreys prior; invariant under reparameterisation, "
                            "slightly favours extreme rates with tiny samples."),
    "weak_centred": (2.0, 8.0, "Weakly informative around a 20% base rate with the "
                               "spread of ~40 observations. Use when your own history "
                               "suggests most profile variants perform modestly."),
    "empirical": (1.0, 1.0, "Replaced at runtime by the user's own pooled historical rate."),
}


def prior_from_history(history_successes: int, history_failures: int,
                       strength: float = 20.0) -> tuple[float, float, str]:
    """Empirical prior with a user-chosen information strength.

    alpha = rate * strength, beta = (1-rate) * strength, so the prior counts as
    `strength` pseudo-observations at the observed historical rate.
    """
    n = history_successes + history_failures
    if n == 0:
        return (1.0, 1.0, "no history available; falling back to the uniform prior")
    rate = (history_successes + 0.5) / (n + 1.0)
    return (rate * strength, (1.0 - rate) * strength,
            f"empirical prior from {n} historical observations at rate {rate:.3f}, "
            f"scaled to {strength:.0f} pseudo-observations")


@dataclass
class Variant:
    id: str
    label: str
    successes: int
    failures: int
    impressions_missing: int = 0   # observations never seen (NOT failures)


def posterior(variant: Variant, prior: tuple[float, float]):
    core = require_native()
    return core.beta_posterior(prior[0], prior[1], variant.successes, variant.failures)


def analyse(variants: list[Variant], prior_name: str = "weak_centred",
            credible_level: float = 0.95, min_observations: int = 30,
            loss_tolerance: float = 0.005, family_level: float = 0.05,
            seed: int = 42) -> dict:
    core = require_native()
    if not variants:
        raise ValueError("analyse: no variants supplied")
    a0, b0, prior_explanation = PRIORS.get(prior_name, PRIORS["uniform"])

    posters = [posterior(v, (a0, b0)) for v in variants]
    summaries = []
    for v, p in zip(variants, posters):
        observed = v.successes + v.failures
        rate = core.conversion_rate(v.successes, observed) if observed > 0 else None
        iv = core.credible_interval(p, credible_level)
        summaries.append({
            "id": v.id,
            "label": v.label,
            "successes": v.successes,
            "failures": v.failures,
            "observed": observed,
            "missing": v.impressions_missing,
            "observed_rate": None if rate is None else round(float(rate), 6),
            "posterior": {"alpha": p.alpha, "beta": p.beta, "mean": p.mean,
                          "variance": p.variance, "mode": p.mode},
            "credible_interval": {"level": credible_level, "lo": iv.lo, "hi": iv.hi,
                                  "width": iv.width},
            "maturity": "insufficient" if observed < min_observations else "usable",
        })

    # Pairwise comparisons against the current leader (highest posterior mean).
    order = sorted(range(len(variants)), key=lambda i: -posters[i].mean)
    leader = order[0]
    pairwise = []
    for i in order[1:]:
        p_sup = core.prob_superiority(posters[leader], posters[i])
        loss_leader = core.expected_loss(posters[leader], posters[i])
        pairwise.append({
            "a": variants[leader].id,
            "b": variants[i].id,
            "prob_a_better": round(float(p_sup), 6),
            "expected_loss_if_a": round(float(loss_leader), 6),
            "expected_loss_if_b": round(float(core.expected_loss(posters[i], posters[leader])), 6),
        })

    decision = None
    if len(variants) == 2:
        d = core.sequential_decision(posters[0], posters[1],
                                     variants[0].successes + variants[0].failures,
                                     variants[1].successes + variants[1].failures,
                                     min_observations, loss_tolerance)
        decision = {
            "stop": bool(d.stop),
            "reason": d.reason,
            "prob_a_better": round(float(d.prob_superiority), 6),
            "expected_loss_choose_a": round(float(d.expected_loss_choose_a), 6),
            "expected_loss_choose_b": round(float(d.expected_loss_choose_b), 6),
        }

    allocation = core.thompson_allocation(posters, 2000, seed)

    k = len(variants)
    multiplicity = {
        "variants": k,
        "bonferroni_alpha": core.bonferroni_level(family_level, k),
        "sidak_alpha": core.sidak_level(family_level, k),
        "note": "With k variants compared simultaneously, an unadjusted 5% level "
                "overstates confidence. These are the per-comparison levels that keep "
                "the family-wise error at the requested level under the normal model.",
    }

    return {
        "prior": {"name": prior_name, "alpha": a0, "beta": b0, "explanation": prior_explanation},
        "variants": summaries,
        "pairwise": pairwise,
        "decision": decision,
        "thompson": {
            "draws": 2000,
            "allocation": [int(c) for c in allocation],
            "share": [round(float(c) / 2000.0, 4) for c in allocation],
            "seed": seed,
        },
        "multiplicity": multiplicity,
        "missing_data_policy": "Variants with zero observed conversions are reported as "
                              "observed_rate=null, never as 0%. 'missing' counts "
                              "observations that never happened and are excluded from "
                              "both numerator and denominator.",
    }


def density_curve(variant_summaries: list[dict], points: int = 200) -> dict:
    """Posterior density samples for plotting (returned by the API, not computed
    in the browser)."""
    core = require_native()
    xs = np.linspace(0.0005, 0.9995, points)
    out = {}
    for s in variant_summaries:
        p = core.BetaParams(s["posterior"]["alpha"], s["posterior"]["beta"])
        out[s["id"]] = {
            "x": [round(float(x), 5) for x in xs],
            "pdf": [float(core.beta_pdf(float(x), p)) for x in xs],
        }
    return out


def hierarchical(variants: list[Variant], prior_name: str = "weak_centred",
                 shrink: float = 0.5, credible_level: float = 0.95) -> dict:
    """Random-effects Beta-Binomial fit via empirical-Bayes moment matching."""
    core = require_native()
    a0, b0, _ = PRIORS.get(prior_name, PRIORS["uniform"])
    fit = core.fit_hierarchical([v.successes for v in variants],
                                [v.failures for v in variants], a0, b0, shrink)
    out = []
    for v, p in zip(variants, fit.posteriors):
        iv = core.credible_interval(p, credible_level)
        out.append({
            "id": v.id,
            "label": v.label,
            "alpha": p.alpha,
            "beta": p.beta,
            "mean": p.mean,
            "credible_interval": {"lo": iv.lo, "hi": iv.hi, "level": credible_level},
        })
    return {
        "population": {"alpha": fit.population_alpha, "beta": fit.population_beta,
                       "mean": fit.population_alpha / (fit.population_alpha + fit.population_beta)},
        "overdispersion": fit.overdispersion,
        "shrinkage": shrink,
        "posteriors": out,
        "note": "Partial pooling borrows strength across variants. It is appropriate when "
                "variants plausibly share a common rate; with very few variants the "
                "between-variant variance estimate is itself noisy.",
    }


def required_n(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> dict:
    core = require_native()
    n = core.required_sample_size(p1, p2, alpha, power)
    return {
        "per_arm": int(n),
        "total": int(2 * n),
        "alpha": alpha, "power": power,
        "assumptions": ["independent Bernoulli observations",
                        "fixed horizon (not sequential)", "two-sided test",
                        "normal approximation to the binomial difference"],
        "note": "Planning estimate only. Sequential monitoring in this product uses a "
                "Bayesian loss rule and does not inherit this guarantee.",
    }
