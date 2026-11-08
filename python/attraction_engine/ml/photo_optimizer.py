"""Module B: photograph ordering optimization, orchestrated in Python on top of
the C++26 optimizer.

Python owns: position weights, narrative slot affinities, objective assembly and
reporting. C++ owns: search (greedy / beam / SA / exhaustive) and evaluation.
The split keeps the research layer in Python and the inner loops in C++.
"""
from __future__ import annotations

import numpy as np

from ..native import require as require_native

STRATEGIES = ["greedy", "beam", "simulated_annealing", "random_restart", "exhaustive"]


def position_weights(n: int, decay: float = 0.35) -> np.ndarray:
    """w_k = (1 + decay*(k-1))^-1 normalised to sum 1.

    Rationale: attention falls off with slot index. The decay is a modelling
    assumption (documented, tunable in the UI), not a fitted parameter.
    """
    k = np.arange(n, dtype=np.float64)
    w = 1.0 / (1.0 + decay * k)
    return w / w.sum()


def slot_targets(n: int) -> np.ndarray:
    """Ideal narrative affinity per slot: rapport-building first, story later."""
    return np.linspace(1.0, 0.0, n) if n > 1 else np.ones(1)


def infer_slot_affinity(photos: list[dict]) -> np.ndarray:
    """Map each photo onto [0,1] storytelling affinity.

    Heuristic (all inputs disclosed): photographs whose dominant scene is a
    close-up portrait or indoor social shot are good rapport-builders early;
    activity/travel/landscape shots carry the "what I do" narrative later.
    When scene labels are missing we fall back to face presence, then to a
    neutral 0.5 so the term cannot silently fabricate signal.
    """
    early = {"portrait_closeup", "indoor_social", "indoor_home", "group_photo"}
    late = {"activity_sport", "travel_landmark", "outdoor_nature", "outdoor_urban"}
    vals = []
    for p in photos:
        scene = p.get("dominant_scene")
        if scene in early:
            vals.append(0.85)
        elif scene in late:
            vals.append(0.35)
        elif p.get("faces") and p["faces"]["count"] == 1 and p["faces"]["largest_fraction"] > 0.06:
            vals.append(0.75)
        else:
            vals.append(0.5)
    return np.asarray(vals, dtype=np.float64)


def build_problem(embeddings: np.ndarray, quality: np.ndarray, slot_affinity: np.ndarray,
                  lam: float = 0.35, mu: float = 0.20,
                  positional_kernel: bool = True):
    native = require_native()
    n = embeddings.shape[0]
    p = native.PhotoProblem()
    p.quality = [float(np.clip(q, 0.0, 1.0)) for q in quality]
    p.slot_affinity = [float(np.clip(s, 0.0, 1.0)) for s in slot_affinity]
    p.position_weight = position_weights(n).tolist()
    p.slot_target = slot_targets(n).tolist()
    p.lambda_ = float(lam)
    p.mu = float(mu)
    p.positional_kernel = positional_kernel
    dist = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * (embeddings @ embeddings.T)))
    p.set_distance_matrix(np.ascontiguousarray(dist, dtype=np.float64))
    return p


def _options(native, strategy: str, seed: int, beam_width: int, sa_iters: int):
    o = native.OptimizeOptions()
    o.strategy = getattr(native.Strategy, strategy)
    o.seed = seed
    o.beam_width = beam_width
    o.sa_iterations = sa_iters
    o.polish = True
    o.restarts = 16
    return o


def optimize_ordering(photos: list[dict], embeddings: np.ndarray, quality: np.ndarray,
                      lam: float = 0.35, mu: float = 0.20, seed: int = 42,
                      strategies: list[str] | None = None,
                      beam_width: int = 8, sa_iters: int = 20000) -> dict:
    """Run every requested strategy, return a comparison table plus the winner."""
    native = require_native()
    n = embeddings.shape[0]
    strategies = strategies or ["greedy", "beam", "simulated_annealing", "random_restart"]
    if n <= 9 and "exhaustive" not in strategies:
        strategies = strategies + ["exhaustive"]

    affinity = infer_slot_affinity(photos)
    problem = build_problem(embeddings, quality, affinity, lam, mu)

    results = []
    best = None
    for strat in strategies:
        opts = _options(native, strat, seed, beam_width, sa_iters)
        try:
            res = native.optimize_ordering(problem, opts)
        except RuntimeError as exc:      # exhaustive limit, etc.
            results.append({"strategy": strat, "error": str(exc)})
            continue
        entry = {
            "strategy": strat,
            "permutation": list(res.permutation),
            "objective": round(float(res.score.total), 6),
            "quality_term": round(float(res.score.quality), 6),
            "diversity_term": round(float(res.score.diversity), 6),
            "coherence_term": round(float(res.score.coherence), 6),
            "redundancy_term": round(float(res.score.redundancy), 6),
            "evaluations": int(res.evaluations),
            "elapsed_ms": round(float(res.elapsed_ms), 3),
            "proved_optimal": bool(res.proved_optimal),
        }
        results.append(entry)
        if best is None or entry["objective"] > best["objective"]:
            best = entry

    baseline_perm = list(range(n))
    baseline = native.evaluate_ordering(problem, baseline_perm)
    best["improvement_vs_upload_order"] = round(float(best["objective"] - baseline.total), 6)

    return {
        "n": n,
        "parameters": {"lambda": lam, "mu": mu, "position_decay": 0.35,
                       "positional_kernel": True, "seed": seed,
                       "beam_width": beam_width, "sa_iterations": sa_iters},
        "slot_affinity": [round(float(v), 4) for v in affinity],
        "position_weights": [round(float(v), 6) for v in position_weights(n)],
        "baseline": {
            "permutation": baseline_perm,
            "objective": round(float(baseline.total), 6),
            "quality_term": round(float(baseline.quality), 6),
            "diversity_term": round(float(baseline.diversity), 6),
            "coherence_term": round(float(baseline.coherence), 6),
        },
        "results": results,
        "best": best,
        "caveat": "The objective is a declared preference function. It is not a validated "
                  "model of how people respond to photographs.",
    }


def sensitivity_analysis(photos: list[dict], embeddings: np.ndarray, quality: np.ndarray,
                         lambdas: list[float] | None = None,
                         mus: list[float] | None = None, seed: int = 42) -> dict:
    native = require_native()
    affinity = infer_slot_affinity(photos)
    problem = build_problem(embeddings, quality, affinity)
    opts = _options(native, "beam", seed, 8, 4000)
    lams = lambdas or [0.0, 0.15, 0.35, 0.55, 0.75]
    mvs = mus or [0.0, 0.2, 0.4]
    pts = native.sensitivity(problem, opts, lams, mvs)
    return {
        "grid": [
            {
                "lambda": p.lambda_,
                "mu": p.mu,
                "fixed_perm_total": round(float(p.fixed_perm_total), 6),
                "optimized_total": round(float(p.optimized_total), 6),
                "position_changes": int(p.position_changes),
                "optimized_permutation": list(p.optimized_permutation),
            }
            for p in pts
        ],
        "note": "Each cell re-optimises the ordering at that (lambda, mu). "
                "position_changes is the number of slots whose photograph changed "
                "relative to the base parameters.",
    }


def ablation(photos: list[dict], embeddings: np.ndarray, quality: np.ndarray,
             seed: int = 42) -> dict:
    native = require_native()
    affinity = infer_slot_affinity(photos)
    problem = build_problem(embeddings, quality, affinity)
    opts = _options(native, "beam", seed, 8, 4000)
    rows = native.ablate_leverage(problem, opts)
    return {
        "rows": [
            {
                "removed_index": int(r.removed_index),
                "reduced_total": round(float(r.reduced_total), 6),
                "baseline_total": round(float(r.baseline_total), 6),
                "best_total_with": round(float(r.best_total_with), 6),
                "loss": round(float(r.loss), 6),
                "best_permutation_with": list(r.best_permutation_with),
            }
            for r in rows
        ],
        "interpretation": "loss is how much objective is lost when that photograph is "
                          "allowed only one slot inside an otherwise re-optimised order. "
                          "A large loss means the photograph carries a lot of the ordering's "
                          "value; a ~0 loss means the order is indifferent to it.",
    }
