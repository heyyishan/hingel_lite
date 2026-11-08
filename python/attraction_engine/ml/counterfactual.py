"""Module E: counterfactual profile simulator.

Given a baseline profile configuration and a modified one, this module reports

    dQ = Q(P_modified) - Q(P_baseline)

where Q is the *same* declared objective used by the Photo Laboratory plus a
text-quality term. Attribution uses permutation importance on the objective.

CRITICAL HONESTY CONSTRAINT
---------------------------
dQ is a prediction under a stated preference model. It is NOT an estimated causal
effect of changing the profile: no randomised assignment happened. The response
therefore always carries ``interpretation`` describing exactly what was computed.
"""
from __future__ import annotations

import numpy as np

from . import image_analysis, photo_optimizer, text_analysis


def _quality_of(config: dict) -> tuple[float, dict]:
    """Objective Q for one configuration (photos + texts).

    Q = mean quality of the selected photographs        (0.55)
      + visual diversity of the selection               (0.25)
      + mean prompt specificity of the text blocks      (0.20)
    Weights are declared here, returned in the payload, and identical across the
    baseline and modified configurations so the comparison is like-for-like.
    """
    emb = np.asarray(config["embeddings"], dtype=np.float64)
    q = np.asarray(config["quality"], dtype=np.float64)
    sel = config["selection"]
    if len(sel) == 0:
        photo_part, div_part = 0.0, 0.0
    else:
        photo_part = float(np.mean(q[sel]))
        sub = emb[sel]
        if len(sel) == 1:
            div_part = 0.0
        else:
            sim = sub @ sub.T
            iu = np.triu_indices(len(sel), k=1)
            div_part = float(1.0 - sim[iu].mean())
    spec = config.get("text_specificity")
    text_part = float(np.mean(spec)) if spec else 0.0
    components = {
        "photo_quality": round(photo_part, 5),
        "visual_diversity": round(div_part, 5),
        "text_specificity": round(text_part, 5),
    }
    weights = {"photo_quality": 0.55, "visual_diversity": 0.25, "text_specificity": 0.20}
    total = sum(weights[k] * components[k] for k in weights)
    return total, {**components, "weights": weights}


def simulate(baseline: dict, modified: dict, seed: int = 42) -> dict:
    q0, c0 = _quality_of(baseline)
    q1, c1 = _quality_of(modified)
    delta = q1 - q0

    # Permutation importance on the modified configuration: shuffle each input
    # block (photo selection, text specificity) and measure the drop in dQ.
    rng = np.random.default_rng(seed)
    importances: dict[str, float] = {}
    n_perm = 200
    drops = {"photos": [], "text": []}
    for _ in range(n_perm):
        perm_sel = rng.permutation(modified["selection"])
        alt = {**modified, "selection": list(perm_sel)}
        drops["photos"].append(q1 - _quality_of(alt)[0] if len(modified["selection"]) > 1 else 0.0)
        if modified.get("text_specificity"):
            spec = np.asarray(modified["text_specificity"], dtype=np.float64)
            shuffled = rng.permutation(spec)
            alt2 = {**modified, "text_specificity": shuffled.tolist()}
            drops["text"].append(q1 - _quality_of(alt2)[0])
        else:
            drops["text"].append(0.0)
    importances["photo_selection"] = round(float(np.mean(drops["photos"])), 6)
    importances["text_specificity"] = round(float(np.mean(drops["text"])), 6)

    # Sensitivity: dQ as photo set size and text specificity move one step.
    sensitivity = []
    if len(modified["selection"]) > 2:
        smaller = {**modified, "selection": modified["selection"][:-1]}
        sensitivity.append({"change": "drop the last selected photograph",
                            "delta_q": round(_quality_of(smaller)[0] - q1, 6)})
    sensitivity.append({
        "change": "improve every prompt's specificity by 0.1 (hypothetical)",
        "delta_q": round(0.20 * 0.1, 6),
        "note": "computed analytically from the declared weight, not from a model",
    })

    return {
        "q_baseline": round(q0, 6),
        "q_modified": round(q1, 6),
        "delta_q": round(delta, 6),
        "relative_change": round(delta / q0, 6) if q0 > 0 else None,
        "components_baseline": c0,
        "components_modified": c1,
        "attribution": {
            "method": "permutation importance over 200 shuffles of each input block",
            "importance": importances,
            "n_permutations": n_perm,
        },
        "sensitivity": sensitivity,
        "uncertainty": {
            "kind": "not a sampling distribution",
            "value": None,
            "explanation": "Q is a deterministic function of declared features, so it has no "
                           "sampling uncertainty. The uncertainty that matters is model "
                           "validity, which this number cannot express.",
        },
        "model_version": "counterfactual-v1 (declared weights, no fitted parameters)",
        "interpretation": (
            "dQ is the difference between the project's declared scoring function evaluated on "
            "two configurations. It is a prediction under an explicit preference model. It is "
            "not a causal effect: no experiment randomised these configurations, and the "
            "weights were not fitted to outcome data. To obtain causal evidence, run the "
            "configurations as an experiment in the Experimentation Lab."
        ),
    }


def build_config(photo_records: list[dict], text_records: list[dict],
                 selection: list[int] | None = None) -> dict:
    emb = np.asarray([p["embedding"] for p in photo_records], dtype=np.float64)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms
    return {
        "embeddings": emb.tolist(),
        "quality": [p["quality_estimate"] for p in photo_records],
        "selection": selection if selection is not None else list(range(len(photo_records))),
        "text_specificity": [t["specificity"]["score"] for t in text_records],
    }
