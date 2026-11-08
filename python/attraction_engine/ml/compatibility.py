"""Module C: compatibility and recommendation engine.

Three layers, each labelled in the API response:
  1. transparent weighted baseline  S(i) = aV + bT + gA + dP
  2. Reciprocal (harmonic) re-ranking with cold-start confidence
  3. optional trained pairwise MLP head (P(Y=1) = sigmoid(MLP(...)))

The MLP is trained offline by ml/train_pair_ranker.py on synthetic pairs; when no
exported weights exist the API reports model_status="untrained" and the ranked
score is the baseline. No claim of real-world validity is made anywhere.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..native import require as require_native

MODEL_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = MODEL_DIR / "pair_ranker.txt"
MODEL_META = MODEL_DIR / "pair_ranker.json"


def model_status() -> dict:
    if not MODEL_PATH.exists():
        return {
            "status": "untrained",
            "reason": "no exported weights found; ranked scores come from the transparent "
                      "baseline only",
            "path": str(MODEL_PATH),
        }
    meta = json.loads(MODEL_META.read_text()) if MODEL_META.exists() else {}
    return {"status": "trained-synthetic", "path": str(MODEL_PATH), **meta}


_ranker_cache: dict[str, object] = {}


def load_ranker():
    native = require_native()
    if "r" in _ranker_cache:
        return _ranker_cache["r"]
    if not MODEL_PATH.exists():
        _ranker_cache["r"] = None
        return None
    text = MODEL_PATH.read_text()
    ranker = native.PairRanker.from_serialised(text)
    _ranker_cache["r"] = ranker
    return ranker


def _cos_to_unit(c: float) -> float:
    return float(np.clip((c + 1.0) * 0.5, 0.0, 1.0))


def _agreement(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    if n == 0:
        return 0.0
    return float(np.mean(1.0 - np.minimum(np.abs(a[:n] - b[:n]), 1.0)))


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or a.size != b.size:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.clip(float(a @ b) / (na * nb), 0.0, 1.0))


def score_pair(viewer: dict, cand: dict, weights: dict | None = None) -> dict:
    w = weights or {"visual": 0.25, "textual": 0.30, "attributes": 0.25, "interests": 0.20}
    z = sum(w.values()) or 1.0
    v = _cos_to_unit(float(np.dot(viewer["visual"], cand["visual"]))) \
        if len(cand["visual"]) and len(viewer["visual"]) else 0.5
    t = _cos_to_unit(float(np.dot(viewer["textual"], cand["textual"]))) \
        if len(cand["textual"]) and len(viewer["textual"]) else 0.5
    a = _agreement(np.asarray(viewer["attributes"]), np.asarray(cand["attributes"]))
    p = _overlap(np.asarray(viewer["interests"]), np.asarray(cand["interests"]))
    s = (w["visual"] * v + w["textual"] * t + w["attributes"] * a + w["interests"] * p) / z
    return {"visual": v, "textual": t, "attributes": a, "interests": p, "score": s}


def rank(viewer: dict, candidates: list[dict], top_k: int = 20,
         weights: dict | None = None, lambda_diversity: float = 0.0,
         use_neural: bool = True) -> dict:
    native = require_native()
    scores = []
    for c in candidates:
        fwd = score_pair(viewer, c, weights)
        bwd = score_pair(c, viewer, weights)
        harmonic = (2 * fwd["score"] * bwd["score"] / (fwd["score"] + bwd["score"])
                    if (fwd["score"] + bwd["score"]) > 0 else 0.0)
        conf = 0.55 if (viewer.get("cold_start") or c.get("cold_start")) else 0.95
        scores.append({
            "index": int(c["index"]),
            "forward": fwd, "backward": bwd["score"], "harmonic": harmonic,
            "confidence": conf, "cold_start": bool(c.get("cold_start", False)),
        })

    ranker = load_ranker() if use_neural else None
    neural_used = False
    if ranker is not None and ranker.is_trained:
        for s, c in zip(scores, candidates):
            ui = np.asarray(viewer["visual"] + viewer["textual"], dtype=np.float64)
            uj = np.asarray(c["visual"] + c["textual"], dtype=np.float64)
            if ui.size and uj.size:
                s["neural_probability"] = float(ranker.predict_symmetric(ui, uj))
                neural_used = True

    scores.sort(key=lambda s: (-(s["harmonic"]), s["index"]))

    if lambda_diversity > 0 and len(scores) > 1:
        from ..native import require as _req

        core = _req()
        by_index = {int(c["index"]): c for c in candidates}
        emb = np.asarray([by_index[s["index"]]["visual"] for s in scores], dtype=np.float64)
        # normalise defensively: cold-start profiles have zero visual vectors
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = np.ascontiguousarray(emb / norms)
        rel = np.asarray([s["harmonic"] for s in scores], dtype=np.float64)
        picked = core.mmr_select(emb, rel, lambda_diversity, min(top_k, len(scores)))
        scores = [scores[p.index] for p in picked]

    out = scores[:top_k]
    return {
        "model_status": model_status()["status"],
        "neural_head_applied": neural_used,
        "weights": weights or {"visual": 0.25, "textual": 0.30, "attributes": 0.25,
                               "interests": 0.20},
        "lambda_diversity": lambda_diversity,
        "ranking": out,
        "formula": "S(i) = (a*V + b*T + g*A + d*P)/sum(weights); "
                   "harmonic = 2*S_forward*S_backward/(S_forward+S_backward)",
        "caveat": "Scores rank candidates under declared features and weights. They do not "
                  "predict whether two people will form a relationship.",
    }


def ranking_metrics(ranking: list[dict], ground_truth: list[float], k: int = 10) -> dict:
    """Precision/Recall/NDCG against synthetic ground truth.

    Only meaningful because the synthetic generator plants a known affinity
    structure; the API labels these numbers as synthetic-evaluation metrics.
    """
    core = require_native()
    order = [r["index"] for r in ranking]
    relevant = {i for i, r in enumerate(ground_truth) if r >= 0.66}
    top = order[:k]
    hits = sum(1 for i in top if i in relevant)
    precision = hits / max(1, len(top))
    recall = hits / max(1, len(relevant))
    ideal_scores = sorted(ground_truth, reverse=True)
    ranked_rel = [ground_truth[i] for i in top] + [0.0] * max(0, k - len(top))
    ideal_rel = ideal_scores[:k] + [0.0] * max(0, k - len(ideal_scores))
    ndcg = core.ndcg_at_k(ranked_rel[:k], ideal_rel[:k], k)
    return {
        "k": k,
        "relevant_definition": "synthetic affinity >= 0.66",
        "precision_at_k": round(precision, 4),
        "recall_at_k": round(recall, 4),
        "ndcg_at_k": round(float(ndcg), 4),
        "n_relevant": len(relevant),
    }


def embedding_visualisation(profiles: list[dict], ground_truth: list[float] | None = None,
                            method: str = "pca") -> dict:
    """2-D projection of the visual embeddings for the explorer view."""
    emb = np.asarray([p["visual"] for p in profiles], dtype=np.float64)
    if emb.size == 0:
        return {"points": []}
    emb = emb - emb.mean(axis=0)
    if method == "pca" or emb.shape[0] < 5:
        u, s, vt = np.linalg.svd(emb, full_matrices=False)
        coords = u[:, :2] * s[:2]
        explained = (s[:2] ** 2) / np.sum(s ** 2) if np.sum(s ** 2) > 0 else np.zeros(2)
    else:
        coords, explained = _tsne_lite(emb)
    pts = []
    for i, p in enumerate(profiles):
        pts.append({
            "index": int(p["index"]),
            "x": round(float(coords[i, 0]), 5),
            "y": round(float(coords[i, 1]), 5),
            "cluster": int(p.get("cluster", -1)),
            "cold_start": bool(p.get("cold_start", False)),
            "truth": None if ground_truth is None else round(float(ground_truth[i]), 4),
        })
    return {
        "points": pts, "method": method,
        "explained_variance": [round(float(e), 4) for e in explained],
        "note": "PCA on visual embeddings. Cluster colours come from the generator, "
                "not from a model fitted on your data." if method == "pca" else
                "2-D projection for inspection only.",
    }


def _tsne_lite(emb: np.ndarray, perplexity: float = 8.0, iters: int = 250) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic t-SNE (exact O(n^2) affinities, no BH tree).

    Implemented directly because scikit-learn is already a dependency but adding
    a second joint-embedding code path would duplicate behaviour; this keeps the
    numerical behaviour explicit and reproducible.
    """
    n = emb.shape[0]
    d2 = np.square(emb[:, None, :] - emb[None, :, :]).sum(-1)
    sigma2 = np.ones(n)
    target = np.log(perplexity)
    for _ in range(50):
        p = np.exp(-d2 / (2 * sigma2[:, None]))
        np.fill_diagonal(p, 0.0)
        row = p.sum(1, keepdims=True) + 1e-12
        p = p / row
        H = -np.sum(p * np.log(p + 1e-12), axis=1)
        sigma2 *= np.exp(0.5 * (H - target))
    P = (p + p.T) / (2 * n)
    rng = np.random.default_rng(0)
    Y = rng.normal(scale=1e-2, size=(n, 2))
    lr = 50.0
    for it in range(iters):
        d = np.square(Y[:, None, :] - Y[None, :, :]).sum(-1)
        num = 1.0 / (1.0 + d)
        np.fill_diagonal(num, 0.0)
        Q = num / (num.sum() + 1e-12)
        grad = 4.0 * ((P - Q) * num)[:, :, None] * (Y[:, None, :] - Y[None, :, :])
        Y -= lr * grad.sum(1) / n
        if it == 100:
            lr = 10.0
    u, s, vt = np.linalg.svd(Y - Y.mean(0), full_matrices=False)
    return u[:, :2] * s[:2], np.zeros(2)
