"""Reproducible synthetic data.

Everything generated here is clearly synthetic. The generator is seeded and the
seed is stored with the dataset so results can be reproduced exactly.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticSpec:
    seed: int = 20260101
    n_candidates: int = 60
    image_dim: int = 32
    text_dim: int = 32
    attr_dim: int = 8
    interest_dim: int = 16
    n_clusters: int = 5


def _rng(seed: int, salt: str) -> np.random.Generator:
    mixed = int.from_bytes(hashlib.sha256(f"{seed}:{salt}".encode()).digest()[:8], "little")
    return np.random.default_rng(mixed)


def make_candidate_pool(spec: SyntheticSpec) -> dict:
    """Mixture-of-Gaussians profiles with a planted latent affinity structure.

    Latent structure (documented so evaluation is not circular):
      * every candidate belongs to one of `n_clusters` activity clusters
      * visual and textual features are noisy draws around the cluster centroid
      * attribute vectors are partially determined by the cluster, partially free
      * interest vectors are sparse, drawn from the cluster's interest support
    A viewer's "true" compatibility is the cosine to the cluster centroid plus
    noise; this lets us measure ranking quality against a known ground truth for
    SYNTHETIC data only. It says nothing about real people.
    """
    rng = _rng(spec.seed, "pool")
    k = spec.n_clusters
    centroids_v = rng.normal(size=(k, spec.image_dim))
    centroids_v /= np.linalg.norm(centroids_v, axis=1, keepdims=True)
    centroids_t = rng.normal(size=(k, spec.text_dim))
    centroids_t /= np.linalg.norm(centroids_t, axis=1, keepdims=True)
    interest_support = [rng.choice(spec.interest_dim, size=max(3, spec.interest_dim // 4),
                                   replace=False) for _ in range(k)]

    profiles = []
    for i in range(spec.n_candidates):
        c = int(rng.integers(0, k))
        v = centroids_v[c] + rng.normal(scale=0.45, size=spec.image_dim)
        t = centroids_t[c] + rng.normal(scale=0.45, size=spec.text_dim)
        v /= np.linalg.norm(v)
        t /= np.linalg.norm(t)
        attrs = np.clip(rng.normal(loc=0.5, scale=0.25, size=spec.attr_dim), 0, 1)
        attrs[: spec.attr_dim // 2] = np.clip(
            attrs[: spec.attr_dim // 2] * 0.4 + (c / max(1, k - 1)) * 0.6, 0, 1
        )
        interests = np.zeros(spec.interest_dim)
        interests[interest_support[c]] = rng.uniform(0.5, 1.0, size=interest_support[c].size)
        cold = bool(rng.random() < 0.08)
        if cold:
            v = np.zeros(spec.image_dim)
            t = np.zeros(spec.text_dim)
        profiles.append({
            "index": i, "cluster": c, "visual": v.tolist(), "textual": t.tolist(),
            "attributes": attrs.tolist(), "interests": interests.tolist(),
            "cold_start": cold,
        })

    # Ground-truth affinity per candidate for a synthetic viewer at cluster 0.
    viewer_cluster = 0
    viewer_v = centroids_v[viewer_cluster] + rng.normal(scale=0.2, size=spec.image_dim)
    viewer_t = centroids_t[viewer_cluster] + rng.normal(scale=0.2, size=spec.text_dim)
    viewer_v /= np.linalg.norm(viewer_v)
    viewer_t /= np.linalg.norm(viewer_t)
    rel = np.array([float(viewer_v @ np.asarray(p["visual"]) + viewer_t @ np.asarray(p["textual"]))
                    for p in profiles])
    rel = (rel - rel.min()) / max(1e-9, rel.max() - rel.min())
    return {
        "spec": spec.__dict__,
        "profiles": profiles,
        "viewer": {
            "visual": viewer_v.tolist(), "textual": viewer_t.tolist(),
            "attributes": [0.5] * spec.attr_dim,
            "interests": [1.0 if j in interest_support[viewer_cluster] else 0.0
                          for j in range(spec.interest_dim)],
        },
        "ground_truth_relevance": rel.tolist(),
        "label": "SYNTHETIC: mixture-of-Gaussians, not real dating data",
    }


def make_conversation(seed: int = 7, n_messages: int = 60) -> dict:
    """Synthetic two-party conversation with a planted reciprocity regime."""
    rng = _rng(seed, "conv")
    t = 0.0
    rows = []
    for i in range(n_messages):
        sender = "me" if (i % 2 == 0 if rng.random() < 0.8 else rng.random() < 0.3) else "them"
        # reply latency: lognormal, shorter for 'me' (the user is the analyst)
        lat = float(rng.lognormal(mean=6.2 if sender == "me" else 6.8, sigma=0.55))
        t += lat
        rows.append({
            "index": i, "sender": sender, "timestamp_offset_s": round(t, 1),
            "latency_s": round(lat, 1), "length": int(max(1, rng.poisson(12 if sender == "me" else 9))),
            "text": f"[synthetic message {i}]",
        })
    return {
        "seed": seed, "messages": rows,
        "label": "SYNTHETIC: generated conversation, not a real person's messages",
    }
