"""Modules A/B/C/E: the ML layer.

Photos are generated procedurally so the suite runs offline. Encoders are tested for
invariants (determinism, normalisation, separation) — not for accuracy, because we
have no labelled photographic dataset and will not invent one.
"""
from __future__ import annotations

import io
import itertools

import numpy as np
import pytest
from PIL import Image

from attraction_engine.ml import (compatibility, counterfactual, fallback,
                                  image_analysis, photo_optimizer, profile_report,
                                  text_analysis)


def png_bytes(rgb, size=64):
    buf = io.BytesIO()
    Image.new("RGB", (size, size), rgb).save(buf, format="PNG")
    return buf.getvalue()


def gradient_image(size=96):
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[..., 0] = np.linspace(0, 255, size)[None, :]
    arr[..., 1] = np.linspace(0, 255, size)[:, None]
    arr[..., 2] = 128
    return Image.fromarray(arr, "RGB")


def load(data):
    return Image.open(io.BytesIO(data)).convert("RGB")


# ------------------------------------------------------- encoders (fallback)
def test_fallback_image_embedding_is_deterministic_and_unit_norm():
    a = fallback.fallback_image_embedding(np.asarray(load(png_bytes((10, 120, 200)))))
    b = fallback.fallback_image_embedding(np.asarray(load(png_bytes((10, 120, 200)))))
    np.testing.assert_allclose(a, b)
    assert a.ndim == 1 and a.size == 256
    assert np.linalg.norm(a) == pytest.approx(1.0, rel=1e-6)


def test_fallback_image_embedding_separates_different_images():
    a = fallback.fallback_image_embedding(np.asarray(load(png_bytes((10, 120, 200)))))
    b = fallback.fallback_image_embedding(np.asarray(gradient_image()))
    assert float(a @ b) < 0.999


def test_fallback_text_embedding_is_unit_norm_and_content_sensitive():
    a = fallback.fallback_text_embedding("I like long walks and film photography")
    b = fallback.fallback_text_embedding("I like long walks and film photography")
    c = fallback.fallback_text_embedding("Competitive powerlifting and cooking")
    np.testing.assert_allclose(a, b)
    assert a.size == 384 and np.linalg.norm(a) == pytest.approx(1.0, rel=1e-6)
    assert float(a @ c) < float(a @ b)


# -------------------------------------------------------------- module A: text
def test_specificity_orders_vague_below_concrete():
    vague = text_analysis.specificity("I like fun and good vibes")["score"]
    concrete = text_analysis.specificity(
        "Saturday: 6am bakery on Mall Road, espresso at Curious Life, then the "
        "sandstone trail at Nahargarh before the Jaipur heat arrives.")["score"]
    assert 0.0 <= vague <= 1.0 and 0.0 <= concrete <= 1.0
    assert concrete > vague


def test_text_block_analysis_reports_components_and_no_sensitive_inference():
    out = text_analysis.analyse_text_block(
        "I run a small photography studio in Jaipur. Weekends are climbing at "
        "Nahargarh or a long drive with a camera.")
    assert 0.0 <= out["specificity"]["score"] <= 1.0
    assert "embedding" in out
    blob = str(out).lower()
    for banned in ("ethnicity", "religion", "sexual orientation", "attachment style",
                   "personality type", "mental health"):
        assert banned not in blob


# ------------------------------------------------------------ module A: photo
def test_image_analysis_reports_quality_without_sensitive_attributes():
    out = image_analysis.analyze_photo(np.asarray(load(png_bytes((120, 170, 90)))))
    assert 0.0 <= out["quality_estimate"] <= 1.0
    # The quality score must be a weighted sum of the reported components.
    comps, weights = out["quality_components"], out["quality_weights"]
    rebuilt = sum(weights[k] * comps[k] for k in weights)
    assert rebuilt == pytest.approx(out["quality_estimate"], abs=1e-9)
    assert out["faces"]["count"] == 0
    emb = np.asarray(out["embedding"])
    assert emb.ndim == 1 and emb.size == 512
    assert out["embedding_encoder"]
    blob = str(sorted(out)).lower()
    for banned in ("ethnicity", "attractiveness", "race", "religion"):
        assert banned not in blob


def test_profile_report_flags_duplicates_and_discloses_limits():
    imgs = [np.asarray(load(png_bytes((20, 60, 200)))),
            np.asarray(load(png_bytes((20, 60, 200)))),
            np.asarray(gradient_image())]
    rep = profile_report.build_report(
        images=imgs,
        texts=[{"text": "I run a small photography studio in Jaipur.",
                "answer": "I run a small photography studio in Jaipur."},
               {"text": "Weekends are climbing or a long drive with a camera.",
                "answer": "Weekends are climbing or a long drive with a camera."}],
        ordering_suggestion=None)
    div = rep["visual_diversity"]
    assert 0.0 <= div["diversity"] <= 1.0
    assert len(rep["photos"]) == 3
    assert rep["limitations"]
    # Identical low-content images must produce at least one recommendation.
    assert len(rep["recommendations"]) >= 1
    # Two identical images must be reported as a redundant pair. Diversity is the
    # mean over ALL pairs (including the one dissimilar image), so it only has to
    # fall well below the 0.5 you would get from three mutually unrelated photos.
    assert div["diversity"] < 0.5
    assert div["redundant_pairs"], "duplicate photographs were not flagged"
    assert div["redundant_pairs"][0]["similarity"] > 0.99
    assert rep["confidence"]["evidence_coverage"] >= 0.0


# --------------------------------------------------------- module B: optimizer
def _embeddings(n=6, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dim))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _photos(n):
    return [{"id": f"p{i}", "position": i} for i in range(n)]


def test_objective_decomposes_exactly():
    emb = _embeddings()
    quality = np.linspace(0.9, 0.2, len(emb))
    prob = photo_optimizer.build_problem(emb, quality, np.linspace(1.0, 0.0, len(emb)),
                                         lam=0.35, mu=0.2)
    from attraction_engine import native

    bd = native.core.evaluate_ordering(prob, list(range(len(emb))))
    assert bd.total == pytest.approx(bd.quality + 0.35 * bd.diversity + 0.2 * bd.coherence,
                                    abs=1e-9)
    assert 0.0 <= bd.diversity <= 1.0 + 1e-12


def test_lambda_mu_zero_reproduces_the_quality_oracle():
    emb = _embeddings()
    quality = np.array([0.2, 0.9, 0.5, 0.7, 0.1, 0.3])
    out = photo_optimizer.optimize_ordering(_photos(6), emb, quality, lam=0.0, mu=0.0,
                                            strategies=["exhaustive"])
    perm = out["results"][0]["permutation"]
    order = [quality[i] for i in perm]
    assert order == sorted(order, reverse=True)
    assert out["results"][0]["proved_optimal"] is True
    assert out["results"][0]["strategy"] == "exhaustive"


def test_diversity_term_changes_the_order():
    emb = _embeddings(4, 8, seed=3)
    emb[1] = emb[0] + 1e-6
    emb[1] /= np.linalg.norm(emb[1])
    quality = np.array([0.95, 0.94, 0.50, 0.40])
    plain = photo_optimizer.optimize_ordering(_photos(4), emb, quality, lam=0.0, mu=0.0,
                                              strategies=["exhaustive"])["results"][0]
    diverse = photo_optimizer.optimize_ordering(_photos(4), emb, quality, lam=0.8, mu=0.0,
                                                strategies=["exhaustive"])["results"][0]
    assert plain["permutation"][:2] == [0, 1]
    assert diverse["permutation"][:2] != [0, 1]
    # Different lambda => different objective, so compare the *diversity term*
    # rather than the totals of two differently-weighted objectives.
    assert diverse["diversity_term"] > plain["diversity_term"]
    assert diverse["quality_term"] <= plain["quality_term"] + 1e-9


def test_exhaustive_beats_every_sampled_permutation():
    emb = _embeddings(5, 16, seed=5)
    quality = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    slot = np.linspace(1.0, 0.0, 5)
    prob = photo_optimizer.build_problem(emb, quality, slot, lam=0.4, mu=0.2)
    from attraction_engine import native

    best = photo_optimizer.optimize_ordering(_photos(5), emb, quality, lam=0.4, mu=0.2,
                                             strategies=["exhaustive"])["results"][0]
    gaps = [native.core.evaluate_ordering(prob, list(p)).total - best["objective"]
            for p in itertools.permutations(range(5))]
    assert max(gaps) <= 1e-9, "exhaustive result was not the maximum"


def test_sensitivity_grid_and_ablation_are_consistent():
    emb = _embeddings(5, 16, seed=6)
    quality = np.array([0.4, 0.5, 0.6, 0.7, 0.8])
    sens = photo_optimizer.sensitivity_analysis(_photos(5), emb, quality,
                                                lambdas=[0.0, 0.3], mus=[0.0, 0.2])
    assert len(sens["grid"]) == 4
    for row in sens["grid"]:
        assert row["optimized_total"] >= row["fixed_perm_total"] - 1e-9
    abl = photo_optimizer.ablation(_photos(5), emb, quality)
    assert len(abl["rows"]) >= 4
    assert all(np.isfinite(r["loss"]) for r in abl["rows"])


# ------------------------------------------------------- module C: ranking
def _pool_record(seed, dim_v=24, dim_t=16, dim_a=6, dim_i=12, cold=False):
    rng = np.random.default_rng(seed)

    def unit(n):
        v = rng.normal(size=n)
        return (v / np.linalg.norm(v)).tolist()

    return {"index": seed, "visual": unit(dim_v), "textual": unit(dim_t),
            "attributes": (rng.random(dim_a)).tolist(),
            "interests": (rng.random(dim_i)).tolist(), "cold_start": cold}


def test_score_pair_is_symmetric_and_bounded():
    a, b = _pool_record(1), _pool_record(2)
    ab = compatibility.score_pair(a, b)
    ba = compatibility.score_pair(b, a)
    assert ab["score"] == pytest.approx(ba["score"], abs=1e-9)
    assert 0.0 <= ab["score"] <= 1.0
    # The declared blend must reconstruct the score to machine precision.
    w = {"visual": 0.25, "textual": 0.30, "attributes": 0.25, "interests": 0.20}
    rebuilt = sum(w[k] * ab[k] for k in w)
    assert rebuilt == pytest.approx(ab["score"], abs=1e-12)
    # Every component is a bounded similarity.
    for k in w:
        assert 0.0 <= ab[k] <= 1.0


def test_ranking_metrics_match_hand_computation():
    # ranking entries carry the candidate index; ground_truth is indexed the same way.
    # Relevance >= 0.66 is the documented "relevant" threshold.
    ranking = [{"index": 0}, {"index": 1}, {"index": 2}, {"index": 3}]
    truth = [0.9, 0.1, 0.8, 0.2]          # 0 and 2 are relevant
    out = compatibility.ranking_metrics(ranking, truth, k=2)
    assert out["n_relevant"] == 2
    assert out["precision_at_k"] == pytest.approx(0.5)   # only index 0 in the top 2
    assert out["recall_at_k"] == pytest.approx(0.5)
    assert 0.0 < out["ndcg_at_k"] < 1.0
    # A perfect ordering (both relevant first) scores nDCG = 1 and full recall.
    perfect = compatibility.ranking_metrics(
        [{"index": 0}, {"index": 2}, {"index": 1}, {"index": 3}], truth, k=2)
    assert perfect["ndcg_at_k"] == pytest.approx(1.0, abs=1e-9)
    assert perfect["precision_at_k"] == pytest.approx(1.0)
    assert perfect["recall_at_k"] == pytest.approx(1.0)


def test_rank_returns_sorted_scores_and_marks_synthetic_pool():
    from attraction_engine.stats import synthetic

    pool = synthetic.make_candidate_pool(synthetic.SyntheticSpec(seed=3, n_candidates=40))
    out = compatibility.rank(pool["viewer"], pool["profiles"], top_k=10)
    scores = [r["forward"]["score"] for r in out["ranking"]]
    assert scores == sorted(scores, reverse=True)
    assert len(out["ranking"]) == 10
    # The disclaimer must travel with every ranking.
    assert "do not predict" in out["caveat"].lower()
    assert out["model_status"] == "trained-synthetic", "neural head provenance"
    assert out["lambda_diversity"] == 0.0
    # And the synthetic pool must say so.
    assert str(pool["label"]).startswith("SYNTHETIC")


# ------------------------------------------------------ module E: counterfactual
def _text_rec(text):
    return {"text": text, "index": 0, "specificity": text_analysis.specificity(text)}


def _photo_recs(quality, emb):
    return [{"id": f"p{i}", "quality_estimate": quality, "position": i,
             "dominant_scene": "outdoor_urban", "embedding": list(emb[i])}
            for i in range(emb.shape[0])]


def test_counterfactual_delta_is_zero_for_an_identical_configuration():
    emb = _embeddings(3, 8, 20)
    photos = _photo_recs(0.5, emb)
    texts = [_text_rec("A weekend built around a ridge climb and a long breakfast.")]
    cfg = counterfactual.build_config(photos, texts)
    out = counterfactual.simulate(cfg, cfg)
    assert out["delta_q"] == pytest.approx(0.0, abs=1e-12)
    assert out["uncertainty"]["value"] is None
    assert "not a causal effect" in out["interpretation"]
    # Q is the declared weighted sum of its own components.
    weights = out["components_modified"]["weights"]
    comps = out["components_modified"]
    rebuilt = sum(weights[k] * comps[k] for k in weights)
    assert rebuilt == pytest.approx(out["q_modified"], abs=1e-6)


def test_counterfactual_detects_improvement_and_reports_attribution():
    emb = _embeddings(3, 32, 4)
    base = counterfactual.build_config(_photo_recs(0.2, emb), [_text_rec("fun")])
    better = counterfactual.build_config(
        _photo_recs(0.9, emb),
        [_text_rec("Saturday 6am bakery on Mall Road then the Nahargarh ridge trail, "
                   "back via the old city market.")])
    out = counterfactual.simulate(base, better)
    assert out["delta_q"] > 0
    assert out["relative_change"] > 0
    # Attribution covers both modifiable blocks and sums to something finite.
    imp = out["attribution"]["importance"]
    assert set(imp) == {"photo_selection", "text_specificity"}
    assert out["attribution"]["n_permutations"] == 200
    assert all(np.isfinite(v) for v in imp.values())
    # The sensitivity block must state how each number was obtained.
    assert out["sensitivity"]
    assert any("analytically" in str(s.get("note", "")) for s in out["sensitivity"])
