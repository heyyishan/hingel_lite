"""Module A orchestrator: turn per-photo and per-text analyses into a profile
report with ranked recommendations, confidence and explicit limitations.

Design rule: the report is a *decision aid*. Every recommendation carries the
evidence it is based on and the direction of the effect; nothing is asserted as
ground truth about how other people will react.
"""
from __future__ import annotations

import numpy as np

from . import image_analysis, text_analysis

RECOMMENDATION_LIBRARY: dict[str, dict] = {
    "low_res_first_photo": {
        "severity": "high",
        "title": "Your first photograph is low resolution",
        "why": "First photographs are rendered small on most clients; under ~640 px on the "
               "short edge detail collapses and the photo reads as low effort.",
        "action": "Replace with the highest-resolution originals you have, or re-export at "
                  "perceptual quality without upscaling.",
    },
    "redundant_photos": {
        "severity": "medium",
        "title": "Two photographs are nearly identical",
        "why": "Pairs with cosine similarity above {threshold} usually show the same scene, "
               "outfit and framing, so the second adds no new information.",
        "action": "Swap one of the pair for a photo that shows a different scene or activity.",
    },
    "low_visual_diversity": {
        "severity": "medium",
        "title": "Visual variety is low",
        "why": "Mean pairwise similarity of {value} means the set explores few distinct "
               "contexts; the profile shows one side of you.",
        "action": "Add one outdoor/activity photo and one social photo if you have them.",
    },
    "weak_first_photo": {
        "severity": "high",
        "title": "Your strongest photo is not first",
        "why": "Position weights decay with slot; the computed objective puts photo {best} "
               "in slot 1 and is {delta} higher than your current order.",
        "action": "Reorder using the Photo Laboratory suggestion, then judge it yourself.",
    },
    "generic_prompt_language": {
        "severity": "medium",
        "title": "A prompt answer uses generic phrasing",
        "why": "Detected stock phrases: {phrases}. They carry little information and appear "
               "on a large fraction of profiles.",
        "action": "Replace with one specific, checkable detail — a place, a routine, a name.",
    },
    "low_specificity": {
        "severity": "medium",
        "title": "A prompt answer is not very specific",
        "why": "Specificity score {value}/1. Concreteness (named places, routines, objects) "
               "is what gives a stranger something to reply to.",
        "action": "Add one concrete noun the reader could ask about.",
    },
    "repetition": {
        "severity": "low",
        "title": "Repeated wording across prompts",
        "why": "Terms repeated: {terms}.",
        "action": "Vary the vocabulary; repetition reads as a short profile even when it is not.",
    },
    "incoherent_narrative": {
        "severity": "low",
        "title": "Your prompt answers read as unrelated",
        "why": "Mean cross-prompt similarity {value}; the blocks do not share themes.",
        "action": "Link answers with a recurring theme (a city, a hobby, a kind of humour).",
    },
    "overcoherent_narrative": {
        "severity": "low",
        "title": "Your prompt answers repeat the same idea",
        "why": "Mean cross-prompt similarity {value} is high; the blocks restate one point.",
        "action": "Use one prompt for the theme and let the others add new information.",
    },
    "text_image_gap": {
        "severity": "medium",
        "title": "Prose and photos evoke different things",
        "why": "Mean CLIP similarity between your text and your best-matching photo is "
               "{value}. This is a description-consistency signal, not a honesty signal.",
        "action": "If your prose talks about the outdoors, make sure a photo shows it.",
    },
    "exposure_issues": {
        "severity": "medium",
        "title": "Exposure problems detected",
        "why": "Photos flagged: {photos}. Clipped highlights or crushed shadows lose detail.",
        "action": "Prefer photos where faces are not clipped to white or lost in shadow.",
    },
}


def _fmt(template: str, **kwargs) -> str:
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError):
        return template


def build_recommendations(report: dict) -> list[dict]:
    recs: list[dict] = []
    photos = report["photos"]
    if not photos:
        return recs

    first = photos[0]
    if first["quality_components"]["resolution"] < 0.15:
        recs.append({"code": "low_res_first_photo", **RECOMMENDATION_LIBRARY["low_res_first_photo"],
                     "evidence": {"quality_components": first["quality_components"]}})

    div = report["visual_diversity"]
    if div.get("redundant_pairs"):
        top = max(div["redundant_pairs"], key=lambda p: p["similarity"])
        recs.append({
            "code": "redundant_photos",
            **{k: (_fmt(v, threshold=div["threshold"]) if k == "why" else v)
               for k, v in RECOMMENDATION_LIBRARY["redundant_photos"].items()},
            "evidence": top,
        })
    if div.get("diversity") is not None and div["diversity"] < 0.35:
        recs.append({
            "code": "low_visual_diversity",
            **{k: (_fmt(v, value=div["diversity"]) if k == "why" else v)
               for k, v in RECOMMENDATION_LIBRARY["low_visual_diversity"].items()},
            "evidence": {"diversity": div["diversity"]},
        })

    ordering = report.get("ordering_suggestion")
    if ordering and ordering["improvement"] > 1e-4 and ordering["best_index"] != 0:
        recs.append({
            "code": "weak_first_photo",
            **{k: (_fmt(v, best=ordering["best_index"] + 1, delta=round(ordering["improvement"], 4))
                  if k == "why" else v)
               for k, v in RECOMMENDATION_LIBRARY["weak_first_photo"].items()},
            "evidence": ordering,
        })

    for block in report["text_blocks"]:
        idx = block["index"]
        gp = block["generic_phrases"]
        if gp["hits"]:
            recs.append({
                "code": "generic_prompt_language", "block_index": idx,
                **{k: (_fmt(v, phrases=", ".join(gp["hits"])) if k == "why" else v)
                   for k, v in RECOMMENDATION_LIBRARY["generic_prompt_language"].items()},
                "evidence": gp,
            })
        sp = block["specificity"]
        if sp["score"] < 0.35:
            recs.append({
                "code": "low_specificity", "block_index": idx,
                **{k: (_fmt(v, value=sp["score"]) if k == "why" else v)
                   for k, v in RECOMMENDATION_LIBRARY["low_specificity"].items()},
                "evidence": sp["components"],
            })
        rp = block["repetition"]
        if rp["repetition_ratio"] > 0.12:
            recs.append({
                "code": "repetition", "block_index": idx,
                **{k: (_fmt(v, terms=", ".join(sorted(rp["repeated_terms"]))) if k == "why" else v)
                   for k, v in RECOMMENDATION_LIBRARY["repetition"].items()},
                "evidence": rp,
            })

    coh = report["coherence"]
    if coh.get("mean_similarity") is not None:
        if coh["mean_similarity"] < 0.12:
            recs.append({"code": "incoherent_narrative",
                         **{k: (_fmt(v, value=coh["mean_similarity"]) if k == "why" else v)
                            for k, v in RECOMMENDATION_LIBRARY["incoherent_narrative"].items()},
                         "evidence": coh})
        elif coh["mean_similarity"] > 0.62:
            recs.append({"code": "overcoherent_narrative",
                         **{k: (_fmt(v, value=coh["mean_similarity"]) if k == "why" else v)
                            for k, v in RECOMMENDATION_LIBRARY["overcoherent_narrative"].items()},
                         "evidence": coh})

    ti = report.get("text_image_consistency")
    if ti and ti["mean_top_image_similarity"] < 0.22:
        recs.append({"code": "text_image_gap",
                     **{k: (_fmt(v, value=ti["mean_top_image_similarity"]) if k == "why" else v)
                        for k, v in RECOMMENDATION_LIBRARY["text_image_gap"].items()},
                     "evidence": ti})

    flagged = [i for i, p in enumerate(photos) if "exposure_clipping" in p["flags"]]
    if flagged:
        recs.append({"code": "exposure_issues",
                     **{k: (_fmt(v, photos=", ".join(str(i + 1) for i in flagged))
                            if k == "why" else v)
                        for k, v in RECOMMENDATION_LIBRARY["exposure_issues"].items()},
                     "evidence": {"photo_indices": flagged}})

    order = {"high": 0, "medium": 1, "low": 2}
    recs.sort(key=lambda r: (order[r["severity"]], -len(str(r.get("evidence", {})))))
    return recs


def build_report(images: list, texts: list[dict], ordering_suggestion: dict | None = None) -> dict:
    """images: list of np.ndarray; texts: list of {"kind","prompt","text"} dicts."""
    photo_records = []
    embeddings = []
    for i, img in enumerate(images):
        rec = image_analysis.analyze_photo(img)
        rec["index"] = i
        photo_records.append(rec)
        embeddings.append(np.asarray(rec["embedding"], dtype=np.float64))

    emb_matrix = np.stack(embeddings) if embeddings else None
    diversity = image_analysis.visual_diversity(emb_matrix) if emb_matrix is not None else {
        "mean_pairwise_similarity": None, "diversity": None, "redundant_pairs": [],
        "nearest_neighbour": [], "threshold": 0.92,
    }

    text_records = []
    for i, t in enumerate(texts):
        rec = text_analysis.analyse_text_block(t["text"])
        rec.update({"index": i, "kind": t.get("kind", "prompt"), "prompt": t.get("prompt")})
        text_records.append(rec)

    coh = text_analysis.coherence([t["text"] for t in texts])
    consistency = text_analysis.text_image_consistency(
        [t["text"] for t in texts], emb_matrix
    )

    strengths = []
    for p in photo_records:
        strengths.append({"index": p["index"], "quality": p["quality_estimate"]})
    strengths.sort(key=lambda x: x["quality"], reverse=True)

    report = {
        "photos": photo_records,
        "text_blocks": text_records,
        "visual_diversity": diversity,
        "coherence": coh,
        "text_image_consistency": consistency,
        "ordering_suggestion": ordering_suggestion,
        "strongest_photos": strengths[:3],
        "weakest_photos": list(reversed(strengths))[:2],
    }
    report["recommendations"] = build_recommendations(report)
    report["confidence"] = _confidence(photo_records, text_records, diversity)
    report["limitations"] = _limitations(report)
    return report


def _confidence(photos: list[dict], texts: list[dict], diversity: dict) -> dict:
    """A conservative, explicitly hand-specified confidence summary.

    These numbers express how much *measured input* the report rests on, not a
    statistical confidence level. The UI labels them "evidence coverage".
    """
    n_photos = len(photos)
    n_texts = len(texts)
    has_scenes = any(p["scene_distribution"] for p in photos)
    has_faces = any(p["faces"] is not None for p in photos)
    coverage = (
        0.30 * min(1.0, n_photos / 6.0)
        + 0.25 * min(1.0, n_texts / 3.0)
        + 0.20 * float(has_scenes)
        + 0.15 * float(has_faces)
        + 0.10 * float(diversity.get("diversity") is not None)
    )
    return {
        "evidence_coverage": round(float(np.clip(coverage, 0.0, 1.0)), 3),
        "inputs": {
            "photographs": n_photos,
            "text_blocks": n_texts,
            "zero_shot_scenes": has_scenes,
            "face_detection": has_faces,
        },
        "interpretation": "Coverage of measured evidence, not a probability that the "
                          "recommendations are correct.",
    }


def _limitations(report: dict) -> list[str]:
    out = [
        "Quality scores are documented pixel heuristics, not human aesthetic judgements "
        "and not a validated model of attraction.",
        "The ordering objective is a preference function chosen by this project; the "
        "weights have not been fitted to outcome data.",
        "Similarity thresholds (e.g. the redundant-photo cut-off) are product decisions.",
    ]
    if report.get("text_image_consistency") is None:
        out.append(
            "Cross-modal text-image consistency is unavailable because at least one of the "
            "pretrained encoders (OpenCLIP / Sentence-Transformers) is not loaded."
        )
    if any(p["scene_distribution"] is None for p in report["photos"]):
        out.append("Scene labels are unavailable for some photos; no fallback taxonomy is "
                   "fabricated in their place.")
    out.append(
        "No inference is made about ethnicity, religion, health, or any other sensitive "
        "characteristic, by design and by construction of the label set."
    )
    return out
