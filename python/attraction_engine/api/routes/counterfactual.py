"""Module E: counterfactual profile simulator."""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from ...db.models import Profile
from ...db.session import session_scope
from ...ml import counterfactual as cf
from ...ml import image_analysis, text_analysis
from ...services.vectors import load_photo_vectors
from ..deps import current_user, rate_limiter
from ..schemas import CounterfactualIn

router = APIRouter(prefix="/counterfactual", tags=["counterfactual"],
                   dependencies=[Depends(rate_limiter)])


def _photo_records(session, profile, photo_ids: list[str] | None) -> list[dict]:
    photos = sorted(profile.photos, key=lambda p: p.position)
    vectors = {v.photo_id: v.vector for v in load_photo_vectors(session, profile.id)}
    if photo_ids is not None:
        by_id = {p.id: p for p in photos}
        missing = [pid for pid in photo_ids if pid not in by_id]
        if missing:
            raise HTTPException(status_code=404, detail=f"unknown photo ids: {missing}")
        photos = [by_id[pid] for pid in photo_ids]
    recs = []
    for p in photos:
        if p.id not in vectors:
            raise HTTPException(status_code=409,
                                detail=f"photo {p.id} has no embedding; re-upload it")
        recs.append({"embedding": vectors[p.id].tolist(), "quality_estimate": p.quality,
                     "dominant_scene": (p.analysis or {}).get("dominant_scene"), "id": p.id,
                     "position": p.position})
    return recs


def _text_records(profile, overrides: list[str | None] | None) -> list[dict]:
    """Build text blocks; a None override keeps the profile's stored text."""
    current = []
    for p in sorted(profile.prompts, key=lambda x: x.position):
        current.append(p.answer)
    if profile.description:
        current.append(profile.description)
    texts = current
    if overrides is not None:
        texts = []
        for i, override in enumerate(overrides):
            if override is None and i < len(current):
                texts.append(current[i])
            elif override is not None:
                texts.append(override)
    return [{"text": t, "specificity": text_analysis.specificity(t), "index": i}
            for i, t in enumerate(texts) if t]


@router.post("/simulate")
def simulate(body: CounterfactualIn, user_id: str = Depends(current_user)):
    with session_scope() as s:
        profile = s.get(Profile, body.profile_id)
        if profile is None or profile.user_id != user_id:
            raise HTTPException(status_code=404, detail="profile not found")
        base_photos = _photo_records(s, profile, body.baseline_photo_ids)
        mod_photos = _photo_records(s, profile, body.modified_photo_ids)
        base_texts = _text_records(profile, body.baseline_prompts)
        mod_texts = _text_records(profile, body.modified_prompts)

    base_cfg = cf.build_config(base_photos, base_texts,
                               selection=list(range(len(base_photos))))
    mod_cfg = cf.build_config(mod_photos, mod_texts,
                              selection=list(range(len(mod_photos))))
    if len(mod_photos) != len(base_photos):
        raise HTTPException(
            status_code=422,
            detail="baseline and modified configurations must contain the same number of "
                   "photographs; build both selections from one photo set")
    result = cf.simulate(base_cfg, mod_cfg, seed=body.seed)
    result["baseline"] = {"photo_ids": [p["id"] for p in base_photos],
                          "texts": [t["text"] for t in base_texts]}
    result["modified"] = {"photo_ids": [p["id"] for p in mod_photos],
                          "texts": [t["text"] for t in mod_texts]}
    return result


@router.post("/preview")
def preview(body: dict, user_id: str = Depends(current_user)):
    """Quick text-only counterfactual: compare two draft prompt answers."""
    original = body.get("original", "")
    revised = body.get("revised", "")
    if not revised:
        raise HTTPException(status_code=422, detail="revised text required")
    a = text_analysis.analyse_text_block(original)
    b = text_analysis.analyse_text_block(revised)
    q = lambda r: 0.55 * r["specificity"]["score"] + 0.30 * r["semantic_richness"]["score"] + 0.15 * r["generic_phrases"]["score"]
    return {
        "original": {"specificity": a["specificity"]["score"],
                     "richness": a["semantic_richness"]["score"],
                     "generic": a["generic_phrases"]["hits"],
                     "q_text": round(q(a), 5)},
        "revised": {"specificity": b["specificity"]["score"],
                    "richness": b["semantic_richness"]["score"],
                    "generic": b["generic_phrases"]["hits"],
                    "q_text": round(q(b), 5)},
        "delta_q_text": round(q(b) - q(a), 5),
        "interpretation": "Declared text-quality function only (specificity 0.55, richness 0.30, "
                          "generic-phrase penalty 0.15). Not an estimated effect on people.",
    }
