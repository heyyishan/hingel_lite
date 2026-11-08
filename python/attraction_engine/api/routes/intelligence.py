"""Module A: multimodal profile intelligence report."""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select

from ...db.models import Photograph, Profile, Prompt
from ...db.session import session_scope
from ...ml import encoders, profile_report
from ...services import cache, storage
from ...services.vectors import load_photo_vectors
from ..deps import current_user, rate_limiter

router = APIRouter(prefix="/intelligence", tags=["intelligence"],
                   dependencies=[Depends(rate_limiter)])


def load_profile_bundle(profile_id: str, user_id: str):
    """Fetch the stored profile as (profile, images, text blocks, photo rows)."""
    with session_scope() as s:
        profile = s.get(Profile, profile_id)
        if profile is None or profile.user_id != user_id:
            raise HTTPException(status_code=404, detail="profile not found")
        photos = sorted(profile.photos, key=lambda p: p.position)
        prompts = sorted(profile.prompts, key=lambda p: p.position)
        images = []
        for p in photos:
            raw = storage.read(user_id, p.sha256, p.mime)
            if raw is None:
                raise HTTPException(status_code=410,
                                    detail=f"photo {p.id} bytes are missing from storage")
            images.append(encoders.decode_image(raw))
        texts = []
        for p in prompts:
            texts.append({"kind": p.kind, "prompt": p.question, "text": p.answer})
        if profile.description:
            texts.append({"kind": "bio", "prompt": None, "text": profile.description})
        vectors = {v.photo_id: v.vector for v in load_photo_vectors(s, profile.id)}
        photo_rows = [{"id": p.id, "position": p.position, "quality": p.quality,
                       "analysis": p.analysis,
                       "embedding": None if p.id not in vectors
                       else vectors[p.id].tolist()} for p in photos]
        return ({"id": profile.id, "name": profile.name,
                 "interests": profile.interests or [],
                 "structured_prefs": profile.structured_prefs or {}},
                images, texts, photo_rows, [p.id for p in photos])


def _ordering_suggestion(images, photo_rows):
    """Cheap beam-search suggestion used by the report (full lab does more)."""
    if len(images) < 3:
        return None
    from ...ml import photo_optimizer

    encs = [np.asarray(r["embedding"] or [], dtype=np.float64) for r in photo_rows]
    dims = {e.size for e in encs}
    if len(dims) != 1 or encs[0].size == 0:
        return None
    emb = np.stack(encs)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms
    photos_meta = [{"dominant_scene": r["analysis"].get("dominant_scene"),
                    "faces": r["analysis"].get("faces")} for r in photo_rows]
    quality = np.asarray([r["quality"] for r in photo_rows], dtype=np.float64)
    res = photo_optimizer.optimize_ordering(photos_meta, emb, quality,
                                            strategies=["beam"], beam_width=8)
    best = res["best"]
    return {
        "permutation": best["permutation"],
        "best_index": best["permutation"][0],
        "improvement": best["improvement_vs_upload_order"],
        "objective": best["objective"],
        "baseline_objective": res["baseline"]["objective"],
    }


@router.post("/{profile_id}/report")
def analyse_profile(profile_id: str, refresh: bool = False,
                    user_id: str = Depends(current_user)):
    profile, images, texts, photo_rows, photo_ids = load_profile_bundle(profile_id, user_id)
    if not images and not texts:
        raise HTTPException(status_code=422,
                            detail="profile has no photographs or text to analyse")
    key = f"ae:report:{profile_id}:{hash(tuple(photo_ids))}"
    if not refresh:
        cached = cache.get(key)
        if cached is not None:
            cached["cached"] = True
            return cached

    ordering = _ordering_suggestion(images, photo_rows)
    report = profile_report.build_report(images, texts, ordering)
    report["photo_ids"] = photo_ids
    report["profile"] = profile
    report["encoders"] = encoders.info().as_dict()
    report["cached"] = False
    report["synthetic_inputs"] = False
    cache.set(key, report)
    return report


@router.get("/{profile_id}/latest")
def latest_report(profile_id: str, user_id: str = Depends(current_user)):
    """Reports are recomputed from stored analyses; this endpoint returns the
    cheap subset that requires no image decoding (per-photo metrics only)."""
    with session_scope() as s:
        profile = s.get(Profile, profile_id)
        if profile is None or profile.user_id != user_id:
            raise HTTPException(status_code=404, detail="profile not found")
        photos = sorted(profile.photos, key=lambda p: p.position)
        return {
            "profile_id": profile_id,
            "photos": [{"id": p.id, "position": p.position, "quality": p.quality,
                        "analysis": p.analysis} for p in photos],
            "note": "POST /report for the full multimodal analysis.",
        }
