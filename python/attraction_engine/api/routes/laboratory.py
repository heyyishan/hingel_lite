"""Module B: photograph ordering laboratory."""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from ...db.models import Photograph, Profile
from ...db.session import session_scope
from ...ml import photo_optimizer
from ...services.vectors import load_photo_vectors, stack_vectors
from ..deps import current_user, rate_limiter
from ..schemas import SensitivityRequest, OptimizeRequest

router = APIRouter(prefix="/laboratory", tags=["laboratory"],
                   dependencies=[Depends(rate_limiter)])


def load_matrix(profile_id: str, user_id: str):
    with session_scope() as s:
        profile = s.get(Profile, profile_id)
        if profile is None or profile.user_id != user_id:
            raise HTTPException(status_code=404, detail="profile not found")
        photos = sorted(profile.photos, key=lambda p: p.position)
        if len(photos) < 2:
            raise HTTPException(status_code=422, detail="need at least two photographs")
        vectors = load_photo_vectors(s, profile.id)
        by_photo = {v.photo_id: v for v in vectors}
        missing = [p.id for p in photos if p.id not in by_photo]
        if missing:
            raise HTTPException(status_code=409,
                                detail=f"photographs without embeddings: {missing}")
        rows = [{"id": p.id, "position": p.position, "quality": p.quality,
                 "dominant_scene": (p.analysis or {}).get("dominant_scene"),
                 "faces": (p.analysis or {}).get("faces"),
                 "embedding_dim": int(by_photo[p.id].vector.size),
                 "encoder": by_photo[p.id].encoder,
                 "flags": (p.analysis or {}).get("flags", []),
                 "width": p.width, "height": p.height} for p in photos]
        try:
            emb = stack_vectors([by_photo[p.id] for p in photos])
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return rows, emb


@router.get("/{profile_id}/photos")
def photo_matrix(profile_id: str, user_id: str = Depends(current_user)):
    rows, emb = load_matrix(profile_id, user_id)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = emb / norms
    sim = unit @ unit.T
    return {
        "photos": rows,
        "similarity_matrix": [[round(float(v), 4) for v in row] for row in sim],
        "dim": int(emb.shape[1]),
    }


@router.post("/{profile_id}/optimize")
def optimize(profile_id: str, body: OptimizeRequest, user_id: str = Depends(current_user)):
    rows, emb = load_matrix(profile_id, user_id)
    quality = np.asarray([r["quality"] for r in rows], dtype=np.float64)
    result = photo_optimizer.optimize_ordering(
        rows, emb, quality, lam=body.lambda_, mu=body.mu, seed=body.seed,
        strategies=body.strategies, beam_width=body.beam_width, sa_iters=body.sa_iterations)
    result["photo_ids"] = [r["id"] for r in rows]
    result["photo_labels"] = [f"#{r['position'] + 1}" for r in rows]
    return result


@router.post("/{profile_id}/sensitivity")
def sensitivity(profile_id: str, body: SensitivityRequest,
                user_id: str = Depends(current_user)):
    rows, emb = load_matrix(profile_id, user_id)
    quality = np.asarray([r["quality"] for r in rows], dtype=np.float64)
    for name, values in (("lambdas", body.lambdas), ("mus", body.mus)):
        if not values or any(not (0.0 <= v <= 1.0) for v in values):
            raise HTTPException(status_code=422,
                                detail=f"{name} must be a non-empty list within [0, 1]")
    return photo_optimizer.sensitivity_analysis(rows, emb, quality,
                                                lambdas=body.lambdas, mus=body.mus,
                                                seed=body.seed)


@router.post("/{profile_id}/ablation")
def ablation(profile_id: str, body: OptimizeRequest, user_id: str = Depends(current_user)):
    rows, emb = load_matrix(profile_id, user_id)
    quality = np.asarray([r["quality"] for r in rows], dtype=np.float64)
    out = photo_optimizer.ablation(rows, emb, quality, seed=body.seed)
    for row in out["rows"]:
        row["photo_id"] = rows[row["removed_index"]]["id"]
        row["position"] = rows[row["removed_index"]]["position"]
    return out


@router.post("/{profile_id}/apply")
def apply_order(profile_id: str, body: dict, user_id: str = Depends(current_user)):
    """Persist an ordering produced by the optimiser (indices -> stored positions)."""
    permutation = body.get("permutation")
    if not isinstance(permutation, list) or not permutation:
        raise HTTPException(status_code=422, detail="permutation (list of indices) required")
    with session_scope() as s:
        profile = s.get(Profile, profile_id)
        if profile is None or profile.user_id != user_id:
            raise HTTPException(status_code=404, detail="profile not found")
        photos = sorted(profile.photos, key=lambda p: p.position)
        if sorted(permutation) != list(range(len(photos))):
            raise HTTPException(status_code=422,
                                detail="permutation must cover every photograph index exactly once")
        ordered = [photos[i] for i in permutation]
        for i, p in enumerate(ordered):
            p.position = -1000 - i
        s.flush()
        for i, p in enumerate(ordered):
            p.position = i
        return {"order": [p.id for p in ordered]}
