"""Profile, prompt and photograph management."""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response
from sqlalchemy import select

from ...db.models import Embedding, Photograph, Profile, Prompt
from ...db.session import session_scope
from ...ml import encoders, image_analysis
from ...services import storage
from ..deps import current_user, rate_limiter
from ..schemas import PhotoOrderIn, ProfileIn, ProfileOut, PromptIn, PromptOut

router = APIRouter(prefix="/profiles", tags=["profiles"],
                   dependencies=[Depends(rate_limiter)])

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}


def _owned_profile(session, profile_id: str, user_id: str) -> Profile:
    profile = session.get(Profile, profile_id)
    if profile is None or profile.user_id != user_id:
        raise HTTPException(status_code=404, detail="profile not found")
    return profile


def _to_out(p: Profile) -> ProfileOut:
    return ProfileOut(
        id=p.id, name=p.name, description=p.description, is_primary=p.is_primary,
        structured_prefs=p.structured_prefs or {}, interests=p.interests or [],
        created_at=p.created_at, photo_count=len(p.photos), prompt_count=len(p.prompts),
    )


@router.post("", response_model=ProfileOut, status_code=201)
def create_profile(body: ProfileIn, user_id: str = Depends(current_user)):
    with session_scope() as s:
        if body.is_primary:
            for other in s.scalars(select(Profile).where(Profile.user_id == user_id)).all():
                other.is_primary = False
        profile = Profile(
            user_id=user_id, name=body.name, description=body.description,
            is_primary=body.is_primary, structured_prefs=body.structured_prefs,
            interests=body.interests,
        )
        s.add(profile)
        s.flush()
        for i, p in enumerate(body.prompts):
            s.add(Prompt(profile_id=profile.id, kind=p.kind, question=p.question,
                         answer=p.answer, position=p.position or i))
        s.flush()
        s.refresh(profile)
        return _to_out(profile)


@router.get("", response_model=list[ProfileOut])
def list_profiles(user_id: str = Depends(current_user)):
    with session_scope() as s:
        rows = s.scalars(select(Profile).where(Profile.user_id == user_id)
                         .order_by(Profile.created_at)).all()
        return [_to_out(p) for p in rows]


@router.get("/{profile_id}", response_model=ProfileOut)
def get_profile(profile_id: str, user_id: str = Depends(current_user)):
    with session_scope() as s:
        return _to_out(_owned_profile(s, profile_id, user_id))


@router.patch("/{profile_id}", response_model=ProfileOut)
def update_profile(profile_id: str, body: ProfileIn, user_id: str = Depends(current_user)):
    with session_scope() as s:
        profile = _owned_profile(s, profile_id, user_id)
        profile.name = body.name
        profile.description = body.description
        profile.structured_prefs = body.structured_prefs
        profile.interests = body.interests
        if body.is_primary:
            for other in s.scalars(select(Profile).where(Profile.user_id == user_id)).all():
                other.is_primary = (other.id == profile.id)
        s.flush()
        s.refresh(profile)
        return _to_out(profile)


@router.delete("/{profile_id}", status_code=204)
def delete_profile(profile_id: str, user_id: str = Depends(current_user)):
    with session_scope() as s:
        profile = _owned_profile(s, profile_id, user_id)
        s.delete(profile)
    return Response(status_code=204)


@router.post("/{profile_id}/prompts", response_model=PromptOut, status_code=201)
def add_prompt(profile_id: str, body: PromptIn, user_id: str = Depends(current_user)):
    with session_scope() as s:
        profile = _owned_profile(s, profile_id, user_id)
        prompt = Prompt(profile_id=profile.id, kind=body.kind, question=body.question,
                        answer=body.answer, position=body.position)
        s.add(prompt)
        s.flush()
        return PromptOut.model_validate(prompt)


@router.get("/{profile_id}/prompts", response_model=list[PromptOut])
def list_prompts(profile_id: str, user_id: str = Depends(current_user)):
    with session_scope() as s:
        profile = _owned_profile(s, profile_id, user_id)
        return [PromptOut.model_validate(p)
                for p in sorted(profile.prompts, key=lambda x: x.position)]


@router.delete("/{profile_id}/prompts/{prompt_id}", status_code=204)
def delete_prompt(profile_id: str, prompt_id: str, user_id: str = Depends(current_user)):
    with session_scope() as s:
        profile = _owned_profile(s, profile_id, user_id)
        prompt = s.get(Prompt, prompt_id)
        if prompt is None or prompt.profile_id != profile.id:
            raise HTTPException(status_code=404, detail="prompt not found")
        s.delete(prompt)
    return Response(status_code=204)


@router.post("/{profile_id}/photos", status_code=201)
async def upload_photo(profile_id: str, file: UploadFile = File(...),
                       user_id: str = Depends(current_user)):
    mime = (file.content_type or "").lower()
    if mime not in ALLOWED_MIME:
        raise HTTPException(status_code=415, detail=f"unsupported content type {mime!r}")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file exceeds 12 MiB")
    if not data:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        img = encoders.decode_image(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"cannot decode image: {exc}") from exc

    analysis = image_analysis.analyze_photo(img)
    emb = np.asarray(analysis.pop("embedding"), dtype=np.float64)
    encoder_name = analysis.pop("embedding_encoder")
    # The vector itself lives in the embeddings table (retrieval source of truth);
    # the analysis JSON keeps only provenance so the two can never disagree.
    analysis["embedding_dim"] = int(emb.size)
    analysis["embedding_encoder"] = encoder_name

    with session_scope() as s:
        profile = _owned_profile(s, profile_id, user_id)
        digest, _path, dedup = storage.store_bytes(user_id, data, mime)
        next_pos = max([p.position for p in profile.photos], default=-1) + 1
        photo = Photograph(
            profile_id=profile.id, position=next_pos, sha256=digest, mime=mime,
            width=analysis["width"], height=analysis["height"], bytes=len(data),
            analysis=analysis, quality=analysis["quality_estimate"],
        )
        s.add(photo)
        s.flush()
        s.add(Embedding(photo_id=photo.id, profile_id=profile.id, kind="image",
                        encoder=encoder_name, dim=int(emb.size), vec=emb.tolist()))
        s.flush()
        return {"id": photo.id, "position": photo.position, "deduplicated": dedup,
                "analysis": analysis, "encoder": encoder_name, "sha256": digest}


@router.get("/{profile_id}/photos")
def list_photos(profile_id: str, user_id: str = Depends(current_user)):
    with session_scope() as s:
        profile = _owned_profile(s, profile_id, user_id)
        return {"photos": [
            {"id": p.id, "position": p.position, "sha256": p.sha256, "mime": p.mime,
             "width": p.width, "height": p.height, "bytes": p.bytes, "quality": p.quality,
             "analysis": p.analysis,
             "url": f"/api/profiles/{profile_id}/photos/{p.id}/raw"}
            for p in sorted(profile.photos, key=lambda x: x.position)
        ]}


@router.get("/{profile_id}/photos/{photo_id}/raw")
def photo_raw(profile_id: str, photo_id: str, user_id: str = Depends(current_user)):
    with session_scope() as s:
        profile = _owned_profile(s, profile_id, user_id)
        photo = s.get(Photograph, photo_id)
        if photo is None or photo.profile_id != profile.id:
            raise HTTPException(status_code=404, detail="photo not found")
        data = storage.read(user_id, photo.sha256, photo.mime)
        if data is None:
            raise HTTPException(status_code=410, detail="photograph bytes no longer stored")
        return Response(content=data, media_type=photo.mime,
                        headers={"Cache-Control": "private, max-age=300"})


@router.post("/{profile_id}/photos/order")
def reorder_photos(profile_id: str, body: PhotoOrderIn, user_id: str = Depends(current_user)):
    with session_scope() as s:
        profile = _owned_profile(s, profile_id, user_id)
        owned = {p.id: p for p in profile.photos}
        if set(body.photo_ids) != set(owned):
            raise HTTPException(status_code=400,
                                detail="photo_ids must be a permutation of the profile's photos")
        # two-phase renumbering avoids transient unique-constraint collisions
        for i, pid in enumerate(body.photo_ids):
            owned[pid].position = -1000 - i
        s.flush()
        for i, pid in enumerate(body.photo_ids):
            owned[pid].position = i
        return {"order": body.photo_ids}


@router.delete("/{profile_id}/photos/{photo_id}", status_code=204)
def delete_photo(profile_id: str, photo_id: str, user_id: str = Depends(current_user)):
    with session_scope() as s:
        profile = _owned_profile(s, profile_id, user_id)
        photo = s.get(Photograph, photo_id)
        if photo is None or photo.profile_id != profile.id:
            raise HTTPException(status_code=404, detail="photo not found")
        s.delete(photo)
        s.flush()
        remaining = sorted({p.id: p for p in profile.photos if p.id != photo_id}.values(),
                           key=lambda x: x.position)
        for i, p in enumerate(remaining):
            p.position = -1000 - i
        s.flush()
        for i, p in enumerate(remaining):
            p.position = i
    return Response(status_code=204)
