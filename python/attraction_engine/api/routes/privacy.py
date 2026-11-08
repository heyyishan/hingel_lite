"""Privacy: export, deletion, retention and audit."""
from __future__ import annotations

import datetime as dt
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, Response
from sqlalchemy import select

from ...db.models import (AnalyticsEvent, Conversation, Experiment, Match, Photograph,
                          Profile, Prompt, User)
from ...db.session import session_scope
from ...services import auth, cache, storage
from ..deps import current_user, rate_limiter

router = APIRouter(prefix="/privacy", tags=["privacy"],
                   dependencies=[Depends(rate_limiter)])


@router.get("/export")
def export(user_id: str = Depends(current_user)):
    """Full data export as a single JSON document (GDPR-style portability)."""
    with session_scope() as s:
        user = s.get(User, user_id)
        profiles = s.scalars(select(Profile).where(Profile.user_id == user_id)).all()
        payload = {
            "export_version": 1,
            "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "user": {"id": user.id, "handle": user.handle,
                     "created_at": user.created_at.isoformat() if user.created_at else None},
            "profiles": [{
                "id": p.id, "name": p.name, "description": p.description,
                "is_primary": p.is_primary, "interests": p.interests,
                "structured_prefs": p.structured_prefs,
                "prompts": [{"kind": q.kind, "question": q.question, "answer": q.answer,
                             "position": q.position} for q in sorted(p.prompts,
                                                                     key=lambda x: x.position)],
                "photos": [{"id": ph.id, "sha256": ph.sha256, "position": ph.position,
                            "width": ph.width, "height": ph.height,
                            "analysis": ph.analysis} for ph in sorted(p.photos,
                                                                      key=lambda x: x.position)],
            } for p in profiles],
            "experiments": [{
                "id": e.id, "name": e.name, "hypothesis": e.hypothesis,
                "variants": [{"label": v.label, "successes": v.successes,
                              "failures": v.failures, "missing": v.missing}
                             for v in e.variants],
            } for e in s.scalars(select(Experiment).where(Experiment.user_id == user_id)).all()],
            "conversations": [{"id": c.id, "label": c.label, "analytics": c.analytics}
                              for c in s.scalars(select(Conversation)
                                                 .where(Conversation.user_id == user_id)).all()],
            "matches": [{"candidate_ref": m.candidate_ref, "score": m.score,
                         "feedback": m.feedback}
                        for m in s.scalars(select(Match).where(Match.user_id == user_id)).all()],
        }
    return JSONResponse(content=payload, headers={
        "Content-Disposition": 'attachment; filename="attraction-engine-export.json"'})


@router.delete("/account")
def delete_account(confirm: str, user_id: str = Depends(current_user)):
    """Irreversible erasure: database rows (cascading) and stored photograph bytes."""
    if confirm != "DELETE":
        raise HTTPException(status_code=400,
                            detail='send confirm="DELETE" to acknowledge irretrievable deletion')
    files = storage.purge_user(user_id)
    removed = cache.invalidate_prefix(f"ae:")
    with session_scope() as s:
        user = s.get(User, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="user not found")
        s.delete(user)          # cascades to profiles, photos, experiments, conversations
    return {"deleted": True, "files_removed": files, "cache_keys_removed": removed}


@router.post("/retention/sweep")
def retention_sweep(user_id: str = Depends(current_user)):
    """Delete photographs whose content has not changed in `retention_days`.

    Retention is opt-in and off by default (365 days in the schema); nothing is
    removed unless the sweep is explicitly triggered.
    """
    cutoff = dt.datetime.now(dt.timezone.utc)
    with session_scope() as s:
        user = s.get(User, user_id)
        days = user.retention_days if user else 365
        horizon = cutoff - dt.timedelta(days=days)
        stale = s.scalars(select(Photograph)
                          .join(Profile, Photograph.profile_id == Profile.id)
                          .where(Profile.user_id == user_id,
                                 Photograph.created_at < horizon)).all()
        removed = []
        for ph in stale:
            removed.append(ph.id)
            s.delete(ph)
    return {"retention_days": days, "cutoff": horizon.isoformat(),
            "photographs_removed": len(removed), "ids": removed}


@router.get("/audit")
def audit(user_id: str = Depends(current_user)):
    with session_scope() as s:
        events = s.scalars(select(AnalyticsEvent)
                           .where(AnalyticsEvent.user_id == user_id)
                           .order_by(AnalyticsEvent.created_at.desc()).limit(100)).all()
        return {"events": [{"kind": e.kind, "payload": e.payload,
                            "at": e.created_at.isoformat() if e.created_at else None}
                           for e in events],
                "storage": storage.disk_usage(user_id)}


@router.get("/policy")
def policy():
    return {
        "local_first": True,
        "data_leaves_the_host": False,
        "third_party_services": [
            "HuggingFace Hub is contacted once to download pretrained encoder weights "
            "(OpenCLIP ViT-B/32, all-MiniLM-L6-v2). No profile data is sent.",
        ],
        "never_collected": ["real names", "email addresses", "phone numbers",
                            "location history", "contacts", "other people's scraped profiles"],
        "sensitive_inference": "No inference about ethnicity, religion, health or sexual "
                               "orientation is performed anywhere in the pipeline.",
        "automation": "No swiping automation, no message generation, no scraping, no platform "
                      "API circumvention.",
        "deletion": "DELETE /api/privacy/account removes rows and stored files irreversibly.",
    }
