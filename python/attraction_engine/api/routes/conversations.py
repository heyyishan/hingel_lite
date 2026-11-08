"""Module F: conversation analytics."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import select

from ...analytics import conversation
from ...db.models import Conversation
from ...db.session import session_scope
from ..deps import current_user, rate_limiter
from ..schemas import ConversationIn

router = APIRouter(prefix="/conversations", tags=["conversations"],
                   dependencies=[Depends(rate_limiter)])


@router.post("/import")
def import_conversation(body: ConversationIn, user_id: str = Depends(current_user)):
    """Analyse a user-supplied conversation export.

    Only sender labels, timestamps and message text are read. Sender identity is
    collapsed to me/them before anything is stored, and the raw text is never
    persisted — only the derived analytics.
    """
    messages = [{"sender": m.sender, "timestamp": m.timestamp, "text": m.text}
                for m in body.messages]
    try:
        result = conversation.analyse(messages, me=body.me)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    with session_scope() as s:
        row = Conversation(user_id=user_id, label=body.label,
                           message_count=len(messages), analytics=result)
        s.add(row)
        s.flush()
        conv_id = row.id
    result["id"] = conv_id
    result["label"] = body.label
    result["privacy"] = {
        "raw_text_stored": False,
        "note": "Only the derived metrics above were stored; message text was discarded "
                "after analysis.",
    }
    return result


@router.get("")
def list_conversations(user_id: str = Depends(current_user)):
    with session_scope() as s:
        rows = s.scalars(select(Conversation).where(Conversation.user_id == user_id)
                         .order_by(Conversation.created_at.desc())).all()
        return {"conversations": [
            {"id": r.id, "label": r.label, "message_count": r.message_count,
             "created_at": r.created_at.isoformat() if r.created_at else None,
             "summary": {k: (r.analytics or {}).get(k) for k in
                         ("messages", "sessions", "initiation_balance", "continuity")}}
            for r in rows]}


@router.get("/{conversation_id}")
def get_conversation(conversation_id: str, user_id: str = Depends(current_user)):
    with session_scope() as s:
        row = s.get(Conversation, conversation_id)
        if row is None or row.user_id != user_id:
            raise HTTPException(status_code=404, detail="conversation not found")
        return {"id": row.id, "label": row.label, "analytics": row.analytics}


@router.delete("/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, user_id: str = Depends(current_user)):
    with session_scope() as s:
        row = s.get(Conversation, conversation_id)
        if row is None or row.user_id != user_id:
            raise HTTPException(status_code=404, detail="conversation not found")
        s.delete(row)
    return Response(status_code=204)


@router.post("/synthetic")
def synthetic_conversation(body: dict | None = None, user_id: str = Depends(current_user)):
    """Generate and analyse a clearly-labelled synthetic conversation."""
    from ...stats import synthetic as syn

    seed = int((body or {}).get("seed", 7))
    n = int((body or {}).get("n_messages", 80))
    n = max(10, min(n, 2000))
    data = syn.make_conversation(seed=seed, n_messages=n)
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
    messages = [{"sender": m["sender"],
                 "timestamp": base + timedelta(seconds=m["timestamp_offset_s"]),
                 "text": m["text"]} for m in data["messages"]]
    result = conversation.analyse(messages)
    result["synthetic"] = True
    result["label"] = data["label"]
    result["seed"] = seed
    return result
