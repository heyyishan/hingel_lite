"""Module C: compatibility, retrieval and ranking."""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, text

from ...db.models import Profile
from ...db.session import IS_POSTGRES, engine, session_scope
from ...ml import compatibility
from ...stats import synthetic
from ..deps import current_user, rate_limiter
from ..schemas import RetrievalRequest

router = APIRouter(prefix="/compatibility", tags=["compatibility"],
                   dependencies=[Depends(rate_limiter)])

_POOL_CACHE: dict[str, dict] = {}


def candidate_pool(seed: int, n: int) -> dict:
    key = f"{seed}:{n}"
    if key not in _POOL_CACHE:
        _POOL_CACHE.clear()          # keep exactly one pool in memory (1.9 GiB host)
        _POOL_CACHE[key] = synthetic.make_candidate_pool(
            synthetic.SyntheticSpec(seed=seed, n_candidates=n))
    return _POOL_CACHE[key]


class PoolRequest(BaseModel):
    seed: int = 20260101
    n_candidates: int = Field(default=60, ge=10, le=400)


@router.post("/pool")
def build_pool(body: PoolRequest, user_id: str = Depends(current_user)):
    pool = candidate_pool(body.seed, body.n_candidates)
    vis = compatibility.embedding_visualisation(pool["profiles"], pool["ground_truth_relevance"])
    return {
        "spec": pool["spec"],
        "viewer": pool["viewer"],
        "profiles": pool["profiles"],
        "ground_truth_relevance": pool["ground_truth_relevance"],
        "projection": vis,
        "label": pool["label"],
        "note": "Synthetic candidate pool for evaluating the ranking pipeline. "
                "Ground-truth affinity exists only because the generator plants it.",
    }


class RankRequest(BaseModel):
    pool_seed: int = 20260101
    n_candidates: int = Field(default=60, ge=10, le=400)
    top_k: int = Field(default=15, ge=1, le=200)
    lambda_diversity: float = Field(default=0.0, ge=0.0, le=1.0)
    weights: dict | None = None
    use_neural: bool = True
    compare_baselines: bool = True


@router.post("/rank")
def rank(body: RankRequest, user_id: str = Depends(current_user)):
    pool = candidate_pool(body.pool_seed, body.n_candidates)
    result = compatibility.rank(pool["viewer"], pool["profiles"], top_k=body.top_k,
                                weights=body.weights,
                                lambda_diversity=body.lambda_diversity,
                                use_neural=body.use_neural)
    result["metrics"] = compatibility.ranking_metrics(result["ranking"],
                                                      pool["ground_truth_relevance"],
                                                      k=min(10, body.top_k))
    if body.compare_baselines:
        # Ablations: each modality alone, to show what the weights actually do.
        ablations = {}
        for name, w in {
            "visual_only": {"visual": 1.0, "textual": 0.0, "attributes": 0.0, "interests": 0.0},
            "text_only": {"visual": 0.0, "textual": 1.0, "attributes": 0.0, "interests": 0.0},
            "structured_only": {"visual": 0.0, "textual": 0.0, "attributes": 1.0,
                                "interests": 1.0},
        }.items():
            alt = compatibility.rank(pool["viewer"], pool["profiles"], top_k=body.top_k,
                                     weights=w, lambda_diversity=body.lambda_diversity,
                                     use_neural=False)
            ablations[name] = compatibility.ranking_metrics(
                alt["ranking"], pool["ground_truth_relevance"], k=min(10, body.top_k))
        result["ablations"] = ablations
    return result


@router.get("/model")
def model_info(user_id: str = Depends(current_user)):
    return compatibility.model_status()


@router.post("/retrieval")
def retrieval(body: RetrievalRequest, user_id: str = Depends(current_user)):
    """Exact vs approximate nearest-neighbour retrieval over stored embeddings.

    PostgreSQL + pgvector: exact scan (`<=>` operator) versus HNSW index scan with
    the index in use. On SQLite the ANN path is unavailable and the response says so.
    """
    if body.profile_id is None:
        raise HTTPException(status_code=422, detail="profile_id is required")
    with session_scope() as s:
        profile = s.get(Profile, body.profile_id)
        if profile is None or profile.user_id != user_id:
            raise HTTPException(status_code=404, detail="profile not found")
        rows = list(s.execute(text(
            "select e.id, e.vec, e.encoder from embeddings e "
            "join photographs p on p.id = e.photo_id where e.profile_id = :pid"), {"pid": profile.id}))
    if not rows:
        return {"exact": [], "ann": None, "note": "no photo embeddings to search"}
    first = rows[0][1]
    if isinstance(first, str):
        vectors = [np.asarray(__import__("json").loads(v[1]), dtype=np.float64) for v in rows]
    else:
        vectors = [np.asarray(list(v[1]), dtype=np.float64) for v in rows]
    mat = np.stack(vectors)
    query = mat[0] if body.query_text is None else mat.mean(axis=0)
    query = query / np.linalg.norm(query)

    import time
    t0 = time.perf_counter()
    exact_scores = mat @ query
    exact_idx = np.argsort(-exact_scores)[: body.top_k]
    exact_ms = (time.perf_counter() - t0) * 1000.0
    exact = [{"id": rows[i][0], "score": round(float(exact_scores[i]), 5)} for i in exact_idx]

    ann = None
    if IS_POSTGRES and body.use_ann:
        try:
            with engine.connect() as conn:
                conn.execute(text("SET LOCAL hnsw.ef_search = :ef"), {"ef": body.ef_search})
                t0 = time.perf_counter()
                ann_rows = conn.execute(text(
                    "select id, 1 - (vec <=> CAST(:q AS vector)) as score from embeddings "
                    "where profile_id = :pid and kind = 'image' "
                    "order by vec <=> CAST(:q AS vector) limit :k"),
                    {"q": "[" + ",".join(f"{x:.8f}" for x in query) + "]",
                     "pid": profile.id, "k": body.top_k}).fetchall()
                ann_ms = (time.perf_counter() - t0) * 1000.0
            ann = {"results": [{"id": r[0], "score": round(float(r[1]), 5)} for r in ann_rows],
                   "elapsed_ms": round(ann_ms, 3), "ef_search": body.ef_search}
            exact_ids = [e["id"] for e in exact]
            ann_ids = [a["id"] for a in ann["results"]]
            hits = len(set(exact_ids) & set(ann_ids))
            ann["recall_at_k"] = round(hits / max(1, len(exact_ids)), 4)
        except Exception as exc:
            ann = {"error": str(exc)}

    return {
        "exact": exact,
        "exact_ms": round(exact_ms, 3),
        "ann": ann,
        "k": body.top_k,
        "note": "Exact scan is a full matrix product; ANN uses the pgvector HNSW index. "
                "recall_at_k compares ANN results against the exact ordering on the same data."
                if IS_POSTGRES else
        "Exact scan only: this database backend has no ANN index.",
    }


@router.post("/feedback")
def feedback(body: dict, user_id: str = Depends(current_user)):
    """Manual preference feedback: +1 like, -1 dislike on a candidate reference.

    Stored as an audit trail. It does *not* silently retrain anything; the
    Experiments and Compatibility screens show the accumulated signal and the
    documented next step (re-fit weights) rather than pretending to learn online.
    """
    from ...db.models import Match

    ref = body.get("candidate_ref")
    value = body.get("feedback")
    if not isinstance(ref, str) or value not in (-1, 0, 1):
        raise HTTPException(status_code=422, detail="candidate_ref and feedback in {-1,0,1} required")
    with session_scope() as s:
        s.add(Match(user_id=user_id, candidate_ref=ref,
                    score=float(body.get("score", 0.0)), feedback=int(value),
                    terms=body.get("terms") or {}))
    return {"ok": True, "stored": {"candidate_ref": ref, "feedback": value}}


@router.get("/feedback")
def feedback_list(user_id: str = Depends(current_user)):
    from ...db.models import Match

    with session_scope() as s:
        rows = s.scalars(select(Match).where(Match.user_id == user_id,
                                            Match.feedback != 0)
                         .order_by(Match.created_at.desc()).limit(200)).all()
        likes = sum(1 for r in rows if r.feedback > 0)
        return {"count": len(rows), "likes": likes, "dislikes": len(rows) - likes,
                "items": [{"candidate_ref": r.candidate_ref, "feedback": r.feedback,
                           "score": r.score, "created_at": r.created_at.isoformat()
                           if r.created_at else None} for r in rows]}
