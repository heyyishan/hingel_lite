"""Single source of truth for stored photo embeddings.

Vectors live in the ``embeddings`` table (pgvector in PostgreSQL). The photograph's
``analysis`` JSON deliberately does NOT duplicate them; it only records the
encoder name and dimensionality so the UI can display provenance. Every caller
that needs vectors goes through this module.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select

from ..db.models import Embedding, Photograph


@dataclass
class PhotoVector:
    photo_id: str
    position: int
    encoder: str
    vector: np.ndarray


def _as_array(vec) -> np.ndarray:
    if isinstance(vec, str):
        return np.asarray(json.loads(vec), dtype=np.float64)
    if hasattr(vec, "to_numpy"):          # pgvector returns numpy on read
        return np.asarray(vec.to_numpy(), dtype=np.float64)
    return np.asarray(list(vec), dtype=np.float64)


def load_photo_vectors(session, profile_id: str) -> list[PhotoVector]:
    """Return vectors ordered by photograph position."""
    rows = session.execute(
        select(Embedding, Photograph)
        .join(Photograph, Photograph.id == Embedding.photo_id)
        .where(Embedding.profile_id == profile_id, Embedding.kind == "image")
        .order_by(Photograph.position)
    ).all()
    return [PhotoVector(photo_id=ph.id, position=ph.position,
                        encoder=emb.encoder, vector=_as_array(emb.vec))
            for emb, ph in rows]


def stack_vectors(vectors: list[PhotoVector]) -> np.ndarray:
    if not vectors:
        raise ValueError("no vectors to stack")
    dims = {v.vector.size for v in vectors}
    if len(dims) != 1:
        raise ValueError(f"mixed embedding dimensionality: {sorted(dims)}; "
                         "re-upload the photographs with one encoder")
    return np.stack([v.vector for v in vectors])


def normalise_embeddings(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms
