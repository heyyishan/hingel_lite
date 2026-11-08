"""Module A, text branch: prompt and description analysis.

All scores are computed features with published formulas (documented in
README §Mathematical foundations 4.2 and returned in ``components``). Nothing
here models a reader's private psychology; the signals are properties of *the
user's own text* such as lexical diversity, specificity, and cross-image
semantic agreement.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from . import fallback
from .encoders import text_embedding

_DIGIT_OR_PROPER = re.compile(r"\b(\d+|(?:[A-Z][a-z]+){1,3})\b")
_FIRST_PERSON = re.compile(r"\b(i|i'm|im|my|me|we|our)\b", re.IGNORECASE)
_TIME_OR_PLACE = re.compile(
    r"\b(20\d\d|19\d\d|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|september|october|november|"
    r"december|morning|evening|weekend|summer|winter|spring|autumn)\b",
    re.IGNORECASE,
)
_CONCRETE = re.compile(
    r"\b(coffee|hike|hiking|trail|beach|concert|gig|record|book|novel|kitchen|recipe|"
    r"climb|climbing|surf|yoga|run|running|cycle|cycling|photograph|gallery|museum|"
    r"film|camera|guitar|piano|chess|pottery|garden|bakery|market|train|road ?trip|"
    r"swim|dance|bake|baking|cook|cooking|board ?game|podcast|jazz|techno|festival|"
    r"volunteer|language|spanish|french|german|japanese|painting|drawing)\b",
    re.IGNORECASE,
)


def specificity(text: str) -> dict:
    """Concreteness and specificity of a prompt answer.

    components
      entities    - count of digits / capitalised tokens, saturating at 4
      concrete    - distinct concrete noun hits, saturating at 5
      temporal    - presence of a time or place anchor
      length      - token count inside the 8-28 useful band
      first_person- verbs anchored to the writer (a proxy for specificity of voice)
    """
    tokens = fallback.tokenize(text)
    n = len(tokens)
    entities = len(_DIGIT_OR_PROPER.findall(text))
    concrete_hits = sorted({m.group(0).lower() for m in _CONCRETE.finditer(text)})
    temporal = bool(_TIME_OR_PLACE.search(text))
    first_person = len(_FIRST_PERSON.findall(text))
    length_score = float(np.clip((n - 4) / 10.0, 0.0, 1.0)) * (
        1.0 if n <= 45 else max(0.0, 1.0 - (n - 45) / 60.0)
    )
    score = (
        0.24 * min(1.0, entities / 4.0)
        + 0.30 * min(1.0, len(concrete_hits) / 5.0)
        + 0.12 * float(temporal)
        + 0.24 * length_score
        + 0.10 * min(1.0, first_person / 3.0)
    )
    return {
        "score": round(float(np.clip(score, 0.0, 1.0)), 4),
        "components": {
            "entity_density": round(min(1.0, entities / 4.0), 4),
            "concrete_terms": concrete_hits,
            "concrete_score": round(min(1.0, len(concrete_hits) / 5.0), 4),
            "time_or_place_anchor": temporal,
            "length_score": round(float(length_score), 4),
            "token_count": n,
            "first_person_anchors": first_person,
        },
    }


def semantic_richness(text: str) -> dict:
    toks = fallback.tokenize(text)
    content = fallback.content_tokens(text)
    unique_ratio = (len(set(content)) / len(content)) if content else 0.0
    avg_word_len = float(np.mean([len(t) for t in toks])) if toks else 0.0
    long_words = sum(1 for t in content if len(t) > 6)
    long_ratio = long_words / len(content) if content else 0.0
    score = 0.45 * unique_ratio + 0.30 * min(1.0, avg_word_len / 6.5) + 0.25 * min(1.0, long_ratio * 4)
    return {
        "score": round(float(np.clip(score, 0.0, 1.0)), 4),
        "components": {
            "unique_content_ratio": round(unique_ratio, 4),
            "avg_word_length": round(avg_word_len, 3),
            "long_word_ratio": round(long_ratio, 4),
            "content_tokens": len(content),
            "total_tokens": len(toks),
        },
    }


def generic_phrases(text: str) -> dict:
    hits = fallback.generic_phrase_hits(text)
    toks = max(1, len(fallback.tokenize(text)))
    density = sum(len(h.split()) for h in hits) / toks
    return {
        "hits": hits,
        "density": round(float(density), 4),
        "score": round(float(np.clip(1.0 - density * 2.5, 0.0, 1.0)), 4),
    }


def repetition(text: str) -> dict:
    toks = fallback.tokenize(text)
    counts = Counter(fallback.content_tokens(text))
    repeats = {t: c for t, c in counts.items() if c > 1}
    bigrams = list(zip(toks, toks[1:]))
    bigram_repeats = sum(1 for _, c in Counter(bigrams).items() if c > 1)
    return {
        "repeated_terms": repeats,
        "repetition_ratio": round(fallback.repetition_ratio(text), 4),
        "repeated_bigrams": bigram_repeats,
        "score": round(float(np.clip(1.0 - fallback.repetition_ratio(text) * 2.0
                                     - bigram_repeats * 0.05, 0.0, 1.0)), 4),
    }


def coherence(texts: list[str]) -> dict:
    """Within-profile narrative coherence.

    coherence = mean cosine similarity between the centred profiles' block
    embeddings. Two readings matter and both are reported:
      * too low  -> the blocks read like unrelated people
      * too high -> the blocks repeat the same content
    The "band" score peaks at an empirical target of 0.35 similarity.
    """
    if len(texts) < 2:
        return {"mean_similarity": None, "score": None, "note": "need at least two text blocks"}
    embs = np.stack([text_embedding(t)[0] for t in texts])
    sim = embs @ embs.T
    iu = np.triu_indices(len(texts), k=1)
    mean_sim = float(sim[iu].mean())
    target = 0.35
    band = 1.0 - min(1.0, abs(mean_sim - target) / 0.55)
    return {
        "mean_similarity": round(mean_sim, 4),
        "score": round(float(np.clip(band, 0.0, 1.0)), 4),
        "target": target,
        "pairwise": [
            {"i": int(i), "j": int(j), "similarity": round(float(sim[i, j]), 4)}
            for i, j in zip(*iu)
        ],
    }


def text_image_consistency(texts: list[str], image_embeddings: np.ndarray | None) -> dict | None:
    """Cross-modal agreement, only meaningful when CLIP embeddings are present.

    The image and text towers of CLIP share a space, so cosine similarity
    between a profile's prose embedding and its photographs is interpretable.
    With the fallback encoders the two spaces are unrelated, so this function
    returns None instead of a meaningless number.
    """
    from .encoders import info as encoder_info

    info = encoder_info()
    if image_embeddings is None or not info.image_encoder.startswith("openclip"):
        return None
    if not info.text_encoder.startswith("sentence-transformers"):
        return None
    # Project prose into the CLIP space using the CLIP text tower, not MiniLM.
    from .encoders import clip_text_embedding

    per_text = []
    for t in texts:
        vec, _ = clip_text_embedding(t)
        if vec is None:
            return None
        per_text.append(vec)
    text_mat = np.stack(per_text)
    sim = text_mat @ image_embeddings.T
    best_per_text = sim.max(axis=1)
    return {
        "mean_top_image_similarity": round(float(best_per_text.mean()), 4),
        "best_match_per_text": [
            {"text_index": int(i), "image_index": int(np.argmax(sim[i])),
             "similarity": round(float(best_per_text[i]), 4)}
            for i in range(sim.shape[0])
        ],
        "matrix": [[round(float(v), 4) for v in row] for row in sim],
        "note": "CLIP cross-modal cosine; low values mean the prose and the photos "
                "evoke different scenes, not that the profile is dishonest.",
    }


def analyse_text_block(text: str) -> dict:
    return {
        "text": text,
        "specificity": specificity(text),
        "semantic_richness": semantic_richness(text),
        "generic_phrases": generic_phrases(text),
        "repetition": repetition(text),
        "embedding": text_embedding(text)[0].tolist(),
        "embedding_encoder": text_embedding(text)[1],
    }
