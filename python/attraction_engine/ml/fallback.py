"""Local, dependency-free feature extractors.

These are the documented fallback when pretrained weights cannot be downloaded.
They are *deterministic* functions of the input pixels, so embeddings, reports
and experiments built on them are fully reproducible — a property the pretrained
encoders do not have across library versions.

They are NOT a substitute for a vision-language model: the descriptor captures
colour layout, local contrast and gradient orientation statistics, not semantics.
Every API response that uses them carries ``encoder: "fallback:..."`` so the UI
can label the provenance.
"""
from __future__ import annotations

import hashlib

import numpy as np


def _blocks(img: np.ndarray, grid: int) -> list[np.ndarray]:
    """Split an (H, W, C) array into a grid x grid list of blocks."""
    h, w = img.shape[:2]
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    out = []
    for gy in range(grid):
        for gx in range(grid):
            block = img[ys[gy]:ys[gy + 1], xs[gx]:xs[gx + 1]]
            if block.size == 0:
                block = np.zeros((1, 1, img.shape[2]), dtype=img.dtype)
            out.append(block.reshape(-1, img.shape[2]))
    return out


def _gradient_histograms(img: np.ndarray, bins: int = 8) -> np.ndarray:
    """Orientation histogram of gradient magnitudes on the luma plane."""
    luma = img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722
    gx = np.zeros_like(luma)
    gy = np.zeros_like(luma)
    gx[:, 1:-1] = luma[:, 2:] - luma[:, :-2]
    gy[1:-1, :] = luma[2:, :] - luma[:-2, :]
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)
    idx = np.clip((ang * bins).astype(int), 0, bins - 1)
    hist = np.bincount(idx.ravel(), weights=mag.ravel(), minlength=bins)
    total = hist.sum()
    return hist / total if total > 0 else hist


def fallback_image_embedding(img: np.ndarray, dim: int = 256) -> np.ndarray:
    """Deterministic visual descriptor for an (H, W, 3) float array in [0, 1].

    Layout (grid=4):
      * 4x4 blocks x (mean R, G, B)       -> 48
      * 4x4 blocks x (luma std, |gx|, |gy|) -> 48
      * 8     gradient-orientation bins   ->  8
      * global colour moments (RGB mean, std) -> 6
      * 3     luminance percentiles (p10, p50, p90) -> 3
    Total 113 values expanded to `dim` by a fixed orthogonal projection seeded
    from the descriptor's own hash — this keeps cross-image comparability while
    matching the requested dimensionality of the pretrained encoders.
    """
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    grid = 4
    feats: list[float] = []
    luma = img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722
    for block in _blocks(img, grid):
        feats.extend(block.mean(axis=0).tolist())
    for block in _blocks(luma[..., None], grid):
        b = block[:, 0]
        gx = np.abs(np.diff(b.reshape(-1))) if b.size > 1 else np.zeros(1)
        feats.extend([float(b.std()), float(gx.mean()), float(b.max() - b.min())])
    feats.extend(_gradient_histograms(img).tolist())
    feats.extend(img.reshape(-1, 3).mean(axis=0).tolist())
    feats.extend(img.reshape(-1, 3).std(axis=0).tolist())
    feats.extend(np.percentile(luma, [10, 50, 90]).tolist())
    base = np.asarray(feats, dtype=np.float64)

    if base.size >= dim:
        vec = base[:dim]
    else:
        # deterministic, data-independent projection matrix
        seed = int.from_bytes(hashlib.blake2b(b"ae-fallback-img-v1", digest_size=8).digest(), "big")
        rng = np.random.default_rng(seed)
        proj = rng.normal(size=(base.size, dim)) / np.sqrt(base.size)
        vec = base @ proj
    vec = vec - vec.mean()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


_GENERIC_PHRASES = {
    "love to laugh", "love travelling", "love travel", "foodie", "live laugh love",
    "looking for the one", "partner in crime", "adventure buddy", "work hard play hard",
    "good vibes only", "fluent in sarcasm", "coffee lover", "dog mom", "dog dad",
    "cat mom", "cat dad", "living my best life", "here for a good time",
    "swipe right", "no drama", "family oriented", "young at heart", "glass half full",
    "if you know you know", "carpe diem", "yolo", "wanderlust", "gym rat",
}

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "for", "with",
    "is", "am", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that",
    "these", "those", "i", "me", "my", "we", "our", "you", "your", "he", "she", "they",
    "them", "as", "by", "from", "about", "into", "over", "after", "so", "than", "too",
    "very", "just", "can", "will", "not", "no", "yes", "do", "does", "did", "have", "has",
}


def tokenize(text: str) -> list[str]:
    cleaned = []
    for ch in text.lower():
        cleaned.append(ch if ch.isalnum() or ch in " '-" else " ")
    tokens = "".join(cleaned).split()
    return [t.strip("'-") for t in tokens if t.strip("'-")]


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 2]


def fallback_text_embedding(text: str, dim: int = 384) -> np.ndarray:
    """Hashed bag-of-bigrams with sublinear term weighting.

    Deterministic and stateless. Quality is far below a sentence encoder — it
    cannot represent negation or word order — which is why the product always
    discloses the encoder used alongside any text-derived score.
    """
    toks = content_tokens(text)
    if not toks:
        return np.zeros(dim)
    vec = np.zeros(dim)
    grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    for i, gram in enumerate(grams):
        h = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = 1.0 if h[4] & 1 else -1.0
        weight = 1.0 if "_" in gram else 1.5
        vec[idx] += sign * weight
    vec = np.sign(vec) * np.log1p(np.abs(vec))
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def generic_phrase_hits(text: str) -> list[str]:
    low = " ".join(tokenize(text))
    low = f" {low} "
    return sorted({p for p in _GENERIC_PHRASES if f" {p} " in low or low.strip() == p})


def lexical_diversity(text: str) -> float:
    toks = tokenize(text)
    if not toks:
        return 0.0
    return len(set(toks)) / len(toks)


def repetition_ratio(text: str) -> float:
    """Fraction of content tokens that are repeats of an earlier token."""
    toks = content_tokens(text)
    if not toks:
        return 0.0
    seen: set[str] = set()
    repeats = 0
    for t in toks:
        if t in seen:
            repeats += 1
        else:
            seen.add(t)
    return repeats / len(toks)
