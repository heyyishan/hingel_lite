"""Multimodal encoders with an explicit two-tier contract.

Tier 1 (preferred)  : OpenCLIP ViT-B-32 (laion2b_s34b_b79k) for images and CLIP
                      text; Sentence-Transformers all-MiniLM-L6-v2 for profile
                      prose. Both are CPU-friendly and were verified to download
                      and run in this environment (see docs/environment.md).
Tier 2 (fallback)   : the deterministic numpy descriptors in fallback.py.

The tier actually used is always reported through ``EncoderInfo`` so that no
score, report or chart can silently mix provenances.
"""
from __future__ import annotations

import functools
import hashlib
import io
import logging
import threading
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from . import fallback

log = logging.getLogger(__name__)

ImageEncoderName = Literal["openclip:ViT-B-32", "fallback:visual-descriptor-v1"]
TextEncoderName = Literal["sentence-transformers:all-MiniLM-L6-v2", "fallback:hashed-bigram-v1"]


@dataclass(frozen=True)
class EncoderInfo:
    image_encoder: str
    text_encoder: str
    image_dim: int
    text_dim: int
    l2_normalized: bool = True
    deterministic: bool = False
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "image_encoder": self.image_encoder,
            "text_encoder": self.text_encoder,
            "image_dim": self.image_dim,
            "text_dim": self.text_dim,
            "l2_normalized": self.l2_normalized,
            "deterministic": self.deterministic,
            "notes": self.notes,
        }


class _State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None
        self.clip_failed = False
        self.st_model = None
        self.st_failed = False
        self.info: EncoderInfo | None = None


_STATE = _State()
_CLIP_NAME = "ViT-B-32"
_CLIP_PRETRAINED = "laion2b_s34b_b79k"
_ST_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _load_clip() -> bool:
    with _STATE.lock:
        if _STATE.clip_model is not None:
            return True
        if _STATE.clip_failed:
            return False
        try:
            import open_clip
            import torch  # noqa: F401  (imported for side-effect: open_clip needs it)

            model, _, preprocess = open_clip.create_model_and_transforms(
                _CLIP_NAME, pretrained=_CLIP_PRETRAINED, device="cpu"
            )
            model.eval()
            _STATE.clip_model = model
            _STATE.clip_preprocess = preprocess
            _STATE.clip_tokenizer = open_clip.get_tokenizer(_CLIP_NAME)
            log.info("OpenCLIP %s/%s loaded", _CLIP_NAME, _CLIP_PRETRAINED)
            return True
        except Exception as exc:  # pragma: no cover - depends on network/weights
            log.warning("OpenCLIP unavailable, using fallback image encoder: %s", exc)
            _STATE.clip_failed = True
            return False


def _load_sentence_transformer() -> bool:
    with _STATE.lock:
        if _STATE.st_model is not None:
            return True
        if _STATE.st_failed:
            return False
        try:
            from sentence_transformers import SentenceTransformer

            _STATE.st_model = SentenceTransformer(_ST_NAME, device="cpu")
            log.info("Sentence-Transformers %s loaded", _ST_NAME)
            return True
        except Exception as exc:  # pragma: no cover
            log.warning("Sentence-Transformers unavailable, using fallback text encoder: %s", exc)
            _STATE.st_failed = True
            return False


def info() -> EncoderInfo:
    if _STATE.info is not None:
        return _STATE.info
    have_clip = _load_clip()
    have_st = _load_sentence_transformer()
    _STATE.info = EncoderInfo(
        image_encoder="openclip:ViT-B-32" if have_clip else "fallback:visual-descriptor-v1",
        text_encoder="sentence-transformers:all-MiniLM-L6-v2" if have_st
        else "fallback:hashed-bigram-v1",
        image_dim=512 if have_clip else 256,
        text_dim=384 if have_st else 384,
        deterministic=not (have_clip and have_st),
        notes=(
            "Pretrained encoders active. CLIP image features are not fine-tuned by "
            "this project; sentence embeddings are used as fixed feature extractors."
            if have_clip and have_st
            else "Fallback encoders active: deterministic numpy descriptors. "
            "Semantic quality is markedly lower than the pretrained models."
        ),
    )
    return _STATE.info


# --------------------------------------------------------------------- images
def decode_image(data: bytes) -> np.ndarray:
    """Decode to an RGB float32 array in [0,1] of shape (H, W, 3)."""
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)  # respect camera orientation
        im = im.convert("RGB")
        return np.asarray(im, dtype=np.float32) / 255.0


def image_embedding(img: np.ndarray) -> tuple[np.ndarray, str]:
    """Return (unit-norm embedding, encoder name)."""
    if _load_clip():
        import torch

        pre = _STATE.clip_preprocess
        # torchvision transforms expect CHW; numpy images arrive as HWC.
        hwc = np.ascontiguousarray(np.clip(img, 0.0, 1.0), dtype=np.float32)
        chw = torch.from_numpy(hwc.transpose(2, 0, 1))
        tensor = pre(chw)
        with torch.no_grad():
            feats = _STATE.clip_model.encode_image(tensor.unsqueeze(0))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].cpu().numpy().astype(np.float64), f"openclip:{_CLIP_NAME}"
    vec = fallback.fallback_image_embedding(img, dim=256)
    return vec, "fallback:visual-descriptor-v1"


def text_embedding(text: str) -> tuple[np.ndarray, str]:
    """Return (unit-norm embedding, encoder name) for a profile text block."""
    if _load_sentence_transformer():
        emb = _STATE.st_model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb[0], dtype=np.float64), "sentence-transformers:all-MiniLM-L6-v2"
    return fallback.fallback_text_embedding(text, dim=384), "fallback:hashed-bigram-v1"


def clip_text_embedding(text: str) -> tuple[np.ndarray, str] | tuple[None, None]:
    """CLIP text embedding in the image space, used for zero-shot scene labels.

    Returns (None, None) when OpenCLIP is unavailable — callers must skip
    zero-shot classification rather than pretending a fallback is equivalent.
    """
    if not _load_clip():
        return None, None
    import torch

    tokens = _STATE.clip_tokenizer([text])
    with torch.no_grad():
        feats = _STATE.clip_model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0].cpu().numpy().astype(np.float64), f"openclip:{_CLIP_NAME}"


@functools.lru_cache(maxsize=4096)
def cached_text_embedding(text: str) -> tuple[bytes, str]:
    vec, enc = text_embedding(text)
    return vec.tobytes(), enc


def text_embedding_cached(text: str) -> tuple[np.ndarray, str]:
    raw, enc = cached_text_embedding(text)
    dim = 384
    return np.frombuffer(raw, dtype=np.float64).copy().reshape(dim), enc


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
