"""Module A, image branch: per-photograph analytics.

Every number returned here is computed from the pixels. Subjective judgements
(quality, "best photo") are never presented as ground truth: the score is an
explicit, documented heuristic whose components are all returned alongside it,
and the report always states that only the user's own preference feedback can
calibrate it.
"""
from __future__ import annotations

import numpy as np

from ..native import require as require_native
from . import encoders, fallback

# Zero-shot scene taxonomy. Labels are intentionally broad and non-sensitive:
# no demographic, religious or health-related categories are present.
SCENE_PROMPTS: dict[str, str] = {
    "outdoor_nature": "a photograph taken outdoors in nature, mountains, forest or beach",
    "outdoor_urban": "a photograph taken outside in a city street or urban setting",
    "indoor_home": "a photograph taken indoors at home, a living room or kitchen",
    "indoor_social": "a photograph at a bar, restaurant or party with other people",
    "activity_sport": "a photograph of someone doing sport, hiking, cycling or climbing",
    "travel_landmark": "a travel photograph in front of a well known landmark",
    "portrait_closeup": "a close-up portrait of a person's face",
    "group_photo": "a group photograph with several people",
    "animal_pet": "a photograph featuring a pet, dog or cat",
    "food_scene": "a photograph of food or a meal",
}


def _luma(img: np.ndarray) -> np.ndarray:
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def _sobel(img: np.ndarray) -> np.ndarray:
    l = _luma(img)
    gx = np.zeros_like(l)
    gy = np.zeros_like(l)
    gx[1:-1, 1:-1] = (
        l[:-2, 2:] + 2 * l[1:-1, 2:] + l[2:, 2:]
        - l[:-2, :-2] - 2 * l[1:-1, :-2] - l[2:, :-2]
    ) / 4.0
    gy[1:-1, 1:-1] = (
        l[2:, :-2] + 2 * l[2:, 1:-1] + l[2:, 2:]
        - l[:-2, :-2] - 2 * l[:-2, 1:-1] - l[:-2, 2:]
    ) / 4.0
    return np.hypot(gx, gy)


def _rule_of_thirds_energy(img: np.ndarray) -> float:
    """Fraction of gradient energy in the four thirds-intersection windows.

    A crude composition proxy: strong subjects tend to sit near the thirds
    intersections. It is a heuristic, not a rule, and is reported as such.
    """
    mag = _sobel(img)
    h, w = mag.shape
    if h < 6 or w < 6:
        return 0.0
    win_h, win_w = max(1, h // 5), max(1, w // 5)
    total = float(mag.sum()) + 1e-12
    pts = [(h // 3, w // 3), (h // 3, 2 * w // 3), (2 * h // 3, w // 3), (2 * h // 3, 2 * w // 3)]
    energy = 0.0
    for cy, cx in pts:
        y0, y1 = max(0, cy - win_h // 2), min(h, cy + win_h // 2 + 1)
        x0, x1 = max(0, cx - win_w // 2), min(w, cx + win_w // 2 + 1)
        energy += float(mag[y0:y1, x0:x1].sum())
    return min(1.0, energy / total)


def _face_stats(img: np.ndarray) -> dict | None:
    """Optional face detection through OpenCV's Haar cascade.

    Returns None when OpenCV or the cascade is unavailable. We never infer
    identity, age, ethnicity or gender from faces — only presence, count and a
    framing/brightness measurement of the detected box.
    """
    try:
        import cv2
    except Exception:
        return None
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return None
        gray = (_luma(img) * 255.0).astype(np.uint8)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(32, 32))
        h, w = gray.shape
        if len(faces) == 0:
            return {"count": 0, "largest_fraction": 0.0, "faces": []}
        boxes = [
            {
                "x": int(x), "y": int(y), "w": int(fw), "h": int(fh),
                "area_fraction": float(fw * fh) / float(w * h),
                "centre_offset": float(
                    abs((x + fw / 2) / w - 0.5) + abs((y + fh / 2) / h - 0.5)
                ),
            }
            for (x, y, fw, fh) in faces
        ]
        boxes.sort(key=lambda b: b["area_fraction"], reverse=True)
        return {"count": len(boxes), "largest_fraction": boxes[0]["area_fraction"], "faces": boxes}
    except Exception:
        return None


def zero_shot_scenes(img: np.ndarray) -> dict[str, float] | None:
    """CLIP zero-shot scene distribution; None when OpenCLIP is unavailable."""
    emb, enc = encoders.image_embedding(img)
    if not enc.startswith("openclip"):
        return None
    logits = {}
    for label, prompt in SCENE_PROMPTS.items():
        temb, tenc = encoders.clip_text_embedding(prompt)
        if temb is None:
            return None
        logits[label] = float(np.dot(emb, temb))
    keys = list(logits)
    vals = np.array([logits[k] for k in keys])
    # CLIP logits are cosine similarities; a temperature of 100 is the standard
    # zero-shot scaling used in the original paper for ViT-B/32.
    z = np.exp((vals - vals.max()) * 100.0)
    probs = z / z.sum()
    return {k: float(p) for k, p in zip(keys, probs)}


def analyze_photo(img: np.ndarray) -> dict:
    """Full per-photo analysis record. All fields are pixel-derived."""
    core = require_native()
    rgb8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    h, w = rgb8.shape[:2]
    stats = core.image_stats if hasattr(core, "image_stats") else None  # not bound; use numpy
    luma = _luma(img)
    sobel = _sobel(img)
    hist, _ = np.histogram((luma * 255).astype(np.uint8), bins=256, range=(0, 255))
    p = hist / max(1, hist.sum())
    entropy = float(-np.sum(p[p > 0] * np.log2(p[p > 0])))
    scenes = zero_shot_scenes(img)
    faces = _face_stats(img)
    emb, enc = encoders.image_embedding(img)

    resolution = int(w * h)
    aspect = float(w) / float(h) if h else 0.0
    megapixels = resolution / 1e6

    # --- component heuristics, each in [0,1], all disclosed in the response ---
    exposure = float(1.0 - min(1.0, (np.mean(luma < 0.02) + np.mean(luma > 0.98)) * 3.0))
    contrast = float(np.clip((luma.std() - 0.05) / 0.25, 0.0, 1.0))
    sharpness = float(np.clip(np.percentile(sobel, 90) / 0.25, 0.0, 1.0))
    resolution_score = float(np.clip((megapixels - 0.3) / 4.0, 0.0, 1.0))
    composition = _rule_of_thirds_energy(img)
    face_score = 0.0
    if faces is not None:
        if faces["count"] == 0:
            face_score = 0.25
        else:
            area = faces["largest_fraction"]
            offset = faces["faces"][0]["centre_offset"]
            face_score = float(np.clip(0.4 + 2.0 * area - 0.35 * offset, 0.0, 1.0))
    background = float(np.clip(1.0 - np.mean(sobel > 0.12) * 1.5, 0.0, 1.0))

    weights = {
        "exposure": 0.18, "contrast": 0.12, "sharpness": 0.22,
        "resolution": 0.10, "composition": 0.13, "face": 0.15, "background": 0.10,
    }
    components = {
        "exposure": exposure, "contrast": contrast, "sharpness": sharpness,
        "resolution": resolution_score, "composition": composition,
        "face": face_score, "background": background,
    }
    quality = float(sum(weights[k] * components[k] for k in weights))

    flags: list[str] = []
    if min(w, h) < 640:
        flags.append("low_resolution")
    if exposure < 0.5:
        flags.append("exposure_clipping")
    if sharpness < 0.25:
        flags.append("likely_blurry")
    if contrast < 0.15:
        flags.append("low_contrast")
    if aspect < 0.6:
        flags.append("extreme_portrait_crop")
    if aspect > 2.4:
        flags.append("extreme_landscape_crop")

    dominant_scene = max(scenes, key=scenes.get) if scenes else None

    return {
        "width": int(w),
        "height": int(h),
        "aspect_ratio": round(aspect, 4),
        "megapixels": round(megapixels, 3),
        "quality_estimate": round(quality, 4),
        "quality_components": {k: round(v, 4) for k, v in components.items()},
        "quality_weights": weights,
        "lighting": {
            "mean_luma": round(float(np.mean(luma)), 4),
            "contrast_rms": round(float(luma.std()), 4),
            "entropy_bits": round(entropy, 4),
            "clip_low_fraction": round(float(np.mean(luma < 0.02)), 4),
            "clip_high_fraction": round(float(np.mean(luma > 0.98)), 4),
            "shadow_fraction": round(float(np.mean(luma < 0.2)), 4),
            "highlight_fraction": round(float(np.mean(luma > 0.85)), 4),
        },
        "composition": {
            "rule_of_thirds_energy": round(composition, 4),
            "edge_density": round(float(np.mean(sobel)), 4),
            "background_complexity": round(float(np.mean(sobel > 0.12)), 4),
        },
        "faces": faces,
        "scene_distribution": {k: round(v, 4) for k, v in scenes.items()} if scenes else None,
        "dominant_scene": dominant_scene,
        "scene_source": "openclip-zero-shot" if scenes else "unavailable",
        "flags": flags,
        "embedding": emb.tolist(),
        "embedding_encoder": enc,
    }


def visual_diversity(embeddings: np.ndarray) -> dict:
    """Within-set visual diversity and redundancy detection.

    Uses cosine similarity; two photographs are flagged redundant when their
    similarity exceeds ``REDUNDANT_THRESHOLD``. The threshold is a product
    decision (not a validated perceptual constant) and is surfaced in the UI.
    """
    n = embeddings.shape[0]
    if n < 2:
        return {"mean_pairwise_similarity": None, "redundant_pairs": [],
                "diversity": None, "nearest_neighbour": [], "threshold": 0.92}
    sim = embeddings @ embeddings.T
    iu = np.triu_indices(n, k=1)
    pair_sims = sim[iu]
    threshold = 0.92
    redundant = [
        {"a": int(i), "b": int(j), "similarity": round(float(sim[i, j]), 4)}
        for i, j in zip(*iu) if sim[i, j] > threshold
    ]
    nearest = []
    for i in range(n):
        row = np.abs(sim[i]).copy()
        row[i] = -np.inf
        j = int(np.argmax(row))
        nearest.append({"index": i, "nearest": j, "similarity": round(float(sim[i, j]), 4)})
    return {
        "mean_pairwise_similarity": round(float(pair_sims.mean()), 4),
        "diversity": round(float(1.0 - pair_sims.mean()), 4),
        "redundant_pairs": redundant,
        "nearest_neighbour": nearest,
        "threshold": threshold,
    }


def select_diverse_subset(embeddings: np.ndarray, k: int) -> list[int]:
    """Farthest-point sampling on cosine distance, seeded with photo 0."""
    n = embeddings.shape[0]
    if n <= k:
        return list(range(n))
    chosen = [0]
    min_dist = 1.0 - embeddings @ embeddings[0]
    while len(chosen) < k:
        idx = int(np.argmax(min_dist))
        chosen.append(idx)
        dist = 1.0 - embeddings @ embeddings[idx]
        min_dist = np.minimum(min_dist, dist)
        min_dist[chosen] = -1.0
    return chosen
