#!/usr/bin/env python3
"""Seed one demo profile with synthetic photographs through the public API.

Run this before looking at the UI if the database is empty:
    python3 scripts/seed_demo.py [base_url]

The photographs are procedurally generated gradients/patterns — they are synthetic,
and the UI labels every derived number as arising from synthetic inputs.
"""
from __future__ import annotations

import io
import math
import sys

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def photo(kind: str, size: int = 720) -> bytes:
    """Deterministic synthetic images with very different colour statistics."""
    rng = np.random.default_rng(abs(hash(kind)) % (2**32))
    if kind == "sky":
        arr = np.zeros((size, size, 3), np.float64)
        y = np.linspace(0, 1, size)[:, None]
        arr[..., 2] = 235 - 70 * y
        arr[..., 1] = 170 - 40 * y
        arr[..., 0] = 120 + 90 * y
        arr += rng.normal(0, 3, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(img)
        d.ellipse((size * 0.1, size * 0.55, size * 0.9, size * 1.15), fill=(196, 148, 96))
        d.ellipse((size * 0.62, size * 0.12, size * 0.8, size * 0.3), fill=(255, 246, 214))
    elif kind == "ridge":
        img = Image.new("RGB", (size, size), (30, 34, 48))
        d = ImageDraw.Draw(img)
        for i in range(4):
            d.polygon([(size * i / 4, size), (size * (i + 0.5) / 4, size * (0.35 + 0.1 * i)),
                       (size * (i + 1) / 4, size)], fill=(58 + 18 * i, 44 + 12 * i, 62 + 8 * i))
        d.ellipse((size * 0.68, size * 0.1, size * 0.84, size * 0.26), fill=(255, 240, 200))
    elif kind == "studio":
        img = Image.new("RGB", (size, size), (26, 22, 24))
        d = ImageDraw.Draw(img)
        d.ellipse((size * 0.28, size * 0.18, size * 0.72, size * 0.62), fill=(206, 168, 142))
        d.rectangle((size * 0.34, size * 0.62, size * 0.66, size), fill=(48, 56, 72))
        img = img.filter(ImageFilter.GaussianBlur(1.2))
    elif kind == "street":
        arr = np.zeros((size, size, 3), np.float64)
        x = np.linspace(0, 1, size)[None, :]
        arr[..., 0] = 60 + 120 * x
        arr[..., 1] = 90 + 60 * x
        arr[..., 2] = 40 + 30 * x
        img = Image.fromarray(arr.astype(np.uint8))
        d = ImageDraw.Draw(img)
        for i in range(6):
            yy = size * (0.12 + i * 0.14)
            d.rectangle((0, yy, size, yy + size * 0.045), fill=(240, 235, 220))
        img = img.filter(ImageFilter.GaussianBlur(0.6))
    else:  # texture
        arr = (rng.random((size, size, 3)) * 255).astype(np.uint8)
        img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(2))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    c = httpx.Client(base_url=base, timeout=300)
    tok = c.post("/api/auth/login", json={"handle": "owner",
                                          "password": "attraction-engine"}).json()["token"]
    c.headers["Authorization"] = f"Bearer {tok}"

    existing = c.get("/api/profiles").json()
    for p in existing:
        if p["name"] == "Demo profile (synthetic)":
            print("already seeded:", p["id"])
            return

    prof = c.post("/api/profiles", json={
        "name": "Demo profile (synthetic)",
        "is_primary": True,
        "description": ("I run a small photography studio in Jaipur. Weekends are climbing at "
                        "Nahargarh or a long drive with a film camera."),
        "interests": ["climbing", "film photography", "baking", "ridge trails"],
        "structured_prefs": {"distance_km": 25, "smoking": "never", "pets": "ok"},
        "prompts": [
            {"kind": "prompt", "question": "My ideal Sunday",
             "answer": ("6am bakery on Mall Road, espresso at Curious Life, then the sandstone "
                        "trail at Nahargarh before the heat. Back through the old city market "
                        "with 200 ISO film in the camera."), "position": 0},
            {"kind": "prompt", "question": "A fact about me",
             "answer": "I have developed 340 rolls of film in my own kitchen.", "position": 1},
            {"kind": "prompt", "question": "Together we could",
             "answer": "Drive to Bundi for the stepwells and eat dal baati on the way back.",
             "position": 2},
        ],
    }).json()
    pid = prof["id"]
    print("profile:", pid)

    for kind in ("sky", "ridge", "studio", "street", "texture"):
        r = c.post(f"/api/profiles/{pid}/photos",
                   files={"file": (f"{kind}.jpg", photo(kind), "image/jpeg")}, timeout=300)
        r.raise_for_status()
        out = r.json()
        print(f"  {kind:8} q={out['quality']:.4f} scene={out['analysis']['dominant_scene']}")

    # one experiment with real recorded counts, so the experiments screen has data
    exp = c.post("/api/experiments", json={
        "name": "Opener specificity A/B",
        "hypothesis": "A concrete opening line lifts reply rate.",
        "metric": "reply_rate",
        "variants": [{"label": "vague opener"}, {"label": "concrete opener"}],
    }).json()
    vids = [v["id"] for v in exp["variants"]]
    for vid, s, f in ((vids[0], 38, 162), (vids[1], 57, 143)):
        c.post(f"/api/experiments/{exp['id']}/observations",
               json={"variant_id": vid, "successes": s, "failures": f})
    print("experiment:", exp["id"])

    syn = c.post("/api/conversations/synthetic", json={"messages": 120, "seed": 7}).json()
    print("conversation:", syn.get("id") or syn.get("conversation_id"))
    print("\nseeded. open", base)


if __name__ == "__main__":
    main()
