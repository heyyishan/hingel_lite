#!/usr/bin/env python3
"""Headless-browser smoke test of all nine screens.

Asserts each screen reaches a heading, renders without a page error, and that the
photo thumbnails actually load (i.e. the signed-URL path works). Exits non-zero on
any failure so verify_all.sh can gate on it.
"""
from __future__ import annotations

import os
import sys
import time

from playwright.sync_api import sync_playwright

BASE = os.environ.get("AE_BASE_URL", "http://127.0.0.1:8000")
PASSWORD = os.environ.get("AE_OWNER_PASSWORD", "attraction-engine")
SHOTS = os.environ.get("AE_SHOT_DIR", "docs/screenshots")

SCREENS = [
    ("/", "Overview", 3),
    ("/profile", "Profile Intelligence", 3),
    ("/laboratory", "Photo Laboratory", 3),
    ("/compatibility", "Compatibility Explorer", 0),
    ("/experiments", "Experimentation Lab", 0),
    ("/counterfactual", "Counterfactual Simulator", 3),
    ("/conversations", "Conversation Analytics", 0),
    ("/benchmarks", "Systems Benchmarking", 0),
    ("/settings", "Settings & Privacy", 0),
]


def main() -> int:
    os.makedirs(SHOTS, exist_ok=True)
    failures: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1500, "height": 1000})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))

        page.goto(BASE + "/", wait_until="networkidle")
        page.fill("#password", PASSWORD)
        page.click("form button")
        page.wait_for_load_state("networkidle")
        time.sleep(2)

        for path, heading, min_images in SCREENS:
            page.goto(BASE + path, wait_until="networkidle")
            time.sleep(2.5)
            body = page.locator("body").inner_text()
            if heading not in body:
                failures.append(f"{path}: missing heading {heading!r}")
            if errors:
                failures.append(f"{path}: page error {errors[0]}")
                errors.clear()
            if min_images:
                loaded = page.evaluate(
                    "() => Array.from(document.images).filter(i => i.complete && i.naturalWidth > 0).length"
                )
                if loaded < min_images:
                    failures.append(f"{path}: only {loaded}/{min_images} images loaded")
            name = path.strip("/").replace("/", "-") or "overview"
            page.screenshot(path=f"{SHOTS}/{name}.png", full_page=True)
            print(f"OK   {path:16} heading={heading!r} images>= {min_images}")

        browser.close()

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print(f"\nsmoke: {len(SCREENS)} screens rendered with no page errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
