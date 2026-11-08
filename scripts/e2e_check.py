"""End-to-end integration test against a running API (real HTTP, real DB).

Run with:  python3 scripts/e2e_check.py [base_url]
It exercises every module and asserts on the actual payloads, printing a compact
pass/fail table at the end. This is the acceptance harness for the delivered app.
"""
from __future__ import annotations

import io
import json
import sys
import time

import httpx
import numpy as np
from PIL import Image

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
PASSWORD = "attraction-engine"
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, bool(ok), detail))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f"  — {detail}" if detail else ""))
    return ok


def png_bytes(width: int, height: int, kind: str, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    if kind == "sky":
        img = np.zeros((height, width, 3))
        img[..., 2] = 0.9 - 0.55 * (yy / height)
        img[..., 1] = 0.75 - 0.35 * (yy / height)
        img[..., 0] = 0.35 + 0.45 * (yy / height)
    elif kind == "dark":
        img = np.full((height, width, 3), 0.13)
        glow = 0.12 * np.exp(-((xx - width * 0.5) ** 2 + (yy - height * 0.45) ** 2)
                             / (0.04 * width * height))
        img += glow[..., None]
    elif kind == "graphic":
        v = ((xx // 40 + yy // 40) % 2).astype(float) * 0.9 + 0.05
        img = np.stack([v, v, v], axis=-1)
    elif kind == "green":
        img = np.zeros((height, width, 3))
        img[..., 1] = 0.55 + 0.25 * np.sin(xx / width * 6.0)
        img[..., 0] = 0.20
        img[..., 2] = 0.15
    else:
        img = rng.random((height, width, 3))
    img = np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1)
    buf = io.BytesIO()
    Image.fromarray((img * 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    client = httpx.Client(base_url=BASE, timeout=180.0)

    # ---------------------------------------------------------------- health
    r = client.get("/healthz")
    check("healthz responds", r.status_code == 200, r.text[:80])

    r = client.get("/api/system/capabilities")
    caps = r.json()
    check("capabilities: native core present", caps["native"].get("native_available") is True,
          str(caps["native"].get("cpu_target") or caps["native"].get("target")))
    check("capabilities: c++26 dialect >= 202302",
          caps["native"]["cxx_dialect"] >= 202302, str(caps["native"]["cxx_dialect"]))
    check("capabilities: encoders reported",
          "image_encoder" in caps["encoders"], caps["encoders"]["image_encoder"])
    print(f"      db={caps['database'].get('url_scheme')} pgvector={caps['database'].get('pgvector')} "
          f"cache={caps['cache']['enabled']}")

    # ------------------------------------------------------------------ auth
    # Deterministic harness: reset the installation password to the dev default.
    import subprocess

    subprocess.run([sys.executable, "scripts/dev_reset_login.py"],
                   cwd="/home/user/attraction-engine", check=False, capture_output=True)
    r = client.post("/api/auth/login", json={"handle": "owner", "password": "wrong"})
    check("auth: wrong password rejected", r.status_code == 401, str(r.status_code))
    r = client.post("/api/auth/login", json={"handle": "owner", "password": PASSWORD})
    check("auth: login succeeds", r.status_code == 200, str(r.status_code))
    token = r.json()["token"]
    client.headers.update({"Authorization": f"Bearer {token}"})
    r = client.get("/api/profiles")
    check("auth: token accepted", r.status_code == 200)
    anon = httpx.get(f"{BASE}/api/profiles")
    check("auth: anonymous rejected", anon.status_code == 401, str(anon.status_code))

    # ------------------------------------------------------------- profiles
    r = client.post("/api/profiles", json={
        "name": "Primary (demo)",
        "description": "I run a small photography studio in Jaipur. Weekends are climbing "
                       "or a long drive with a camera and too much coffee.",
        "is_primary": True,
        "interests": ["climbing", "film photography", "specialty coffee"],
        "structured_prefs": {"age_range": [28, 38], "max_distance_km": 25, "smoking": "no"},
        "prompts": [
            {"kind": "prompt", "question": "My simple pleasures",
             "answer": "The 6am bakery on Mall Road and the first espresso before the trail.",
             "position": 0},
            {"kind": "prompt", "question": "Together we could",
             "answer": "Love to laugh and travel", "position": 1},
        ],
    })
    check("profile: created", r.status_code == 201, r.text[:120] if r.status_code != 201 else "")
    pid = r.json()["id"]

    r = client.post("/api/profiles", json={"name": "Bad prompt", "prompts": [
        {"kind": "prompt", "answer": "", "position": 0}]})
    check("profile: empty prompt answer rejected (422)", r.status_code == 422, str(r.status_code))

    # -------------------------------------------------------------- photos
    uploads = [
        ("sky.png", png_bytes(1200, 900, "sky", 1), "image/png"),
        ("dark.png", png_bytes(800, 800, "dark", 2), "image/png"),
        ("green.png", png_bytes(1000, 1000, "green", 3), "image/png"),
        ("graphic.png", png_bytes(700, 700, "graphic", 4), "image/png"),
    ]
    photo_ids = []
    t0 = time.perf_counter()
    for name, data, mime in uploads:
        r = client.post(f"/api/profiles/{pid}/photos",
                        files={"file": (name, data, mime)})
        if r.status_code != 201:
            check(f"photo: upload {name}", False, r.text[:160])
            continue
        body = r.json()
        photo_ids.append(body["id"])
        check(f"photo: upload {name}", True,
              f"q={body['analysis']['quality_estimate']} scene={body['analysis']['dominant_scene']}")
    upload_seconds = time.perf_counter() - t0

    r = client.post(f"/api/profiles/{pid}/photos",
                    files={"file": ("bad.txt", b"not an image", "text/plain")})
    check("photo: non-image rejected (415)", r.status_code == 415, str(r.status_code))

    r = client.get(f"/api/profiles/{pid}/photos")
    photos = r.json()["photos"]
    check("photo: listed with analyses", len(photos) == len(uploads) and
          all("quality_estimate" in p["analysis"] for p in photos), f"{len(photos)} photos")
    check("photo: encoders are real (CLIP)",
          photos[0]["analysis"]["scene_source"] == "openclip-zero-shot",
          photos[0]["analysis"]["scene_source"])

    raw = client.get(f"/api/profiles/{pid}/photos/{photo_ids[0]}/raw")
    check("photo: raw bytes served", raw.status_code == 200 and raw.content[:4] == b"\x89PNG")

    # ------------------------------------------------------- intelligence
    r = client.post(f"/api/intelligence/{pid}/report")
    rep = r.json()
    check("intelligence: report generated", r.status_code == 200 and "recommendations" in rep,
          f"{len(rep.get('recommendations', []))} recommendations")
    check("intelligence: visual diversity computed",
          rep["visual_diversity"]["diversity"] is not None,
          f"diversity={rep['visual_diversity']['diversity']}")
    check("intelligence: text blocks analysed", len(rep["text_blocks"]) == 3,
          f"{len(rep['text_blocks'])} blocks")
    check("intelligence: limitations disclosed", len(rep["limitations"]) >= 3)
    check("intelligence: confidence is evidence coverage",
          "evidence_coverage" in rep["confidence"])

    # ---------------------------------------------------------- laboratory
    r = client.post(f"/api/laboratory/{pid}/optimize",
                    json={"profile_id": pid, "lambda": 0.35, "mu": 0.2, "seed": 42})
    opt = r.json()
    check("laboratory: optimization ran", r.status_code == 200 and "best" in opt,
          f"{len(opt.get('results', []))} strategies")
    check("laboratory: exhaustive proved optimal",
          any(x.get("proved_optimal") for x in opt["results"]),
          f"n={opt['n']} (n! enumeration feasible)")
    check("laboratory: all strategies agree on optimum",
          len({x["objective"] for x in opt["results"] if "objective" in x}) == 1,
          str({x["strategy"]: x.get("objective") for x in opt["results"]}))

    r = client.post(f"/api/laboratory/{pid}/sensitivity",
                    json={"profile_id": pid, "seed": 42})
    sens = r.json()
    check("laboratory: sensitivity grid", len(sens["grid"]) == 15, f"{len(sens['grid'])} cells")

    r = client.post(f"/api/laboratory/{pid}/ablation", json={"profile_id": pid, "seed": 42})
    abl = r.json()
    check("laboratory: ablation rows", len(abl["rows"]) == len(uploads),
          f"worst loss={max((r['loss'] for r in abl['rows']), default=0):.4f}")

    best_perm = opt["best"]["permutation"]
    r = client.post(f"/api/laboratory/{pid}/apply", json={"permutation": best_perm})
    check("laboratory: ordering persisted", r.status_code == 200)
    r = client.get(f"/api/profiles/{pid}/photos")
    check("laboratory: persisted order matches",
          r.json()["photos"][0]["id"] == photo_ids[best_perm[0]])

    # ------------------------------------------------------- compatibility
    r = client.post("/api/compatibility/pool", json={"seed": 20260101, "n_candidates": 60})
    pool = r.json()
    check("compatibility: synthetic pool", r.status_code == 200 and len(pool["profiles"]) == 60)
    check("compatibility: pool labelled synthetic", "SYNTHETIC" in pool["label"])

    r = client.post("/api/compatibility/rank", json={"top_k": 10, "lambda_diversity": 0.0})
    rank = r.json()
    check("compatibility: ranking produced", r.status_code == 200 and len(rank["ranking"]) == 10)
    check("compatibility: metrics computed", "ndcg_at_k" in rank["metrics"],
          json.dumps(rank["metrics"]))
    check("compatibility: trained neural head applied", rank["neural_head_applied"] is True,
          rank["model_status"])

    r = client.post("/api/compatibility/rank", json={"top_k": 10, "lambda_diversity": 0.7})
    div = r.json()
    check("compatibility: diversity reranking changes order",
          [x["index"] for x in div["ranking"]] != [x["index"] for x in rank["ranking"]])

    r = client.post("/api/compatibility/retrieval", json={"profile_id": pid, "top_k": 4})
    ret = r.json()
    check("compatibility: ANN retrieval executed", r.status_code == 200,
          f"exact={ret.get('exact_ms')}ms ann={ret.get('ann', {}).get('elapsed_ms')}ms "
          f"recall={ret.get('ann', {}).get('recall_at_k')}")

    # ----------------------------------------------------------- experiments
    r = client.post("/api/experiments", json={
        "name": "First photo: portrait vs outdoor",
        "hypothesis": "An outdoor activity photo first produces more matches than a close-up.",
        "profile_id": pid, "prior": "weak_centred", "min_observations": 30,
        "variants": [
            {"label": "outdoor_first", "configuration": {"first": photo_ids[0]}},
            {"label": "portrait_first", "configuration": {"first": photo_ids[2]}},
        ],
    })
    exp = r.json()
    check("experiments: created", r.status_code == 201, exp.get("id", "")[:8])
    eid = exp["id"]
    r = client.post(f"/api/experiments/{eid}/observations",
                    json={"successes": 41, "failures": 139, "missing": 12})
    check("experiments: observations recorded", r.status_code == 200)

    r = client.get(f"/api/experiments/{eid}/analysis")
    ana = r.json()
    check("experiments: posterior computed", "variants" in ana and
          abs(sum(v["posterior"]["mean"] for v in ana["variants"]) > 0), str(r.status_code))
    # verify Beta-Binomial posterior mean exactly
    a0, b0 = ana["prior"]["alpha"], ana["prior"]["beta"]
    expected = (a0 + 41) / (a0 + b0 + 41 + 139)
    got = ana["variants"][0]["posterior"]["mean"]
    check("experiments: posterior mean matches Beta(α+s, β+f)",
          abs(got - expected) < 1e-12, f"{got:.12f} vs {expected:.12f}")
    check("experiments: missing != failure", ana["variants"][0]["missing"] == 12 and
          ana["variants"][0]["observed"] == 180)
    check("experiments: credible interval present",
          ana["variants"][0]["credible_interval"]["hi"] > ana["variants"][0]["credible_interval"]["lo"])
    check("experiments: thompson allocation sums", sum(ana["thompson"]["allocation"]) == 2000)
    check("experiments: multiplicity adjusted", ana["multiplicity"]["variants"] == 2,
          f"bonferroni={ana['multiplicity']['bonferroni_alpha']:.4f}")
    check("experiments: density curves for plotting", len(ana["density"]) == 2)
    check("experiments: hierarchical model fitted",
          "population" in ana["hierarchical"] and ana["hierarchical"]["overdispersion"] >= 0)

    r = client.post("/api/experiments/simulate", json={"seed": 9})
    sim = r.json()
    check("experiments: synthetic simulation labelled", sim.get("synthetic") is True)

    # ------------------------------------------------------- counterfactual
    r = client.post("/api/counterfactual/simulate", json={
        "profile_id": pid,
        "baseline_photo_ids": photo_ids,
        "modified_photo_ids": list(reversed(photo_ids)),
        "modified_prompts": ["The 6am bakery on Mall Road, then espresso at Curious Life "
                             "before the sandstone trail at Nahargarh. I return via the "
                             "old city market with 200 ISO film in the camera.",
                             "A weekend built around a ridge climb, a long breakfast and "
                             "developing the film from the day before.",
                             "I run a small photography studio in Jaipur. Weekends are "
                             "climbing at Nahargarh or a long drive with a camera."],
    })
    cfs = r.json()
    ok_cf = r.status_code == 200 and "delta_q" in cfs
    check("counterfactual: simulation ran", ok_cf,
          f"ΔQ={cfs.get('delta_q')} (baseline {cfs.get('q_baseline')})" if ok_cf else r.text[:160])
    if ok_cf:
        check("counterfactual: honesty statement present",
              "not a causal effect" in cfs["interpretation"])
        check("counterfactual: attribution computed",
              "importance" in cfs["attribution"])
        check("counterfactual: model version labelled", "version" in cfs["model_version"] or
              "counterfactual-v1" in cfs["model_version"])

    # ------------------------------------------------------------ conversations
    from datetime import datetime, timedelta, timezone
    base = datetime(2026, 3, 1, 18, 0, tzinfo=timezone.utc)
    msgs = []
    for i in range(40):
        sender = "me" if i % 2 == 0 else "Aarav"
        msgs.append({"sender": sender,
                     "timestamp": (base + timedelta(minutes=i * 7.5)).isoformat(),
                     "text": f"message {i} about climbing and coffee"})
    r = client.post("/api/conversations/import",
                    json={"label": "e2e import", "me": "me", "messages": msgs})
    conv = r.json()
    check("conversations: imported and analysed", r.status_code == 200 and
          conv["messages"] == 40, f"reciprocity={conv.get('reciprocity', {}).get('index')}")
    check("conversations: raw text not stored",
          conv["privacy"]["raw_text_stored"] is False)
    check("conversations: refusal of psychological inference",
          any("no inference" in x.lower() for x in conv["limitations"]))
    r = client.get(f"/api/conversations/{conv['id']}")
    check("conversations: retrievable", r.status_code == 200)

    r = client.post("/api/conversations/synthetic", json={"seed": 3, "n_messages": 60})
    check("conversations: synthetic labelled", r.json().get("synthetic") is True)

    # ------------------------------------------------------------ benchmarks
    r = client.post("/api/benchmarks/run",
                    json={"sizes": [512, 4096], "dim": 128, "k": 10, "repeats": 3})
    bench = r.json()
    check("benchmarks: executed and stored", r.status_code == 200 and bench.get("run_id"),
          f"{len(bench.get('results', []))} measurements")
    if r.status_code == 200:
        heap = [b for b in bench["results"] if b.get("strategy") == "heap_bounded"]
        sortf = [b for b in bench["results"] if b.get("strategy") == "sort_full"]
        if heap and sortf:
            check("benchmarks: heap faster than full sort at n=4096",
                  heap[-1]["median_ms"] <= sortf[-1]["median_ms"],
                  f"heap={heap[-1]['median_ms']}ms sort={sortf[-1]['median_ms']}ms")

    # ---------------------------------------------------------------- privacy
    r = client.get("/api/privacy/export")
    export = r.json()
    check("privacy: export contains profiles", len(export["profiles"]) >= 1,
          f"{len(export['profiles'])} profiles, {len(export.get('experiments', []))} experiments")
    r = client.get("/api/privacy/policy")
    check("privacy: policy published", r.json()["local_first"] is True)

    # ------------------------------------------------------------- cleanup
    r = client.delete(f"/api/profiles/{pid}")
    check("profile: deleted (204)", r.status_code == 204)

    print(f"\nupload+analysis wall time for {len(uploads)} photos: {upload_seconds:.1f}s")
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\n{passed}/{total} checks passed")
    if passed != total:
        print("failures:")
        for name, ok, detail in results:
            if not ok:
                print(f"  - {name}: {detail}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
