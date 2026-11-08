"""End-to-end API tests against the real application, database and native core.

Nothing is mocked: the tests hit FastAPI routes that touch PostgreSQL (pgvector),
the Redis cache and the pybind11 extension. Photos are generated in-memory.
"""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.api


def png(rgb, size=200, noise=0):
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2] = rgb
    if noise:
        arr = np.clip(arr.astype(int) + np.random.default_rng(size).integers(
            -noise, noise, arr.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def profile(client, auth):
    r = client.post("/api/profiles", headers=auth, json={
        "name": "Test Profile",
        "description": "I run a small photography studio in Jaipur.",
        "interests": ["climbing", "film", "baking"],
        "structured_prefs": {"distance_km": 25, "smoking": "never"},
        "prompts": [
            {"kind": "prompt", "question": "My ideal Sunday",
             "answer": "Bakery at 6am, then the Nahargarh ridge.", "position": 0},
            {"kind": "prompt", "question": "Two truths and a lie",
             "answer": "I have 200 rolls of film and 3 cameras.", "position": 1},
        ],
    })
    assert r.status_code == 201, r.text
    pid = r.json()["id"]
    for name, colour in (("a", (20, 80, 200)), ("b", (200, 90, 20)),
                         ("c", (30, 160, 90)), ("d", (140, 140, 140))):
        client.post(f"/api/profiles/{pid}/photos", headers=auth,
                    files={"file": (f"{name}.png", png(colour), "image/png")})
    yield client.get(f"/api/profiles/{pid}", headers=auth).json()
    client.delete(f"/api/profiles/{pid}", headers=auth)


def test_health_and_capabilities(client):
    h = client.get("/healthz")
    assert h.status_code == 200 and h.json()["status"] == "ok"
    caps = client.get("/api/system/capabilities").json()
    assert caps["native"]["native_available"] is True
    assert int(caps["native"]["cxx_dialect"]) >= 202302
    assert caps["encoders"]["image_encoder"].startswith("openclip")
    assert caps["encoders"]["image_dim"] == 512
    assert caps["disclaimer"]


def test_authentication_is_enforced(client):
    assert client.get("/api/profiles").status_code == 401
    bad = client.post("/api/auth/login", json={"handle": "owner", "password": "nope"})
    assert bad.status_code == 401


def test_profile_crud_roundtrip(client, auth, profile):
    pid = profile["id"]
    got = client.get(f"/api/profiles/{pid}", headers=auth).json()
    assert got["name"] == profile["name"]
    assert got["photo_count"] == 4
    patched = client.patch(f"/api/profiles/{pid}", headers=auth,
                           json={"description": "Now in Udaipur."}).json()
    assert patched["description"] == "Now in Udaipur."
    assert client.get(f"/api/profiles/does-not-exist", headers=auth).status_code == 404


def test_empty_prompt_is_rejected(client, auth):
    r = client.post("/api/profiles", headers=auth,
                    json={"name": "x", "prompts": [{"question": "Q", "answer": ""}]})
    assert r.status_code == 422


def test_photo_upload_rejects_non_images(client, auth, profile):
    r = client.post(f"/api/profiles/{profile['id']}/photos", headers=auth,
                    files={"file": ("t.txt", b"not an image", "text/plain")})
    assert r.status_code == 415


def test_photo_analysis_and_raw_bytes(client, auth, profile):
    photos = client.get(f"/api/profiles/{profile['id']}/photos", headers=auth).json()["photos"]
    assert len(photos) == 4
    p = photos[0]
    assert 0.0 <= p["quality"] <= 1.0
    assert p["analysis"]["embedding_encoder"].startswith("openclip")
    raw = client.get(f"/api/profiles/{profile['id']}/photos/{p['id']}/raw", headers=auth)
    assert raw.status_code == 200 and raw.headers["content-type"].startswith("image/")


def test_intelligence_report(client, auth, profile):
    r = client.post(f"/api/intelligence/{profile['id']}/report", headers=auth, json={})
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["visual_diversity"]["diversity"] >= 0.0
    assert rep["recommendations"]
    assert rep["limitations"]
    assert rep["confidence"]["evidence_coverage"] is not None
    # Every recommendation must carry an explanation, not just a label.
    for rec in rep["recommendations"]:
        assert rec["why"] and rec["action"]


def test_laboratory_optimize_and_apply(client, auth, profile):
    pid = profile["id"]
    r = client.post(f"/api/laboratory/{pid}/optimize", headers=auth,
                    json={"profile_id": pid, "lambda": 0.35, "mu": 0.2, "seed": 7})
    assert r.status_code == 200, r.text
    out = r.json()
    assert len(out["results"]) >= 4
    # For n=4 the exhaustive search is feasible and must be marked optimal.
    exhaustive = [x for x in out["results"] if x["strategy"] == "exhaustive"]
    assert exhaustive and exhaustive[0]["proved_optimal"] is True
    best = min(out["results"], key=lambda x: -x["objective"])
    # No strategy may report an objective above the proven optimum.
    assert all(x["objective"] <= exhaustive[0]["objective"] + 1e-9 for x in out["results"])
    applied = client.post(f"/api/laboratory/{pid}/apply", headers=auth,
                          json={"profile_id": pid, "permutation": best["permutation"]})
    assert applied.status_code == 200
    after = client.get(f"/api/profiles/{pid}/photos", headers=auth).json()["photos"]
    assert [p["id"] for p in after] == [out["photo_ids"][i] for i in best["permutation"]]


def test_laboratory_sensitivity_and_ablation(client, auth, profile):
    pid = profile["id"]
    sens = client.post(f"/api/laboratory/{pid}/sensitivity", headers=auth,
                       json={"profile_id": pid, "lambdas": [0.0, 0.2, 0.4],
                             "mus": [0.0, 0.2]})
    assert sens.status_code == 200, sens.text
    grid = sens.json()["grid"]
    assert len(grid) == 6
    assert {round(g["lambda"], 6) for g in grid} == {0.0, 0.2, 0.4}
    assert {round(g["mu"], 6) for g in grid} == {0.0, 0.2}
    bad = client.post(f"/api/laboratory/{pid}/sensitivity", headers=auth,
                      json={"profile_id": pid, "lambdas": [-1.0]})
    assert bad.status_code == 422
    abl = client.post(f"/api/laboratory/{pid}/ablation", headers=auth,
                      json={"profile_id": pid})
    assert abl.status_code == 200
    rows = abl.json()["rows"]
    assert rows and all("loss" in r for r in rows)


def test_compatibility_rank_and_metrics(client, auth, profile):
    r = client.post("/api/compatibility/rank", headers=auth,
                    json={"profile_id": profile["id"], "top_k": 10, "seed": 11,
                          "pool_size": 200})
    assert r.status_code == 200, r.text
    out = r.json()
    scores = [x["forward"]["score"] for x in out["ranking"]]
    assert scores == sorted(scores, reverse=True)
    assert 0.0 <= out["metrics"]["ndcg_at_k"] <= 1.0
    assert out["metrics"]["k"] == 10
    assert out["weights"] and abs(sum(out["weights"].values()) - 1.0) < 1e-9
    assert out["ablations"], "weight ablations must be reported"
    assert "do not predict" in out["caveat"].lower()


def test_experiments_posterior_matches_closed_form(client, auth):
    r = client.post("/api/experiments", headers=auth, json={
        "name": "test-exp", "hypothesis": "New opener improves reply rate",
        "metric": "reply_rate",
        "variants": [{"label": "control"}, {"label": "treatment"}]})
    assert r.status_code == 201, r.text
    exp = r.json()
    eid = exp["id"]
    vids = [v["id"] for v in exp["variants"]]
    for vid, s_, f_ in ((vids[0], 40, 60), (vids[1], 60, 140)):
        r = client.post(f"/api/experiments/{eid}/observations", headers=auth,
                        json={"variant_id": vid, "successes": s_, "failures": f_,
                              "missing": 0})
        assert r.status_code == 200, r.text
    # An unknown variant id must be rejected, not silently applied everywhere.
    bad = client.post(f"/api/experiments/{eid}/observations", headers=auth,
                      json={"variant_id": "nope", "successes": 1, "failures": 0})
    assert bad.status_code == 404
    analysis = client.get(f"/api/experiments/{eid}/analysis", headers=auth).json()
    prior = analysis["prior"]
    v0 = analysis["variants"][0]
    # Per-arm attribution: each variant received its own counts.
    assert v0["successes"] == 40 and v0["failures"] == 60
    assert v0["posterior"]["alpha"] == pytest.approx(prior["alpha"] + 40)
    assert v0["posterior"]["beta"] == pytest.approx(prior["beta"] + 60)
    v1 = analysis["variants"][1]
    assert v1["successes"] == 60 and v1["failures"] == 140
    assert v1["posterior"]["alpha"] == pytest.approx(prior["alpha"] + 60)
    assert 0 < analysis["pairwise"][0]["prob_a_better"] < 1
    assert analysis["multiplicity"]["bonferroni_alpha"] == pytest.approx(0.025)
    client.delete(f"/api/experiments/{eid}", headers=auth)


def test_counterfactual_simulate(client, auth, profile):
    pid = profile["id"]
    photos = client.get(f"/api/profiles/{pid}/photos", headers=auth).json()["photos"]
    # Both selections must be the same size: Q averages over the selection, so a
    # smaller modified set would compare different quantities.
    ids = [p["id"] for p in photos]
    r = client.post("/api/counterfactual/simulate", headers=auth, json={
        "profile_id": pid,
        "baseline_photo_ids": ids,
        "modified_photo_ids": list(reversed(ids)),
        "modified_prompts": [
            "Saturday 6am bakery on Mall Road, then the Nahargarh ridge trail "
            "before the heat, back via the old city market.",
            "I have 200 rolls of film and 3 cameras.", None],
        "seed": 5,
    })
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["uncertainty"]["value"] is None
    assert "not a causal effect" in out["interpretation"]


def test_conversation_analytics_never_stores_raw_text(client, auth):
    messages = [{"sender": "me" if i % 2 == 0 else "them",
                 "timestamp": f"2026-01-01T{i % 24:02d}:00:00",
                 "text": f"message number {i}"} for i in range(20)]
    r = client.post("/api/conversations/import", headers=auth,
                    json={"label": "unit-test", "me": "me", "messages": messages})
    assert r.status_code == 200, r.text
    cid = r.json()["id"]
    got = client.get(f"/api/conversations/{cid}", headers=auth).json()
    blob = json.dumps(got)
    assert "message number 7" not in blob, "raw message text must not be persisted"
    assert got["analytics"]["messages"] == 20
    assert "sessions" in got["analytics"]
    client.delete(f"/api/conversations/{cid}", headers=auth)


def test_privacy_export_and_deletion(client, auth):
    created = client.post("/api/profiles", headers=auth,
                          json={"name": "deletable"}).json()
    export = client.get("/api/privacy/export", headers=auth).json()
    assert any(p["id"] == created["id"] for p in export["profiles"])
    assert client.delete(f"/api/profiles/{created['id']}", headers=auth).status_code == 204
    assert client.get(f"/api/profiles/{created['id']}", headers=auth).status_code == 404
    policy = client.get("/api/privacy/policy", headers=auth).json()
    assert policy, "privacy policy must be published"


def test_benchmark_endpoint_stores_measurements(client, auth):
    r = client.post("/api/benchmarks/run", headers=auth,
                    json={"sizes": [4096], "k": 10, "dim": 64, "repeats": 2,
                          "threads": [1]})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["run_id"]
    assert all(x["median_ms"] > 0 for x in out["results"])
    hist = client.get("/api/benchmarks/history", headers=auth).json()
    assert hist["runs"] and hist["runs"][0]["n_results"] > 0


def test_photo_list_survives_a_missing_signing_secret(client, auth, profile, monkeypatch):
    """A missing AE_MEDIA_SECRET must degrade, not 500.

    Signing exists so <img> tags can load bytes; it is an optimisation, not a
    requirement. An earlier version raised RuntimeError from the listing route, which
    turned an unset optional setting into a 500 on every photo list.
    """
    from attraction_engine.services import auth as auth_service

    monkeypatch.delenv("AE_MEDIA_SECRET", raising=False)
    monkeypatch.delenv("AE_OWNER_PASSWORD", raising=False)
    monkeypatch.setattr(auth_service, "_media_key_warned", True)

    r = client.get(f"/api/profiles/{profile['id']}/photos", headers=auth)
    assert r.status_code == 200, r.text
    photos = r.json()["photos"]
    assert len(photos) == 4
    assert all("url" not in p for p in photos), "no signature means no url, not '?sig='"
    # The credentialed endpoint still works.
    raw = client.get(f"/api/profiles/{profile['id']}/photos/{photos[0]['id']}/raw",
                     headers=auth)
    assert raw.status_code == 200


def test_photo_raw_rejects_a_tampered_signature(client, auth, profile):
    photos = client.get(f"/api/profiles/{profile['id']}/photos", headers=auth).json()["photos"]
    url = photos[0]["url"]                    # "<expires>.<mac>"
    expires, mac = url.split("sig=")[1].split(".")
    flipped = ("0" if mac[0] != "0" else "1") + mac[1:]
    r = client.get(f"/api/profiles/{profile['id']}/photos/{photos[0]['id']}/raw"
                   f"?sig={expires}.{flipped}")
    assert r.status_code == 401
