"""Shared fixtures.

The API tests run against the real application object (no mocks of the database or
of the native core) using a dedicated PostgreSQL database, so a passing test means
the same SQL and the same pgvector operators as production.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg2://ae:ae_dev_password@localhost:5432/attraction_engine_test")
os.environ.setdefault("AE_STORE_DIR", str(ROOT / "data" / "test_photos"))
os.environ.setdefault("AE_CACHE_TTL", "1")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6379/9")


@pytest.fixture(scope="session")
def live_server(tmp_path_factory):
    """A real uvicorn process on a free port, pointed at the test database.

    Using TestClient is not viable with the installed FastAPI version (its
    middleware stack requires an ASGI lifespan context that TestClient does not
    provide), and a real socket is closer to production anyway.
    """
    import socket
    import subprocess
    import time
    from pathlib import Path

    import httpx

    root = Path(__file__).resolve().parents[2]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**os.environ,
           "DATABASE_URL": "postgresql+psycopg2://ae:ae_dev_password@localhost:5432/attraction_engine_test",
           "AE_STORE_DIR": str(tmp_path_factory.mktemp("photos")),
           "PYTHONPATH": str(root / "python")}
    # Migrate the test database to head before starting.
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"],
                   cwd=root, env=env, check=True, capture_output=True)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "attraction_engine.api.main:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 120
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("server exited: " + proc.stdout.read().decode()[:4000])
        try:
            if httpx.get(base + "/healthz", timeout=1.0).status_code == 200:
                break
        except Exception:
            time.sleep(0.25)
    else:
        raise RuntimeError("server did not become healthy")
    yield base
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def client(live_server):
    import httpx

    with httpx.Client(base_url=live_server, timeout=180) as c:
        yield c


@pytest.fixture(scope="session")
def auth(client):
    import subprocess

    subprocess.run([sys.executable, str(ROOT / "scripts" / "dev_reset_login.py")],
                   check=True, capture_output=True,
                   env={**os.environ,
                        "DATABASE_URL": "postgresql+psycopg2://ae:ae_dev_password"
                                        "@localhost:5432/attraction_engine_test",
                        "AE_OWNER_PASSWORD": "attraction-engine"})
    r = client.post("/api/auth/login", json={"handle": "owner", "password": "attraction-engine"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


@pytest.fixture(scope="session")
def native_core():
    from attraction_engine import native

    pytest.importorskip("attraction_engine.native")
    if native.core is None:
        pytest.skip("native extension not built")
    return native.core
