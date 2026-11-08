"""ATTRACTION ENGINE — FastAPI application.

Endpoint groups map 1:1 to the product modules:
  /api/auth            authentication
  /api/profiles        profile CRUD, prompts, photographs
  /api/intelligence    Module A multimodal profile report
  /api/laboratory      Module B ordering optimisation, sensitivity, ablation
  /api/compatibility   Module C retrieval + reciprocal ranking
  /api/experiments     Module D Bayesian experimentation
  /api/counterfactual  Module E simulator
  /api/conversations   Module F analytics
  /api/benchmarks      backend benchmarks (real timings, stored)
  /api/system          capabilities, health, metrics
  /api/privacy         export, deletion, retention
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, REGISTRY, generate_latest

from .. import native
from ..db.session import backend_info, engine
from ..services import auth, cache
from .routes import (auth_routes, compatibility, conversations, counterfactual, experiments,
                     intelligence, laboratory, privacy, profiles, system)

log = logging.getLogger("attraction_engine.api")

REQUESTS = Counter("ae_http_requests_total", "HTTP requests", ["method", "route", "status"])
LATENCY = Histogram("ae_http_request_seconds", "Request latency", ["route"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Migrations are applied by scripts/setup.sh or the container entrypoint; the
    # application verifies the schema rather than mutating it on boot.
    owner = auth.ensure_owner()
    log.info("installation owner ready: %s", owner.handle)
    yield


app = FastAPI(
    title="ATTRACTION ENGINE",
    version="1.0.0",
    description=(
        "Privacy-preserving multimodal dating intelligence. Local-first; analyses only "
        "data the user supplies. Predictions are model outputs, not causal facts."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("AE_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def observability(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:  # convert unhandled errors into the structured envelope
        log.exception("unhandled error request_id=%s", request_id)
        body = {"error": {"code": "internal_error", "message": "unexpected server error",
                          "request_id": request_id, "detail": type(exc).__name__}}
        response = JSONResponse(status_code=500, content=body)
    elapsed = time.perf_counter() - started
    route = request.scope.get("route")
    label = getattr(route, "path", request.url.path)
    LATENCY.labels(route=label).observe(elapsed)
    REQUESTS.labels(method=request.method, route=label, status=str(response.status_code)).inc()
    response.headers["x-request-id"] = request_id
    response.headers["x-response-time-ms"] = f"{elapsed * 1000:.1f}"
    response.headers["x-engine-native"] = "1" if native.NATIVE_AVAILABLE else "0"
    return response


api = APIRouter(prefix="/api")
api.include_router(auth_routes.router)
api.include_router(system.router)
api.include_router(profiles.router)
api.include_router(intelligence.router)
api.include_router(laboratory.router)
api.include_router(compatibility.router)
api.include_router(experiments.router)
api.include_router(counterfactual.router)
api.include_router(conversations.router)
api.include_router(privacy.router)
app.include_router(api)


@app.get("/healthz", tags=["system"])
def healthz():
    return {"status": "ok", "native": native.NATIVE_AVAILABLE,
            "db": backend_info()["is_postgres"], "cache": cache.status()["enabled"]}


@app.get("/metrics", tags=["system"])
def metrics():
    return PlainTextResponse(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


# Serve the built frontend when present so a single process can host everything.
_dist = Path(os.environ.get("AE_FRONTEND_DIST", Path(__file__).resolve().parents[3] / "frontend" / "dist"))
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
