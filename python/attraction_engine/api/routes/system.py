"""System, capabilities, health and benchmark endpoints."""
from __future__ import annotations

import os
import platform
import time

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ... import native
from ...db.models import BenchmarkRun
from ...db.session import backend_info, session_scope
from ...services import cache
from ..deps import current_user, rate_limiter

router = APIRouter(tags=["system"], dependencies=[Depends(rate_limiter)])


@router.get("/system/capabilities")
def capabilities():
    from ...ml import encoders
    from ...services import storage

    return {
        "native": native.build_info(),
        "encoders": encoders.info().as_dict(),
        "database": backend_info(),
        "cache": cache.status(),
        "storage_dir": str(storage.STORE_DIR),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "disclaimer": "All profile, compatibility and counterfactual scores are model outputs "
                      "under declared assumptions. Nothing here is a causal claim about people.",
    }


@router.get("/system/health")
def health(user_id: str = Depends(current_user)):
    return {
        "status": "ok",
        "user_id": user_id,
        "native_available": native.NATIVE_AVAILABLE,
        "database": backend_info(),
        "cache": cache.status(),
    }


class BenchmarkRequest(BaseModel):
    sizes: list[int] = [128, 512, 2048]
    dim: int = 384
    k: int = 10
    repeats: int = 5
    strategies: list[str] | None = None
    threads: list[int] = [1, 2]


@router.post("/benchmarks/run")
def run_benchmark(req: BenchmarkRequest, user_id: str = Depends(current_user)):
    """Execute top-k and similarity benchmarks in-process and persist the results.

    These are real timings measured on the machine that serves the request; the UI
    labels them with the CPU reported by /system/capabilities.
    """
    import numpy as np

    core = native.require()
    rng = np.random.default_rng(1234)
    strategies = req.strategies or ["sort_full", "partial_sort", "nth_element", "heap_bounded"]
    results = []
    machine = {"cpu_count": os.cpu_count(), "platform": platform.platform(),
               "python": platform.python_version(), "processor": platform.processor()}

    # Bound the working set: an (n x dim) float64 matrix plus a float64 similarity
    # matrix is 8*n*dim + 8*64*n bytes. On the reference 1.9 GiB host a request for
    # n=65536, dim=384 would need ~230 MB per array and repeated allocation across
    # repeats killed the process. Budget 256 MB and clamp accordingly, reporting the
    # clamp in the response rather than silently shrinking the workload.
    BUDGET_BYTES = 256 * 1024 * 1024
    requested = [int(v) for v in req.sizes]
    clamped: list[dict] = []
    sizes: list[int] = []
    for n_raw in requested:
        n = max(req.k + 1, min(int(n_raw), 200000))
        per_row = 8 * req.dim + 8 * 64
        cap = max(req.k + 1, BUDGET_BYTES // max(1, per_row))
        if n > cap:
            clamped.append({"requested": n, "used": cap,
                            "reason": f"working set would exceed {BUDGET_BYTES // (1024*1024)} MB"})
            n = cap
        sizes.append(n)

    for n in sizes:
        scores = rng.random(n)
        x = rng.normal(size=(n, req.dim))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        for strat in strategies:
            times = []
            for _ in range(req.repeats):
                t0 = time.perf_counter()
                core.top_k(scores, req.k, getattr(core.TopKStrategy, strat))
                times.append((time.perf_counter() - t0) * 1000.0)
            times.sort()
            results.append({
                "benchmark": "top_k", "n": n, "k": req.k, "strategy": strat,
                "median_ms": round(times[len(times) // 2], 4),
                "min_ms": round(times[0], 4),
                "p95_ms": round(times[min(len(times) - 1, int(0.95 * (len(times) - 1)))], 4),
                "repeats": req.repeats,
            })
        q = np.ascontiguousarray(x[: min(64, n)])
        for kernel in ("eigen", "inhouse"):
            for threads in req.threads:
                times = []
                for _ in range(req.repeats):
                    t0 = time.perf_counter()
                    if kernel == "eigen":
                        core.batch_cosine_eigen(q, x, threads)
                    else:
                        core.batch_cosine_unit(q, x, threads)
                    times.append((time.perf_counter() - t0) * 1000.0)
                times.sort()
                med = times[len(times) // 2]
                results.append({
                    "benchmark": "batch_cosine", "kernel": kernel, "n": n, "dim": req.dim,
                    "threads": threads, "queries": int(q.shape[0]),
                    "median_ms": round(med, 4), "min_ms": round(times[0], 4),
                    "pairs_per_ms": round((q.shape[0] * n) / max(1e-9, med), 1),
                    "repeats": req.repeats,
                })
        # NumPy/BLAS baseline on the same arrays, so a stored run is self-contained.
        for _ in range(max(1, req.repeats // 2)):
            t0 = time.perf_counter()
            _ = q @ x.T
        times = []
        for _ in range(req.repeats):
            t0 = time.perf_counter()
            q @ x.T
            times.append((time.perf_counter() - t0) * 1000.0)
        times.sort()
        results.append({"benchmark": "batch_cosine", "kernel": "numpy_blas", "n": n,
                        "dim": req.dim, "threads": 0, "queries": int(q.shape[0]),
                        "median_ms": round(times[len(times) // 2], 4),
                        "min_ms": round(times[0], 4),
                        "pairs_per_ms": round((q.shape[0] * n) / max(1e-9, times[len(times) // 2]), 1),
                        "repeats": req.repeats})

    with session_scope() as s:
        run = BenchmarkRun(suite="api-inline", machine=machine, results=results)
        s.add(run)
        s.flush()
        run_id = run.id
    return {"run_id": run_id, "machine": machine, "results": results,
            "clamped": clamped,
            "note": "Measured on the API host for this request; includes pybind11 call "
                    "overhead from Python." + (
                        " Some sizes were clamped to keep the working set inside the "
                        "process memory budget; see `clamped`." if clamped else "")}


@router.get("/benchmarks/history")
def benchmark_history(limit: int = 10, user_id: str = Depends(current_user)):
    from sqlalchemy import select

    with session_scope() as s:
        rows = s.scalars(select(BenchmarkRun).order_by(BenchmarkRun.created_at.desc())
                         .limit(max(1, min(limit, 50)))).all()
        return {"runs": [
            {"id": r.id, "suite": r.suite, "machine": r.machine,
             "created_at": r.created_at.isoformat() if r.created_at else None,
             "n_results": len(r.results or []), "results": r.results}
            for r in rows
        ]}
