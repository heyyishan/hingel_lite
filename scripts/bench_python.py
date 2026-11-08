#!/usr/bin/env python3
"""NumPy / Python baselines for the operations the native core implements.

Deliberately uses the *optimized* NumPy path for each operation (argpartition for
top-k, BLAS-backed matmul for cosine) so the comparison is fair. Each workload
uses the same data the C++ benchmark uses.
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np


def timed(fn, repeats: int) -> dict:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "median_ms": round(times[len(times) // 2], 5),
        "min_ms": round(times[0], 5),
        "p95_ms": round(times[min(len(times) - 1, int(0.95 * (len(times) - 1)))], 5),
        "repeats": repeats,
    }


def bench_topk(repeats: int) -> list[dict]:
    out = []
    rng = np.random.default_rng(99)
    for n, k in [(4096, 10), (65536, 10), (65536, 100), (1048576, 10)]:
        scores = rng.random(n)
        out.append({
            "benchmark": "top_k", "implementation": "numpy_argpartition",
            "n": n, "k": k,
            **timed(lambda: np.argpartition(scores, -k)[-k:], repeats),
        })
        out.append({
            "benchmark": "top_k", "implementation": "numpy_full_argsort",
            "n": n, "k": k,
            **timed(lambda: np.argsort(scores)[-k:], repeats),
        })
    return out


def bench_batch_cosine(repeats: int) -> list[dict]:
    out = []
    rng = np.random.default_rng(4)
    for n in (512, 4096, 32768):
        q = rng.normal(size=(64, 384))
        x = rng.normal(size=(n, 384))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        res = timed(lambda: q @ x.T, repeats)
        res.update({"benchmark": "batch_cosine", "implementation": "numpy_blas_matmul",
                    "n": n, "dim": 384, "queries": 64,
                    "pairs_per_ms": round(64 * n / max(1e-9, res["median_ms"]), 1)})
        out.append(res)
    return out


def bench_dot(repeats: int) -> list[dict]:
    rng = np.random.default_rng(11)
    out = []
    for dim in (128, 384, 512, 768, 1536):
        a = rng.normal(size=dim)
        b = rng.normal(size=dim)
        out.append({"benchmark": "dot_product", "implementation": "numpy_dot",
                    "n": dim, **timed(lambda: float(a @ b), repeats)})
    return out


def bench_mmr(repeats: int) -> list[dict]:
    """Greedy MMR in pure NumPy — the fair baseline for the C++ MMR loop."""
    out = []
    rng = np.random.default_rng(21)
    for n in (200, 2000):
        cand = rng.normal(size=(n, 384))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        rel = rng.random(n)
        k, lam = 20, 0.7

        def run():
            chosen = [int(np.argmax(rel))]
            max_sim = cand @ cand[chosen[0]]
            for _ in range(k - 1):
                score = lam * rel - (1 - lam) * max_sim
                score[chosen] = -np.inf
                nxt = int(np.argmax(score))
                chosen.append(nxt)
                max_sim = np.maximum(max_sim, cand @ cand[nxt])
            return chosen

        out.append({"benchmark": "mmr", "implementation": "numpy_greedy",
                    "n": n, "k": k, **timed(run, max(1, repeats // 2))})
    return out


def bench_beta(repeats: int) -> list[dict]:
    from scipy import stats

    out = []
    out.append({"benchmark": "credible_interval", "implementation": "scipy_beta_ppf",
                **timed(lambda: (stats.beta.ppf(0.025, 41, 161),
                                 stats.beta.ppf(0.975, 41, 161)), repeats)})

    def superiority():
        # P(X > Y) for independent Betas via quadrature
        xs = np.linspace(0, 1, 4096)
        return float(np.trapezoid(stats.beta.pdf(xs, 40, 160) * stats.beta.cdf(xs, 60, 140), xs))

    out.append({"benchmark": "prob_superiority", "implementation": "scipy_quadrature",
                **timed(superiority, max(1, repeats // 2))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()
    payload = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "results": (bench_dot(args.repeats) + bench_topk(args.repeats)
                    + bench_batch_cosine(args.repeats) + bench_mmr(args.repeats)
                    + bench_beta(args.repeats)),
        "method": "time.perf_counter around the optimized NumPy/SciPy call, "
                  "median of N repeats, ascending sort, p95 by index",
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
