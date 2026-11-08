#!/usr/bin/env python3
"""Merge Google Benchmark JSON with the NumPy baseline into one report.

Emits, next to the raw data, a Markdown fragment (docs/benchmarks_table.md) that
the README includes. Nothing is hand-edited: if the numbers change, re-running the
harness rewrites the table.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_name(name: str) -> dict:
    """BM_TopK<ae::TopKStrategy::heap_bounded>/65536/10 -> structured fields."""
    m = re.match(r"BM_(\w+)(?:<[^:]*::([\w:]+)>)?(?:/(.+))?$", name)
    if not m:
        return {"family": name, "args": []}
    family, variant, args = m.group(1), m.group(2), m.group(3)
    arg_list = []
    if args:
        for a in args.split("/"):
            # Google Benchmark appends _median/_mean/... to the LAST argument of
            # aggregate rows, so strip the suffix before parsing.
            for suffix in ("_median", "_mean", "_stddev", "_cv", "_min", "_max"):
                if a.endswith(suffix):
                    a = a[: -len(suffix)]
                    break
            try:
                arg_list.append(float(a) if "." in a else int(a))
            except ValueError:
                arg_list.append(a)
    for suffix in ("_median", "_mean", "_stddev", "_cv", "_min", "_max"):
        if family.endswith(suffix):
            family = family[: -len(suffix)]
    return {"family": family, "variant": variant, "args": arg_list}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp", required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--environment", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="quick")
    args = ap.parse_args()

    cpp = json.loads(Path(args.cpp).read_text())
    py = json.loads(Path(args.python).read_text())
    env = json.loads(Path(args.environment).read_text())

    native = []
    for bench in cpp.get("benchmarks", []):
        name = bench["name"]
        aggregate = bench.get("aggregate_name", "single")
        if aggregate not in ("median", "single"):
            continue
        meta = parse_name(name)
        native.append({
            "name": name,
            "family": meta["family"],
            "variant": meta.get("variant"),
            "args": meta.get("args", []),
            "aggregate": aggregate,
            "real_ms": round(bench["real_time"] / 1e6, 6) if bench["time_unit"] == "ns" else bench["real_time"],
            "cpu_ms": round(bench["cpu_time"] / 1e6, 6) if bench["time_unit"] == "ns" else bench["cpu_time"],
            "time_unit": "ms",
            "iterations": bench.get("iterations"),
        })

    # Attach like-for-like comparisons where an equivalent NumPy baseline exists.
    comparisons = []
    py_by_family: dict[str, list[dict]] = {}
    for r in py["results"]:
        py_by_family.setdefault(r["benchmark"], []).append(r)

    for row in native:
        fam = row["family"]
        if fam == "TopK":
            k = int(row["args"][1]) if len(row["args"]) > 1 else 10
            n = int(row["args"][0]) if row["args"] else 0
            for base in py_by_family.get("top_k", []):
                if base["n"] == n and base["k"] == k:
                    comparisons.append({
                        "workload": f"top_k n={n} k={k}",
                        "native": row["variant"],
                        "native_ms": row["real_ms"],
                        "baseline": base["implementation"],
                        "baseline_ms": base["median_ms"],
                        "speedup": round(base["median_ms"] / row["real_ms"], 2)
                        if row["real_ms"] else None,
                    })
        elif fam in ("BatchCosine", "BatchCosineEigen"):
            n = int(row["args"][0])
            for base in py_by_family.get("batch_cosine", []):
                if base["n"] == n:
                    comparisons.append({
                        "workload": f"batch_cosine 64x{n} d=384",
                        "native": ("eigen+threads=%d" if fam == "BatchCosineEigen"
                                   else "inhouse+threads=%d") % int(row["args"][1]),
                        "native_ms": row["real_ms"],
                        "baseline": base["implementation"],
                        "baseline_ms": base["median_ms"],
                        "speedup": round(base["median_ms"] / row["real_ms"], 2)
                        if row["real_ms"] else None,
                    })
        elif fam == "DotProduct":
            dim = int(row["args"][0])
            for base in py_by_family.get("dot_product", []):
                if base["n"] == dim:
                    comparisons.append({
                        "workload": f"dot {dim}",
                        "native": "dispatched",
                        "native_ms": row["real_ms"],
                        "baseline": base["implementation"],
                        "baseline_ms": base["median_ms"],
                        "speedup": round(base["median_ms"] / row["real_ms"], 2)
                        if row["real_ms"] else None,
                    })
        elif fam == "Mmr":
            n = int(row["args"][0])
            for base in py_by_family.get("mmr", []):
                if base["n"] == n:
                    comparisons.append({
                        "workload": f"mmr n={n} k=20",
                        "native": "cpp_greedy",
                        "native_ms": row["real_ms"],
                        "baseline": base["implementation"],
                        "baseline_ms": base["median_ms"],
                        "speedup": round(base["median_ms"] / row["real_ms"], 2)
                        if row["real_ms"] else None,
                    })
        elif fam == "ProbSuperiority":
            for base in py_by_family.get("prob_superiority", []):
                comparisons.append({
                    "workload": "P(theta_A > theta_B)",
                    "native": "cpp_simpson_4096",
                    "native_ms": row["real_ms"],
                    "baseline": base["implementation"],
                    "baseline_ms": base["median_ms"],
                    "speedup": round(base["median_ms"] / row["real_ms"], 2)
                    if row["real_ms"] else None,
                })
        elif fam == "CredibleInterval":
            for base in py_by_family.get("credible_interval", []):
                comparisons.append({
                    "workload": "95% credible interval",
                    "native": "cpp_bisection",
                    "native_ms": row["real_ms"],
                    "baseline": base["implementation"],
                    "baseline_ms": base["median_ms"],
                    "speedup": round(base["median_ms"] / row["real_ms"], 2)
                    if row["real_ms"] else None,
                })

    report = {
        "mode": args.mode,
        "environment": env,
        "native_results": native,
        "python_results": py["results"],
        "comparisons": comparisons,
        "counts": {"native": len(native), "python": len(py["results"]),
                   "comparisons": len(comparisons)},
    }
    Path(args.out).write_text(json.dumps(report, indent=2))

    # ---- Markdown fragment -------------------------------------------------
    lines = ["<!-- generated by scripts/bench_combine.py; do not edit by hand -->",
             f"_{env['cpu_model']}, {env['cpu_count_logical']} vCPU, "
             f"gcc {env['compiler']['gxx'] if 'gxx' in env['compiler'] else env['compiler']['gcc']}, "
             f"mode={args.mode}, "
             f"{env['timestamp_utc']}_", ""]
    lines.append("### Native core vs NumPy baseline\n")
    lines.append("| workload | native impl | native (ms) | baseline | baseline (ms) | speedup |")
    lines.append("|---|---|---:|---|---:|---:|")
    for c in sorted(comparisons, key=lambda c: (c["workload"], str(c["native"]))):
        lines.append(f"| {c['workload']} | {c['native']} | {c['native_ms']:.5f} | "
                     f"{c['baseline']} | {c['baseline_ms']:.5f} | {c['speedup']}× |")
    lines.append("")
    lines.append("### Native micro-benchmarks\n")
    lines.append("| benchmark | variant | args | median (ms) |")
    lines.append("|---|---|---|---:|")
    for r in native:
        if r["aggregate"] != "median" and args.mode != "quick":
            continue
        args_str = ", ".join(str(int(a)) if isinstance(a, float) and a.is_integer() else str(a)
                             for a in r["args"])
        lines.append(f"| {r['family']} | {r['variant'] or '-'} | {args_str} | {r['real_ms']:.6f} |")
    table_path = Path(args.out).with_name("benchmarks_table.md")
    table_path.write_text("\n".join(lines) + "\n")
    print(json.dumps({"wrote": str(args.out), "table": str(table_path),
                      "comparisons": len(comparisons), "native_rows": len(native)}, indent=2))


if __name__ == "__main__":
    main()
