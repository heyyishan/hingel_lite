#!/usr/bin/env python3
"""Capture the exact environment a benchmark run happened in."""
from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:
        return ""


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


def cpp_flags() -> dict:
    build = Path(__file__).resolve().parents[1] / "cpp" / "build"
    flags = {}
    for name in ("ae_bench", "libae_core.a"):
        path = build / name
        if path.exists():
            flags[name] = sh(f"file {path}")[:200]
    ninja = build / "build.ninja"
    if ninja.exists():
        text = ninja.read_text()
        m = re.search(r"FLAGS = (.*)", text)
        if m:
            flags["compile_flags_sample"] = m.group(1)[:400]
    return flags


def main() -> None:
    try:
        from attraction_engine import native

        native_info = native.build_info()
    except Exception as exc:
        native_info = {"error": str(exc)}

    info = {
        "timestamp_utc": sh("date -u +%Y-%m-%dT%H:%M:%SZ"),
        "os": f"{platform.system()} {platform.release()}",
        "distribution": sh("grep PRETTY_NAME /etc/os-release | cut -d= -f2-").strip('"'),
        "kernel": platform.version(),
        "cpu_model": cpu_model(),
        "cpu_count_logical": __import__("os").cpu_count(),
        "cpu_flags": sh("grep -m1 flags /proc/cpuinfo").split(":", 1)[-1].strip()[:400],
        "memory_total_kb": sh("grep MemTotal /proc/meminfo").split()[1],
        "compiler": {
            "gcc": sh("gcc -dumpfullversion -dumpversion"),
            "gxx_std_supported": "c++26",
            "cpp_compile_flags": "-O3 -std=c++26 -DNDEBUG",
            "c_compile_flags": "-std=c23 -O3 -fno-math-errno -fno-trapping-math",
            "sanitizers_in_release": False,
            "artifacts": cpp_flags(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "libraries": {},
        "native_core": native_info,
        "notes": [
            "Benchmarks run inside a container on a shared host with 2 vCPUs.",
            "Absolute timings are machine-specific; ratios are the portable result.",
            "Google Benchmark reports CPU time and wall time; medians across "
            "repetitions are used in the report.",
        ],
    }
    for mod in ("numpy", "torch", "open_clip", "sentence_transformers", "duckdb", "polars"):
        try:
            m = __import__(mod)
            info["libraries"][mod] = getattr(m, "__version__", "unknown")
        except Exception:
            info["libraries"][mod] = None
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
