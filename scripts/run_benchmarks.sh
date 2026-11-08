#!/usr/bin/env bash
# Reproducible benchmark harness.
#
#   ./scripts/run_benchmarks.sh [quick|full]
#
# Produces machine-readable artifacts under benchmarks/results/:
#   cpp_<timestamp>.json      Google Benchmark JSON (native core)
#   python_<timestamp>.json   NumPy baseline timings on identical data
#   combined.json             the merged report consumed by docs/benchmarks.md
#   environment.json          CPU, compiler, flags and library versions
# Nothing in docs/benchmarks.md is written by hand.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-quick}"
OUT=benchmarks/results
mkdir -p "$OUT"
TS=$(date -u +%Y%m%dT%H%M%SZ)

MIN_TIME=$([ "$MODE" = "full" ] && echo 0.5s || echo 0.1s)
REPEATS=$([ "$MODE" = "full" ] && echo 5 || echo 2)

if [ ! -x cpp/build/ae_bench ]; then
  echo "ae_bench not built; configure with -DAE_BUILD_BENCH=ON" >&2
  exit 1
fi

echo "== environment =="
python3 scripts/bench_environment.py > "$OUT/environment_$TS.json"
cat "$OUT/environment_$TS.json"

echo "== C++26 / C23 native core =="
./cpp/build/ae_bench \
    --benchmark_format=json \
    --benchmark_min_time="$MIN_TIME" \
    --benchmark_repetitions="$REPEATS" \
    --benchmark_report_aggregates_only=true \
    > "$OUT/cpp_$TS.json"

echo "== NumPy baselines (identical workloads) =="
python3 scripts/bench_python.py --repeats "$((REPEATS * 5))" > "$OUT/python_$TS.json"

echo "== merging =="
python3 scripts/bench_combine.py \
    --cpp "$OUT/cpp_$TS.json" \
    --python "$OUT/python_$TS.json" \
    --environment "$OUT/environment_$TS.json" \
    --out "$OUT/combined.json" \
    --mode "$MODE"

echo "results written to $OUT/combined.json"
