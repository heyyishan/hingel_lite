#!/usr/bin/env bash
# One command that produces every verification number quoted in the README.
# Re-runnable: each section prints PASS/FAIL and the script exits non-zero if any
# section fails, so it can gate a release.
set -uo pipefail
cd "$(dirname "$0")/.."
FAIL=0
say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
check() { if [ "$1" -eq 0 ]; then echo "PASS: $2"; else echo "FAIL: $2"; FAIL=1; fi; }

say "0/8 Build native artefacts if absent"
if [ ! -x c23/build/release/aenum_tests ]; then
  cmake -S c23 -B c23/build/release -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1 \
    && cmake --build c23/build/release -j"$(nproc)" > /dev/null 2>&1
fi
if [ ! -x cpp/build/ae_tests ]; then
  cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1 \
    && cmake --build cpp/build -j"$(nproc)" > /dev/null 2>&1
fi
check $? "native builds present"

say "1/8 C23 numerical library (Release)"
./c23/build/release/aenum_tests > /tmp/c23.log 2>&1
check $? "$(tail -1 /tmp/c23.log)"

say "2/8 C++26 core (Release)"
./cpp/build/ae_tests > /tmp/cpp.log 2>&1
check $? "$(tail -1 /tmp/cpp.log)"

say "3/8 Python + native bindings + API integration"
python3 -m pytest python/tests -q > /tmp/py.log 2>&1
check $? "$(tail -1 /tmp/py.log)"

say "4/8 End-to-end acceptance harness (live server on :8000)"
python3 scripts/e2e_check.py > /tmp/e2e.log 2>&1
check $? "$(tail -1 /tmp/e2e.log)"

say "5/8 Benchmarks (native vs NumPy, reproducible)"
if [ -x scripts/run_benchmarks.sh ]; then
  bash scripts/run_benchmarks.sh quick > /tmp/bench.log 2>&1
  check $? "$(grep -o '"comparisons": [0-9]*' benchmarks/results/combined.json | tail -1)"
else
  echo "SKIP: runner missing"; FAIL=1
fi

say "6/8 Frontend typecheck + production build"
( cd frontend && npx tsc --noEmit > /tmp/tsc.log 2>&1 && npm run build >> /tmp/tsc.log 2>&1 )
check $? "$(grep -c 'built in' /tmp/tsc.log) production build(s)"

say "7/8 Frontend smoke test in headless Chromium"
python3 scripts/ui_smoke.py > /tmp/ui.log 2>&1
check $? "$(tail -1 /tmp/ui.log)"

say "8/8 Summary"
echo "artifacts: /tmp/{c23,cpp,py,e2e,bench,tsc,ui}.log"
[ "$FAIL" -eq 0 ] && echo "ALL CHECKS PASSED" || echo "SOME CHECKS FAILED"
exit "$FAIL"
