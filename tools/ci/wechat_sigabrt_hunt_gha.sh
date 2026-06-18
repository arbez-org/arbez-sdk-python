#!/usr/bin/env bash
# Hunt WeChat SIGABRT on GH Actions ubuntu-24.04-arm with core dumps + gdb.
#
# Usage:
#   bash tools/ci/wechat_sigabrt_hunt_gha.sh [attempts] [mode]
#   mode: gdb | raw | both   (default: both)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

ATTEMPTS="${1:-50}"
MODE="${2:-both}"
LOG_DIR="${LOG_DIR:-$ROOT/wechat-sigabrt-hunt-logs}"
CORE_DIR="${CORE_DIR:-$ROOT/wechat-sigabrt-cores}"
mkdir -p "$LOG_DIR" "$CORE_DIR"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export MALLOC_CHECK_=3
export GLIBCXX_ASSERTIONS=1

ulimit -c unlimited || true

# Core dumps into the artifact directory (writable on GHA runners).
if command -v sysctl >/dev/null 2>&1; then
  sudo sysctl -w "kernel.core_pattern=$CORE_DIR/core.%e.%p.%t" 2>/dev/null || true
fi

python -c "import cv2, sys; print('cv2', cv2.__version__, cv2.__file__); print('python', sys.version)"

_stress() {
  python - <<'PY' >/dev/null 2>&1 || true
import concurrent.futures
import qrcode
from arbez.engines.wechat import WeChatEngine
from arbez.engines.arbez import ArbezEngine

qr = qrcode.QRCode(version=2, box_size=10, border=4)
qr.add_data("gha-hunt")
qr.make(fit=True)
img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

arbez = ArbezEngine()
for _ in range(10):
    arbez.detect_and_decode(img)

engine = WeChatEngine()

def scan(_: int):
    return engine.detect_and_decode(img)

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    [f.result() for f in [pool.submit(scan, i) for i in range(100)]]
PY
}

_gdb_on_core() {
  local core
  core="$(find "$CORE_DIR" -name 'core.*' -type f 2>/dev/null | head -1 || true)"
  if [[ -z "$core" ]]; then
    echo "no core file found under $CORE_DIR"
    return 0
  fi
  echo "=== gdb backtrace from core: $core ==="
  gdb -batch \
    -ex "set pagination off" \
    -ex "bt full 80" \
    -ex "thread apply all bt 40" \
    -ex "quit" \
    "$(command -v python)" "$core" 2>&1 || true
}

_run_raw() {
  local attempt="$1"
  local log="$LOG_DIR/raw-attempt-${attempt}.log"
  rm -f "$CORE_DIR"/core.* 2>/dev/null || true
  set +e
  pytest -q tests/ 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ "$rc" -eq 134 || "$rc" -eq -11 ]]; then
    echo "RAW SIGABRT/SIGSEGV on attempt $attempt (exit $rc)"
    _gdb_on_core | tee "$LOG_DIR/raw-attempt-${attempt}-core-bt.log"
    return 134
  fi
  if grep -qE 'Fatal Python error: Aborted|Signal 6|SIGABRT' "$log"; then
    echo "RAW abort detected in log on attempt $attempt"
    _gdb_on_core | tee "$LOG_DIR/raw-attempt-${attempt}-core-bt.log"
    return 134
  fi
  return 0
}

_run_gdb() {
  local attempt="$1"
  local log="$LOG_DIR/gdb-attempt-${attempt}.log"
  rm -f "$CORE_DIR"/core.* 2>/dev/null || true
  gdb -batch \
    -ex "set pagination off" \
    -ex "set print thread-events off" \
    -ex "catch signal SIGABRT" \
    -ex "run" \
    -ex "echo \\n=== SIGABRT backtrace (main thread) ===\\n" \
    -ex "bt full 80" \
    -ex "echo \\n=== all threads ===\\n" \
    -ex "thread apply all bt full 40" \
    -ex "quit" \
    --args python -m pytest -q tests/ -x 2>&1 | tee "$log"
  if grep -q "Program received signal SIGABRT" "$log"; then
    echo "GDB caught SIGABRT on attempt $attempt"
    _gdb_on_core | tee "$LOG_DIR/gdb-attempt-${attempt}-core-bt.log"
    return 134
  fi
  return 0
}

for i in $(seq 1 "$ATTEMPTS"); do
  echo "########## GHA hunt $i / $ATTEMPTS (mode=$MODE) ##########"
  _stress

  if [[ "$MODE" == "raw" || "$MODE" == "both" ]]; then
    set +e
    _run_raw "$i"
    raw_rc=$?
    set -e
    if [[ "$raw_rc" -eq 134 ]]; then
      echo "SUCCESS: reproduced via raw pytest on attempt $i"
      exit 134
    fi
  fi

  if [[ "$MODE" == "gdb" || "$MODE" == "both" ]]; then
    set +e
    _run_gdb "$i"
    gdb_rc=$?
    set -e
    if [[ "$gdb_rc" -eq 134 ]]; then
      echo "SUCCESS: reproduced via gdb on attempt $i"
      exit 134
    fi
  fi
done

echo "No SIGABRT in $ATTEMPTS attempts (mode=$MODE). Logs: $LOG_DIR"
exit 0