#!/usr/bin/env bash
# Run the derivative-overview data jobs sequentially in the background.
#
# Each job is a resumable run_tangent_sample_scaling.py study with its own
# directory and driver.log under ROOT.  A failed job is recorded and the next
# job still starts.  Monitor with:
#   .venv/bin/python scripts/monitor_tangent_sample_scaling.py --root ROOT status
#   .venv/bin/python scripts/monitor_tangent_sample_scaling.py --root ROOT --plot watch
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/.venv/bin/python"
RUNNER="$REPO/scripts/run_tangent_sample_scaling.py"
ROOT="${1:-$REPO/results/aot_derivative_overview}"
WORKERS="${WORKERS:-40}"
mkdir -p "$ROOT"
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-derivative-overview}"

# log10 g = -1.3, -1.2, ..., 1.0 (24 points, includes -1, 0, 1).
G_VALUES="$("$PY" -c 'print(" ".join(f"{10 ** (k / 10):.12g}" for k in range(-13, 11)))')"
# s = gamma*T with T = 1: the Aug-23 Dressel points plus 5 in between.
S_VALUES="0.1 0.15 0.2 0.3 0.4 0.55 0.7 1.0 1.5 2.0 2.5 3.2 4.0"

cat > "$ROOT/jobs.json" <<EOF
[
  {"name": "dressel_scan", "dir": "dressel_scan", "estimate_minutes": 1},
  {"name": "dressel_histogram_s0.1", "dir": "dressel_histogram", "estimate_minutes": 2},
  {"name": "driven_qubit", "dir": "driven_qubit", "estimate_minutes": 6},
  {"name": "chain_L2", "dir": "chain_L2", "estimate_minutes": 8},
  {"name": "chain_L2_burnin20_check", "dir": "chain_L2_burnin20", "estimate_minutes": 2},
  {"name": "chain_L8", "dir": "chain_L8", "estimate_minutes": 15}
]
EOF
echo $$ > "$ROOT/orchestrator.pid"

run_job() {
  local name="$1" dir="$ROOT/$2"
  shift 2
  mkdir -p "$dir"
  echo "[$(date -Is)] START $name" | tee -a "$ROOT/orchestrator.log"
  "$PY" "$RUNNER" "$@" --workers "$WORKERS" --output-dir "$dir" > "$dir/driver.log" 2>&1
  local code=$?
  echo "exit=$code" >> "$dir/driver.log"
  if [ "$code" -eq 0 ]; then
    echo "[$(date -Is)] DONE $name" | tee -a "$ROOT/orchestrator.log"
  else
    echo "[$(date -Is)] FAILED $name exit=$code (see $dir/driver.log)" | tee -a "$ROOT/orchestrator.log"
  fi
}

run_job dressel_scan dressel_scan --system single-qubit --J 0 --gamma-values $S_VALUES \
  --dt 0.00025 --burn-in-time 0 --sample-time 1 --n-trajectories 16384 --chunk-size 2048 \
  --save-final-z --seed 20261005
run_job dressel_histogram_s0.1 dressel_histogram --system single-qubit --J 0 --gamma-values 0.1 \
  --dt 0.00025 --burn-in-time 0 --sample-time 1 --n-trajectories 1048576 --chunk-size 16384 \
  --save-final-z --seed 20261105
run_job driven_qubit driven_qubit --system single-qubit --J 1 --g-values $G_VALUES \
  --dt 0.00125 --burn-in-time 20 --sample-time 10 --n-trajectories 16384 --chunk-size 2048 \
  --reference --reference-timeout 1200 --seed 20261205
run_job chain_L2 chain_L2 --system chain --L 2 --J 1 --g-values $G_VALUES \
  --dt 0.00125 --burn-in-time 5 --sample-time 10 --n-trajectories 2048 --chunk-size 64 \
  --first-chunk-size 64 --seed 20261305
run_job chain_L2_burnin20_check chain_L2_burnin20 --system chain --L 2 --J 1 --g-values 0.1 1 10 \
  --dt 0.00125 --burn-in-time 20 --sample-time 10 --n-trajectories 2048 --chunk-size 64 \
  --first-chunk-size 64 --seed 20261405
run_job chain_L8 chain_L8 --system chain --L 8 --J 1 --g-values $G_VALUES \
  --dt 0.00125 --burn-in-time 5 --sample-time 10 --n-trajectories 512 --chunk-size 16 \
  --first-chunk-size 16 --seed 20261505

echo "[$(date -Is)] ALL_JOBS_FINISHED" | tee -a "$ROOT/orchestrator.log"
rm -f "$ROOT/orchestrator.pid"
