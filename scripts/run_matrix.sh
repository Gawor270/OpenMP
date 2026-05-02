#!/usr/bin/env bash
set -euo pipefail

BASE_SIZE=${1:-1000000}
BASE_SEED=${2:-123456789}
ALGORITHM=${3:-1}
MODE=${4:-weak}
CSV_DIR=${5:-results}
REPS=${6:-5}

THREADS=(1 2 4 8 16 24 32 40 48)

submit_job() {
  local threads=$1
  local size=$2

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "sbatch --array=1-${REPS} --cpus-per-task=${threads} scripts/sbatch_run.sh ${size} ${BASE_SEED} ${ALGORITHM} ${CSV_DIR}"
    return 0
  fi

  sbatch --array=1-"${REPS}" --cpus-per-task="${threads}" scripts/sbatch_run.sh "${size}" "${BASE_SEED}" "${ALGORITHM}" "${CSV_DIR}"
}

for threads in "${THREADS[@]}"; do
  if [[ "$MODE" == "weak" ]]; then
    size=$((BASE_SIZE * threads))
  else
    size=$BASE_SIZE
  fi

  submit_job "$threads" "$size"
done
