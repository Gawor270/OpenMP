#!/bin/bash -l
#SBATCH --job-name=bucket-sweep
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --partition=plgrid-now
#SBATCH --account=plgmpr26-cpu

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  git pull --ff-only
  sbatch "$0"
  exit 0
fi

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${ROOT_DIR}"

if ! command -v cmake >/dev/null 2>&1; then
  module purge
  module load cmake gcc
fi

cmake -S . -B build
cmake --build build

MULTS=(1 2 4 8 16 32 64 128)
THREADS_LIST=(1 2 4 8 16 24 32 40 48)
REPEATS=5
BASE_SEED=123456789
BASE_SIZE=1000000

mkdir -p results_bucket_sweep

for mult in "${MULTS[@]}"; do
  for t in "${THREADS_LIST[@]}"; do
    export OMP_NUM_THREADS="$t"
    for r in $(seq 1 "$REPEATS"); do
      ./build/openmp_bucket_sort "$BASE_SIZE" "$BASE_SEED" \
        --alg 2 --bucket-mult "$mult" \
        --csv "results_bucket_sweep/mult_${mult}_t${t}_r${r}.csv"
    done
  done
done
