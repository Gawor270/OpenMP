#!/bin/bash -l
#SBATCH --job-name=openmp-bucket
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --time=08:00:00
#SBATCH --partition=plgrid-now
#SBATCH --account=plgmpr26-cpu

set -euo pipefail

# If run on login node, update repo and submit this script.
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

THREADS_LIST=(1 2 4 8 16 24 32 40 48)
REPEATS=5
BASE_SEED=123456789
BASE_SIZE=1000000

mkdir -p results_strong_alg1 results_strong_alg2 results_weak_alg1 results_weak_alg2

BUCKET_MULT=64   # optimised from sweep: best elbow of gain curve

# Strong scaling
for alg in 1 2; do
  for t in "${THREADS_LIST[@]}"; do
    export OMP_NUM_THREADS="$t"
    for r in $(seq 1 "$REPEATS"); do
      ./build/openmp_bucket_sort "$BASE_SIZE" "$BASE_SEED" \
        --alg "$alg" --bucket-mult "$BUCKET_MULT" \
        --csv "results_strong_alg${alg}/run_${t}_${r}.csv"
    done
  done
done

# Weak scaling
for alg in 1 2; do
  for t in "${THREADS_LIST[@]}"; do
    export OMP_NUM_THREADS="$t"
    size=$((BASE_SIZE * t))
    for r in $(seq 1 "$REPEATS"); do
      ./build/openmp_bucket_sort "$size" "$BASE_SEED" \
        --alg "$alg" --bucket-mult "$BUCKET_MULT" \
        --csv "results_weak_alg${alg}/run_${t}_${r}.csv"
    done
  done
done
