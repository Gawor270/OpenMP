#!/usr/bin/env bash
#SBATCH --job-name=bucket_sort
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
set -euo pipefail

VECTOR_SIZE=${1:-1000000}
BASE_SEED=${2:-123456789}
ALGORITHM=${3:-1}
CSV_DIR=${4:-results}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-1}}

mkdir -p "$CSV_DIR"

CSV_PATH="$CSV_DIR/run_${SLURM_JOB_ID:-local}.csv"

cmake -S . -B build
cmake --build build

./build/openmp_bucket_sort "$VECTOR_SIZE" "$BASE_SEED" --alg "$ALGORITHM" --csv "$CSV_PATH"
