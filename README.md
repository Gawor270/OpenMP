# OpenMP

## Build

```sh
cmake -S . -B build
cmake --build build
```

## Run

```sh
./build/openmp_bucket_sort <vector_size> [base_seed] [--alg 1|2] [--csv <path>] [--test]
```

- `--alg 1|2` selects the bucket sort algorithm (default: 1)
- `--csv <path>` appends a CSV row with SLURM metadata if present
- Set `OMP_NUM_THREADS` to control OpenMP thread count

## SLURM (sbatch)

Neutral scripts are provided in `scripts/`:

- `scripts/sbatch_run.sh` submits a single job
- `scripts/run_matrix.sh` submits a scaling matrix up to 48 threads

Example single run:

```sh
sbatch --cpus-per-task=8 scripts/sbatch_run.sh 1000000 123456789 1 results
```

Weak scaling (size scales with threads) or strong scaling (fixed size):

```sh
scripts/run_matrix.sh 1000000 123456789 1 weak results
scripts/run_matrix.sh 1000000 123456789 1 strong results
```
