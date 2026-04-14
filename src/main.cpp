#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <fstream>

#include <omp.h>

int main(int argc, char** argv) {
  constexpr std::uint64_t kDefaultSeed = 123456789ull;

  if (argc < 3 || argc > 4) {
    std::cerr << "Usage: " << argv[0] << " <schedule_type> <chunk_size> [base_seed]\n";
    return 1;
  }

  const std::vector<std::size_t> vector_sizes = {
    5'000'000,
    2'0'000'000,
    5'0'000'000,
    1'00'000'000,
    2'00'000'000,
  };

  std::string schedule_type = argv[1];
  int chunk = std::stoi(argv[2]);
  const std::uint64_t base_seed = (argc == 4)
                                      ? static_cast<std::uint64_t>(std::stoull(argv[3]))
                                      : kDefaultSeed;

  omp_sched_t kind;
  if (schedule_type == "static") kind = omp_sched_static;
  else if (schedule_type == "dynamic") kind = omp_sched_dynamic;
  else if (schedule_type == "guided") kind = omp_sched_guided;
  else kind = omp_sched_auto;

  omp_set_schedule(kind, chunk);

  std::ofstream outfile("dynamic_1_t_8.txt", std::ios::app); // <- TUTAJ ZMIEN NAZWE PLIKU
  if (!outfile) {
    std::cerr << "Error: could not open file\n";
    return 1;
  }

  for (const std::size_t vector_size : vector_sizes) {
    std::vector<double> values(vector_size);

    double start = omp_get_wtime();

#pragma omp parallel
    {
      const int thread_id = omp_get_thread_num();
      std::mt19937_64 engine(base_seed + static_cast<std::uint64_t>(thread_id));
      std::uniform_real_distribution<double> distribution(0.0, 1.0);

#pragma omp for schedule(runtime)
      for (long long i = 0; i < static_cast<long long>(values.size()); ++i) {
        values[static_cast<std::size_t>(i)] = distribution(engine);
      }
    }

    double sum = 0.0;

#pragma omp parallel for reduction(+ : sum)
    for (long long i = 0; i < static_cast<long long>(values.size()); ++i) {
      sum += values[static_cast<std::size_t>(i)];
    }

    double end = omp_get_wtime();
    double elapsed = end - start;

    const double mean = sum / static_cast<double>(values.size());

    outfile
        << "Size: " << values.size() << "\n"
        << "Threads: " << omp_get_max_threads() << "\n"
        << "Schedule: " << schedule_type << "\n"
        << "Chunk size: " << chunk << "\n"
        << "Time: " << elapsed << "\n"
        << "---\n";
  }

  return 0;
}