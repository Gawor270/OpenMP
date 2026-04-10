#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <omp.h>

int main(int argc, char** argv) {
  constexpr std::uint64_t kDefaultSeed = 123456789ull;

  if (argc < 2 || argc > 3) {
    std::cerr << "Usage: " << argv[0] << " <vector_size> [base_seed]\n";
    return 1;
  }

  const std::size_t vector_size = static_cast<std::size_t>(std::stoull(argv[1]));
  const std::uint64_t base_seed = (argc == 3)
                                      ? static_cast<std::uint64_t>(std::stoull(argv[2]))
                                      : kDefaultSeed;

  std::vector<double> values(vector_size);

#pragma omp parallel
  {
    const int thread_id = omp_get_thread_num();
    std::mt19937_64 engine(base_seed + static_cast<std::uint64_t>(thread_id));
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

#pragma omp for schedule(static)
    for (long long i = 0; i < static_cast<long long>(values.size()); ++i) {
      values[static_cast<std::size_t>(i)] = distribution(engine);
    }
  }

  double sum = 0.0;

#pragma omp parallel for reduction(+ : sum)
  for (long long i = 0; i < static_cast<long long>(values.size()); ++i) {
    sum += values[static_cast<std::size_t>(i)];
  }

  const double mean = sum / static_cast<double>(values.size());

  std::cout << "Generated " << values.size() << " random values in [0, 1).\n";
  std::cout << "Threads used (max): " << omp_get_max_threads() << "\n";
  std::cout << "Base seed: " << base_seed << "\n";
  std::cout << "Mean: " << mean << "\n";

  return 0;
}
