#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <omp.h>

std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

double to_unit_interval(std::uint64_t value) {
  constexpr double kInvPow53 = 1.0 / 9007199254740992.0;
  return static_cast<double>(value >> 11) * kInvPow53;
}

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
#pragma omp for schedule(static)
    for (long long i = 0; i < static_cast<long long>(values.size()); ++i) {
      const std::uint64_t sample_seed = base_seed + static_cast<std::uint64_t>(i);
      values[static_cast<std::size_t>(i)] = to_unit_interval(splitmix64(sample_seed));
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
