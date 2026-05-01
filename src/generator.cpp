#include "generator.h"

#include <utility>

#include <omp.h>

namespace {
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
}

GeneratedData generate_values(std::size_t size, std::uint64_t base_seed) {
  std::vector<double> values(size);
  const double start_seconds = omp_get_wtime();

#pragma omp parallel for schedule(static)
  for (long long i = 0; i < static_cast<long long>(values.size()); ++i) {
    const std::uint64_t sample_seed = base_seed + static_cast<std::uint64_t>(i);
    values[static_cast<std::size_t>(i)] = to_unit_interval(splitmix64(sample_seed));
  }

  const double seconds = omp_get_wtime() - start_seconds;
  return {std::move(values), seconds};
}
