#include "bucketSort.h"
#include "generator.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <algorithm>

#include <omp.h>

int main(int argc, char** argv) {
  constexpr std::uint64_t kDefaultSeed = 123456789ull;
  std::vector<double> copy_for_verification;

  if (argc < 2 || argc > 4) {
    std::cerr << "Usage: " << argv[0] << " <vector_size> [base_seed] [--test]\n";
    return 1;
  }

  const std::size_t vector_size = static_cast<std::size_t>(std::stoull(argv[1]));
  const std::uint64_t base_seed = (argc == 3)
                                      ? static_cast<std::uint64_t>(std::stoull(argv[2]))
                                      : kDefaultSeed;
  const bool test_mode = (argc == 4) && (std::string(argv[3]) == "--test");

  const GeneratedData generated = generate_values(vector_size, base_seed);

  if (test_mode) {
      copy_for_verification = generated.values;
  }

  BucketSortTimings timings = bucketSort(const_cast<std::vector<double>&>(generated.values));

  if (test_mode) {
    std::sort(copy_for_verification.begin(), copy_for_verification.end());
    
    if (generated.values != copy_for_verification) {
      std::cerr << "Error: Values are not sorted correctly or data was corrupted.\n";
      return 1;
    }
    std::cout << "[PASS] Array is perfectly sorted!\n";
  }

  std::cout << "Generated " << generated.values.size() << " random values in [0, 1).\n";
  std::cout << "Threads used (max): " << omp_get_max_threads() << "\n";
  std::cout << "Base seed: " << base_seed << "\n";
  std::cout << "Generate time: " << generated.seconds << " s\n";
  std::cout << "Division time: " << timings.distribute_seconds << " s\n";
  std::cout << "Sort time: " << timings.sort_seconds << " s\n";
  std::cout << "Rewrite time: " << timings.rewrite_seconds << " s\n";
  std::cout << "Total algo time: " << timings.total_seconds << " s\n";

  return 0;
}
