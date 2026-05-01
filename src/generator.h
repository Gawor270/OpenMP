#ifndef GENERATOR_H
#define GENERATOR_H

#include <cstddef>
#include <cstdint>
#include <vector>

struct GeneratedData {
  std::vector<double> values;
  double seconds = 0.0;
};

GeneratedData generate_values(std::size_t size, std::uint64_t base_seed);

#endif
