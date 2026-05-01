#include "generator.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

void fail(const char* message) {
  std::cerr << "[FAIL] " << message << "\n";
  std::exit(1);
}

}  // namespace

int main() {
  constexpr std::size_t kSampleSize = 200000;
  constexpr std::uint64_t kBaseSeed = 123456789ull;
  constexpr std::size_t kBins = 50;
  constexpr double kExpectedMean = 0.5;
  constexpr double kExpectedVariance = 1.0 / 12.0;
  constexpr double kMeanTolerance = 0.005;
  constexpr double kVarianceTolerance = 0.005;
  constexpr double kSigmaMultiplier = 8.0;

  const GeneratedData generated = generate_values(kSampleSize, kBaseSeed);
  const std::vector<double>& values = generated.values;

  if (values.size() != kSampleSize) {
    fail("Unexpected sample size");
  }

  double min_value = 1.0;
  double max_value = 0.0;
  double sum = 0.0;
  double sum_sq = 0.0;
  std::vector<std::size_t> bins(kBins, 0);

  for (double value : values) {
    if (value < 0.0 || value >= 1.0) {
      fail("Value outside [0, 1) interval");
    }

    if (value < min_value) {
      min_value = value;
    }
    if (value > max_value) {
      max_value = value;
    }

    sum += value;
    sum_sq += value * value;

    const std::size_t bin = static_cast<std::size_t>(value * kBins);
    const std::size_t clamped = (bin >= kBins) ? (kBins - 1) : bin;
    ++bins[clamped];
  }

  const double mean = sum / static_cast<double>(values.size());
  const double variance = (sum_sq / static_cast<double>(values.size())) - (mean * mean);

  if (std::fabs(mean - kExpectedMean) > kMeanTolerance) {
    fail("Mean deviates from expected 0.5");
  }

  if (std::fabs(variance - kExpectedVariance) > kVarianceTolerance) {
    fail("Variance deviates from expected 1/12");
  }

  const double expected = static_cast<double>(values.size()) / static_cast<double>(kBins);
  const double p = 1.0 / static_cast<double>(kBins);
  const double stddev = std::sqrt(static_cast<double>(values.size()) * p * (1.0 - p));
  const double max_allowed_deviation = kSigmaMultiplier * stddev;

  for (std::size_t count : bins) {
    const double deviation = std::fabs(static_cast<double>(count) - expected);
    if (deviation > max_allowed_deviation) {
      fail("Histogram bin deviates too much from expected count");
    }
  }

  std::cout << "[PASS] Generator uniformity checks passed."
            << " Min=" << min_value << " Max=" << max_value << "\n";
  return 0;
}
