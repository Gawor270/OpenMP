#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

struct BenchmarkConfig {
  std::string schedule_type;
  int chunk_size;
};

omp_sched_t parse_schedule_kind(const std::string& schedule_type) {
  if (schedule_type == "static") {
    return omp_sched_static;
  }
  if (schedule_type == "dynamic") {
    return omp_sched_dynamic;
  }
  if (schedule_type == "guided") {
    return omp_sched_guided;
  }
  if (schedule_type == "auto") {
    return omp_sched_auto;
  }

  throw std::runtime_error("schedule_type must be one of: static, dynamic, guided, auto");
}

std::vector<double> generate_values(
    std::size_t vector_size,
    std::uint64_t base_seed,
    const BenchmarkConfig& config,
    int* threads_used = nullptr) {
  omp_set_schedule(parse_schedule_kind(config.schedule_type), config.chunk_size);
  omp_set_dynamic(0);

  std::vector<double> values(vector_size);
  int local_threads_used = 0;

#pragma omp parallel
  {
#pragma omp single
    { local_threads_used = omp_get_num_threads(); }

#pragma omp for schedule(runtime)
    for (long long i = 0; i < static_cast<long long>(values.size()); ++i) {
      const std::uint64_t sample_seed = base_seed + static_cast<std::uint64_t>(i);
      values[static_cast<std::size_t>(i)] = to_unit_interval(splitmix64(sample_seed));
    }
  }

  if (threads_used != nullptr) {
    *threads_used = local_threads_used;
  }

  return values;
}

void dump_vector_to_file(const std::vector<double>& values, const std::string& output_path) {
  const std::filesystem::path path(output_path);
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }

  std::ofstream outfile(output_path);
  if (!outfile) {
    throw std::runtime_error("could not open file: " + output_path);
  }

  outfile << std::fixed << std::setprecision(17);
  for (double value : values) {
    outfile << value << "\n";
  }
}

void run_benchmark(const BenchmarkConfig& config, std::uint64_t base_seed, int repeats) {
  const std::vector<std::size_t> vector_sizes = {
      5'000'000,
      2'0'000'000,
      5'0'000'000,
      1'00'000'000,
      2'00'000'000,
  };

  std::filesystem::create_directories("src/wyniki");
  const std::string result_path =
      "src/wyniki/" + config.schedule_type + "_" + std::to_string(config.chunk_size) +
      "_t_" + std::to_string(omp_get_max_threads()) + ".txt";

  std::ofstream outfile(result_path, std::ios::app);
  if (!outfile) {
    throw std::runtime_error("could not open file: " + result_path);
  }

  for (const std::size_t vector_size : vector_sizes) {
    for (int run = 1; run <= repeats; ++run) {
      int threads_used = 0;

      const double start = omp_get_wtime();
      std::vector<double> values = generate_values(vector_size, base_seed, config, &threads_used);

      const double elapsed = omp_get_wtime() - start;

      double checksum = 0.0;
      const std::size_t step = (values.size() > 1024) ? values.size() / 1024 : 1;
      for (std::size_t i = 0; i < values.size(); i += step) {
        checksum += values[i];
      }

      outfile << std::fixed << std::setprecision(9)
              << "Size: " << values.size() << "\n"
              << "Threads: " << threads_used << "\n"
              << "Schedule: " << config.schedule_type << "\n"
              << "Chunk size: " << config.chunk_size << "\n"
              << "Run: " << run << "\n"
              << "Time: " << elapsed << "\n"
              << "Checksum: " << checksum << "\n"
              << "---\n";
    }
  }
}

void run_dump_mode(
    std::size_t vector_size,
    const std::string& output_path,
    const BenchmarkConfig& config,
    std::uint64_t base_seed) {
  int threads_used = 0;
  std::vector<double> values = generate_values(vector_size, base_seed, config, &threads_used);
  dump_vector_to_file(values, output_path);

  std::cout << "Dumped " << values.size() << " values to " << output_path << "\n"
            << "Schedule: " << config.schedule_type << ", chunk=" << config.chunk_size
            << ", threads=" << threads_used << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  constexpr std::uint64_t kDefaultSeed = 123456789ull;
  constexpr int kDefaultRepeats = 10;

  const std::vector<BenchmarkConfig> kDefaultBatch = {
      {"static", 1},
      {"dynamic", 1},
      {"dynamic", 1000},
      {"guided", 1000},
      {"auto", 1},
  };

  if (argc >= 2 && std::string(argv[1]) == "dump") {
    if (argc < 4 || argc == 6 || argc > 7) {
      std::cerr << "Usage: " << argv[0]
                << " dump <vector_size> <output_file> [base_seed] [schedule_type chunk_size]\n";
      return 1;
    }

    std::size_t vector_size = 0;
    std::string output_path = argv[3];
    std::uint64_t base_seed = kDefaultSeed;
    BenchmarkConfig config{"static", 1};

    try {
      vector_size = static_cast<std::size_t>(std::stoull(argv[2]));

      if (argc >= 5) {
        base_seed = static_cast<std::uint64_t>(std::stoull(argv[4]));
      }

      if (argc == 7) {
        config.schedule_type = argv[5];
        config.chunk_size = std::stoi(argv[6]);
      }
    } catch (const std::exception&) {
      std::cerr << "Error: invalid numeric argument(s).\n";
      return 1;
    }

    if (vector_size == 0) {
      std::cerr << "Error: vector_size must be > 0.\n";
      return 1;
    }

    try {
      run_dump_mode(vector_size, output_path, config, base_seed);
    } catch (const std::exception& error) {
      std::cerr << "Error: " << error.what() << "\n";
      return 1;
    }

    return 0;
  }

  if (argc != 1 && (argc < 3 || argc > 5)) {
    std::cerr << "Usage: " << argv[0] << " [<schedule_type> <chunk_size> [base_seed] [repeats]]\n";
    return 1;
  }

  std::uint64_t base_seed = kDefaultSeed;
  int repeats = kDefaultRepeats;

  try {
    if (argc >= 4) {
      base_seed = static_cast<std::uint64_t>(std::stoull(argv[3]));
    }
    if (argc == 5) {
      repeats = std::stoi(argv[4]);
    }
  } catch (const std::exception&) {
    std::cerr << "Error: invalid numeric argument(s).\n";
    return 1;
  }

  if (repeats < 1) {
    std::cerr << "Error: repeats must be >= 1.\n";
    return 1;
  }

  try {
    if (argc == 1) {
      for (const BenchmarkConfig& config : kDefaultBatch) {
        run_benchmark(config, base_seed, repeats);
      }
    } else {
      const BenchmarkConfig config{argv[1], std::stoi(argv[2])};
      run_benchmark(config, base_seed, repeats);
    }
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }

  return 0;
}