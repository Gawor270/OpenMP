#include "bucketSort.h"
#include "generator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <omp.h>

namespace {

struct RunOptions {
  std::size_t vector_size = 0;
  std::uint64_t base_seed = 123456789ull;
  bool test_mode = false;
  int algorithm = 1;
  std::string csv_path;
};

std::string get_env(const char* name) {
  const char* value = std::getenv(name);
  return value ? value : "";
}

std::string current_timestamp() {
  std::time_t now = std::time(nullptr);
  std::tm local_time = *std::localtime(&now);
  char buffer[32];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &local_time);
  return buffer;
}

std::string csv_escape(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 2);
  out.push_back('"');
  for (char ch : value) {
    if (ch == '"') {
      out.push_back('"');
    }
    out.push_back(ch);
  }
  out.push_back('"');
  return out;
}

void write_csv_row(
    const std::string& path,
    int algorithm,
    std::size_t vector_size,
    std::uint64_t base_seed,
    int threads,
    const GeneratedData& generated,
    const BucketSortTimings& timings,
    bool test_mode) {
  const std::filesystem::path csv_path(path);
  if (!csv_path.parent_path().empty()) {
    std::filesystem::create_directories(csv_path.parent_path());
  }

  const bool exists = std::filesystem::exists(csv_path);
  std::ofstream out(path, std::ios::app);
  if (!out) {
    std::cerr << "Failed to open CSV output: " << path << "\n";
    return;
  }

  if (!exists) {
    out << "timestamp,slurm_job_id,slurm_nodelist,slurm_cpus_per_task,algorithm,"
           "vector_size,base_seed,threads,generate_seconds,distribute_seconds,"
           "sort_seconds,rewrite_seconds,total_seconds,test_mode\n";
  }

  out << csv_escape(current_timestamp()) << ','
      << csv_escape(get_env("SLURM_JOB_ID")) << ','
      << csv_escape(get_env("SLURM_NODELIST")) << ','
      << csv_escape(get_env("SLURM_CPUS_PER_TASK")) << ','
      << algorithm << ','
      << vector_size << ','
      << base_seed << ','
      << threads << ','
      << std::fixed << std::setprecision(9)
      << generated.seconds << ','
      << timings.distribute_seconds << ','
      << timings.sort_seconds << ','
      << timings.rewrite_seconds << ','
      << timings.total_seconds << ','
      << (test_mode ? 1 : 0) << "\n";
}

bool parse_args(int argc, char** argv, RunOptions* options) {
  bool vector_set = false;
  bool seed_set = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);

    if (arg == "--test") {
      options->test_mode = true;
      continue;
    }

    if (arg.rfind("--alg=", 0) == 0) {
      options->algorithm = std::stoi(arg.substr(6));
      continue;
    }
    if (arg == "--alg") {
      if (i + 1 >= argc) {
        return false;
      }
      options->algorithm = std::stoi(argv[++i]);
      continue;
    }

    if (arg.rfind("--csv=", 0) == 0) {
      options->csv_path = arg.substr(6);
      continue;
    }
    if (arg == "--csv") {
      if (i + 1 >= argc) {
        return false;
      }
      options->csv_path = argv[++i];
      continue;
    }

    if (!arg.empty() && arg[0] == '-') {
      return false;
    }

    if (!vector_set) {
      options->vector_size = static_cast<std::size_t>(std::stoull(arg));
      vector_set = true;
      continue;
    }

    if (!seed_set) {
      options->base_seed = static_cast<std::uint64_t>(std::stoull(arg));
      seed_set = true;
      continue;
    }

    return false;
  }

  return vector_set && (options->algorithm == 1 || options->algorithm == 2);
}

void print_usage(const char* program) {
  std::cerr << "Usage: " << program
            << " <vector_size> [base_seed] [--alg 1|2] [--csv <path>] [--test]\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<double> copy_for_verification;

  RunOptions options;
  try {
    if (!parse_args(argc, argv, &options)) {
      print_usage(argv[0]);
      return 1;
    }
  } catch (const std::exception&) {
    print_usage(argv[0]);
    return 1;
  }

  const GeneratedData generated = generate_values(options.vector_size, options.base_seed);

  if (options.test_mode) {
    copy_for_verification = generated.values;
  }

  BucketSortTimings timings =
      bucketSort(const_cast<std::vector<double>&>(generated.values), options.algorithm);

  if (options.test_mode) {
    std::sort(copy_for_verification.begin(), copy_for_verification.end());

    if (generated.values != copy_for_verification) {
      std::cout << "[FAIL] Array is not sorted correctly!\n startup params: vector_size="
                << options.vector_size << ", base_seed=" << options.base_seed
                << ", algorithm=" << options.algorithm << "\n";
      return 1;
    }
    std::cout << "[PASS] Array is perfectly sorted!\n";
  }

  if (!options.csv_path.empty()) {
    write_csv_row(
        options.csv_path,
        options.algorithm,
        options.vector_size,
        options.base_seed,
        omp_get_max_threads(),
        generated,
        timings,
        options.test_mode);
  }

  std::cout << "Generated " << generated.values.size() << " random values in [0, 1).\n";
  std::cout << "Threads used (max): " << omp_get_max_threads() << "\n";
  std::cout << "Algorithm: " << options.algorithm << "\n";
  std::cout << "Base seed: " << options.base_seed << "\n";
  std::cout << "Generate time: " << generated.seconds << " s\n";
  std::cout << "Division time: " << timings.distribute_seconds << " s\n";
  std::cout << "Sort time: " << timings.sort_seconds << " s\n";
  std::cout << "Rewrite time: " << timings.rewrite_seconds << " s\n";
  std::cout << "Total algo time: " << timings.total_seconds << " s\n";

  return 0;
}
