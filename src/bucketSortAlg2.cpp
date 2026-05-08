#include "bucketSortAlg2.h"

#include <algorithm>
#include <cstddef>
#include <vector>

#include <omp.h>

BucketSortTimings bucketSortAlg2(std::vector<double>& values, int bucket_mult) {
  BucketSortTimings timings;
  const int numThreads = omp_get_max_threads();
  const int numBuckets = numThreads * bucket_mult;
  std::vector<omp_lock_t> bucket_locks(numBuckets);
  for (int i = 0; i < numBuckets; ++i) {
    omp_init_lock(&bucket_locks[i]);
  }
  std::vector<std::vector<double>> buckets(numBuckets);
  const std::size_t expected = values.size() / static_cast<std::size_t>(numBuckets);
  for (int i = 0; i < numBuckets; ++i) {
    buckets[i].reserve(expected * 2 + 1);
  }
  std::vector<std::size_t> offsets(numBuckets, 0);

  double t_b_start = 0.0;
  double t_b_end = 0.0;
  double t_c_start = 0.0;
  double t_c_end = 0.0;
  double t_d_start = 0.0;
  double t_d_end = 0.0;
  double t_total_start = 0.0;
  double t_total_end = 0.0;

#pragma omp parallel
  {
#pragma omp single
    t_total_start = omp_get_wtime();

    const int tid = omp_get_thread_num();
    const std::size_t start_index = static_cast<std::size_t>(tid) * values.size() /
                                    static_cast<std::size_t>(numThreads);
    const std::size_t end_index = static_cast<std::size_t>(tid + 1) * values.size() /
                                  static_cast<std::size_t>(numThreads);

#pragma omp barrier
#pragma omp single
    t_b_start = omp_get_wtime();

    for (std::size_t i = start_index; i < end_index; ++i) {
      const int bucket_index =
          std::min(static_cast<int>(values[i] * numBuckets), numBuckets - 1);
      omp_set_lock(&bucket_locks[bucket_index]);
      buckets[bucket_index].push_back(values[i]);
      omp_unset_lock(&bucket_locks[bucket_index]);
    }

#pragma omp barrier
#pragma omp single
    {
      t_b_end = omp_get_wtime();
      t_c_start = t_b_end;
    }

#pragma omp for schedule(static)
    for (int i = 0; i < numBuckets; ++i) {
      std::sort(buckets[i].begin(), buckets[i].end());
    }

#pragma omp barrier
#pragma omp single
    {
      t_c_end = omp_get_wtime();
      t_d_start = t_c_end;
      offsets[0] = 0;
      for (int i = 1; i < numBuckets; ++i) {
        offsets[static_cast<std::size_t>(i)] =
            offsets[static_cast<std::size_t>(i - 1)] + buckets[static_cast<std::size_t>(i - 1)].size();
      }
    }

#pragma omp for schedule(static)
    for (int i = 0; i < numBuckets; ++i) {
      const std::size_t offset = offsets[static_cast<std::size_t>(i)];
      for (std::size_t j = 0; j < buckets[static_cast<std::size_t>(i)].size(); ++j) {
        values[offset + j] = buckets[static_cast<std::size_t>(i)][j];
      }
    }

#pragma omp barrier
#pragma omp single
    {
      t_d_end = omp_get_wtime();
      t_total_end = t_d_end;
    }
  }

  for (int i = 0; i < numBuckets; ++i) {
    omp_destroy_lock(&bucket_locks[i]);
  }

  timings.distribute_seconds = t_b_end - t_b_start;
  timings.sort_seconds = t_c_end - t_c_start;
  timings.rewrite_seconds = t_d_end - t_d_start;
  timings.total_seconds = t_total_end - t_total_start;
  return timings;
}
