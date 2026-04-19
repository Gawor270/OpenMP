#include "bucketSort.h"

#include <algorithm>
#include <iostream>
#include <omp.h>

BucketSortTimings bucketSort(std::vector<double>& values) {
    BucketSortTimings timings;
    const int numThreads = omp_get_max_threads();
    std::cout << "Number of threads: " << numThreads << std::endl;
    const double threadRange = 1.0 / numThreads;

    std::vector counts(numThreads, 0);
    std::vector offsets(numThreads, 0);

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
        const double rangeMin = tid * threadRange;
        const double rangeMax = (tid == numThreads - 1) ? 1.0 : (tid + 1) * threadRange;

        const int numLocalBuckets = static_cast<int>(values.size()) / numThreads;
        std::vector<std::vector<double>> localBuckets(numLocalBuckets);

        #pragma omp barrier
        #pragma omp single
            t_b_start = omp_get_wtime();

        for (int i = 0; i < static_cast<int>(values.size()); i++) {
            if (double v = values[i]; v >= rangeMin && v < rangeMax) {
                const double localPos = (v - rangeMin) / threadRange;
                const int bucketIdx = std::min(
                    static_cast<int>(localPos * numLocalBuckets),
                    numLocalBuckets - 1);
                localBuckets[bucketIdx].push_back(v);
            }
        }

        #pragma omp barrier
        #pragma omp single
        {
            t_b_end = omp_get_wtime();
            t_c_start = t_b_end;
        }

        for (auto& bucket : localBuckets) {
            std::sort(bucket.begin(), bucket.end());
        }

        #pragma omp barrier
        #pragma omp single
        {
            t_c_end = omp_get_wtime();
            t_d_start = t_c_end;
        }

        int localCount = 0;
        for (auto& bucket : localBuckets) {
            localCount += static_cast<int>(bucket.size());
        }
        counts[tid] = localCount;

        #pragma omp barrier
        #pragma omp single
        {
            offsets[0] = 0;
            for (int i = 1; i < numThreads; i++) {
                offsets[i] = offsets[i - 1] + counts[i - 1];
            }
        }

        int pos = offsets[tid];
        for (const auto& bucket : localBuckets) {
            for (const double v : bucket) {
                values[pos++] = v;
            }
        }

        #pragma omp barrier
        #pragma omp single
            t_d_end = omp_get_wtime();
            t_total_end = t_d_end;
    }

    timings.distribute_seconds = t_b_end - t_b_start;
    timings.sort_seconds = t_c_end - t_c_start;
    timings.rewrite_seconds = t_d_end - t_d_start;
    timings.total_seconds = t_total_end - t_total_start;
    return timings;
}
