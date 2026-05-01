#include "bucketSort.h"

#include <algorithm>
#include <omp.h>

BucketSortTimings bucketSort(std::vector<double>& values) {
    BucketSortTimings timings;
    const int numThreads = omp_get_max_threads();
    const int numBuckets =  numThreads*4;
    std::vector<omp_lock_t> bucket_locks(numBuckets);
    for (int i = 0; i < numBuckets; ++i) {
        omp_init_lock(&bucket_locks[i]);
    }
    std::vector<std::vector<double>> buckets(numBuckets);

    double t_b_start = 0.0;
    double t_b_end = 0.0;
    double t_c_start = 0.0;
    double t_c_end = 0.0;
    double t_d_start = 0.0;
    double t_d_end = 0.0;
    double t_total_start = 0.0;

    #pragma omp parallel
    {
        #pragma omp single
        {
            t_total_start = omp_get_wtime();
        }

        int tid = omp_get_thread_num();
        int start_index = tid * values.size() / numThreads;
        int end_index = (tid + 1) * values.size() / numThreads;

        #pragma omp barrier
        #pragma omp single
        {
            t_b_start = omp_get_wtime();
        }

        for (int i = start_index; i < end_index; ++i) {
            int bucket_index = std::min(static_cast<int>(values[i] * numBuckets), numBuckets - 1);
            omp_set_lock(&bucket_locks[bucket_index]);
            buckets[bucket_index].push_back(values[i]);
            omp_unset_lock(&bucket_locks[bucket_index]);
        }

        #pragma omp barrier
        #pragma omp single
        {
            t_b_end = omp_get_wtime();
        }
    }

    for (int i = 0; i < numBuckets; ++i) {
        omp_destroy_lock(&bucket_locks[i]);
    }

    timings.distribute_seconds = t_b_end - t_b_start;
	return timings;
}
