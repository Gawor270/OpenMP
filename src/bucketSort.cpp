#include "bucketSort.h"

#include <algorithm>
#include <cstddef>
#include <vector>

#include <omp.h>

namespace {

BucketSortTimings bucketSortAlg1(std::vector<double>& values) {
	BucketSortTimings timings;
	const int numThreads = omp_get_max_threads();
	const double threadRange = 1.0 / static_cast<double>(numThreads);

	std::vector<std::size_t> counts(static_cast<std::size_t>(numThreads), 0);
	std::vector<std::size_t> offsets(static_cast<std::size_t>(numThreads), 0);

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
		const double rangeMin = static_cast<double>(tid) * threadRange;
		const double rangeMax = (tid == numThreads - 1) ? 1.0 : (tid + 1) * threadRange;

		const std::size_t numLocalBuckets =
				std::max<std::size_t>(1, values.size() / static_cast<std::size_t>(numThreads));
		std::vector<std::vector<double>> localBuckets(numLocalBuckets);

#pragma omp barrier
#pragma omp single
		t_b_start = omp_get_wtime();

		for (std::size_t i = 0; i < values.size(); ++i) {
			const double v = values[i];
			if (v >= rangeMin && v < rangeMax) {
				const double localPos = (v - rangeMin) / threadRange;
				const std::size_t bucketIdx = std::min<std::size_t>(
						static_cast<std::size_t>(localPos * static_cast<double>(numLocalBuckets)),
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

		std::size_t localCount = 0;
		for (const auto& bucket : localBuckets) {
			localCount += bucket.size();
		}
		counts[static_cast<std::size_t>(tid)] = localCount;

#pragma omp barrier
#pragma omp single
		{
			offsets[0] = 0;
			for (int i = 1; i < numThreads; ++i) {
				offsets[static_cast<std::size_t>(i)] =
						offsets[static_cast<std::size_t>(i - 1)] + counts[static_cast<std::size_t>(i - 1)];
			}
		}

		std::size_t pos = offsets[static_cast<std::size_t>(tid)];
		for (const auto& bucket : localBuckets) {
			for (double v : bucket) {
				values[pos++] = v;
			}
		}

#pragma omp barrier
#pragma omp single
		{
			t_d_end = omp_get_wtime();
			t_total_end = t_d_end;
		}
	}

	timings.distribute_seconds = t_b_end - t_b_start;
	timings.sort_seconds = t_c_end - t_c_start;
	timings.rewrite_seconds = t_d_end - t_d_start;
	timings.total_seconds = t_total_end - t_total_start;
	return timings;
}

BucketSortTimings bucketSortAlg2(std::vector<double>& values) {
	BucketSortTimings timings;
	const int numThreads = omp_get_max_threads();
	const int numBuckets = numThreads * 4;
	std::vector<omp_lock_t> bucket_locks(numBuckets);
	for (int i = 0; i < numBuckets; ++i) {
		omp_init_lock(&bucket_locks[i]);
	}
	std::vector<std::vector<double>> buckets(numBuckets);
	std::vector<std::size_t> offsets(numBuckets, 0);

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
		t_d_end = omp_get_wtime();
	}

	for (int i = 0; i < numBuckets; ++i) {
		omp_destroy_lock(&bucket_locks[i]);
	}

	timings.distribute_seconds = t_b_end - t_b_start;
	timings.sort_seconds = t_c_end - t_c_start;
	timings.rewrite_seconds = t_d_end - t_d_start;
	timings.total_seconds = t_d_end - t_total_start;
	return timings;
}

}  // namespace

BucketSortTimings bucketSort(std::vector<double>& values, int algorithm) {
	if (algorithm == 2) {
		return bucketSortAlg2(values);
	}
	return bucketSortAlg1(values);
}

BucketSortTimings bucketSort(std::vector<double>& values) {
	return bucketSort(values, 1);
}
