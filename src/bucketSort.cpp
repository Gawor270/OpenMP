#include "bucketSort.h"

#include <algorithm>
#include <omp.h>

BucketSortTimings bucketSort(std::vector<double>& values) {
	BucketSortTimings timings;
	const double total_start = omp_get_wtime();

	const double sort_start = total_start;
	std::sort(values.begin(), values.end());
	const double sort_end = omp_get_wtime();

	timings.sort_seconds = sort_end - sort_start;
	timings.total_seconds = sort_end - total_start;
	return timings;
}
