#ifndef BUCKET_SORT_H
#define BUCKET_SORT_H

#include <vector>

struct BucketSortTimings {
  double distribute_seconds = 0.0;
  double sort_seconds = 0.0;
  double rewrite_seconds = 0.0;
  double total_seconds = 0.0;
};

BucketSortTimings bucketSort(std::vector<double>& values);

#endif
