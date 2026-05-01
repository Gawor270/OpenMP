#include "bucketSort.h"

#include "bucketSortAlg1.h"
#include "bucketSortAlg2.h"

BucketSortTimings bucketSort(std::vector<double>& values, int algorithm) {
  if (algorithm == 2) {
    return bucketSortAlg2(values);
  }
  return bucketSortAlg1(values);
}
