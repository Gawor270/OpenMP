cmake_minimum_required(VERSION 3.16)
project(OpenMPBucketSort LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP REQUIRED)

add_executable(openmp_bucket_sort
  src/main.cpp
  src/bucketSort.cpp
  src/generator.cpp
)
target_link_libraries(openmp_bucket_sort PRIVATE OpenMP::OpenMP_CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(openmp_bucket_sort PRIVATE -O2 -Wall -Wextra -Wpedantic)
endif()
