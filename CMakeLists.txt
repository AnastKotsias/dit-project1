cmake_minimum_required(VERSION 3.20)
project(image_search LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# k-means and BoVW construction are far too slow without optimization
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# OpenCV only reads images (imgcodecs) and computes SIFT descriptors (features2d)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs features2d)

add_executable(search
    src/main.cpp
    src/dataset.cpp
    src/image_features.cpp
    src/kmeans.cpp
    src/vocabulary.cpp
    src/pipeline.cpp
    src/experiment.cpp
    src/search/common.cpp
    src/search/exact.cpp
    src/search/lsh.cpp
    src/search/hypercube.cpp
    src/search/ivfflat.cpp
    src/search/ivfpq.cpp
    src/search/output.cpp
)

target_include_directories(search PRIVATE include)
target_link_libraries(search PRIVATE ${OpenCV_LIBS})

# optional: only parallelizes image preprocessing, results are identical without it
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(search PRIVATE OpenMP::OpenMP_CXX)
endif()
