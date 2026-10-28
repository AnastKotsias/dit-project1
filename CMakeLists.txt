cmake_minimum_required(VERSION 3.20)
project(image_search LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# default to an optimized build, k-means and BoW construction are far too slow without optimization
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)

    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        add_compile_options(-g -fsanitize=address,undefined)
        add_link_options(-fsanitize=address,undefined)
    endif()
endif()

# OpenCV is used only to read images (imgcodecs) and compute SIFT descriptors (features2d)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs features2d)

add_executable(search
    src/main.cpp
    src/image_features.cpp
    src/vocabulary.cpp
    src/kmeans.cpp
    src/dataset.cpp
    src/pipeline.cpp
    src/search/common.cpp
    src/search/exact.cpp
    src/search/lsh.cpp
    src/search/hypercube.cpp
    src/search/ivfflat.cpp
    src/search/ivfpq.cpp
    src/search/output.cpp
    src/experiment.cpp
)

target_include_directories(search PRIVATE include)
target_link_libraries(search PRIVATE ${OpenCV_LIBS})

# OpenMP is optional and only parallelizes image preprocessing (SIFT extraction and BoW vectors)
# without it the same code runs serially and produces identical results
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(search PRIVATE OpenMP::OpenMP_CXX)
endif()