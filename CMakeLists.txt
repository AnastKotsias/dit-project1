cmake_minimum_required(VERSION 3.20)
project(image_search LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(CTest)
option(BUILD_SEARCH_APP "Build the image search executable" ON)

# k-means and BoVW construction are far too slow without optimization
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Eigen only computes the SVD of the DLT matrix; DLT, matching and RANSAC are project code
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# everything except the program entry points, shared by search, verify and the tests
set(IMAGE_SEARCH_SOURCES
    src/app/command_line.cpp
    src/app/search_setup.cpp
    src/data/dataset.cpp
    src/data/pipeline.cpp
    src/bovw/image_features.cpp
    src/bovw/kmeans.cpp
    src/bovw/vocabulary.cpp
    src/search/common.cpp
    src/search/exact.cpp
    src/search/lsh.cpp
    src/search/hypercube.cpp
    src/search/ivfflat.cpp
    src/search/ivfpq.cpp
    src/search/output.cpp
    src/geometry/local_features.cpp
    src/geometry/descriptor_matching.cpp
    src/geometry/homography.cpp
    src/geometry/geometric_verification.cpp
    src/evaluation/geometric_evaluation.cpp
    src/evaluation/verification_output.cpp
)

if(BUILD_SEARCH_APP)
    # OpenCV only reads images (imgcodecs) and computes SIFT descriptors (features2d)
    find_package(OpenCV REQUIRED COMPONENTS core imgcodecs features2d)

    add_executable(search src/main.cpp src/app/experiment.cpp ${IMAGE_SEARCH_SOURCES})

    # assignment 2: TopC candidates of the same index, SIFT matching, RANSAC and reranking
    add_executable(verify src/verify_main.cpp ${IMAGE_SEARCH_SOURCES})

    # optional: only parallelizes image preprocessing, results are identical without it
    find_package(OpenMP)
    foreach(app search verify)
        target_include_directories(${app} PRIVATE include)
        target_link_libraries(${app} PRIVATE ${OpenCV_LIBS} Eigen3::Eigen)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(${app} PRIVATE OpenMP::OpenMP_CXX)
        endif()
    endforeach()
endif()

if(BUILD_TESTING)
    find_package(OpenCV REQUIRED COMPONENTS core imgcodecs features2d)

    add_executable(image_search_tests tests/unit_tests.cpp ${IMAGE_SEARCH_SOURCES})

    target_include_directories(image_search_tests PRIVATE include src/search)
    target_link_libraries(image_search_tests PRIVATE ${OpenCV_LIBS} Eigen3::Eigen)
    add_test(NAME image_search_unit_tests COMMAND image_search_tests)
endif()
