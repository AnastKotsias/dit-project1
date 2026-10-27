#ifndef KMEANS_HPP
#define KMEANS_HPP

#include <cstddef>
#include <vector>

#include "matrix.hpp"

namespace image_search {

// output of a k-means run
template <typename T>
struct KMeansResult {
    // one center per row
    Matrix<T> centers;
    // index of the nearest row of centers for every input point
    std::vector<std::size_t> assignments;
    // number of lloyd iterations performed
    int iterations = 0;
    // true if the assignments stopped changing before max_iterations was reached
    bool converged = false;
};

// squared euclidean distance between two arrays of dim elements
template <typename T>
double squared_distance(const T* a, const T* b, std::size_t dim);

// index of the row of centers closest to point, ties go to the lowest index
template <typename T>
std::size_t nearest_center(const T* point, const Matrix<T>& centers);

// clusters the rows of points into the given number of clusters with lloyd's algorithm
// centers are initialized with k-means++ and every random choice is driven by seed
// stops when no assignment changes, or after max_iterations update and assignment steps
template <typename T>
KMeansResult<T> kmeans(const Matrix<T>& points, std::size_t clusters, int seed,
                       int max_iterations = 100);

}  // namespace image_search

#endif
