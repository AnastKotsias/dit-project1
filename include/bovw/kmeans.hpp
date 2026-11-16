#ifndef KMEANS_HPP
#define KMEANS_HPP

#include <cstddef>
#include <vector>

#include "matrix.hpp"

namespace image_search {

// the functions below are available for T = float (SIFT descriptors) and T = double (BoVW vectors)

template <typename T>
struct KMeansResult {
    // one center per row
    Matrix<T> centers;
    // for every input point, the index of the center it belongs to, always one of the returned centers
    std::vector<std::size_t> assignments;
    // number of Lloyd iterations performed
    int iterations = 0;
    // true if the assignments stopped changing before max_iterations was reached
    bool converged = false;
};

// squared Euclidean distance between two arrays of dim elements
template <typename T>
double squared_distance(const T* a, const T* b, std::size_t dim);

// index of the row of centers closest to point, ties go to the lowest index
// throws if centers is empty
template <typename T>
std::size_t nearest_center(const T* point, const Matrix<T>& centers);

// clusters the rows of points into the given number of clusters with Lloyd's algorithm
// the centers start from k-means++ seeding, and every random choice is driven by seed
// an empty cluster is moved onto the point farthest from its own center
// stops when no assignment changes, or after max_iterations update and assignment steps
// throws if points is empty, clusters is zero or larger than the number of points,
// or max_iterations is negative
template <typename T>
KMeansResult<T> kmeans(const Matrix<T>& points, std::size_t clusters, int seed,
                       int max_iterations = 100);

}  // namespace image_search

#endif
