#ifndef SEARCH_COMMON_HPP
#define SEARCH_COMMON_HPP

// internal helpers shared by the search methods in src/search/
// this header is private to the search module and is not part of the public include/ API

#include "search.hpp"

#include <chrono>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace image_search::detail {

// a BoW vector or any other dense point
using Vector = std::vector<double>;
// pair of (score, database index), sorted ascending so the smallest distance comes first
using ScoredId = std::pair<double, std::size_t>;

// Calculates the squared Euclidean distance between two vectors
double squaredEuclideanDistance(const Vector& a, const Vector& b);

// Finds the "count" nearest centers to a given query vector
std::vector<std::size_t> nearestCenters(const Vector& query, const std::vector<Vector>& centers, int count);

// simple lloyd k-means with random initial centers, used for the IVF coarse clusters and PQ codebooks
std::vector<Vector> kmeans(const std::vector<Vector>& points, int requested, int seed, int iterations = 8);

// Common result-building function used by Exact, LSH, Hypercube, IVFFlat, and IVFPQ search
SearchResult buildResult(const std::vector<Vector>& database, const std::vector<std::string>& ids, const std::vector<std::size_t>& candidates, const Vector& query, const std::string& queryId, const std::string& sequenceId, std::chrono::high_resolution_clock::time_point start);

// base class shared by all indexes that owns the database vectors and validates them
struct DatabaseIndex {
    // BoW vector of every database image
    std::vector<Vector> database;
    // identifier of each database image, at the same position as its vector
    std::vector<std::string> ids;

    // Constructor to initialize the database index with vectors and their corresponding identifiers
    DatabaseIndex(const std::vector<Vector>& values, const std::vector<std::string>& names);
};

} // namespace image_search::detail

#endif
