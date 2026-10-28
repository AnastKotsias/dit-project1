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

// centers and per-point assignments of a clustering of BoW vectors
struct Clustering {
    std::vector<Vector> centers;
    // index of the nearest center of every input point
    std::vector<std::size_t> assignments;
};

// trains the IVF coarse clusters of IVFFlat and IVFPQ: the shared k-means of kmeans.hpp runs on a seeded
// random sample of ceil(sqrt(n)) points, then all n points are assigned to their nearest center
// the number of clusters is capped at the sample size
Clustering trainIVFClusters(const std::vector<Vector>& points, int requested, int seed);

// Checks that the query has the database dimension and only finite values, throws otherwise
void validateQuery(const std::vector<Vector>& database, const Vector& query);

// Finishes a query whose candidates are already in rank order: keeps the first 10 distinct ones,
// stops the timer that started at start and computes the metrics
// examined is the number of distances the index computed to produce the ranking
SearchResult buildRankedResult(const std::vector<std::string>& ids, const std::vector<std::size_t>& ranked, std::size_t examined, const std::string& queryId, const std::string& sequenceId, std::chrono::steady_clock::time_point start);

// Common result-building function used by Exact, LSH, Hypercube and IVFFlat search
// ranks the candidates by exact euclidean distance to the query, then calls buildRankedResult
SearchResult buildResult(const std::vector<Vector>& database, const std::vector<std::string>& ids, const std::vector<std::size_t>& candidates, const Vector& query, const std::string& queryId, const std::string& sequenceId, std::chrono::steady_clock::time_point start);

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
