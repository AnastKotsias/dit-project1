#ifndef SEARCH_COMMON_HPP
#define SEARCH_COMMON_HPP

// internal helpers shared by the search methods in src/search/
// this header is private to the search module and is not part of the public include/ API

#include "search.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace image_search::detail {

// a BoVW vector or any other dense point
using Vector = std::vector<double>;
// (score, database index), sorted ascending so the smallest distance comes first
using ScoredId = std::pair<double, std::size_t>;

// squared Euclidean distance, computed by the k-means distance routine, so every method
// (and the IVFPQ lookup tables) pays the same price per distance
// throws if the vectors have different sizes
double squaredEuclideanDistance(const Vector& a, const Vector& b);

// indices of the count centers nearest to the query, nearest first
// count is capped at the number of centers
std::vector<std::size_t> nearestCenters(const Vector& query, const std::vector<Vector>& centers, int count);

struct Clustering {
    std::vector<Vector> centers;
    // index of the nearest center of every input point
    std::vector<std::size_t> assignments;
};

// trains the coarse clusters of IVFFlat and IVFPQ: k-means runs on a seeded random sample of
// ceil(sqrt(n)) points, then all n points, the sampled ones included, are assigned to their
// nearest center, so no point is left out of the inverted lists
// the number of clusters is capped at the sample size
// returns an empty clustering for no points or a non-positive cluster count
// throws if the points have different dimensions
Clustering trainIVFClusters(const std::vector<Vector>& points, int requested, int seed);

// checks that the query has the database dimension and only finite values, throws otherwise
// every search calls it before generating candidates, so hashing never sees an invalid query
void validateQuery(const std::vector<Vector>& database, const Vector& query);

// window number floor(projection / width) of the LSH and hypercube hash functions
// throws if it does not fit in 64 bits, which a finite query can only cause with an extremely small width
std::int64_t windowIndex(double projection, double width);

// returns nprobe of IVFFlat and IVFPQ, throws unless at least one list is searched
int checkedProbes(int probes);

// finishes a query whose candidates are already in rank order: keeps the first 10 distinct ones,
// stops the timer that started at start and computes the metrics
// examined is the number of distances the index computed to produce the ranking
SearchResult buildRankedResult(const std::vector<std::string>& ids, const std::vector<std::size_t>& ranked, std::size_t examined, const std::string& queryId, const std::string& sequenceId, std::chrono::steady_clock::time_point start);

// result of Exact, LSH, Hypercube and IVFFlat: ranks the candidates by exact distance to the
// query, then calls buildRankedResult
// only the candidates are ranked, an empty candidate set gives an empty result, not an exact search
// candidates must be distinct valid database indices, since buildResult does not deduplicate them,
// only LSH can produce duplicates and removes them itself
SearchResult buildResult(const std::vector<Vector>& database, const std::vector<std::string>& ids, const std::vector<std::size_t>& candidates, const Vector& query, const std::string& queryId, const std::string& sequenceId, std::chrono::steady_clock::time_point start);

// base class of all indexes, which owns the database vectors and validates them
// throws if the database is empty, the sizes of vectors and identifiers differ, the vectors have
// zero or inconsistent dimensions or non-finite values, or an identifier is empty
struct DatabaseIndex {
    // BoVW vector of every database image
    std::vector<Vector> database;
    // identifier of each database image, at the same position as its vector
    std::vector<std::string> ids;

    DatabaseIndex(const std::vector<Vector>& values, const std::vector<std::string>& names);
};

} // namespace image_search::detail

#endif
