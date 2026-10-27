#include "common.hpp"

#include "kmeans.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <set>
#include <stdexcept>
#include <unordered_set>

namespace image_search {
namespace detail {

// Calculates the squared Euclidean distance between two vectors
double squaredEuclideanDistance(const Vector& a, const Vector& b) {
    if(a.size() != b.size()) {
        throw std::invalid_argument("search vectors have inconsistent dimensions");
    };
    double result = 0.0;
    for(std::size_t i = 0; i < a.size(); ++i) {
        const double difference = a[i] - b[i];
        result += difference * difference;
    };
    return result;
}

// Finds the "count" nearest centers to a given query vector
std::vector<std::size_t> nearestCenters(const Vector& query, const std::vector<Vector>& centers, int count) {
    // distance of the query to every center, paired with the center's index
    std::vector<ScoredId> distances;
    distances.reserve(centers.size());
    for(std::size_t i = 0; i < centers.size(); ++i) {
        distances.emplace_back(squaredEuclideanDistance(query, centers[i]), i);
    };
    // Ensure we don't exceed the number of available centers
    count = std::max(0, std::min(count, static_cast<int>(distances.size())));
    // pairs sort by distance first, so the nearest centers come first
    // only the first count positions are needed, so the rest is left unsorted
    std::partial_sort(distances.begin(), distances.begin() + count, distances.end());
    std::vector<std::size_t> result;

    // Collect the nearest centers
    for(int i = 0; i < count; ++i) {
        result.push_back(distances[i].second);
    };

    return result;
}

// runs the shared k-means on the points, converting them to and from the row-major Matrix it works on
Clustering clusterVectors(const std::vector<Vector>& points, int requested, int seed) {
    // nothing to cluster
    if(points.empty() || requested <= 0)
        return {};

    // copy the points into one contiguous matrix
    Matrix<double> matrix(points.size(), points.front().size());
    for(std::size_t i = 0; i < points.size(); ++i) {
        if(points[i].size() != matrix.cols)
            throw std::invalid_argument("clustered vectors have inconsistent dimensions");
        std::copy(points[i].begin(), points[i].end(), matrix.row(i));
    }

    // Ensure we don't request more clusters than available points
    const std::size_t k = std::min(static_cast<std::size_t>(requested), points.size());
    auto result = image_search::kmeans(matrix, k, seed);

    // copy the centers back into one vector per center
    Clustering clustering;
    clustering.centers.reserve(result.centers.rows);
    for(std::size_t c = 0; c < result.centers.rows; ++c) {
        const double* center = result.centers.row(c);
        clustering.centers.emplace_back(center, center + result.centers.cols);
    }
    clustering.assignments = std::move(result.assignments);
    return clustering;
}

// Common result-building function used by Exact, LSH, Hypercube, IVFFlat, and IVFPQ search
SearchResult buildResult(const std::vector<Vector>& database, const std::vector<std::string>& ids, const std::vector<std::size_t>& candidates, const Vector& query, const std::string& queryId, const std::string& sequenceId, std::chrono::high_resolution_clock::time_point start) {
    validateQuery(database, query);

    // candidates scored by their distance to the query
    std::vector<ScoredId> ranked;
    // Use a set to ensure unique candidates and avoid duplicates
    std::unordered_set<std::size_t> unique;
    // Rank only candidates returned by the index. Approximate methods must not
    // silently turn an empty candidate set into an exact search.
    for(const auto index : candidates) {
        if(index < database.size() && unique.insert(index).second) {
            ranked.emplace_back(squaredEuclideanDistance(query, database[index]), index);
        }
    }

    // Select the top 10 among the candidates actually returned by the index.
    // Missing positions are non-relevant by definition.
    const std::size_t limit = std::min<std::size_t>(10, ranked.size());
    std::partial_sort(ranked.begin(), ranked.begin() + limit, ranked.end());
    std::vector<std::size_t> order;
    for(std::size_t i = 0; i < limit; ++i) {
        order.push_back(ranked[i].second);
    }
    return buildRankedResult(ids, order, queryId, sequenceId, start);
}

// Checks that a query can be compared with the database vectors
void validateQuery(const std::vector<Vector>& database, const Vector& query) {
    // debug-only sanity checks, the throws below also protect release builds
    assert(!database.empty());
    assert(query.size() == database.front().size());
    // the query must have the same dimension K as the database vectors
    if(query.size() != database.front().size())
        throw std::invalid_argument("query vector has inconsistent dimensions");
    // NaN or infinite values would make every distance meaningless
    if(!std::all_of(query.begin(), query.end(),
                    [](double value) { return std::isfinite(value); }))
        throw std::invalid_argument("query vector contains non-finite values");
}

// Finishes a query whose candidates are already in rank order and computes its metrics
SearchResult buildRankedResult(const std::vector<std::string>& ids, const std::vector<std::size_t>& ranked, const std::string& queryId, const std::string& sequenceId, std::chrono::high_resolution_clock::time_point start) {
    // identifiers of the first 10 distinct valid candidates, in rank order
    std::vector<std::string> top;
    std::unordered_set<std::size_t> seen;
    for(const auto index : ranked) {
        if(top.size() == 10)
            break;
        if(index < ids.size() && seen.insert(index).second)
            top.push_back(ids[index]);
    }
    const int limit = static_cast<int>(top.size());

    // Stop timing once the ranked top-10 result identifiers are available.
    // Ground-truth construction and metric calculation are intentionally
    // outside the measured query interval.
    const auto end = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double, std::milli>(
        end - start).count();

    // Build the set of relevant images for the given sequence
    std::set<std::string> relevant;
    for(int i = 2; i <= 6; ++i) {
        relevant.insert(sequenceId + "/" + std::to_string(i));
    }

    // matches5 and matches10 count relevant hits in the top 5 and top 10, found counts them so far for AP
    int matches5 = 0;
    int matches10 = 0;
    double apSum = 0.0;
    int found = 0;
    for(int i = 0; i < limit; ++i) {
        // a hit is an image of the query's own sequence, ranks are 1-based so position i has rank i + 1
        if(relevant.count(top[i]) != 0) {
            if(i < 5) ++matches5;
            ++matches10;
            ++found;
            // precision at this rank, namely hits so far divided by rank
            apSum += static_cast<double>(found) / static_cast<double>(i + 1);
        }
    }

    // recall divides by the 5 relevant images, and AP also divides by 5
    return {queryId, top, matches5 / 5.0, matches10 / 5.0, apSum / 5.0, elapsed};
}

// Constructor to initialize the database index with vectors and their corresponding identifiers
DatabaseIndex::DatabaseIndex(const std::vector<Vector>& values, const std::vector<std::string>& names)
            : database(values), ids(names) {
    if(database.empty())
        throw std::invalid_argument("search database is empty");
    if(database.size() != ids.size())
        throw std::invalid_argument("database vectors and identifiers have different sizes");
    if(database.front().empty())
        throw std::invalid_argument("search vectors have zero dimensions");
    for(const auto& vector : database) {
        if(vector.size() != database.front().size())
            throw std::invalid_argument("database vectors have inconsistent dimensions");
        if(!std::all_of(vector.begin(), vector.end(),
                        [](double value) { return std::isfinite(value); }))
            throw std::invalid_argument("database contains non-finite values");
    }
    // identifiers are used to judge relevance, so they cannot be blank
    for(const auto& id : ids) {
        if(id.empty())
            throw std::invalid_argument("database contains an empty identifier");
    }
}

} // namespace detail

// public wrapper that takes the square root of the internal squared distance
double computeEuclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
    return std::sqrt(detail::squaredEuclideanDistance(a, b));
}

} // namespace image_search
