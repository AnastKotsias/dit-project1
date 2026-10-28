#include "common.hpp"

#include "kmeans.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <unordered_set>

namespace image_search {
namespace detail {

// Calculates the squared Euclidean distance between two vectors
// uses the same four-accumulator loop as k-means and the IVFPQ lookup tables,
// so every method pays the same price per distance
double squaredEuclideanDistance(const Vector& a, const Vector& b) {
    if(a.size() != b.size()) {
        throw std::invalid_argument("search vectors have inconsistent dimensions");
    };
    return squared_distance(a.data(), b.data(), a.size());
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

// IVF build step of the lecture: runs the shared k-means on a random sample X' of ceil(sqrt(n)) points,
// then assigns every point, the sampled ones included, to its nearest center, so no point is left out of the lists
Clustering trainIVFClusters(const std::vector<Vector>& points, int requested, int seed) {
    // nothing to cluster
    if(points.empty() || requested <= 0)
        return {};
    const std::size_t dim = points.front().size();
    for(const auto& point : points) {
        if(point.size() != dim)
            throw std::invalid_argument("clustered vectors have inconsistent dimensions");
    }

    // draw the sample with the seeded generator, as the first ceil(sqrt(n)) entries of a shuffled index list
    const std::size_t sampleSize = ivfTrainingSampleSize(points.size());
    std::vector<std::size_t> order(points.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(order.begin(), order.end(), rng);

    // copy the sampled points into one contiguous matrix, the row-major form k-means works on
    Matrix<double> sample(sampleSize, dim);
    for(std::size_t s = 0; s < sampleSize; ++s)
        std::copy(points[order[s]].begin(), points[order[s]].end(), sample.row(s));

    // Ensure we don't request more clusters than sampled points
    const std::size_t k = std::min(static_cast<std::size_t>(requested), sampleSize);
    const auto centers = image_search::kmeans(sample, k, seed).centers;

    // copy the centers back into one vector per center
    Clustering clustering;
    clustering.centers.reserve(centers.rows);
    for(std::size_t c = 0; c < centers.rows; ++c) {
        const double* center = centers.row(c);
        clustering.centers.emplace_back(center, center + centers.cols);
    }
    // assign all n points, not only the sample, to their nearest center
    clustering.assignments.reserve(points.size());
    for(const auto& point : points)
        clustering.assignments.push_back(nearest_center(point.data(), centers));
    return clustering;
}

// Common result-building function used by Exact, LSH, Hypercube, IVFFlat, and IVFPQ search
SearchResult buildResult(const std::vector<Vector>& database, const std::vector<std::string>& ids, const std::vector<std::size_t>& candidates, const Vector& query, const std::string& queryId, const std::string& sequenceId, std::chrono::steady_clock::time_point start) {
    validateQuery(database, query);

    // candidates scored by their distance to the query
    std::vector<ScoredId> ranked;
    ranked.reserve(candidates.size());
    // seen[i] marks database vector i as already scored, so duplicates are skipped
    // a flag array avoids the allocation and hashing a set would cost per candidate
    std::vector<char> seen(database.size(), 0);
    // Rank only candidates returned by the index. Approximate methods must not
    // silently turn an empty candidate set into an exact search.
    for(const auto index : candidates) {
        if(index < database.size() && !seen[index]) {
            seen[index] = 1;
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
    // every distinct valid candidate cost one exact distance computation
    return buildRankedResult(ids, order, ranked.size(), queryId, sequenceId, start);
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
SearchResult buildRankedResult(const std::vector<std::string>& ids, const std::vector<std::size_t>& ranked, std::size_t examined, const std::string& queryId, const std::string& sequenceId, std::chrono::steady_clock::time_point start) {
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
    const auto end = std::chrono::steady_clock::now();
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
    return {queryId, top, matches5 / 5.0, matches10 / 5.0, apSum / 5.0, elapsed, examined};
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

// size of the random sample X' of the lecture's IVF build step
std::size_t ivfTrainingSampleSize(std::size_t database_size) {
    return static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<double>(database_size))));
}

// public wrapper that takes the square root of the internal squared distance
double computeEuclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
    return std::sqrt(detail::squaredEuclideanDistance(a, b));
}

} // namespace image_search
