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

double squaredEuclideanDistance(const Vector& a, const Vector& b) {
    if(a.size() != b.size()) {
        throw std::invalid_argument("search vectors have inconsistent dimensions");
    }
    return squared_distance(a.data(), b.data(), a.size());
}

std::vector<std::size_t> nearestCenters(const Vector& query, const std::vector<Vector>& centers, int count) {
    std::vector<ScoredId> distances;
    distances.reserve(centers.size());
    for(std::size_t i = 0; i < centers.size(); ++i) {
        distances.emplace_back(squaredEuclideanDistance(query, centers[i]), i);
    }
    count = std::max(0, std::min(count, static_cast<int>(distances.size())));
    // pairs compare by distance first, and only the first count positions need to be sorted
    std::partial_sort(distances.begin(), distances.begin() + count, distances.end());
    std::vector<std::size_t> result;
    for(int i = 0; i < count; ++i) {
        result.push_back(distances[i].second);
    }

    return result;
}

Clustering trainIVFClusters(const std::vector<Vector>& points, int requested, int seed) {
    if(points.empty() || requested <= 0)
        return {};
    const std::size_t dim = points.front().size();
    for(const auto& point : points) {
        if(point.size() != dim)
            throw std::invalid_argument("clustered vectors have inconsistent dimensions");
    }

    // the sample is the first ceil(sqrt(n)) entries of a seeded shuffle of the point indices
    const std::size_t sampleSize = ivfTrainingSampleSize(points.size());
    std::vector<std::size_t> order(points.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(order.begin(), order.end(), rng);

    // k-means works on a row-major matrix
    Matrix<double> sample(sampleSize, dim);
    for(std::size_t s = 0; s < sampleSize; ++s)
        std::copy(points[order[s]].begin(), points[order[s]].end(), sample.row(s));

    // k-means cannot form more clusters than it has points
    const std::size_t k = std::min(static_cast<std::size_t>(requested), sampleSize);
    const auto centers = image_search::kmeans(sample, k, seed).centers;

    Clustering clustering;
    clustering.centers.reserve(centers.rows);
    for(std::size_t c = 0; c < centers.rows; ++c) {
        const double* center = centers.row(c);
        clustering.centers.emplace_back(center, center + centers.cols);
    }
    // all n points are assigned, not only the sample
    clustering.assignments.reserve(points.size());
    for(const auto& point : points)
        clustering.assignments.push_back(nearest_center(point.data(), centers));
    return clustering;
}

// the cost depends only on the number of candidates and not on the database size, so the query
// times of the approximate methods have no O(n) term
SearchResult buildResult(const std::vector<Vector>& database, const std::vector<std::string>& ids, const std::vector<std::size_t>& candidates, const Vector& query, const std::string& queryId, const std::string& sequenceId, std::chrono::steady_clock::time_point start) {
    std::vector<ScoredId> ranked;
    ranked.reserve(candidates.size());
    for(const auto index : candidates) {
        assert(index < database.size());
        ranked.emplace_back(squaredEuclideanDistance(query, database[index]), index);
    }

    // positions left unfilled by fewer than 10 candidates count as non-relevant
    const std::size_t limit = std::min<std::size_t>(10, ranked.size());
    std::partial_sort(ranked.begin(), ranked.begin() + limit, ranked.end());
    std::vector<std::size_t> order;
    for(std::size_t i = 0; i < limit; ++i) {
        order.push_back(ranked[i].second);
    }
    // every candidate cost one exact distance computation
    return buildRankedResult(ids, order, ranked.size(), queryId, sequenceId, start);
}

void validateQuery(const std::vector<Vector>& database, const Vector& query) {
    // debug-only checks, the throws below also protect release builds
    assert(!database.empty());
    assert(query.size() == database.front().size());
    if(query.size() != database.front().size())
        throw std::invalid_argument("query vector has inconsistent dimensions");
    // NaN or infinite values make every distance meaningless
    if(!std::all_of(query.begin(), query.end(),
                    [](double value) { return std::isfinite(value); }))
        throw std::invalid_argument("query vector contains non-finite values");
}

std::int64_t windowIndex(double projection, double width) {
    const double window = std::floor(projection / width);
    // converting a double outside the int64 range is undefined behaviour, so it is rejected instead
    // -2^63 is the smallest int64 and 2^63 the first value above the largest, both exact as doubles
    // a NaN fails both comparisons
    constexpr double limit = 9223372036854775808.0;
    if(!(window >= -limit && window < limit))
        throw std::range_error("hash window number does not fit in 64 bits, the width w is too small");
    return static_cast<std::int64_t>(window);
}

int checkedProbes(int probes) {
    if(probes <= 0)
        throw std::invalid_argument("IVF nprobe must be strictly positive");
    return probes;
}

SearchResult buildRankedResult(const std::vector<std::string>& ids, const std::vector<std::size_t>& ranked, std::size_t examined, const std::string& queryId, const std::string& sequenceId, std::chrono::steady_clock::time_point start) {
    // identifiers of the first 10 distinct candidates, in rank order
    std::vector<std::string> top;
    std::unordered_set<std::size_t> seen;
    for(const auto index : ranked) {
        if(top.size() == 10)
            break;
        if(index < ids.size() && seen.insert(index).second)
            top.push_back(ids[index]);
    }
    const int limit = static_cast<int>(top.size());

    // timing stops once the top-10 identifiers are available, the metrics are not part of the query time
    const auto end = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double, std::milli>(
        end - start).count();

    // the relevant images of a query are images 2 to 6 of its own sequence
    std::set<std::string> relevant;
    for(int i = 2; i <= 6; ++i) {
        relevant.insert(sequenceId + "/" + std::to_string(i));
    }

    // relevant hits in the top 5 and top 10, and hits so far for the average precision
    int matches5 = 0;
    int matches10 = 0;
    double apSum = 0.0;
    int found = 0;
    for(int i = 0; i < limit; ++i) {
        // ranks are 1-based, so position i has rank i + 1
        if(relevant.count(top[i]) != 0) {
            if(i < 5) ++matches5;
            ++matches10;
            ++found;
            // precision at this rank, hits so far divided by rank
            apSum += static_cast<double>(found) / static_cast<double>(i + 1);
        }
    }

    // recall and AP both divide by the 5 relevant images
    return {queryId, top, matches5 / 5.0, matches10 / 5.0, apSum / 5.0, elapsed, examined};
}

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
    // identifiers decide relevance, so they cannot be blank
    for(const auto& id : ids) {
        if(id.empty())
            throw std::invalid_argument("database contains an empty identifier");
    }
}

} // namespace detail

std::size_t ivfTrainingSampleSize(std::size_t database_size) {
    return static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<double>(database_size))));
}

} // namespace image_search
