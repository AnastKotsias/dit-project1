#include "common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>

namespace image_search {
using namespace detail;

// hypercube index: each of the d' projections h_i(p) = floor((p.v + t) / w) contributes one bit, so a
// point maps to a vertex of the d'-dimensional cube, which is its bucket
struct HypercubeSearch::Impl : DatabaseIndex {
    // dimension d' of the cube
    int projections;
    // candidate budget M
    int maxCandidates;
    // maximum number of vertices visited per query
    int probes;
    // window width w
    double width;
    // Gaussian projection vector v of each dimension of the cube
    std::vector<Vector> vectors;
    // shift t of each projection, in [0, w)
    std::vector<double> offsets;
    // seed of the window-number-to-bit mapping, so different seeds give different cubes
    std::uint64_t bitSeed;
    // maps a vertex, encoded as a d'-bit integer, to the database indices stored in it
    std::unordered_map<int, std::vector<std::size_t>> buckets;

    Impl(const std::vector<Vector>& db, const std::vector<std::string>& names, int projectionCount, double widthValue, int maxCount, int probeCount, int seed)
            : DatabaseIndex{db, names}, projections(projectionCount), maxCandidates(maxCount),
              probes(probeCount), width(widthValue),
              offsets(projectionCount), bitSeed(static_cast<std::uint64_t>(
                  static_cast<std::int64_t>(seed))) {

        // a vertex is a d'-bit integer key and the probe count is bounded by 2^d', so d' is capped
        if(projections <= 0 || projections > kMaxHypercubeProjections)
            throw std::invalid_argument("Hypercube kproj must be in the range 1.." +
                                        std::to_string(kMaxHypercubeProjections));
        // the window width divides the projection, so it must be positive and finite
        if(width <= 0.0 || !std::isfinite(width))
            throw std::invalid_argument("Hypercube width must be finite and positive");
        // both budgets must allow at least one candidate
        if(maxCandidates <= 0)
            throw std::invalid_argument("Hypercube M must be strictly positive");
        if(probes <= 0)
            throw std::invalid_argument("Hypercube probes must be strictly positive");

        std::mt19937 rng(seed);
        std::normal_distribution<double> normal(0.0, 1.0);
        std::uniform_real_distribution<double> offset(0.0, width);
        vectors.resize(projections, Vector(database.front().size()));

        for(int i = 0; i < projections; ++i) {
            for(double& value : vectors[i])
                value = normal(rng);
            offsets[i] = offset(rng);
        }

        // every database vector goes into its vertex
        for(std::size_t i = 0; i < database.size(); ++i)
            buckets[hash(database[i])].push_back(i);
    }

    // pseudo-randomly maps the window number of a projection to 0 or 1 with a seeded integer mixing function
    int mapToBit(int projection, std::int64_t bucket) const {
        std::uint64_t value = bitSeed ^
            (static_cast<std::uint64_t>(projection) +
             0x9e3779b97f4a7c15ULL);
        value ^= static_cast<std::uint64_t>(bucket) +
            0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
        value ^= value >> 30;
        value *= 0xbf58476d1ce4e5b9ULL;
        value ^= value >> 27;
        value *= 0x94d049bb133111ebULL;
        value ^= value >> 31;
        return static_cast<int>(value & 1U);
    }

    // vertex of a point as a d'-bit integer, bit i comes from projection i
    int hash(const Vector& point) const {
        int value = 0;
        for(int i = 0; i < projections; ++i) {
            const double dot = std::inner_product(point.begin(), point.end(), vectors[i].begin(), 0.0);
            const std::int64_t bucket = windowIndex(dot + offsets[i], width);
            if(mapToBit(i, bucket) != 0)
                value |= (1 << i);
        }
        return value;
    }

    // appends to keys the vertices at Hamming distance flips from queryHash, in ascending order, until keys holds limit vertices
    // recursion over the bits from the highest down: bit is the position still to decide, prefix the bits chosen above it
    void verticesAtDistance(int queryHash, int bit, int flips, int prefix, std::size_t limit, std::vector<int>& keys) const {
        if(keys.size() >= limit)
            return;
        // every bit is decided, and the pruning below guarantees exactly flips of them differ from the query
        if(bit < 0) {
            keys.push_back(prefix);
            return;
        }
        const int queryBit = (queryHash >> bit) & 1;
        // choosing 0 before 1 at the highest open position yields the vertices in ascending order
        for(int value = 0; value <= 1; ++value) {
            const int remaining = flips - (value != queryBit ? 1 : 0);
            // only continue if the remaining flips still fit in the bit positions below this one
            if(remaining >= 0 && remaining <= bit)
                verticesAtDistance(queryHash, bit - 1, remaining, prefix | (value << bit), limit, keys);
        }
    }

    // vectors of the first `probes` vertices around the query, at most maxCandidates of them
    std::vector<std::size_t> candidates(const Vector& query) const {
        const int queryHash = hash(query);

        // the vertices are visited in increasing Hamming distance from the query vertex, ties in ascending vertex order
        // generating them one distance at a time instead of sorting all 2^d' makes the cost grow with probes, not 2^d'
        const std::size_t limit = std::min(static_cast<std::size_t>(probes), std::size_t{1} << projections);
        std::vector<int> keys;
        keys.reserve(limit);
        for(int flips = 0; flips <= projections && keys.size() < limit; ++flips)
            verticesAtDistance(queryHash, projections - 1, flips, 0, limit, keys);

        std::vector<std::size_t> result;
        for(std::size_t i = 0; i < keys.size(); ++i) {
            const auto found = buckets.find(keys[i]);
            if(found != buckets.end()) {
                for(const auto index : found->second) {
                    result.push_back(index);
                    // stop once the candidate budget M is reached
                    if(static_cast<int>(result.size()) >= maxCandidates)
                        return result;
                }
            }
        }
        return result;
    }
};

HypercubeSearch::HypercubeSearch(const std::vector<Vector>& db, const std::vector<std::string>& ids, int projections, double width, int maxCandidates, int probes, int seed)
    : impl(std::make_shared<Impl>(db, ids, projections, width, maxCandidates, probes, seed)) {}

SearchResult HypercubeSearch::search(const Vector& query, const std::string& queryId, const std::string& sequenceId) const {
    const auto start = std::chrono::steady_clock::now();
    // checked before hashing, since a non-finite query cannot be mapped to a vertex
    validateQuery(impl->database, query);
    return buildResult(impl->database, impl->ids, impl->candidates(query), query, queryId, sequenceId, start);
}

} // namespace image_search
