#include "common.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>

namespace image_search {
// shared helpers and aliases from common.hpp
using namespace detail;

// hypercube index, where each of the d' projections contributes one bit of the bucket key
struct HypercubeSearch::Impl : DatabaseIndex {
    int projections;        // Number of projections
    int maxCandidates;      // Maximum number of candidates to consider
    int probes;             // Number of probes to perform in the hypercube
    double width;           // Width parameter for the hypercube projections
    // random gaussian projection vector of each dimension of the cube
    std::vector<Vector> vectors;
    // random shift of each projection
    std::vector<double> offsets;
    // seed mixed into the bucket-to-bit mapping so different seeds give different cubes
    std::uint64_t bitSeed;
    // maps a vertex of the cube, encoded as an integer, to the database indices stored in it
    std::unordered_map<int, std::vector<std::size_t>> buckets;

    // Constructor to initialize the Hypercube index with database vectors, identifiers, and hypercube parameters
    Impl(const std::vector<Vector>& db, const std::vector<std::string>& names, int projectionCount, double widthValue, int maxCount, int probeCount, int seed)
            : DatabaseIndex{db, names}, projections(projectionCount), maxCandidates(maxCount),
              probes(probeCount), width(widthValue),
              offsets(projectionCount), bitSeed(static_cast<std::uint64_t>(
                  static_cast<std::int64_t>(seed))) {

        // 2^d' vertices are enumerated per query, so d' is capped
        if(projections <= 0 || projections > 20)
            throw std::invalid_argument("Hypercube kproj must be in the range 1..20");
        // the window width divides the projection, so it must be positive and finite
        if(width <= 0.0 || !std::isfinite(width))
            throw std::invalid_argument("Hypercube width must be finite and positive");

        std::mt19937 rng(seed);
        std::normal_distribution<double> normal(0.0, 1.0);
        std::uniform_real_distribution<double> offset(0.0, width);
        vectors.resize(projections, Vector(database.front().size()));

        // draw the gaussian projection vector and the shift of each dimension
        for(int i = 0; i < projections; ++i) {
            for(double& value : vectors[i])
                value = normal(rng);
            offsets[i] = offset(rng);
        }

        // place every database vector into the vertex it hashes to
        for(std::size_t i = 0; i < database.size(); ++i)
            buckets[hash(database[i])].push_back(i);
    }

    // pseudo-randomly maps an integer bucket number to 0 or 1 using a seeded integer mixing function
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

    // computes the vertex of the cube for a point as a d'-bit integer
    int hash(const Vector& point) const {
        int value = 0;
        for(int i = 0; i < projections; ++i) {
            // projection of the point onto the i-th random vector
            const double dot = std::inner_product(point.begin(), point.end(), vectors[i].begin(), 0.0);
            // window number of the shifted projection
            const auto bucket = static_cast<std::int64_t>(
                std::floor((dot + offsets[i]) / width));
            // set bit i when the window number maps to 1
            if(mapToBit(i, bucket) != 0)
                value |= (1 << i);
        }
        return value;
    }

    // Function to find candidate vectors for a given query
    std::vector<std::size_t> candidates(const Vector& query) const {
        // Compute the hash value for the query vector
        const int queryHash = hash(query);
        std::vector<int> keys(1 << projections);    // Generate all possible keys for the hypercube
        std::iota(keys.begin(), keys.end(), 0);

        // Sort the keys based on their Hamming distance to the query hash (popcount of XOR to count differing bits)
        std::stable_sort(keys.begin(), keys.end(), [queryHash](int a, int b) {
            return std::popcount(static_cast<unsigned>(a ^ queryHash)) < std::popcount(static_cast<unsigned>(b ^ queryHash));});

        // candidates in the order the vertices are visited
        std::vector<std::size_t> result;
        // Limit the number of probes to the specified maximum and collect candidates from the corresponding buckets
        for(int i = 0; i < std::min(probes, static_cast<int>(keys.size())); ++i) {
            const auto found = buckets.find(keys[i]);
            if(found != buckets.end()) {
                for(const auto index : found->second) {
                    result.push_back(index);
                    // stop early once the candidate budget M is reached
                    if(static_cast<int>(result.size()) >= maxCandidates)
                        return result;
                }
            }
        }
        return result;
    }
};

// HypercubeSearch constructor
HypercubeSearch::HypercubeSearch(const std::vector<Vector>& db, const std::vector<std::string>& ids, int projections, double width, int maxCandidates, int probes, int seed)
    : impl(std::make_shared<Impl>(db, ids, projections, width, maxCandidates, probes, seed)) {}

// HypercubeSearch search function
SearchResult HypercubeSearch::search(const Vector& query, const std::string& queryId, const std::string& sequenceId) const {
    const auto start = std::chrono::high_resolution_clock::now();
    return buildResult(impl->database, impl->ids, impl->candidates(query), query, queryId, sequenceId, start);
}

} // namespace image_search
