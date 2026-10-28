#include "common.hpp"

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

// LSHSearch class implementation
struct LSHSearch::Impl : DatabaseIndex {
    // large prime M used to combine the k bucket numbers into one key
    static constexpr std::uint64_t modulus = 1000000007ULL;
    int k;          // Number of hash functions per table
    int tables;     // Number of hash tables
    double width;   // Width parameter for the hash functions
    std::vector<std::vector<Vector>> projections;   // Projections for each hash table
    std::vector<std::vector<double>> offsets;       // Offsets for each hash function in each table
    // random integers r_i that weight each hash function in g(p) = sum r_i h_i(p) mod M
    std::vector<std::vector<std::uint64_t>> coefficients;
    // per table, maps a combined key g(p) to the database indices stored in that bucket
    std::vector<std::unordered_map<std::uint64_t, std::vector<std::size_t>>> buckets;

    // Constructor to initialize the LSH index with database vectors, identifiers, and LSH parameters
    Impl(const std::vector<Vector>& db, const std::vector<std::string>& names, int kValue, int tableValue, double widthValue, int seed)
            : DatabaseIndex{db, names}, k(kValue), tables(tableValue), width(widthValue) {
        // at least one hash function and one table are required
        if(k <= 0 || tables <= 0)
            throw std::invalid_argument("LSH k and L must be strictly positive");
        // the window width divides the projection, so it must be positive and finite
        if(width <= 0.0 || !std::isfinite(width))
            throw std::invalid_argument("LSH width must be finite and positive");

        // seeded generator makes the index reproducible
        std::mt19937 rng(seed);
        // gaussian entries give the random projection vectors v of the h(p) = floor((p.v + t) / w) family
        std::normal_distribution<double> normal(0.0, 1.0);
        // random shift t in [0, w)
        std::uniform_real_distribution<double> offset(0.0, width);
        // non-zero coefficients below the modulus
        std::uniform_int_distribution<std::uint64_t> coefficient(1, modulus - 1);
        projections.resize(tables);
        offsets.resize(tables);
        coefficients.resize(tables);
        buckets.resize(tables);

        for(int t = 0; t < tables; ++t) {
            // Initialize projections and offsets for each hash table
            projections[t].resize(k, Vector(database.front().size()));
            offsets[t].resize(k);
            coefficients[t].resize(k);
            // draw the random parameters of each of the k hash functions
            for(int h = 0; h < k; ++h) {
                for(double& value : projections[t][h]) {
                    value = normal(rng);
                }
                offsets[t][h] = offset(rng);
                coefficients[t][h] = coefficient(rng);
            }
            // Insert database vectors into the hash buckets based on their hash values
            for(std::size_t i = 0; i < database.size(); ++i) {
                buckets[t][hash(database[i], t)].push_back(i);
            }
        }
    }

    // computes the bucket key g(p) of a point in one table
    std::uint64_t hash(const Vector& point, int table) const {
        std::uint64_t combined = 0;
        for(int h = 0; h < k; ++h) {
            // p.v + t, since inner_product starts its sum from the offset
            const double dot = std::inner_product(
                point.begin(), point.end(), projections[table][h].begin(),
                offsets[table][h]);
            // h_i(p), which can be negative
            const auto bucket = static_cast<std::int64_t>(
                std::floor(dot / width));
            // shifts the bucket number into [0, M) even when it is negative
            const auto normalized = static_cast<std::uint64_t>(
                ((bucket % static_cast<std::int64_t>(modulus)) +
                 static_cast<std::int64_t>(modulus)) %
                static_cast<std::int64_t>(modulus));
            // accumulate r_i * h_i(p) modulo M
            combined = (combined +
                        coefficients[table][h] * normalized) % modulus;
        }
        return combined;
    }

    // collects the database indices that share a bucket with the query in at least one table
    std::vector<std::size_t> candidates(const Vector& query) const {
        std::vector<std::size_t> result;
        for(int t = 0; t < tables; ++t) {
            // Look up matching candidates in the hash bucket and add them to the result
            const auto found = buckets[t].find(hash(query, t));
            if(found != buckets[t].end())
                result.insert(result.end(), found->second.begin(), found->second.end());
        }
        return result;
    }
};

// LSHSearch constructor
LSHSearch::LSHSearch(const std::vector<Vector>& db, const std::vector<std::string>& ids, int k, int tables, double width, int seed)
    : impl(std::make_shared<Impl>(db, ids, k, tables, width, seed)) {}

// LSHSearch search function
SearchResult LSHSearch::search(const Vector& query, const std::string& queryId, const std::string& sequenceId) const {
    const auto start = std::chrono::steady_clock::now();
    return buildResult(impl->database, impl->ids, impl->candidates(query), query, queryId, sequenceId, start);
}

} // namespace image_search
