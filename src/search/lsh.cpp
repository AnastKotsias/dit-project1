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

// LSH index: each table maps the combined key g(p) of a point to the database vectors in that bucket
// a hash function is h(p) = floor((p.v + t) / w) with a Gaussian vector v and a shift t in [0, w)
struct LSHSearch::Impl : DatabaseIndex {
    // large prime M that the combined key is reduced modulo
    static constexpr std::uint64_t modulus = 1000000007ULL;
    // hash functions per table
    int k;
    int tables;
    // window width w
    double width;
    // projections[t][h] is the vector v of hash function h of table t
    std::vector<std::vector<Vector>> projections;
    // offsets[t][h] is its shift t
    std::vector<std::vector<double>> offsets;
    // random integers r_i that weight the hash functions in g(p) = sum r_i h_i(p) mod M
    std::vector<std::vector<std::uint64_t>> coefficients;
    // per table, maps a combined key g(p) to the database indices stored in that bucket
    std::vector<std::unordered_map<std::uint64_t, std::vector<std::size_t>>> buckets;

    Impl(const std::vector<Vector>& db, const std::vector<std::string>& names, int kValue, int tableValue, double widthValue, int seed)
            : DatabaseIndex{db, names}, k(kValue), tables(tableValue), width(widthValue) {
        if(k <= 0 || tables <= 0)
            throw std::invalid_argument("LSH k and L must be strictly positive");
        // the window width divides the projection, so it must be positive and finite
        if(width <= 0.0 || !std::isfinite(width))
            throw std::invalid_argument("LSH width must be finite and positive");

        std::mt19937 rng(seed);
        std::normal_distribution<double> normal(0.0, 1.0);
        std::uniform_real_distribution<double> offset(0.0, width);
        // non-zero coefficients below the modulus
        std::uniform_int_distribution<std::uint64_t> coefficient(1, modulus - 1);
        projections.resize(tables);
        offsets.resize(tables);
        coefficients.resize(tables);
        buckets.resize(tables);

        for(int t = 0; t < tables; ++t) {
            projections[t].resize(k, Vector(database.front().size()));
            offsets[t].resize(k);
            coefficients[t].resize(k);
            for(int h = 0; h < k; ++h) {
                for(double& value : projections[t][h]) {
                    value = normal(rng);
                }
                offsets[t][h] = offset(rng);
                coefficients[t][h] = coefficient(rng);
            }
            // every database vector goes into its bucket of this table
            for(std::size_t i = 0; i < database.size(); ++i) {
                buckets[t][hash(database[i], t)].push_back(i);
            }
        }
    }

    // bucket key g(p) = sum r_i h_i(p) mod M of a point in one table
    std::uint64_t hash(const Vector& point, int table) const {
        std::uint64_t combined = 0;
        for(int h = 0; h < k; ++h) {
            // p.v + t, since inner_product starts its sum from the offset
            const double dot = std::inner_product(
                point.begin(), point.end(), projections[table][h].begin(),
                offsets[table][h]);
            // h_i(p), which can be negative
            const std::int64_t bucket = windowIndex(dot, width);
            // maps h_i(p) into [0, M) even when it is negative
            const auto normalized = static_cast<std::uint64_t>(
                ((bucket % static_cast<std::int64_t>(modulus)) +
                 static_cast<std::int64_t>(modulus)) %
                static_cast<std::int64_t>(modulus));
            combined = (combined +
                        coefficients[table][h] * normalized) % modulus;
        }
        return combined;
    }

    // collects the distinct database indices that share a bucket with the query in at least one table
    std::vector<std::size_t> candidates(const Vector& query) const {
        std::vector<std::size_t> result;
        for(int t = 0; t < tables; ++t) {
            const auto found = buckets[t].find(hash(query, t));
            if(found != buckets[t].end())
                result.insert(result.end(), found->second.begin(), found->second.end());
        }
        // a point sharing the query's bucket in several tables is listed once per table, sorting
        // removes the repeats at a cost that depends only on the number of candidates
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        return result;
    }
};

LSHSearch::LSHSearch(const std::vector<Vector>& db, const std::vector<std::string>& ids, int k, int tables, double width, int seed)
    : impl(std::make_shared<Impl>(db, ids, k, tables, width, seed)) {}

SearchResult LSHSearch::search(const Vector& query, const std::string& queryId, const std::string& sequenceId) const {
    const auto start = std::chrono::steady_clock::now();
    // checked before hashing, since a non-finite query cannot be mapped to a bucket
    validateQuery(impl->database, query);
    return buildResult(impl->database, impl->ids, impl->candidates(query), query, queryId, sequenceId, start);
}

} // namespace image_search
