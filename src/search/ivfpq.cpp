#include "common.hpp"

#include "kmeans.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace image_search {
// shared helpers and aliases from common.hpp
using namespace detail;

// helpers below are private to this translation unit
namespace {

// Calculates the number of unique values that can be represented with a given number of bits
int checkedCodeCount(int bits) {
    // limits the codebook size to 2^16 entries, so every code fits in a uint16_t
    if(bits <= 0 || bits > 16)
        throw std::invalid_argument("IVFPQ nbits must be in the range 1..16");
    return 1 << bits;
}

} // namespace

// IVFPQ index, which stores each vector as the quantized residual from its cluster center
// queries are ranked by the asymmetric distance (ADC) to the stored codes,
// so the full database vectors are not used at query time
struct IVFPQSearch::Impl : DatabaseIndex {
    std::size_t subquantizers;  // Number of subquantizers (M)
    std::size_t width;          // Length of each subvector, the vector dimension divided by M
    std::vector<Vector> centers;                        // Coarse cluster centers obtained from k-means clustering
    std::vector<std::vector<std::size_t>> lists;        // Inverted lists for each coarse cluster center
    std::vector<Matrix<double>> codebooks;              // codebooks[m] holds the centroids of subspace m, one per row
    std::vector<std::uint16_t> codes;                   // codes[i * M + m] is the codebook entry of subvector m of vector i

    // trains the coarse clusters, the PQ codebooks and encodes the whole database
    Impl(const std::vector<Vector>& db, const std::vector<std::string>& names, int clusterCount, int subCount, int bits, int seed)
            : DatabaseIndex{db, names} {
        const int codeCount = checkedCodeCount(bits);
        // Check for valid subquantizer count and ensure it divides the vector dimension
        if(subCount <= 0)
            throw std::invalid_argument("IVFPQ M must be positive");
        subquantizers = static_cast<std::size_t>(subCount);
        if(database.front().size() % subquantizers != 0)
            throw std::invalid_argument("IVFPQ subquantizers must divide vector dimension");
        width = database.front().size() / subquantizers;

        auto coarse = clusterVectors(database, clusterCount, seed);
        centers = std::move(coarse.centers);
        // k-means already assigned each database vector to its nearest coarse centroid
        const std::vector<std::size_t> coarse_assignments = std::move(coarse.assignments);
        lists.resize(centers.size());
        for(std::size_t i = 0; i < database.size(); ++i) {
            lists[coarse_assignments[i]].push_back(i);
        }

        // residual of database vector i, namely the vector minus its coarse center
        const auto residualOf = [&](std::size_t i, Vector& residual) {
            const auto& center = centers[coarse_assignments[i]];
            for(std::size_t d = 0; d < residual.size(); ++d)
                residual[d] = database[i][d] - center[d];
        };

        // codebooks are trained on a random sample of at most 2048 vectors to keep building fast
        const std::size_t sampleLimit =
            std::min<std::size_t>(database.size(), 2048);
        std::vector<std::size_t> sampleIndices(database.size());
        std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
        std::mt19937 rng(seed);
        std::shuffle(sampleIndices.begin(), sampleIndices.end(), rng);

        // training[m] holds subvector m of every sampled residual, one per row
        std::vector<Matrix<double>> training(subquantizers, Matrix<double>(sampleLimit, width));
        Vector residual(database.front().size());
        for(std::size_t s = 0; s < sampleLimit; ++s) {
            residualOf(sampleIndices[s], residual);
            for(std::size_t m = 0; m < subquantizers; ++m)
                std::copy(residual.begin() + m * width, residual.begin() + (m + 1) * width,
                          training[m].row(s));
        }

        // one k-means codebook of 2^nbits entries per subquantizer, fewer if there are fewer samples
        const std::size_t codebookSize = std::min(static_cast<std::size_t>(codeCount), sampleLimit);
        for(std::size_t m = 0; m < subquantizers; ++m)
            codebooks.push_back(image_search::kmeans(
                training[m], codebookSize, seed + static_cast<int>(m) + 1).centers);

        // Encode each database residual with the corresponding PQ codebooks.
        codes.resize(database.size() * subquantizers);
        for(std::size_t i = 0; i < database.size(); ++i) {
            residualOf(i, residual);
            for(std::size_t m = 0; m < subquantizers; ++m) {
                codes[i * subquantizers + m] = static_cast<std::uint16_t>(
                    nearest_center(residual.data() + m * width, codebooks[m]));
            }
        }
    }

    // returns the database indices of the 10 smallest ADC distances among the probed lists, nearest first
    // examined receives the number of ADC distances computed
    std::vector<std::size_t> candidates(const Vector& query, int probes, std::size_t& examined) const {
        const auto selected = nearestCenters(query, centers, probes);   // Find the nearest coarse cluster centers to the query vector
        const std::size_t codebookSize = codebooks.front().rows;
        // stored vectors with their ADC distance to the query
        std::vector<ScoredId> ranked;
        Vector queryResidual(query.size());
        // lookup[m * s + h] is the squared distance of the query's m-th residual subvector to entry h of codebook m
        std::vector<double> lookup(subquantizers * codebookSize);

        // Build one residual-PQ lookup table for each selected coarse cell.
        for(const auto center : selected) {
            // residual of the query relative to this cluster's center
            for(std::size_t d = 0; d < query.size(); ++d)
                queryResidual[d] = query[d] - centers[center][d];
            for(std::size_t m = 0; m < subquantizers; ++m) {
                for(std::size_t h = 0; h < codebookSize; ++h)
                    lookup[m * codebookSize + h] = squared_distance(
                        queryResidual.data() + m * width, codebooks[m].row(h), width);
            }

            // ADC distance of each stored vector, the sum of its M table lookups
            for(const auto index : lists[center]) {
                const std::uint16_t* code = codes.data() + index * subquantizers;
                double score = 0.0;
                for(std::size_t m = 0; m < subquantizers; ++m)
                    score += lookup[m * codebookSize + code[m]];
                ranked.emplace_back(score, index);
            }
        }

        examined = ranked.size();
        // only the 10 nearest by ADC are returned, there is no reranking by exact distance
        const std::size_t limit = std::min<std::size_t>(10, ranked.size());
        std::partial_sort(ranked.begin(), ranked.begin() + limit, ranked.end());
        std::vector<std::size_t> result;
        for(std::size_t i = 0; i < limit; ++i)
            result.push_back(ranked[i].second);
        return result;
    }
};

// IVFPQSearch constructor
IVFPQSearch::IVFPQSearch(const std::vector<Vector>& db, const std::vector<std::string>& ids, int clusters, int probes, int subquantizers, int bits, int seed)
    : impl(std::make_shared<Impl>(db, ids, clusters, subquantizers, bits, seed)), probes(probes) {}

// copies the handle, so the trained index is shared rather than rebuilt
IVFPQSearch IVFPQSearch::withProbes(int probe_count) const {
    IVFPQSearch copy = *this;
    copy.probes = probe_count;
    return copy;
}

// IVFPQSearch search function
SearchResult IVFPQSearch::search(const Vector& query, const std::string& queryId, const std::string& sequenceId) const {
    const auto start = std::chrono::high_resolution_clock::now();
    // checked first, since the lookup tables index the query by the database dimension
    validateQuery(impl->database, query);
    std::size_t examined = 0;
    const auto ranked = impl->candidates(query, probes, examined);
    return buildRankedResult(impl->ids, ranked, examined, queryId, sequenceId, start);
}

} // namespace image_search
