#include "common.hpp"

#include <algorithm>
#include <chrono>
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
    // limits the codebook size to 2^16 entries
    if(bits <= 0 || bits > 16)
        throw std::invalid_argument("IVFPQ nbits must be in the range 1..16");
    return 1 << bits;
}

} // namespace

// IVFPQ index, which stores each vector as the quantized residual from its cluster center
struct IVFPQSearch::Impl : DatabaseIndex {
    int probes;             // Number of probes to perform in the IVFPQ search
    int subquantizers;      // Number of subquantizers (M)
    int codeCount;          // Number of unique codes that can be represented with the specified number of bits
    std::vector<Vector> centers;                        // Coarse cluster centers obtained from k-means clustering
    std::vector<std::vector<std::size_t>> lists;        // Inverted lists for each coarse cluster center
    std::vector<std::vector<Vector>> codebooks;         // Codebooks for each subquantizer
    std::vector<std::vector<std::vector<int>>> codes;   // Quantized codes for each database vector

    // trains the coarse clusters, the PQ codebooks and encodes the whole database
    Impl(const std::vector<Vector>& db, const std::vector<std::string>& names, int clusterCount, int probeCount, int subCount, int bits, int seed)
            : DatabaseIndex{db, names}, probes(probeCount), subquantizers(subCount), codeCount(checkedCodeCount(bits)) {
        // Check for valid subquantizer count and ensure it divides the vector dimension
        if(subquantizers <= 0)
            throw std::invalid_argument("IVFPQ M must be positive");
        if(database.front().size() % static_cast<std::size_t>(subquantizers) != 0)
            throw std::invalid_argument("IVFPQ subquantizers must divide vector dimension");

        auto coarse = clusterVectors(database, clusterCount, seed);
        centers = std::move(coarse.centers);
        // k-means already assigned each database vector to its nearest coarse centroid
        const std::vector<std::size_t> coarse_assignments = std::move(coarse.assignments);
        lists.resize(centers.size());
        for(std::size_t i = 0; i < database.size(); ++i) {
            lists[coarse_assignments[i]].push_back(i);
        }

        // training subvectors of each subquantizer
        std::vector<std::vector<Vector>> training(subquantizers);
        // length of each subvector, which is the dimension divided by M
        const std::size_t width =
            database.front().size() / subquantizers;
        // codebooks are trained on a random sample of at most 2048 vectors to keep building fast
        const std::size_t sampleLimit =
            std::min<std::size_t>(database.size(), 2048);
        std::vector<std::size_t> sampleIndices(database.size());
        std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
        std::mt19937 rng(seed);
        std::shuffle(sampleIndices.begin(), sampleIndices.end(), rng);

        // Train PQ codebooks on subvectors of coarse-centroid residuals.
        for(std::size_t i = 0; i < sampleLimit; ++i) {
            const std::size_t databaseIndex = sampleIndices[i];
            // residual is the vector minus its coarse center
            Vector residual(database[databaseIndex].size());
            const auto& center = centers[coarse_assignments[databaseIndex]];
            for(std::size_t d = 0; d < database[databaseIndex].size(); ++d)
                residual[d] = database[databaseIndex][d] - center[d];

            // slice the residual into M consecutive subvectors
            for(int m = 0; m < subquantizers; ++m) {
                training[m].emplace_back(
                    residual.begin() + m * width,
                    residual.begin() + (m + 1) * width);
            }
        }

        // one k-means codebook of 2^nbits entries per subquantizer
        for(int m = 0; m < subquantizers; ++m)
            codebooks.push_back(clusterVectors(training[m], codeCount, seed + m + 1).centers);

        // codes[i][m] holds the codebook entry chosen for subvector m of vector i
        codes.resize(database.size());

        // Encode each database residual with the corresponding PQ codebooks.
        for(std::size_t i = 0; i < database.size(); ++i) {
            Vector residual(database[i].size());
            const auto& center = centers[coarse_assignments[i]];
            for(std::size_t d = 0; d < database[i].size(); ++d)
                residual[d] = database[i][d] - center[d];

            codes[i].resize(subquantizers);
            for(int m = 0; m < subquantizers; ++m) {
                Vector part(
                    residual.begin() + m * width,
                    residual.begin() + (m + 1) * width);
                // store the index of the nearest codebook entry as a one-element vector
                codes[i][m] = {static_cast<int>(
                    nearestCenters(part, codebooks[m], 1).front())};
            }
        }
    }

    // Function to find candidate vectors for a given query
    std::vector<std::size_t> candidates(const Vector& query) const {
        const auto selected = nearestCenters(query, centers, probes);   // Find the nearest coarse cluster centers to the query vector
        // candidates with their approximate distances, summed over all probed clusters
        std::vector<ScoredId> ranked;
        // length of each subvector
        const std::size_t width = query.size() / subquantizers;

        // Build one residual-PQ lookup table for each selected coarse cell.
        std::vector<std::size_t> result;
        for(const auto center : selected) {
            // residual of the query relative to this cluster's center
            Vector query_residual(query.size());
            for(std::size_t d = 0; d < query.size(); ++d)
                query_residual[d] = query[d] - centers[center][d];

            // lookup[m][c] is the distance of the query's m-th subvector to codebook entry c
            std::vector<std::vector<double>> lookup(subquantizers);
            for(int m = 0; m < subquantizers; ++m) {
                Vector part(
                    query_residual.begin() + m * width,
                    query_residual.begin() + (m + 1) * width);
                for(const auto& centroid : codebooks[m])
                    lookup[m].push_back(squaredEuclideanDistance(part, centroid));
            }

            // approximate distance of each stored vector, the sum of its M table lookups
            for(const auto index : lists[center]) {
                double score = 0.0;
                for(int m = 0; m < subquantizers; ++m) {
                    score += lookup[m][codes[index][m][0]];
                }
                ranked.emplace_back(score, index);
            }
        }

        // Sort the candidates based on their scores and return the sorted indices
        std::sort(ranked.begin(), ranked.end());
        // return indices only, since the final ranking recomputes exact distances
        for(const auto& item : ranked)
            result.push_back(item.second);
        return result;
    }
};

// IVFPQSearch constructor
IVFPQSearch::IVFPQSearch(const std::vector<Vector>& db, const std::vector<std::string>& ids, int clusters, int probes, int subquantizers, int bits, int seed)
    : impl(std::make_shared<Impl>(db, ids, clusters, probes, subquantizers, bits, seed)) {}

// IVFPQSearch search function
SearchResult IVFPQSearch::search(const Vector& query, const std::string& queryId, const std::string& sequenceId) const {
    const auto start = std::chrono::high_resolution_clock::now();
    return buildResult(impl->database, impl->ids, impl->candidates(query), query, queryId, sequenceId, start);
}

} // namespace image_search
