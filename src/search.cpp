#include "search.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace image_search {
namespace {

using Vector = std::vector<double>;
using ScoredId = std::pair<double, std::size_t>;

// Calculates the squared Euclidean distance between two vectors
double Euclidiandistance(const Vector& a, const Vector& b) {
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

// Generates a unique string key for a vector of integers, used for hashing in LSH
std::string integerKey(const std::vector<int>& values) {
    std::ostringstream key;
    for(const int value : values) {
        key << value << ',';
    };
    return key.str();
}

// Finds the "count" nearest centers to a given query vector
std::vector<std::size_t> nearestCenters(const Vector& query, const std::vector<Vector>& centers, int count) {
    std::vector<ScoredId> distances;
    distances.reserve(centers.size());
    for(std::size_t i = 0; i < centers.size(); ++i) {
        distances.emplace_back(Euclidiandistance(query, centers[i]), i);
    };
    std::sort(distances.begin(), distances.end());
    
    // Ensure we don't exceed the number of available centers
    count = std::min(count, static_cast<int>(distances.size()));
    std::vector<std::size_t> result;
    
    // Collect the nearest centers
    for(int i = 0; i < count; ++i) {
        result.push_back(distances[i].second);
    };

    return result;
}

std::vector<Vector> kmeans(const std::vector<Vector>& points, int requested, int seed, int iterations = 8) {
    if (points.empty()) 
        return {};
    
    // Ensure we don't request more clusters than available points
    const int k = std::min(requested, static_cast<int>(points.size()));
    
    // Initialize random number generator for shuffling
    std::mt19937 rng(seed);
    std::vector<std::size_t> order(points.size());
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    
    // Select initial centers from the shuffled points
    std::vector<Vector> centers;
    for(int i = 0; i < k; ++i) {
        centers.push_back(points[order[i]]);
    }

    // Vector to hold the assignment of each point to a center
    std::vector<int> assignments(points.size(), 0);
    for(int iteration = 0; iteration < iterations; ++iteration) {
        std::vector<Vector> sums(k, Vector(points.front().size(), 0.0));
        std::vector<int> counts(k, 0);
        for(std::size_t i = 0; i < points.size(); ++i) {
            // Assign each point to the nearest center
            assignments[i] = static_cast<int>(nearestCenters(points[i], centers, 1).front());
            ++counts[assignments[i]];
            
            for(std::size_t d = 0; d < points[i].size(); ++d) {
                sums[assignments[i]][d] += points[i][d];
            }
        }
        // Update centers based on the mean of assigned points
        for(int c = 0; c < k; ++c) {
            if(counts[c] != 0) {
                for(double& value : sums[c]) value /= counts[c];
                centers[c] = std::move(sums[c]);
            }
        }
    }
    return centers;
}

// Calculates the number of unique values that can be represented with a given number of bits
int checkedCodeCount(int bits) {
    if(bits <= 0 || bits > 16)
        throw std::invalid_argument("IVFPQ nbits must be in the range 1..16");
    return 1 << bits;
}

// Common result-building function used by Exact, LSH, Hypercube, IVFFlat, and IVFPQ search
SearchResult buildResult(const std::vector<Vector>& database, const std::vector<std::string>& ids, const std::vector<std::size_t>& candidates, const Vector& query, const std::string& queryId, const std::string& sequenceId, std::chrono::high_resolution_clock::time_point start) {
    std::vector<ScoredId> ranked;
    // Use a set to ensure unique candidates and avoid duplicates
    std::unordered_set<std::size_t> unique;
    // Rank only candidates returned by the index. Approximate methods must not
    // silently turn an empty candidate set into an exact search.
    for(const auto index : candidates) {
        if(index < database.size() && unique.insert(index).second) {
            ranked.emplace_back(Euclidiandistance(query, database[index]), index);
        }
    }

    // Select the top 10 among the candidates actually returned by the index.
    // Missing positions are non-relevant by definition.
    std::sort(ranked.begin(), ranked.end());
    const int limit = std::min(10, static_cast<int>(ranked.size()));
    std::vector<std::string> top;
    for(int i = 0; i < limit; ++i) {
        top.push_back(ids[ranked[i].second]);
    }
    
    // Build the set of relevant images for the given sequence
    std::set<std::string> relevant;
    for(int i = 2; i <= 6; ++i) {
        relevant.insert(sequenceId + "/" + std::to_string(i));
    }

    // Initialize and calculate counters for recall and average precision calculations
    int matches5 = 0;
    int matches10 = 0;
    double apSum = 0.0;
    int found = 0;
    for(int i = 0; i < limit; ++i) {
        if(relevant.count(top[i]) != 0) {
            if(i < 5) ++matches5;
            ++matches10;
            ++found;
            apSum += static_cast<double>(found) / static_cast<double>(i + 1);
        }
    }

    // Return the search results
    const auto end = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    return {queryId, top, matches5 / 5.0, matches10 / 5.0, apSum / 5.0, elapsed};
}

struct DatabaseIndex {
    std::vector<Vector> database;
    std::vector<std::string> ids;

    // Constructor to initialize the database index with vectors and their corresponding identifiers
    DatabaseIndex(const std::vector<Vector>& values, const std::vector<std::string>& names)
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
        }
    }
};

} // namespace

double computeEuclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
    return std::sqrt(Euclidiandistance(a, b));
}

// ExactSearch class implementation
SearchResult ExactSearch::search(const Vector& query, const std::string& queryId, const std::string& sequenceId) {
    const auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::size_t> all(database_bows.size());
    std::iota(all.begin(), all.end(), 0);
    return buildResult(database_bows, database_ids, all, query, queryId, sequenceId, start);
}

// LSHSearch class implementation
struct LSHSearch::Impl : DatabaseIndex {
    int k;          // Number of hash functions per table
    int tables;     // Number of hash tables
    double width;   // Width parameter for the hash functions
    std::vector<std::vector<Vector>> projections;   // Projections for each hash table
    std::vector<std::vector<double>> offsets;       // Offsets for each hash function in each table
    std::vector<std::unordered_map<std::string, std::vector<std::size_t>>> buckets;

    // Constructor to initialize the LSH index with database vectors, identifiers, and LSH parameters
    Impl(const std::vector<Vector>& db, const std::vector<std::string>& names, int kValue, int tableValue, double widthValue, int seed)
            : DatabaseIndex{db, names}, k(kValue), tables(tableValue), width(widthValue) {
        
        std::mt19937 rng(seed);
        std::normal_distribution<double> normal(0.0, 1.0);
        std::uniform_real_distribution<double> offset(0.0, width);
        projections.resize(tables);
        offsets.resize(tables);
        buckets.resize(tables);

        for(int t = 0; t < tables; ++t) {
            // Initialize projections and offsets for each hash table
            projections[t].resize(k, Vector(database.front().size()));
            offsets[t].resize(k);
            for(int h = 0; h < k; ++h) {
                for(double& value : projections[t][h]) {
                    value = normal(rng);
                }
                offsets[t][h] = offset(rng);
            }
            // Insert database vectors into the hash buckets based on their hash values
            for(std::size_t i = 0; i < database.size(); ++i) {
                std::vector<int> hash;
                for(int h = 0; h < k; ++h) {
                    double dot = std::inner_product(database[i].begin(), database[i].end(), projections[t][h].begin(), offsets[t][h]);
                    hash.push_back(static_cast<int>(std::floor(dot / width)));
                }
                buckets[t][integerKey(hash)].push_back(i);
            }
        }
    }

    // Querying the LSH index
    std::vector<std::size_t> candidates(const Vector& query) const {
        std::vector<std::size_t> result;
        for(int t = 0; t < tables; ++t) {
            std::vector<int> hash;
            // For each hash function in the table, compute the hash value for the query vector
            for(int h = 0; h < k; ++h) {
                double dot = std::inner_product(query.begin(), query.end(), projections[t][h].begin(), offsets[t][h]);
                hash.push_back(static_cast<int>(std::floor(dot / width)));
            }
            // Look up matching candidates in the hash bucket and add them to the result
            const auto found = buckets[t].find(integerKey(hash));
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
    const auto start = std::chrono::high_resolution_clock::now();
    return buildResult(impl->database, impl->ids, impl->candidates(query), query, queryId, sequenceId, start);
}

struct HypercubeSearch::Impl : DatabaseIndex {
    int projections;        // Number of projections
    int maxCandidates;      // Maximum number of candidates to consider
    int probes;             // Number of probes to perform in the hypercube
    double width;           // Width parameter for the hypercube projections
    std::vector<Vector> vectors;
    std::vector<double> offsets;
    std::uint64_t bitSeed;
    std::unordered_map<int, std::vector<std::size_t>> buckets;

    // Constructor to initialize the Hypercube index with database vectors, identifiers, and hypercube parameters
    Impl(const std::vector<Vector>& db, const std::vector<std::string>& names, int projectionCount, double widthValue, int maxCount, int probeCount, int seed)
            : DatabaseIndex{db, names}, projections(projectionCount), maxCandidates(maxCount),
              probes(probeCount), width(widthValue),
              offsets(projectionCount), bitSeed(static_cast<std::uint64_t>(
                  static_cast<std::int64_t>(seed))) {
        
        if(projections <= 0 || projections > 20)
            throw std::invalid_argument("Hypercube kproj must be in the range 1..20");
        if(width <= 0.0 || !std::isfinite(width))
            throw std::invalid_argument("Hypercube width must be finite and positive");

        std::mt19937 rng(seed);
        std::normal_distribution<double> normal(0.0, 1.0);
        std::uniform_real_distribution<double> offset(0.0, width);
        vectors.resize(projections, Vector(database.front().size()));

        for(int i = 0; i < projections; ++i) {
            for(double& value : vectors[i])
                value = normal(rng);
            offsets[i] = offset(rng);
        }

        for(std::size_t i = 0; i < database.size(); ++i)
            buckets[hash(database[i])].push_back(i);
    }

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

    int hash(const Vector& point) const {
        int value = 0;
        for(int i = 0; i < projections; ++i) {
            const double dot = std::inner_product(point.begin(), point.end(), vectors[i].begin(), 0.0);
            const auto bucket = static_cast<std::int64_t>(
                std::floor((dot + offsets[i]) / width));
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

        std::vector<std::size_t> result;
        // Limit the number of probes to the specified maximum and collect candidates from the corresponding buckets
        for(int i = 0; i < std::min(probes, static_cast<int>(keys.size())); ++i) {
            const auto found = buckets.find(keys[i]);
            if(found != buckets.end()) {
                for(const auto index : found->second) {
                    result.push_back(index);
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

// IVFFlatSearch class implementation
struct IVFFlatSearch::Impl : DatabaseIndex {
    int probes;
    std::vector<Vector> centers;
    std::vector<std::vector<std::size_t>> lists;

    // Constructor to initialize the IVFFlat index
    Impl(const std::vector<Vector>& db, const std::vector<std::string>& names, int clusterCount, int probeCount, int seed)
            : DatabaseIndex{db, names}, probes(probeCount), centers(kmeans(db, clusterCount, seed)) {
        
        // Create inverted lists for each cluster center
        lists.resize(centers.size());
        for(std::size_t i = 0; i < database.size(); ++i) {
            lists[nearestCenters(database[i], centers, 1).front()].push_back(i);
        }
    }

    // Function to find candidate vectors for a given query
    std::vector<std::size_t> candidates(const Vector& query) const {
        const auto selected = nearestCenters(query, centers, probes);
        std::vector<std::size_t> result;
        for(const auto center : selected) 
            result.insert(result.end(), lists[center].begin(), lists[center].end());
        return result;
    }
};

// IVFFlatSearch constructor
IVFFlatSearch::IVFFlatSearch(const std::vector<Vector>& db, const std::vector<std::string>& ids, int clusters, int probes, int seed)
    : impl(std::make_shared<Impl>(db, ids, clusters, probes, seed)) {}

// IVFFlatSearch search function
SearchResult IVFFlatSearch::search(const Vector& query, const std::string& queryId, const std::string& sequenceId) const {
    const auto start = std::chrono::high_resolution_clock::now();
    return buildResult(impl->database, impl->ids, impl->candidates(query), query, queryId, sequenceId, start);
}

struct IVFPQSearch::Impl : DatabaseIndex {
    int probes;             // Number of probes to perform in the IVFPQ search
    int subquantizers;      // Number of subquantizers (M)
    int codeCount;          // Number of unique codes that can be represented with the specified number of bits
    std::vector<Vector> centers;                        // Coarse cluster centers obtained from k-means clustering
    std::vector<std::vector<std::size_t>> lists;        // Inverted lists for each coarse cluster center
    std::vector<std::vector<Vector>> codebooks;         // Codebooks for each subquantizer
    std::vector<std::vector<std::vector<int>>> codes;   // Quantized codes for each database vector

    Impl(const std::vector<Vector>& db, const std::vector<std::string>& names, int clusterCount, int probeCount, int subCount, int bits, int seed)
            : DatabaseIndex{db, names}, probes(probeCount), subquantizers(subCount), codeCount(checkedCodeCount(bits)), centers(kmeans(db, clusterCount, seed)) {
        // Check for valid subquantizer count and ensure it divides the vector dimension
        if(subquantizers <= 0)
            throw std::invalid_argument("IVFPQ M must be positive");
        if(database.front().size() % static_cast<std::size_t>(subquantizers) != 0)
            throw std::invalid_argument("IVFPQ subquantizers must divide vector dimension");
        
        // Assign each database vector to its nearest coarse centroid.
        lists.resize(centers.size());
        std::vector<std::size_t> coarse_assignments(database.size());
        for(std::size_t i = 0; i < database.size(); ++i) {
            coarse_assignments[i] = nearestCenters(
                database[i], centers, 1).front();
            lists[coarse_assignments[i]].push_back(i);
        }

        std::vector<std::vector<Vector>> training(subquantizers);
        const std::size_t width =
            database.front().size() / subquantizers;
        const std::size_t sampleLimit =
            std::min<std::size_t>(database.size(), 2048);
        std::vector<std::size_t> sampleIndices(database.size());
        std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
        std::mt19937 rng(seed);
        std::shuffle(sampleIndices.begin(), sampleIndices.end(), rng);

        // Train PQ codebooks on subvectors of coarse-centroid residuals.
        for(std::size_t i = 0; i < sampleLimit; ++i) {
            const std::size_t databaseIndex = sampleIndices[i];
            Vector residual(database[databaseIndex].size());
            const auto& center = centers[coarse_assignments[databaseIndex]];
            for(std::size_t d = 0; d < database[databaseIndex].size(); ++d)
                residual[d] = database[databaseIndex][d] - center[d];

            for(int m = 0; m < subquantizers; ++m) {
                training[m].emplace_back(
                    residual.begin() + m * width,
                    residual.begin() + (m + 1) * width);
            }
        }

        for(int m = 0; m < subquantizers; ++m)
            codebooks.push_back(kmeans(training[m], codeCount, seed + m + 1, 5));

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
                codes[i][m] = {static_cast<int>(
                    nearestCenters(part, codebooks[m], 1).front())};
            }
        }
    }

    // Function to find candidate vectors for a given query
    std::vector<std::size_t> candidates(const Vector& query) const {
        const auto selected = nearestCenters(query, centers, probes);   // Find the nearest coarse cluster centers to the query vector
        std::vector<ScoredId> ranked;
        const std::size_t width = query.size() / subquantizers;

        // Build one residual-PQ lookup table for each selected coarse cell.
        std::vector<std::size_t> result;
        for(const auto center : selected) {
            Vector query_residual(query.size());
            for(std::size_t d = 0; d < query.size(); ++d)
                query_residual[d] = query[d] - centers[center][d];

            std::vector<std::vector<double>> lookup(subquantizers);
            for(int m = 0; m < subquantizers; ++m) {
                Vector part(
                    query_residual.begin() + m * width,
                    query_residual.begin() + (m + 1) * width);
                for(const auto& centroid : codebooks[m])
                    lookup[m].push_back(Euclidiandistance(part, centroid));
            }

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

void evaluateAndWriteResults(const std::string& outputPath, const std::string& methodName, const std::vector<SearchResult>& results) {
    // Open the output file for writing and check if it was successful
    std::ofstream output(outputPath);
    if(!output) 
        throw std::runtime_error("could not open output file: " + outputPath);
    
    if(results.empty()) 
        return;

    // Initialize accumulators for recall, average precision, and query time
    double recall5 = 0.0, recall10 = 0.0, ap10 = 0.0, time = 0.0;
    for(const auto& result : results) {
        recall5 += result.recall_5;
        recall10 += result.recall_10;
        ap10 += result.ap_10;
        time += result.query_time_ms;
    }
    const double count = static_cast<double>(results.size());   // Number of queries processed
    output << std::fixed << std::setprecision(4);               // Set output formatting for floating-point numbers
    // Write the results for each query to the output file as specified in the project requirements
    for(const auto& result : results) {
        output << methodName << "\n\nQuery: " << result.query_id << "\n\n";
        for(std::size_t i = 0; i < result.top_10_ids.size(); ++i)
            output << "Rank-" << i + 1 << ": " << result.top_10_ids[i] << "\n";
        output << "\nRecall@5: " << result.recall_5
               << "\nRecall@10: " << result.recall_10
               << "\nAP@10: " << result.ap_10
               << "\n\nMean Recall@5: " << recall5 / count
               << "\n\nMean Recall@10: " << recall10 / count
               << "\n\nmAP@10: " << ap10 / count
               << "\n\nAverage query time: " << time / count << "\n\n";
    }
}

} // namespace image_search