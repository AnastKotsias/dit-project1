#ifndef SEARCH_HPP
#define SEARCH_HPP

#include <vector>
#include <string>
#include <cstddef>
#include <memory>

namespace image_search {

// every search class takes the BoVW vector of each database image and its identifier
// a constructor throws if the database is empty, the vectors have inconsistent dimensions or non-finite
// values, an identifier is empty, or a parameter is out of range (the ranges are given per class)
// search() takes the query vector, its identifier and the sequence it belongs to, which defines its
// five relevant images; it throws if the query has the wrong dimension or non-finite values
// copying a search object shares the same index, and all randomness derives from the seed

// largest hypercube dimension d', since a vertex is stored as an int key and probes are bounded by 2^d'
constexpr int kMaxHypercubeProjections = 20;
// largest IVFPQ code size in bits, since every code is stored in a uint16_t
constexpr int kMaxPQBits = 16;

// outcome of one query: the returned ranking, its quality metrics and the time it took
struct SearchResult {
    std::string query_id;
    // fewer than 10 if the index returned fewer candidates, the missing ranks count as non-relevant
    std::vector<std::string> top_10_ids;
    double recall_5 = 0.0;
    double recall_10 = 0.0;
    double ap_10 = 0.0;
    // search time in milliseconds, excluding feature extraction and metric computation
    double query_time_ms = 0.0;
    // distinct database vectors whose distance to the query was computed, exact distances for
    // every method except IVFPQ, which counts ADC distances; a deterministic measure of search cost
    std::size_t candidates_examined = 0;
};

// number of vectors the IVF k-means is trained on, ceil(sqrt(n)) for a database of n vectors
// it is also the largest number of IVF clusters that can be formed, larger requests are capped to it
std::size_t ivfTrainingSampleSize(std::size_t database_size);

// exhaustive search that compares the query with every database vector
class ExactSearch {
private:
    // the implementation is hidden in src/search/ (pimpl idiom)
    struct Impl;
    std::shared_ptr<Impl> impl;

public:
    ExactSearch(const std::vector<std::vector<double>>& db_bows,
                const std::vector<std::string>& db_ids);

    // the exact top 10 neighbours of the query, ranked by Euclidean distance
    SearchResult search(const std::vector<double>& query_bow, const std::string& query_id, const std::string& sequence_id) const;
};

// locality sensitive hashing with `tables` hash tables, each combining k hash functions of window width w
// k, tables and width must be positive and width finite
class LSHSearch {
private:
    struct Impl;
    std::shared_ptr<Impl> impl;

public:
    LSHSearch(const std::vector<std::vector<double>>& db_bows, const std::vector<std::string>& db_ids, int k, int tables, double width, int seed = 1);

    // the top 10 among the vectors sharing a bucket with the query in any table, ranked by Euclidean distance
    SearchResult search(const std::vector<double>& query_bow, const std::string& query_id, const std::string& sequence_id) const;
};

// random projection to a hypercube of `projections` dimensions, where each vertex is a bucket
// a query checks at most `probes` vertices and `max_candidates` vectors
// projections must be in 1..kMaxHypercubeProjections, the others positive and width finite
class HypercubeSearch {
private:
    struct Impl;
    std::shared_ptr<Impl> impl;

public:
    HypercubeSearch(const std::vector<std::vector<double>>& db_bows, const std::vector<std::string>& db_ids, int projections, double width, int max_candidates, int probes, int seed = 1);

    // the top 10 among the vectors of the probed vertices, ranked by Euclidean distance
    SearchResult search(const std::vector<double>& query_bow, const std::string& query_id, const std::string& sequence_id) const;
};

// inverted file index that stores the full vectors in the lists of their nearest k-means cluster
// clusters and probes must be positive, clusters is capped at ivfTrainingSampleSize
class IVFFlatSearch {
private:
    struct Impl;
    std::shared_ptr<Impl> impl;
    // number of nearest clusters searched per query, kept outside Impl since it does not affect training
    int probes;

public:
    IVFFlatSearch(const std::vector<std::vector<double>>& db_bows, const std::vector<std::string>& db_ids, int clusters, int probes, int seed = 1);

    // a copy that shares the trained index but searches the given number of clusters
    IVFFlatSearch withProbes(int probe_count) const;

    // the top 10 among the vectors of the `probes` nearest clusters, ranked by Euclidean distance
    SearchResult search(const std::vector<double>& query_bow, const std::string& query_id, const std::string& sequence_id) const;
};

// inverted file index that stores each vector as the product-quantized residual from its cluster center
// subquantizers (M) must divide the vector dimension, bits must be in 1..kMaxPQBits (2^bits centroids
// per subspace), clusters and probes must be positive, clusters is capped at ivfTrainingSampleSize
class IVFPQSearch {
private:
    struct Impl;
    std::shared_ptr<Impl> impl;
    int probes;

public:
    IVFPQSearch(const std::vector<std::vector<double>>& db_bows, const std::vector<std::string>& db_ids, int clusters, int probes, int subquantizers, int bits, int seed = 1);

    // a copy that shares the trained index but searches the given number of clusters
    IVFPQSearch withProbes(int probe_count) const;

    // the 10 vectors of the probed clusters with the smallest asymmetric distance (ADC) to the query
    // there is no reranking by exact distance
    SearchResult search(const std::vector<double>& query_bow, const std::string& query_id, const std::string& sequence_id) const;
};

// writes the method name, every query's ranking and metrics, and the mean metrics to the output file
// throws if the file cannot be opened
void evaluateAndWriteResults(
    const std::string& output_path,
    const std::string& method_name,
    const std::vector<SearchResult>& results);

} // namespace image_search

#endif
