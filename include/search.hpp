#ifndef SEARCH_HPP
#define SEARCH_HPP

#include <vector>
#include <string>
#include <cstddef>
#include <memory>

namespace image_search {

// outcome of one query: the returned ranking, its quality metrics and the time it took
struct SearchResult {
    std::string query_id;
    // fewer than 10 if the index returned fewer candidates
    std::vector<std::string> top_10_ids;
    double recall_5;
    double recall_10;
    double ap_10;
    // search time in milliseconds, excluding feature extraction and metric computation
    double query_time_ms;
};

// returns the euclidean distance between two vectors of equal dimension
double computeEuclideanDistance(const std::vector<double>& a,
                                const std::vector<double>& b);

// exhaustive search that compares the query against every database vector
class ExactSearch {
private:
    // implementation details are hidden in search.cpp (pimpl idiom)
    struct Impl;
    // shared pointer keeps the class cheap to copy while sharing one index
    std::shared_ptr<Impl> impl;

public:
    // Ιnitialization of the ExactSearch class with database BoW vectors and their identifiers
    ExactSearch(const std::vector<std::vector<double>>& db_bows,
                const std::vector<std::string>& db_ids);

    // returns the exact top 10 neighbours of the query together with its metrics
    SearchResult search(const std::vector<double>& query_bow, const std::string& query_id, const std::string& sequence_id);
};

// locality sensitive hashing with L hash tables, each using k hash functions of width w
class LSHSearch {
private:
    struct Impl;
    std::shared_ptr<Impl> impl;
    
public:
    // Initialization of the LSHSearch class with database BoW vectors, their identifiers, and LSH parameters
    LSHSearch(const std::vector<std::vector<double>>& db_bows, const std::vector<std::string>& db_ids, int k, int tables, double width, int seed = 1);
    
    // returns the approximate top 10 neighbours found among the index candidates, plus metrics
    SearchResult search(const std::vector<double>& query_bow, const std::string& query_id, const std::string& sequence_id) const;
};

// random projection to a hypercube, where each vertex of the cube is a bucket
class HypercubeSearch {
private:
    struct Impl;
    std::shared_ptr<Impl> impl;
    
public:
    // Initialization of the HypercubeSearch class with database BoW vectors, their identifiers, and hypercube parameters
    HypercubeSearch(const std::vector<std::vector<double>>& db_bows, const std::vector<std::string>& db_ids, int projections, double width, int max_candidates, int probes, int seed = 1);
    
    // returns the approximate top 10 neighbours found among the index candidates, plus metrics
    SearchResult search(const std::vector<double>& query_bow, const std::string& query_id, const std::string& sequence_id) const;

};

// inverted file index that stores full vectors in the lists of their nearest k-means cluster
class IVFFlatSearch {
private:
    struct Impl;
    std::shared_ptr<Impl> impl;

public:
    // Initialization of the IVFFlatSearch class with database BoW vectors, their identifiers, and IVF parameters
    IVFFlatSearch(const std::vector<std::vector<double>>& db_bows, const std::vector<std::string>& db_ids, int clusters, int probes, int seed = 1);
    
    // returns the approximate top 10 neighbours found among the index candidates, plus metrics
    SearchResult search(const std::vector<double>& query_bow, const std::string& query_id, const std::string& sequence_id) const;

};

// inverted file index that stores product-quantized residuals and ranks candidates by approximate distance
class IVFPQSearch {
private:
    struct Impl;
    std::shared_ptr<Impl> impl;

public:
    // Initialization of the IVFPQSearch class with database BoW vectors, their identifiers, and IVFPQ parameters
    IVFPQSearch(const std::vector<std::vector<double>>& db_bows, const std::vector<std::string>& db_ids, int clusters, int probes, int subquantizers, int bits, int seed = 1);
    
    // returns the approximate top 10 neighbours found among the index candidates, plus metrics
    SearchResult search(const std::vector<double>& query_bow, const std::string& query_id, const std::string& sequence_id) const;
};

// writes every query's ranking and metrics followed by the mean metrics to the output file
void evaluateAndWriteResults(
    const std::string& output_path,
    const std::string& method_name,
    const std::vector<SearchResult>& results);

} // namespace image_search

#endif