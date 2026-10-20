#ifndef SEARCH_HPP
#define SEARCH_HPP

#include <vector>
#include <string>
#include <cstddef>
#include <memory>

namespace image_search {

struct SearchResult {
    std::string query_id;
    std::vector<std::string> top_10_ids;
    double recall_5;
    double recall_10;
    double ap_10;
    double query_time_ms;
};

double computeEuclideanDistance(const std::vector<double>& a,
                                const std::vector<double>& b);

class ExactSearch {
private:
    struct Impl;
    std::shared_ptr<Impl> impl;

public:
    // Ιnitialization of the ExactSearch class with database BoW vectors and their identifiers
    ExactSearch(const std::vector<std::vector<double>>& db_bows,
                const std::vector<std::string>& db_ids);

    // Search function
    SearchResult search(const std::vector<double>& query_bow, const std::string& query_id, const std::string& sequence_id);
};

class LSHSearch {
private:
    struct Impl;
    std::shared_ptr<Impl> impl;
    
public:
    // Initialization of the LSHSearch class with database BoW vectors, their identifiers, and LSH parameters
    LSHSearch(const std::vector<std::vector<double>>& db_bows, const std::vector<std::string>& db_ids, int k, int tables, double width, int seed = 1);
    
    SearchResult search(const std::vector<double>& query_bow, const std::string& query_id, const std::string& sequence_id) const;
};

class HypercubeSearch {
private:
    struct Impl;
    std::shared_ptr<Impl> impl;
    
public:
    // Initialization of the HypercubeSearch class with database BoW vectors, their identifiers, and hypercube parameters
    HypercubeSearch(const std::vector<std::vector<double>>& db_bows, const std::vector<std::string>& db_ids, int projections, double width, int max_candidates, int probes, int seed = 1);
    
    SearchResult search(const std::vector<double>& query_bow, const std::string& query_id, const std::string& sequence_id) const;

};

class IVFFlatSearch {
private:
    struct Impl;
    std::shared_ptr<Impl> impl;

public:
    // Initialization of the IVFFlatSearch class with database BoW vectors, their identifiers, and IVF parameters
    IVFFlatSearch(const std::vector<std::vector<double>>& db_bows, const std::vector<std::string>& db_ids, int clusters, int probes, int seed = 1);
    
    SearchResult search(const std::vector<double>& query_bow, const std::string& query_id, const std::string& sequence_id) const;

};

class IVFPQSearch {
private:
    struct Impl;
    std::shared_ptr<Impl> impl;

public:
    // Initialization of the IVFPQSearch class with database BoW vectors, their identifiers, and IVFPQ parameters
    IVFPQSearch(const std::vector<std::vector<double>>& db_bows, const std::vector<std::string>& db_ids, int clusters, int probes, int subquantizers, int bits, int seed = 1);
    
    SearchResult search(const std::vector<double>& query_bow, const std::string& query_id, const std::string& sequence_id) const;
};

// Evaluates the search results and writes them to a specified output file
void evaluateAndWriteResults(
    const std::string& output_path,
    const std::string& method_name,
    const std::vector<SearchResult>& results);

} // namespace image_search

#endif