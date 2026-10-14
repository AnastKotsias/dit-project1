#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>
#include <string>
#include "dataset.hpp"

namespace image_search {

struct SearchResult {
    std::string query_id;
    std::vector<std::string> top_10_ids;
    double recall_5;
    double recall_10;
    double ap_10;
    double query_time_ms;
};

class ExactSearch {
private:
    std::vector<std::vector<double>> database_bows;
    std::vector<std::string> database_ids;

public:
    // Ιnitialization of the ExactSearch class with database BoW vectors and their identifiers
    ExactSearch(const std::vector<std::vector<double>>& db_bows, const std::vector<std::string>& db_ids)
        : database_bows(db_bows), database_ids(db_ids) {}

    // Search function
    SearchResult search(const std::vector<double>& query_bow, const std::string& query_id, const std::string& sequence_id);
};

// Evaluates the search results and writes them to a specified output file
void evaluateAndWriteResults(
    const std::string& output_path,
    const std::string& method_name,
    const std::vector<SearchResult>& results);

} // namespace image_search

#endif