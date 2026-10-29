#include "search.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

namespace image_search {
// helpers below are private to this translation unit
namespace {

// printed for a rank the index did not fill, it can never be an image identifier
const std::string kMissingRank = "-";

}  // namespace

// writes per-query results and the aggregate means to the output file
void evaluateAndWriteResults(const std::string& outputPath, const std::string& methodName, const std::vector<SearchResult>& results) {
    // Open the output file for writing and check if it was successful
    std::ofstream output(outputPath);
    if(!output)
        throw std::runtime_error("could not open output file: " + outputPath);

    // nothing to report, the file is left empty
    if(results.empty())
        return;

    // Initialize accumulators for recall, average precision, and query time
    double recall5 = 0.0, recall10 = 0.0, ap10 = 0.0, time = 0.0;
    // sum the metrics over all queries
    for(const auto& result : results) {
        recall5 += result.recall_5;
        recall10 += result.recall_10;
        ap10 += result.ap_10;
        time += result.query_time_ms;
    }
    const double count = static_cast<double>(results.size());   // Number of queries processed
    output << std::fixed << std::setprecision(4);               // Set output formatting for floating-point numbers
    // Write the results for each query to the output file:
    // the method name once, then one block per query with no blank lines inside it, blocks separated by one blank line
    output << methodName << "\n";
    for(const auto& result : results) {
        output << "Query: " << result.query_id << "\n";
        // every block has Rank-1 to Rank-10 as in the required format, positions the index did not fill
        // are printed as kMissingRank and count as non-relevant in the metrics
        for(std::size_t i = 0; i < 10; ++i)
            output << "Rank-" << i + 1 << ": "
                   << (i < result.top_10_ids.size() ? result.top_10_ids[i] : kMissingRank) << "\n";
        output << "Recall@5: " << result.recall_5 << "\n"
               << "Recall@10: " << result.recall_10 << "\n"
               << "AP@10: " << result.ap_10 << "\n\n";
    }

    // aggregate means over all queries, after the blank line that ends the last query block
    output << "Mean Recall@5: " << recall5 / count << "\n"
           << "Mean Recall@10: " << recall10 / count << "\n"
           << "mAP@10: " << ap10 / count << "\n"
           << "Average query time: " << time / count << "\n";
}

} // namespace image_search
