#include "search.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace image_search {

// writes per-query results and the aggregate means in the output format required by the assignment
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
    // Write the results for each query to the output file as specified in the project requirements
    for(const auto& result : results) {
        output << methodName << "\n\nQuery: " << result.query_id << "\n\n";
        for(std::size_t i = 0; i < result.top_10_ids.size(); ++i)
            output << "Rank-" << i + 1 << ": " << result.top_10_ids[i] << "\n";
        output << "\nRecall@5: " << result.recall_5
               << "\nRecall@10: " << result.recall_10
               << "\nAP@10: " << result.ap_10 << "\n\n";
    }

    // aggregate means over all queries
    output << "Mean Recall@5: " << recall5 / count
           << "\n\nMean Recall@10: " << recall10 / count
           << "\nmAP@10: " << ap10 / count
           << "\n\nAverage query time: " << time / count << "\n\n";
}

} // namespace image_search
