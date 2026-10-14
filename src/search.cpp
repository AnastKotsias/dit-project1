#include "search.hpp"
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>

namespace image_search {

// Helper function to compute Euclidean distance between two vectors
double computeEuclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

SearchResult ExactSearch::search(const std::vector<double>& query_bow, const std::string& query_id, const std::string& sequence_id) {
    // To compute the search time, we can use std::chrono to measure the duration of the search process.
    auto start_time = std::chrono::high_resolution_clock::now();

    // Compute distances from the query Bow vector to all database Bow vectors
    std::vector<std::pair<double, std::string>> distances;
    distances.reserve(database_bows.size());

    for (size_t i = 0; i < database_bows.size(); ++i) {
        double dist = computeEuclideanDistance(query_bow, database_bows[i]);
        distances.push_back({dist, database_ids[i]});
    }

    // Sort the distances to find the closest matches
    std::sort(distances.begin(), distances.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // Select the top 10 closest matches
    std::vector<std::string> top_10;
    int limit = std::min(10, (int)distances.size());
    for (int i = 0; i < limit; ++i) {
        top_10.push_back(distances[i].second);
    }

    // Measure the end time of the search
    auto end_time = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // Create a set of relevant database image identifiers for the given query sequence
    std::set<std::string> relevant_set;
    for (int i = 2; i <= 6; ++i) {
        relevant_set.insert(sequence_id + "/" + std::to_string(i));
    }

    // Calculate recall@5, recall@10, and AP@10
    int matches_5 = 0;
    int matches_10 = 0;
    
    for (int i = 0; i < limit; ++i) {
        if (relevant_set.count(top_10[i])) {
            if (i < 5) matches_5++;
            matches_10++;
        }
    }

    double recall_5 = static_cast<double>(matches_5) / 5.0;
    double recall_10 = static_cast<double>(matches_10) / 5.0;

    // Calculate Average Precision at 10 (AP@10)
    double ap_sum = 0.0;
    int found_count = 0;
    for (int i = 0; i < limit; ++i) {
        if (relevant_set.count(top_10[i])) {
            found_count++;
            double rank = i + 1.0;
            ap_sum += static_cast<double>(found_count) / rank;
        }
    }
    double ap_10 = ap_sum / 5.0;

    return {query_id, top_10, recall_5, recall_10, ap_10, duration_ms};
}

// Implementation of the evaluateAndWriteResults function
void evaluateAndWriteResults(
    const std::string& output_path,
    const std::string& method_name,
    const std::vector<SearchResult>& results) 
{
    std::ofstream outfile(output_path);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open output file for writing: " << output_path << std::endl;
        return;
    }

    if (results.empty()) {
        std::cerr << "Warning: No search results provided to write." << std::endl;
        return;
    }

    // Calculating mean metrics across all queries
    double total_recall_5 = 0.0;
    double total_recall_10 = 0.0;
    double total_ap_10 = 0.0;
    double total_time_ms = 0.0;

    for (const auto& res : results) {
        total_recall_5 += res.recall_5;
        total_recall_10 += res.recall_10;
        total_ap_10 += res.ap_10;
        total_time_ms += res.query_time_ms;
    }

    size_t num_queries = results.size();
    double mean_recall_5 = total_recall_5 / num_queries;
    double mean_recall_10 = total_recall_10 / num_queries;
    double map_10 = total_ap_10 / num_queries;
    double avg_query_time = total_time_ms / num_queries;

    // Setting precision for floats/doubles
    outfile << std::fixed << std::setprecision(4);

    // Writing results for each query according to the format
    for (const auto& res : results) {
        outfile << method_name << "\n\n";
        outfile << "Query: " << res.query_id << "\n\n";

        for (size_t i = 0; i < res.top_10_ids.size(); ++i) {
            outfile << "Rank-" << (i + 1) << ": " << res.top_10_ids[i] << "\n";
        }
        outfile << "\n";

        outfile << "Recall@5: " << res.recall_5 << "\n";
        outfile << "Recall@10: " << res.recall_10 << "\n";
        outfile << "AP@10: " << res.ap_10 << "\n\n";

        outfile << "Mean Recall@5: " << mean_recall_5 << "\n\n";
        outfile << "Mean Recall@10: " << mean_recall_10 << "\n\n";
        outfile << "mAP@10: " << map_10 << "\n\n";

        outfile << "Average query time: " << avg_query_time << "\n\n";
    }
    // Debugging message to indicate successful writing of results
    // std::cout << "Results successfully written to " << output_path << std::endl;
}

} // namespace image_search