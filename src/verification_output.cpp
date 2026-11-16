#include "verification_output.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace image_search {
namespace {

// printed for a rank a short TopC did not fill
const std::string kMissingRank = "-";
// printed for a missing median or mean corner error
const std::string kNotAvailable = "NA";

}  // namespace

void write_verification_results(const std::string& output_path,
                                const std::string& method_name,
                                const std::vector<QueryEvaluation>& queries) {
    // means and the 5 |Q| pair denominator need at least one query
    if(queries.empty())
        throw std::invalid_argument("verification output needs at least one query");
    std::ofstream output(output_path);
    if(!output)
        throw std::runtime_error("could not open output file: " + output_path);

    // sums over all queries, and every relevant pair for the rates
    RetrievalMetrics before_sum;
    RetrievalMetrics after_sum;
    double candidate_time = 0.0, verification_time = 0.0, total_time = 0.0;
    std::vector<PairEvaluation> pairs;

    output << std::fixed << std::setprecision(4);
    output << method_name << "\n";
    for(const auto& query : queries) {
        output << "Query: " << query.sequence << "\n"
               << "Candidates: " << query.candidate_count << "\n";
        for(std::size_t i = 0; i < 10; ++i)
            output << "Before-Rank-" << i + 1 << ": "
                   << (i < query.before_ids.size() ? query.before_ids[i] : kMissingRank)
                   << "\n";
        for(std::size_t i = 0; i < 10; ++i) {
            const std::string rank = std::to_string(i + 1);
            // no image at this rank: no inliers or median either
            if(i >= query.after.size()) {
                output << "After-Rank-" << rank << ": " << kMissingRank << "\n"
                       << "After-Inliers-" << rank << ": " << kMissingRank << "\n"
                       << "After-MedianError-" << rank << ": " << kMissingRank << "\n";
                continue;
            }
            const auto& verification = query.after[i];
            output << "After-Rank-" << rank << ": " << verification.candidate.id << "\n"
                   // S(I), zero on RANSAC failure
                   << "After-Inliers-" << rank << ": "
                   << final_inlier_count(verification) << "\n"
                   << "After-MedianError-" << rank << ": ";
            // NA without a valid homography, never 0
            if(verification.homography.valid && verification.homography.median_error)
                output << *verification.homography.median_error << "\n";
            else
                output << kNotAvailable << "\n";
        }
        output << "Before-Recall@5: " << query.before_metrics.recall_5 << "\n"
               << "Before-Recall@10: " << query.before_metrics.recall_10 << "\n"
               << "Before-AP@10: " << query.before_metrics.ap_10 << "\n"
               << "After-Recall@5: " << query.after_metrics.recall_5 << "\n"
               << "After-Recall@10: " << query.after_metrics.recall_10 << "\n"
               << "After-AP@10: " << query.after_metrics.ap_10 << "\n\n";

        before_sum.recall_5 += query.before_metrics.recall_5;
        before_sum.recall_10 += query.before_metrics.recall_10;
        before_sum.ap_10 += query.before_metrics.ap_10;
        after_sum.recall_5 += query.after_metrics.recall_5;
        after_sum.recall_10 += query.after_metrics.recall_10;
        after_sum.ap_10 += query.after_metrics.ap_10;
        candidate_time += query.candidate_time_ms;
        verification_time += query.verification_time_ms;
        total_time += query.total_time_ms;
        pairs.insert(pairs.end(), query.pairs.begin(), query.pairs.end());
    }

    // mAP@10 = 1/|Q| sum AP@10(Q)
    const double count = static_cast<double>(queries.size());
    output << "MeanBeforeRecall@5: " << before_sum.recall_5 / count << "\n"
           << "MeanBeforeRecall@10: " << before_sum.recall_10 / count << "\n"
           << "MeanBeforeMAP@10: " << before_sum.ap_10 / count << "\n"
           << "MeanAfterRecall@5: " << after_sum.recall_5 / count << "\n"
           << "MeanAfterRecall@10: " << after_sum.recall_10 / count << "\n"
           << "MeanAfterMAP@10: " << after_sum.ap_10 / count << "\n"
           << "AverageCandidateTimeMs: " << candidate_time / count << "\n"
           << "AverageVerificationTimeMs: " << verification_time / count << "\n"
           << "AverageTotalTimeMs: " << total_time / count << "\n";

    // rates over 5 |Q| pairs, mean over valid pairs only
    const HomographyAccuracy accuracy = summarize_homography_accuracy(pairs);
    output << "ValidHomographyRate: " << accuracy.valid_rate << "\n"
           << "AccurateHomographyRate5px: " << accuracy.accurate_rates[2] << "\n"
           << "MeanCornerErrorPx: ";
    if(accuracy.mean_corner_error)
        output << *accuracy.mean_corner_error << "\n";
    else
        output << kNotAvailable << "\n";
    // 1 and 3 px thresholds of the experimental study
    output << "AccurateHomographyRate1px: " << accuracy.accurate_rates[0] << "\n"
           << "AccurateHomographyRate3px: " << accuracy.accurate_rates[1] << "\n";
}

}  // namespace image_search
