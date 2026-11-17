#include "evaluation/verification_output.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <utility>

namespace image_search {
namespace {

// printed for a rank a short TopC did not fill
const std::string kMissingRank = "-";
// printed for a missing median or mean corner error
const std::string kNotAvailable = "NA";

// summary lines of these queries, each name preceded by prefix
void write_summary(std::ofstream& output, const std::vector<QueryEvaluation>& queries,
                   std::size_t candidate_count, const std::string& prefix) {
    const VerificationSummary summary = summarize_queries(queries);
    output << prefix << "MeanBeforeRecall@5: " << summary.before.recall_5 << "\n"
           << prefix << "MeanBeforeRecall@10: " << summary.before.recall_10 << "\n"
           << prefix << "MeanBeforeMAP@10: " << summary.before.ap_10 << "\n"
           << prefix << "MeanAfterRecall@5: " << summary.after.recall_5 << "\n"
           << prefix << "MeanAfterRecall@10: " << summary.after.recall_10 << "\n"
           << prefix << "MeanAfterMAP@10: " << summary.after.ap_10 << "\n"
           << prefix << "AverageCandidateTimeMs: " << summary.candidate_time_ms << "\n"
           << prefix << "AverageVerificationTimeMs: " << summary.verification_time_ms << "\n"
           << prefix << "AverageTotalTimeMs: " << summary.total_time_ms << "\n"
           << prefix << "ValidHomographyRate: " << summary.accuracy.valid_rate << "\n"
           << prefix << "AccurateHomographyRate5px: " << summary.accuracy.accurate_rates[2]
           << "\n"
           << prefix << "MeanCornerErrorPx: ";
    if(summary.accuracy.mean_corner_error)
        output << *summary.accuracy.mean_corner_error << "\n";
    else
        output << kNotAvailable << "\n";
    // 1 and 3 px thresholds, then the match-count ranking
    output << prefix << "AccurateHomographyRate1px: " << summary.accuracy.accurate_rates[0]
           << "\n"
           << prefix << "AccurateHomographyRate3px: " << summary.accuracy.accurate_rates[1]
           << "\n"
           << prefix << "MeanMatchCountRecall@5: " << summary.match_count.recall_5 << "\n"
           << prefix << "MeanMatchCountRecall@10: " << summary.match_count.recall_10 << "\n"
           << prefix << "MeanMatchCountMAP@10: " << summary.match_count.ap_10 << "\n";
    // |TopC(Q)| < C: the method returned fewer candidates
    const auto below = std::count_if(queries.begin(), queries.end(),
                                     [&](const QueryEvaluation& query) {
                                         return query.candidate_count < candidate_count;
                                     });
    output << prefix << "QueriesBelowC: " << below << "\n";
}

}  // namespace

void write_verification_results(const std::string& output_path,
                                const std::string& method_name,
                                std::size_t candidate_count,
                                const std::vector<QueryEvaluation>& queries) {
    // means and the 5 |Q| pair denominator need at least one query
    if(queries.empty())
        throw std::invalid_argument("verification output needs at least one query");
    if(candidate_count == 0)
        throw std::invalid_argument("verification output needs a positive C");
    std::ofstream output(output_path);
    if(!output)
        throw std::runtime_error("could not open output file: " + output_path);

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
               << "After-AP@10: " << query.after_metrics.ap_10 << "\n"
               // T_total = T_candidates + T_verify of this query
               << "CandidateTimeMs: " << query.candidate_time_ms << "\n"
               << "VerificationTimeMs: " << query.verification_time_ms << "\n"
               << "TotalTimeMs: " << query.total_time_ms << "\n"
               << "MatchCount-Recall@5: " << query.match_count_metrics.recall_5 << "\n"
               << "MatchCount-Recall@10: " << query.match_count_metrics.recall_10 << "\n"
               << "MatchCount-AP@10: " << query.match_count_metrics.ap_10 << "\n\n";
    }

    write_summary(output, queries, candidate_count, "");
    // the same lines per sequence type, after the summary of all queries
    for(const auto& [prefix, group] : {std::pair{"i_", "Illumination-"},
                                       std::pair{"v_", "Viewpoint-"}}) {
        std::vector<QueryEvaluation> members;
        for(const auto& query : queries)
            if(query.sequence.rfind(prefix, 0) == 0)
                members.push_back(query);
        if(members.empty())
            continue;
        output << group << "Queries: " << members.size() << "\n";
        write_summary(output, members, candidate_count, group);
    }
}

}  // namespace image_search
