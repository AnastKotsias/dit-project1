#ifndef GEOMETRIC_EVALUATION_HPP
#define GEOMETRIC_EVALUATION_HPP

#include "geometry/geometric_verification.hpp"
#include "geometry/homography.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace image_search {

// HPatches ground truth, used only in evaluation, never in verification

// <hp_dir>/<sequence>/H_1_j, the reference homography of the pair (I1, Ij)
// throws unless 2 <= j <= 6
std::filesystem::path ground_truth_homography_path(
    const std::string& hp_dir, const std::string& sequence, int image_number);

// H_1_j^GT, 3 x 3 row by row, maps I1 pixels to Ij pixels
// throws unless the file holds exactly 9 finite numbers
Matrix3 load_ground_truth_homography(const std::filesystem::path& path);

// a_1, ..., a_4: centers of the corner pixels of a width x height image,
// (0,0), (w-1,0), (w-1,h-1), (0,h-1), as in SIFT keypoint coordinates
// throws if width or height is zero
std::array<Point2, 4> image_corners(std::size_t width, std::size_t height);

// E_corner = 1/4 sum_i ||pi(H a_i) - pi(H_GT a_i)||_2, in Ij pixels
// a_i the corners of I1; nullopt if a corner of H does not project
// to finite coordinates (no valid estimate)
// throws if a corner of H_GT does not project
std::optional<double> corner_error(const Matrix3& estimated,
                                   const Matrix3& ground_truth,
                                   std::size_t width, std::size_t height);

// accuracy of one relevant pair (Q = I1, Ij)
struct PairEvaluation {
    // j, 2 <= j <= 6
    int image_number = 0;
    // Ij belongs to TopC(Q), so RANSAC ran on it
    bool retrieved = false;
    // 1-based ranks in TopC(Q) before and after reranking, 0 if not retrieved
    std::size_t before_rank = 0;
    std::size_t after_rank = 0;
    // M Lowe matches and S(I) final inliers, 0 if not retrieved
    std::size_t match_count = 0;
    std::size_t inlier_count = 0;
    // E_corner, set only for a valid estimate H_Q->Ij
    std::optional<double> corner_error;
};

// the 5 relevant pairs (I1, Ij), j = 2..6, of the query sequence
// ranking: the verified TopC(Q) candidates after reranking
// query_width x query_height: size of I1, for the corners a_i
// outside TopC or RANSAC failure: no corner error, never zero
// H_1_j^GT loaded only for a valid estimate
std::array<PairEvaluation, 5> evaluate_relevant_pairs(
    const std::vector<CandidateVerification>& ranking,
    const std::string& hp_dir, const std::string& sequence,
    std::size_t query_width, std::size_t query_height);

// one query Q = I1 before and after geometric reranking
struct QueryEvaluation {
    // sequence of Q, also its output identifier
    std::string sequence;
    // |TopC(Q)|, fewer than C for some ANN methods
    std::size_t candidate_count = 0;
    // first 10 of TopC(Q), original order
    std::vector<std::string> before_ids;
    // first 10 after reranking, with inliers and median error
    std::vector<CandidateVerification> after;
    RetrievalMetrics before_metrics;
    RetrievalMetrics after_metrics;
    // first 10 by Lowe match count, the ranking compared with RANSAC
    RetrievalMetrics match_count_metrics;
    // T_total = T_candidates + T_verify
    double candidate_time_ms = 0.0;
    double verification_time_ms = 0.0;
    double total_time_ms = 0.0;
    // the 5 relevant pairs (I1, Ij), from all C candidates
    std::array<PairEvaluation, 5> pairs;
};

// candidates: TopC(Q) with T_candidates
// verification: the same candidates reranked, with T_verify
// throws if verification does not hold every TopC candidate
QueryEvaluation evaluate_query(
    const CandidateSearchResult& candidates,
    const VerificationResult& verification,
    const std::string& hp_dir, const std::string& sequence,
    std::size_t query_width, std::size_t query_height);

// accurate estimate: valid with E_corner < 1, 3 or 5 px
constexpr std::array<double, 3> kCornerErrorThresholds = {1.0, 3.0, 5.0};

// homography accuracy over the relevant pairs of a set
struct HomographyAccuracy {
    // valid pairs / all pairs
    double valid_rate = 0.0;
    // accurate pairs / all pairs, one per kCornerErrorThresholds
    std::array<double, 3> accurate_rates{};
    // mean E_corner of the valid pairs, nullopt (NA) if none
    std::optional<double> mean_corner_error;
};

// pairs: the 5 relevant pairs of every query, 23 * 5 = 115 per split
// invalid pairs stay in the denominator, out of the mean
// throws if pairs is empty
HomographyAccuracy summarize_homography_accuracy(
    const std::vector<PairEvaluation>& pairs);

// summary lines of the output, over all queries or one group of them
struct VerificationSummary {
    std::size_t query_count = 0;
    // mean Recall@5, Recall@10 and mAP@10 of each ranking
    RetrievalMetrics before;
    RetrievalMetrics after;
    RetrievalMetrics match_count;
    // average times per query
    double candidate_time_ms = 0.0;
    double verification_time_ms = 0.0;
    double total_time_ms = 0.0;
    // over the 5 |Q| relevant pairs of these queries
    HomographyAccuracy accuracy;
};

// mAP@10 = 1/|Q| sum AP@10(Q), likewise the other means
// throws if queries is empty
VerificationSummary summarize_queries(const std::vector<QueryEvaluation>& queries);

}  // namespace image_search

#endif
