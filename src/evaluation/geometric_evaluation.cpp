#include "evaluation/geometric_evaluation.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace image_search {

std::filesystem::path ground_truth_homography_path(
    const std::string& hp_dir, const std::string& sequence, int image_number) {
    // only I2, ..., I6 are relevant to the query I1
    if(image_number < 2 || image_number > 6)
        throw std::invalid_argument("HPatches ground truth exists only for images 2 to 6");
    return std::filesystem::path(hp_dir) / sequence /
           ("H_1_" + std::to_string(image_number));
}

Matrix3 load_ground_truth_homography(const std::filesystem::path& path) {
    std::ifstream file(path);
    if(!file)
        throw std::runtime_error("could not open ground truth homography " + path.string());

    // 3 rows of 3 numbers, any whitespace between them
    Matrix3 homography{};
    for(double& value : homography) {
        if(!(file >> value) || !std::isfinite(value))
            throw std::runtime_error("ground truth homography " + path.string() +
                                     " does not hold 9 finite numbers");
    }
    // only whitespace may follow the 9th number
    std::string extra;
    if(file >> extra)
        throw std::runtime_error("ground truth homography " + path.string() +
                                 " holds more than 9 numbers");
    return homography;
}

std::array<Point2, 4> image_corners(std::size_t width, std::size_t height) {
    if(width == 0 || height == 0)
        throw std::invalid_argument("image corners need a non-empty image");
    // last pixel center at (w-1, h-1)
    const double right = static_cast<double>(width - 1);
    const double bottom = static_cast<double>(height - 1);
    return {{{0.0, 0.0}, {right, 0.0}, {right, bottom}, {0.0, bottom}}};
}

std::optional<double> corner_error(const Matrix3& estimated,
                                   const Matrix3& ground_truth,
                                   std::size_t width, std::size_t height) {
    double total = 0.0;
    for(const auto& corner : image_corners(width, height)) {
        // pi(H_GT a_i), the true position of a_i in Ij
        const auto expected = project_point(ground_truth, corner);
        if(!expected.has_value())
            throw std::runtime_error("ground truth homography does not project an image corner");
        // pi(H a_i), checked: |c| > eps ||H_3|| ||a_i||
        const auto projected = project_point(estimated, corner);
        if(!projected.has_value())
            return std::nullopt;
        // ||pi(H a_i) - pi(H_GT a_i)||_2
        const double dx = (*projected)[0] - (*expected)[0];
        const double dy = (*projected)[1] - (*expected)[1];
        total += std::sqrt(dx * dx + dy * dy);
    }
    // mean over the 4 corners
    return total / 4.0;
}

std::array<PairEvaluation, 5> evaluate_relevant_pairs(
    const std::vector<CandidateVerification>& ranking,
    const std::string& hp_dir, const std::string& sequence,
    std::size_t query_width, std::size_t query_height) {
    std::array<PairEvaluation, 5> pairs;
    for(int j = 2; j <= 6; ++j) {
        PairEvaluation& pair = pairs[static_cast<std::size_t>(j - 2)];
        pair.image_number = j;
        // R(Q) = {sequence/2, ..., sequence/6}
        const std::string id = sequence + "/" + std::to_string(j);
        const auto found = std::find_if(
            ranking.begin(), ranking.end(),
            [&](const CandidateVerification& verification) {
                return verification.candidate.id == id;
            });
        // Ij not in TopC(Q): not verified, counts as a failure
        if(found == ranking.end())
            continue;
        pair.retrieved = true;
        // RANSAC failure: invalid and not accurate, no corner error
        if(!found->homography.valid)
            continue;

        // H_GT enters only here, after verification
        const Matrix3 ground_truth = load_ground_truth_homography(
            ground_truth_homography_path(hp_dir, sequence, j));
        // a_i corners of I1, nullopt if pi(H a_i) is not finite
        pair.corner_error = corner_error(found->homography.matrix, ground_truth,
                                         query_width, query_height);
    }
    return pairs;
}

QueryEvaluation evaluate_query(
    const CandidateSearchResult& candidates,
    const VerificationResult& verification,
    const std::string& hp_dir, const std::string& sequence,
    std::size_t query_width, std::size_t query_height) {
    // reranking only reorders TopC(Q)
    if(verification.ranking.size() != candidates.candidates.size())
        throw std::invalid_argument("verification does not hold every TopC candidate");

    QueryEvaluation result;
    result.sequence = sequence;
    result.candidate_count = candidates.candidates.size();
    // first 10, or all of them when fewer
    const std::size_t shown = std::min<std::size_t>(result.candidate_count, 10);
    std::vector<std::string> after_ids;
    for(std::size_t rank = 0; rank < shown; ++rank) {
        result.before_ids.push_back(candidates.candidates[rank].id);
        result.after.push_back(verification.ranking[rank]);
        after_ids.push_back(verification.ranking[rank].candidate.id);
    }
    // same Recall/AP definitions before and after
    result.before_metrics = computeRetrievalMetrics(result.before_ids, sequence);
    result.after_metrics = computeRetrievalMetrics(after_ids, sequence);
    // same candidates by Lowe matches only, untimed: not part of T_verify
    const auto by_matches = rank_by_match_count(verification.ranking);
    std::vector<std::string> match_count_ids;
    for(std::size_t rank = 0; rank < shown; ++rank)
        match_count_ids.push_back(by_matches[rank].candidate.id);
    result.match_count_metrics = computeRetrievalMetrics(match_count_ids, sequence);

    // T_total(Q) = T_candidates(Q) + T_verify(Q)
    result.candidate_time_ms = candidates.query_time_ms;
    result.verification_time_ms = verification.verification_time_ms;
    result.total_time_ms = result.candidate_time_ms + result.verification_time_ms;

    // all C candidates: a relevant Ij may lie below rank 10
    result.pairs = evaluate_relevant_pairs(verification.ranking, hp_dir, sequence,
                                           query_width, query_height);
    return result;
}

HomographyAccuracy summarize_homography_accuracy(
    const std::vector<PairEvaluation>& pairs) {
    // denominator of both rates
    if(pairs.empty())
        throw std::invalid_argument("homography accuracy needs at least one relevant pair");

    std::size_t valid = 0;
    std::array<std::size_t, 3> accurate{};
    double error_sum = 0.0;
    for(const auto& pair : pairs) {
        // outside TopC or RANSAC failure: invalid, not accurate
        if(!pair.corner_error.has_value())
            continue;
        ++valid;
        error_sum += *pair.corner_error;
        // strict E_corner < threshold
        for(std::size_t t = 0; t < kCornerErrorThresholds.size(); ++t) {
            if(*pair.corner_error < kCornerErrorThresholds[t])
                ++accurate[t];
        }
    }

    HomographyAccuracy result;
    const double total = static_cast<double>(pairs.size());
    result.valid_rate = static_cast<double>(valid) / total;
    for(std::size_t t = 0; t < accurate.size(); ++t)
        result.accurate_rates[t] = static_cast<double>(accurate[t]) / total;
    // mean over valid pairs only, NA when there are none
    if(valid > 0)
        result.mean_corner_error = error_sum / static_cast<double>(valid);
    return result;
}

}  // namespace image_search
