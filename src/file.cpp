#include "geometric_verification.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace image_search {
namespace {

constexpr std::size_t kMinimumSampleSize = 4;

// Euclidean transfer error in destination-image pixels
double reprojection_error(const DescriptorMatch& match,
                          const Matrix3& homography) {
    // points at infinity cannot produce a finite inlier error
    const auto projected = project_point(homography, match.query_point);
    if(!projected.has_value())
        return std::numeric_limits<double>::infinity();

    // compare the projected query point with its matched candidate point
    const double dx = (*projected)[0] - match.candidate_point[0];
    const double dy = (*projected)[1] - match.candidate_point[1];
    return std::sqrt(dx * dx + dy * dy);
}

// median transfer error among one hypothesis's inliers
std::optional<double> median_inlier_error(
    const std::vector<double>& errors,
    const std::vector<std::size_t>& inlier_indices) {
    if(inlier_indices.empty())
        return std::nullopt;

    // collect only inlier errors because outliers must not affect the tie-break
    std::vector<double> inlier_errors;
    inlier_errors.reserve(inlier_indices.size());
    for(const std::size_t index : inlier_indices)
        inlier_errors.push_back(errors[index]);
    std::sort(inlier_errors.begin(), inlier_errors.end());
    const std::size_t middle = inlier_errors.size() / 2;
    if(inlier_errors.size() % 2 == 1)
        return inlier_errors[middle];
    return (inlier_errors[middle - 1] + inlier_errors[middle]) / 2.0;
}

// update the maximum number of trials from the current inlier ratio
std::size_t adaptive_limit(std::size_t current_limit,
                           std::size_t inlier_count,
                           std::size_t match_count,
                           double confidence,
                           std::size_t completed_trials) {
    const double inlier_ratio =
        static_cast<double>(inlier_count) / static_cast<double>(match_count);
    // zero inliers provide no estimate of the probability of a clean sample
    if(inlier_ratio <= 0.0)
        return current_limit;
    // a perfect hypothesis needs no additional trials
    if(inlier_ratio >= 1.0)
        return completed_trials;

    // four correspondences are needed for a homography hypothesis
    const double sample_probability = std::pow(inlier_ratio, 4.0);
    // log1p avoids loss of precision when the sample probability is small
    const double denominator = std::log1p(-sample_probability);
    const double numerator = std::log1p(-confidence);
    if(!std::isfinite(numerator) || !std::isfinite(denominator) ||
       denominator == 0.0)
        return current_limit;

    // invert p = 1 - (1 - w^4)^T and round upward
    const double required_trials = std::ceil(numerator / denominator);
    if(!std::isfinite(required_trials) || required_trials < 1.0)
        return current_limit;
    const auto required = static_cast<std::size_t>(
        std::min(required_trials,
                 static_cast<double>(std::numeric_limits<std::size_t>::max())));
    return std::min(current_limit, std::max(required, completed_trials));
}

}  // namespace

HomographyResult estimate_homography_ransac(
    const std::vector<DescriptorMatch>& matches,
    const RansacConfig& config) {
    if(!std::isfinite(config.tau) || config.tau <= 0.0)
        throw std::invalid_argument("tau must be finite and positive");
    if(config.maximum_trials == 0)
        throw std::invalid_argument("maximum_trials must be positive");
    if(!std::isfinite(config.confidence) || config.confidence <= 0.0 ||
       config.confidence >= 1.0)
        throw std::invalid_argument(
            "confidence must be finite and strictly between 0 and 1");

    HomographyResult failure;
    // RANSAC cannot form its minimum projective sample otherwise
    if(matches.size() < kMinimumSampleSize)
        return failure;

    std::mt19937_64 generator(config.seed);
    // the adaptive limit can only reduce the configured maximum
    std::size_t trial_limit = config.maximum_trials;
    bool has_best = false;
    HomographyResult best;
    double best_median = std::numeric_limits<double>::infinity();

    // each iteration proposes one homography hypothesis
    for(std::size_t trial = 0; trial < trial_limit; ++trial) {
        // sample four distinct accepted matches without replacement
        std::vector<std::size_t> sample_indices;
        sample_indices.reserve(kMinimumSampleSize);
        std::uniform_int_distribution<std::size_t> distribution(
            0, matches.size() - 1);
        while(sample_indices.size() < kMinimumSampleSize) {
            const std::size_t index = distribution(generator);
            if(std::find(sample_indices.begin(), sample_indices.end(), index) ==
               sample_indices.end())
                sample_indices.push_back(index);
        }
    }

}

}  // namespace image_search
