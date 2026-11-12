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

}  // namespace
}  // namespace image_search