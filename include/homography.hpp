#ifndef HOMOGRAPHY_HPP
#define HOMOGRAPHY_HPP

#include "local_features.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

namespace image_search {

using Matrix3 = std::array<double, 9>;

struct PointCorrespondence {
    Point2 source;
    Point2 destination;
};

struct HomographyResult {
    bool valid = false;
    Matrix3 matrix{};
    std::vector<std::size_t> inlier_indices;
    std::optional<double> median_error;
};

// four-point RANSAC sample check
// repeated points or collinear triple in either image
bool is_degenerate_sample(
    const std::vector<PointCorrespondence>& sample);

// normalized DLT result for a set of point correspondences
// repeated points allowed for refits
HomographyResult estimate_homography(
    const std::vector<PointCorrespondence>& correspondences);

// projection result with denominator and finiteness checks
std::optional<Point2> project_point(const Matrix3& homography,
                                    const Point2& point);

}  // namespace image_search

#endif
