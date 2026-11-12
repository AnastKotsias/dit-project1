#include "homography.hpp"

#include <cmath>

namespace image_search {
namespace {

// minimum valid homogeneous denominator
constexpr double kMinimumProjectionDenominator = 1e-12;

bool finite_point(const Point2& point) {
    return std::isfinite(point[0]) && std::isfinite(point[1]);
}

}  // namespace

std::optional<Point2> project_point(const Matrix3& homography,
                                    const Point2& point) {
    // reject non-finite inputs before homogeneous arithmetic
    if(!finite_point(point))
        return std::nullopt;
    for(const double value : homography) {
        if(!std::isfinite(value))
            return std::nullopt;
    }

    // homogeneous projection denominator
    const double denominator = homography[6] * point[0] +
                               homography[7] * point[1] +
                               homography[8];
    // points at or too close to infinity cannot be projected reliably
    if(!std::isfinite(denominator) ||
       std::abs(denominator) <= kMinimumProjectionDenominator) {
        return std::nullopt;
    }

    // Cartesian coordinates after homogeneous division
    const double x = (homography[0] * point[0] +
                      homography[1] * point[1] + homography[2]) /
                     denominator;
    const double y = (homography[3] * point[0] +
                      homography[4] * point[1] + homography[5]) /
                     denominator;
    if(!std::isfinite(x) || !std::isfinite(y))
        return std::nullopt;
    return Point2{x, y};
}

}  // namespace image_search
