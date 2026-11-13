#include "homography.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace image_search {
namespace {

// minimum valid homogeneous denominator
constexpr double kMinimumProjectionDenominator = 1e-12;
// minimum Frobenius norm of H
constexpr double kMinimumMatrixNorm = 1e-12;
// minimum mean centroid distance in pixels, above rounding noise
constexpr double kMinimumMeanDistance = 1e-8;
// minimum accepted squared distance between distinct points
constexpr double kMinimumPointSeparationSquared = 1e-10;
// minimum normalized triangle area for non-collinearity
constexpr double kMinimumCollinearity = 1e-4;
// epsilon_rank, minimum sigma3 / sigma1 of H
constexpr double kMinimumHomographySingularValueRatio = 1e-12;

bool finite_point(const Point2& point) {
    return std::isfinite(point[0]) && std::isfinite(point[1]);
}

struct Normalization {
    Point2 center{};
    // mean distance d from the centroid
    double mean_distance = 0.0;
    double scale = 0.0;
};

Normalization normalize_points(const std::vector<PointCorrespondence>& points,
                               bool source) {
    Normalization result;
    // Hartley normalization starts by translating the centroid to the origin
    for(const auto& correspondence : points) {
        // choose source or destination coordinates
        const Point2& point = source ? correspondence.source : correspondence.destination;
        result.center[0] += point[0];
        result.center[1] += point[1];
    }
    result.center[0] /= static_cast<double>(points.size());
    result.center[1] /= static_cast<double>(points.size());

    // mean distance d from the centroid
    for(const auto& correspondence : points) {
        const Point2& point = source ? correspondence.source : correspondence.destination;
        const double dx = point[0] - result.center[0];
        const double dy = point[1] - result.center[1];
        result.mean_distance += std::sqrt(dx * dx + dy * dy);
    }
    result.mean_distance /= static_cast<double>(points.size());
    // s = sqrt(2) / d, near-zero d rejected by the caller
    if(result.mean_distance > kMinimumMeanDistance)
        result.scale = std::sqrt(2.0) / result.mean_distance;
    return result;
}

Point2 normalized_point(const Point2& point, const Normalization& normalization) {
    // x normalized equals scale times x minus centroid
    return {(point[0] - normalization.center[0]) * normalization.scale,
            (point[1] - normalization.center[1]) * normalization.scale};
}

double triangle_cross_product(const Point2& first, const Point2& second,
                              const Point2& third) {
    return std::abs((second[0] - first[0]) * (third[1] - first[1]) -
                    (second[1] - first[1]) * (third[0] - first[0]));
}

double squared_distance(const Point2& first, const Point2& second) {
    const double dx = first[0] - second[0];
    const double dy = first[1] - second[1];
    return dx * dx + dy * dy;
}

bool has_duplicate_points(const std::vector<Point2>& points) {
    // repeated points reduce the rank of the DLT system
    for(std::size_t first = 0; first < points.size(); ++first) {
        for(std::size_t second = first + 1; second < points.size(); ++second) {
            if(squared_distance(points[first], points[second]) <
               kMinimumPointSeparationSquared)
                return true;
        }
    }
    return false;
}

bool has_collinear_triple(const std::vector<Point2>& points) {
    // collinear triples invalidate a four-point projective sample
    for(std::size_t first = 0; first < points.size(); ++first) {
        for(std::size_t second = first + 1; second < points.size(); ++second) {
            for(std::size_t third = second + 1; third < points.size(); ++third) {
                // normalize the cross product by the largest pairwise length
                const double maximum_squared_distance = std::max(
                    {squared_distance(points[first], points[second]),
                     squared_distance(points[first], points[third]),
                     squared_distance(points[second], points[third])});
                if(maximum_squared_distance <= 0.0 ||
                   triangle_cross_product(points[first], points[second],
                                          points[third]) /
                           maximum_squared_distance <
                       kMinimumCollinearity)
                    return true;
            }
        }
    }
    return false;
}

bool is_invertible_homography(const Matrix3& matrix) {
    // singular values of H itself, sigma1 >= sigma2 >= sigma3 >= 0
    // H^T H would square the condition number
    Eigen::Matrix3d homography;
    homography << matrix[0], matrix[1], matrix[2],
                  matrix[3], matrix[4], matrix[5],
                  matrix[6], matrix[7], matrix[8];
    // singular values only, no U or V
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(homography);
    const Eigen::Vector3d& singular_values = svd.singularValues();
    if(!singular_values.allFinite())
        return false;

    // sigma1 > 0 and sigma3 / sigma1 > epsilon_rank
    // scale invariant: lambda H has singular values |lambda| sigma_i
    const double largest = singular_values(0);
    const double smallest = singular_values(2);
    return largest > 0.0 &&
           smallest / largest > kMinimumHomographySingularValueRatio;
}

}  // namespace

bool is_degenerate_sample(const std::vector<PointCorrespondence>& sample) {
    // minimal sample only
    if(sample.size() != 4)
        throw std::invalid_argument("a minimal homography sample has four correspondences");

    // query-image and candidate-image points
    std::vector<Point2> source_points;
    std::vector<Point2> destination_points;
    source_points.reserve(sample.size());
    destination_points.reserve(sample.size());
    for(const auto& correspondence : sample) {
        // NaN passes every comparison below
        if(!finite_point(correspondence.source) ||
           !finite_point(correspondence.destination))
            return true;
        source_points.push_back(correspondence.source);
        destination_points.push_back(correspondence.destination);
    }

    // four distinct points per image
    if(has_duplicate_points(source_points) ||
       has_duplicate_points(destination_points))
        return true;
    // collinear triple gives rank(A) < 8
    return has_collinear_triple(source_points) ||
           has_collinear_triple(destination_points);
}

HomographyResult estimate_homography(
    const std::vector<PointCorrespondence>& correspondences) {
    HomographyResult result;
    // four point correspondences provide the minimum eight independent equations
    if(correspondences.size() < 4)
        return result;

    // finite coordinates only
    for(const auto& correspondence : correspondences) {
        if(!finite_point(correspondence.source) ||
           !finite_point(correspondence.destination))
            return result;
    }
    // repeated points only repeat rows of A

    // T and T' from these points, also on refit
    const auto source_normalization = normalize_points(correspondences, true);
    const auto destination_normalization = normalize_points(correspondences, false);
    // near-zero d gives no reliable s = sqrt(2) / d
    if(!std::isfinite(source_normalization.mean_distance) ||
       !std::isfinite(destination_normalization.mean_distance) ||
       source_normalization.mean_distance <= kMinimumMeanDistance ||
       destination_normalization.mean_distance <= kMinimumMeanDistance)
        return result;

    // A in R^(2k x 9) from normalized points T p and T' p'
    const auto rows = static_cast<Eigen::Index>(2 * correspondences.size());
    Eigen::MatrixXd dlt_matrix(rows, 9);
    Eigen::Index row = 0;
    for(const auto& correspondence : correspondences) {
        const auto source = normalized_point(correspondence.source,
                                              source_normalization);
        const auto destination = normalized_point(correspondence.destination,
                                                   destination_normalization);
        const double x = source[0];
        const double y = source[1];
        const double u = destination[0];
        const double v = destination[1];
        // u (h31 x + h32 y + h33) - (h11 x + h12 y + h13) = 0
        dlt_matrix.row(row++) << -x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u;
        // v (h31 x + h32 y + h33) - (h21 x + h22 y + h23) = 0
        dlt_matrix.row(row++) << 0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v;
    }

    // SVD A = U Sigma V^T directly on A
    // full V: for k = 4, A is 8 x 9 and thin V lacks column 9
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(dlt_matrix, Eigen::ComputeFullV);
    // h = argmin ||A h||_2, ||h||_2 = 1: last column of V (sigma9)
    // rank(A) = 8 holds: samples pass the degeneracy checks,
    // refit inliers contain the non-degenerate sample
    const Eigen::VectorXd h = svd.matrixV().col(8);
    if(!h.allFinite())
        return result;

    // h reshaped row by row into Hn
    std::array<double, 9> normalized_matrix{};
    for(Eigen::Index index = 0; index < 9; ++index)
        normalized_matrix[static_cast<std::size_t>(index)] = h(index);

    // H equals inverse destination transform times normalized H times source transform
    const double source_scale = source_normalization.scale;
    const double destination_scale = destination_normalization.scale;
    const double tx = source_normalization.center[0];
    const double ty = source_normalization.center[1];
    const double ux = destination_normalization.center[0];
    const double uy = destination_normalization.center[1];
    Matrix3 matrix = {
        normalized_matrix[0] * source_scale / destination_scale +
            ux * normalized_matrix[6] * source_scale,
        normalized_matrix[1] * source_scale / destination_scale +
            ux * normalized_matrix[7] * source_scale,
        (normalized_matrix[2] - normalized_matrix[0] * source_scale * tx -
         normalized_matrix[1] * source_scale * ty) / destination_scale +
            ux * (normalized_matrix[8] - normalized_matrix[6] * source_scale * tx -
                  normalized_matrix[7] * source_scale * ty),
        normalized_matrix[3] * source_scale / destination_scale +
            uy * normalized_matrix[6] * source_scale,
        normalized_matrix[4] * source_scale / destination_scale +
            uy * normalized_matrix[7] * source_scale,
        (normalized_matrix[5] - normalized_matrix[3] * source_scale * tx -
         normalized_matrix[4] * source_scale * ty) / destination_scale +
            uy * (normalized_matrix[8] - normalized_matrix[6] * source_scale * tx -
                  normalized_matrix[7] * source_scale * ty),
        normalized_matrix[6] * source_scale,
        normalized_matrix[7] * source_scale,
        normalized_matrix[8] - normalized_matrix[6] * source_scale * tx -
            normalized_matrix[7] * source_scale * ty};

    double matrix_norm = 0.0;
    for(const double value : matrix)
        matrix_norm += value * value;
    matrix_norm = std::sqrt(matrix_norm);
    if(!std::isfinite(matrix_norm) || matrix_norm <= kMinimumMatrixNorm)
        return result;
    // homogeneous matrices are equivalent up to any non-zero scale
    for(double& value : matrix)
        value /= matrix_norm;
    // reject non-invertible estimates using a scale-invariant singular ratio
    if(!is_invertible_homography(matrix))
        return result;
    result.valid = true;
    result.matrix = matrix;
    return result;
}

std::optional<Point2> project_point(const Matrix3& homography,
                                    const Point2& point) {
    // non-finite input rejection
    if(!finite_point(point))
        return std::nullopt;
    // reject non-finite matrix coefficients
    for(const double value : homography) {
        if(!std::isfinite(value))
            return std::nullopt;
    }

    // projected x and y share the homogeneous denominator
    const double denominator = homography[6] * point[0] +
                               homography[7] * point[1] +
                               homography[8];
    const double row_norm = std::sqrt(
        homography[6] * homography[6] + homography[7] * homography[7] +
        homography[8] * homography[8]);
    const double point_norm =
        std::sqrt(point[0] * point[0] + point[1] * point[1] + 1.0);
    const double minimum_denominator =
        kMinimumProjectionDenominator * row_norm * point_norm;
    // reject points at or numerically near infinity
    if(!std::isfinite(denominator) ||
       !std::isfinite(minimum_denominator) ||
       std::abs(denominator) <= minimum_denominator) {
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
