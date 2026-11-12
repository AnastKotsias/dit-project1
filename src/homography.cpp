#include "homography.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace image_search {
namespace {

// minimum valid homogeneous denominator
constexpr double kMinimumProjectionDenominator = 1e-12;
// minimum accepted normalization scale
constexpr double kMinimumNormalizationScale = 1e-12;
// minimum accepted point separation and triangle area
constexpr double kMinimumTriangleArea = 1e-10;
// Jacobi rotation convergence threshold
constexpr double kMinimumEigenvalueGap = 1e-12;

bool finite_point(const Point2& point) {
    return std::isfinite(point[0]) && std::isfinite(point[1]);
}

using DltRow = std::array<double, 9>;
using DltMatrix = std::vector<DltRow>;

struct Normalization {
    Point2 center{};
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

    // isotropic scale for a mean distance of square root of two
    double mean_distance = 0.0;
    for(const auto& correspondence : points) {
        const Point2& point = source ? correspondence.source : correspondence.destination;
        const double dx = point[0] - result.center[0];
        const double dy = point[1] - result.center[1];
        mean_distance += std::sqrt(dx * dx + dy * dy);
    }
    mean_distance /= static_cast<double>(points.size());
    result.scale = std::sqrt(2.0) / mean_distance;
    return result;
}

Point2 normalized_point(const Point2& point, const Normalization& normalization) {
    // x normalized equals scale times x minus centroid
    return {(point[0] - normalization.center[0]) * normalization.scale,
            (point[1] - normalization.center[1]) * normalization.scale};
}

double triangle_area(const Point2& first, const Point2& second,
                     const Point2& third) {
    return std::abs((second[0] - first[0]) * (third[1] - first[1]) -
                    (second[1] - first[1]) * (third[0] - first[0]));
}

bool degenerate(const std::vector<Point2>& points) {
    // repeated points and collinear triples make the DLT system rank deficient
    for(std::size_t first = 0; first < points.size(); ++first) {
        for(std::size_t second = first + 1; second < points.size(); ++second) {
            const double dx = points[first][0] - points[second][0];
            const double dy = points[first][1] - points[second][1];
            if(dx * dx + dy * dy < kMinimumTriangleArea)
                return true;
            for(std::size_t third = second + 1; third < points.size(); ++third) {
                if(triangle_area(points[first], points[second], points[third]) <
                   kMinimumTriangleArea)
                    return true;
            }
        }
    }
    return false;
}

std::array<double, 9> smallest_right_singular_vector(const DltMatrix& matrix) {
    DltMatrix transformed = matrix;
    std::array<std::array<double, 9>, 9> right_vectors{};
    for(std::size_t index = 0; index < 9; ++index) {
        right_vectors[index][index] = 1.0;
    }

    // one sided Jacobi SVD orthogonalizes the columns of A
    for(std::size_t iteration = 0; iteration < 9 * 9 * 20; ++iteration) {
        double largest_correlation = 0.0;
        for(std::size_t first = 0; first < 9; ++first) {
            for(std::size_t second = first + 1; second < 9; ++second) {
                double first_norm = 0.0;
                double second_norm = 0.0;
                double correlation = 0.0;
                for(const auto& row : transformed) {
                    first_norm += row[first] * row[first];
                    second_norm += row[second] * row[second];
                    correlation += row[first] * row[second];
                }
                // zero columns cannot contribute a Jacobi rotation
                if(first_norm == 0.0 || second_norm == 0.0)
                    continue;
                largest_correlation = std::max(
                    largest_correlation,
                    std::abs(correlation) /
                        std::sqrt(first_norm * second_norm));
                if(std::abs(correlation) <= kMinimumEigenvalueGap)
                    continue;

                // rotation angle that removes the column correlation
                const double angle = 0.5 * std::atan2(
                    2.0 * correlation, second_norm - first_norm);
                const double cosine = std::cos(angle);
                const double sine = std::sin(angle);
                for(auto& row : transformed) {
                    const double first_value = row[first];
                    const double second_value = row[second];
                    row[first] = cosine * first_value - sine * second_value;
                    row[second] = sine * first_value + cosine * second_value;
                }
                for(std::size_t index = 0; index < 9; ++index) {
                    const double first_value = right_vectors[index][first];
                    const double second_value = right_vectors[index][second];
                    right_vectors[index][first] =
                        cosine * first_value - sine * second_value;
                    right_vectors[index][second] =
                        sine * first_value + cosine * second_value;
                }
            }
        }
        if(largest_correlation <= kMinimumEigenvalueGap)
            break;
    }

    // smallest column norm identifies the right singular vector for sigma nine
    std::size_t smallest = 0;
    double smallest_norm = std::numeric_limits<double>::infinity();
    for(std::size_t index = 0; index < 9; ++index) {
        double norm = 0.0;
        for(const auto& row : transformed)
            norm += row[index] * row[index];
        if(norm < smallest_norm) {
            smallest_norm = norm;
            smallest = index;
        }
    }
    std::array<double, 9> result{};
    for(std::size_t index = 0; index < 9; ++index)
        result[index] = right_vectors[index][smallest];
    return result;
}

double determinant(const Matrix3& matrix) {
    // determinant used for singularity validation
    return matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
           matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
           matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
}

}  // namespace

HomographyResult estimate_homography(
    const std::vector<PointCorrespondence>& correspondences) {
    HomographyResult result;
    // four point correspondences provide the minimum eight independent equations
    if(correspondences.size() < 4)
        return result;

    // collect and validate both point sets
    std::vector<Point2> source_points;
    std::vector<Point2> destination_points;
    source_points.reserve(correspondences.size());
    destination_points.reserve(correspondences.size());
    for(const auto& correspondence : correspondences) {
        if(!finite_point(correspondence.source) ||
           !finite_point(correspondence.destination))
            return result;
        source_points.push_back(correspondence.source);
        destination_points.push_back(correspondence.destination);
    }
    if(degenerate(source_points) || degenerate(destination_points))
        return result;

    // zero mean distance produces an undefined normalization transform
    const auto source_normalization = normalize_points(correspondences, true);
    const auto destination_normalization = normalize_points(correspondences, false);
    if(!std::isfinite(source_normalization.scale) ||
       !std::isfinite(destination_normalization.scale) ||
       source_normalization.scale <= kMinimumNormalizationScale ||
       destination_normalization.scale <= kMinimumNormalizationScale)
        return result;

    // explicit DLT matrix with two rows for each correspondence
    DltMatrix dlt_matrix;
    dlt_matrix.reserve(2 * correspondences.size());
    for(const auto& correspondence : correspondences) {
        const auto source = normalized_point(correspondence.source,
                                              source_normalization);
        const auto destination = normalized_point(correspondence.destination,
                                                   destination_normalization);
        // h11 x plus h12 y plus h13 minus u times h31 x plus h32 y plus h33
        const DltRow first = {
            -source[0], -source[1], -1.0, 0.0, 0.0, 0.0,
            destination[0] * source[0], destination[0] * source[1],
            destination[0]};
        // h21 x plus h22 y plus h23 minus v times h31 x plus h32 y plus h33
        const DltRow second = {
            0.0, 0.0, 0.0, -source[0], -source[1], -1.0,
            destination[1] * source[0], destination[1] * source[1],
            destination[1]};
        dlt_matrix.push_back(first);
        dlt_matrix.push_back(second);
    }

    // SVD A equals U Sigma V transpose
    const auto normalized_matrix =
        smallest_right_singular_vector(dlt_matrix);
    double norm = 0.0;
    for(const double value : normalized_matrix)
        norm += value * value;
    norm = std::sqrt(norm);
    if(!std::isfinite(norm) || norm <= kMinimumNormalizationScale)
        return result;

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
    if(!std::isfinite(matrix_norm) || matrix_norm <= kMinimumNormalizationScale)
        return result;
    // homogeneous matrices are equivalent up to any non-zero scale
    for(double& value : matrix)
        value /= matrix_norm;
    // reject non-invertible estimates
    if(!std::isfinite(determinant(matrix)) ||
       std::abs(determinant(matrix)) <= kMinimumNormalizationScale)
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
    // invalid points at or near infinity
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
