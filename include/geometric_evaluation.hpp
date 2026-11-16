#ifndef GEOMETRIC_EVALUATION_HPP
#define GEOMETRIC_EVALUATION_HPP

#include "homography.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>

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

}  // namespace image_search

#endif
