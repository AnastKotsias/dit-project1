#ifndef GEOMETRIC_EVALUATION_HPP
#define GEOMETRIC_EVALUATION_HPP

#include "homography.hpp"

#include <filesystem>
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

}  // namespace image_search

#endif
