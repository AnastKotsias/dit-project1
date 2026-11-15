#include "geometric_evaluation.hpp"

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

}  // namespace image_search
