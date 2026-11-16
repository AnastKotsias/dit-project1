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

}  // namespace image_search
