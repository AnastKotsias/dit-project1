#include "geometry/local_features.hpp"

#include <cmath>
#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <utility>

namespace image_search {

LocalFeatureSet extract_sift_features(const std::filesystem::path& image_path) {
    cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);
    if(image.empty())
        return {{}, {}, SiftExtractionStatus::ImageUnreadable};
    // image size, needed for the corners a_i of the corner error
    const auto width = static_cast<std::size_t>(image.cols);
    const auto height = static_cast<std::size_t>(image.rows);

    const auto sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

    if(descriptors.empty())
        return {{}, {}, SiftExtractionStatus::NoDescriptors, width, height};
    if(descriptors.type() != CV_32F)
        throw std::runtime_error("SIFT returned descriptors that are not 32-bit floats");
    if(descriptors.rows != static_cast<int>(keypoints.size()) ||
       descriptors.cols != 128) {
        throw std::runtime_error("SIFT keypoints and descriptors are misaligned");
    }

    LocalFeatureSet result;
    result.points.reserve(keypoints.size());
    result.descriptors = LocalDescriptorMatrix(
        static_cast<std::size_t>(descriptors.rows),
        static_cast<std::size_t>(descriptors.cols));
    for(std::size_t row = 0; row < result.descriptors.rows; ++row) {
        result.points.push_back({
            static_cast<double>(keypoints[row].pt.x),
            static_cast<double>(keypoints[row].pt.y)});
        const float* source = descriptors.ptr<float>(static_cast<int>(row));
        std::uint8_t* target = result.descriptors.row(row);
        for(std::size_t column = 0; column < result.descriptors.cols; ++column) {
            // byte copy is exact only for whole numbers 0..255, NaN fails the range test
            const float value = source[column];
            if(!(value >= 0.0F && value <= 255.0F) || value != std::floor(value))
                throw std::runtime_error("SIFT descriptor value is not a whole number 0..255");
            target[column] = static_cast<std::uint8_t>(value);
        }
    }
    result.status = SiftExtractionStatus::Success;
    result.width = width;
    result.height = height;
    return result;
}

}  // namespace image_search
