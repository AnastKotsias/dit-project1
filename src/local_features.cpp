#include "local_features.hpp"

#include <algorithm>
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
    result.descriptors = DescriptorMatrix(
        static_cast<std::size_t>(descriptors.rows),
        static_cast<std::size_t>(descriptors.cols));
    for(std::size_t row = 0; row < result.descriptors.rows; ++row) {
        result.points.push_back({
            static_cast<double>(keypoints[row].pt.x),
            static_cast<double>(keypoints[row].pt.y)});
        const float* source = descriptors.ptr<float>(static_cast<int>(row));
        std::copy(source, source + result.descriptors.cols,
                  result.descriptors.row(row));
    }
    result.status = SiftExtractionStatus::Success;
    result.width = width;
    result.height = height;
    return result;
}

}  // namespace image_search
