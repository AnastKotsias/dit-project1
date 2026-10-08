#include "image_features.hpp"

#include <vector>

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>

namespace image_search {

SiftExtractionResult extract_sift_descriptors(
    const std::filesystem::path& image_path) {
    cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);

    if(image.empty()) {
        return {{}, SiftExtractionStatus::ImageUnreadable};
    }

    const auto sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

    if(descriptors.empty()) {
        return {{}, SiftExtractionStatus::NoDescriptors};
    }

    return {descriptors, SiftExtractionStatus::Success};
}

}  // namespace image_search