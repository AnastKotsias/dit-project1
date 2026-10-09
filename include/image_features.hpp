#ifndef IMAGE_FEATURES_HPP
#define IMAGE_FEATURES_HPP

#include <filesystem>

#include <opencv2/core.hpp>

namespace image_search {

enum class SiftExtractionStatus {
    Success,
    ImageUnreadable,
    NoDescriptors,
};

struct SiftExtractionResult {
    cv::Mat descriptors;
    SiftExtractionStatus status;
};

SiftExtractionResult extract_sift_descriptors(
    const std::filesystem::path& image_path,
    int max_features = 0,
    int seed = 1);

}  // namespace image_search

#endif