#ifndef IMAGE_FEATURES_HPP
#define IMAGE_FEATURES_HPP

#include <filesystem>
#include <random>
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
    const std::filesystem::path& image_path);

// sub-samples descriptors for training k-means using a shared prng state
cv::Mat sample_training_descriptors(
    const cv::Mat& descriptors,
    int max_features,
    std::mt19937& rng);

}  // namespace image_search

#endif