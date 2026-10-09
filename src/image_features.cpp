#include "image_features.hpp"
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <stdexcept>
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

cv::Mat sample_training_descriptors(
    const cv::Mat& descriptors, int max_features, std::mt19937& rng) {

    if(max_features <= 0) {
        throw std::invalid_argument("max_features must be strictly positive");
    }
    if(descriptors.empty()) {
        throw std::invalid_argument("descriptors must not be empty");
    }

    if(descriptors.rows <= max_features) {
        return descriptors.clone();
    }

    std::vector<int> indices(descriptors.rows);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    cv::Mat sampled(max_features, descriptors.cols, descriptors.type());
    for(int i = 0 ; i < max_features ; ++i) {
        descriptors.row(indices[i]).copyTo(sampled.row(i));
    }

    return sampled;
}

}  // namespace image_search