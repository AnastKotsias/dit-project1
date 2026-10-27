#include "image_features.hpp"
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>

namespace image_search {

// maps each status to a human readable description
const char* sift_extraction_status_name(SiftExtractionStatus status) {
    switch(status) {
    case SiftExtractionStatus::Success:
        return "success";
    case SiftExtractionStatus::ImageUnreadable:
        return "image unreadable or missing";
    case SiftExtractionStatus::NoDescriptors:
        return "no SIFT descriptors";
    }
    // only reached if an invalid enum value is passed in
    return "unknown extraction status";
}

// loads an image as grayscale and computes its SIFT descriptors
SiftExtractionResult extract_sift_descriptors(
    const std::filesystem::path& image_path) {
    // imread returns an empty matrix instead of throwing when loading fails
    cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);

    if(image.empty()) {
        return {{}, SiftExtractionStatus::ImageUnreadable};
    }

    // default OpenCV SIFT parameters, each descriptor has 128 values
    const auto sift = cv::SIFT::create();
    // keypoints are required by the API but not needed afterwards
    std::vector<cv::KeyPoint> keypoints;
    // output matrix with one descriptor per row
    cv::Mat descriptors;

    // cv::noArray() means no mask is used, so keypoints are detected over the whole image
    sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

    // images without any detected keypoint, such as blank ones, yield no descriptors
    if(descriptors.empty()) {
        return {{}, SiftExtractionStatus::NoDescriptors};
    }
    // OpenCV SIFT always produces 32-bit float descriptors
    if(descriptors.type() != CV_32F) {
        throw std::runtime_error("SIFT returned descriptors that are not 32-bit floats");
    }

    // copy the descriptors out of the cv::Mat, so the rest of the program does not depend on OpenCV
    DescriptorMatrix result(static_cast<std::size_t>(descriptors.rows),
                            static_cast<std::size_t>(descriptors.cols));
    for(std::size_t r = 0; r < result.rows; ++r) {
        // rows are copied one by one, since a cv::Mat is not guaranteed to be continuous
        const float* source = descriptors.ptr<float>(static_cast<int>(r));
        std::copy(source, source + result.cols, result.row(r));
    }

    return {std::move(result), SiftExtractionStatus::Success};
}

// randomly keeps at most max_features descriptor rows, the S limit per training image
DescriptorMatrix sample_training_descriptors(
    const DescriptorMatrix& descriptors, int max_features, std::mt19937& rng) {

    // a non-positive limit would make the sample meaningless
    if(max_features <= 0) {
        throw std::invalid_argument("max_features must be strictly positive");
    }
    // callers are expected to skip images that have no descriptors
    if(descriptors.empty()) {
        throw std::invalid_argument("descriptors must not be empty");
    }

    const auto limit = static_cast<std::size_t>(max_features);
    // images with at most S descriptors keep all of them
    if(descriptors.rows <= limit) {
        return descriptors;
    }

    // shuffle all row indices and take the first max_features, which samples without replacement
    std::vector<std::size_t> indices(descriptors.rows);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    // output keeps the same column count as the input
    DescriptorMatrix sampled(limit, descriptors.cols);
    // copy the selected rows so the result does not share memory with the input
    for(std::size_t i = 0; i < limit; ++i) {
        const float* source = descriptors.row(indices[i]);
        std::copy(source, source + descriptors.cols, sampled.row(i));
    }

    return sampled;
}

}  // namespace image_search