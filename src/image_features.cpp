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

const char* sift_extraction_status_name(SiftExtractionStatus status) {
    switch(status) {
    case SiftExtractionStatus::Success:
        return "success";
    case SiftExtractionStatus::ImageUnreadable:
        return "image unreadable or missing";
    case SiftExtractionStatus::NoDescriptors:
        return "no SIFT descriptors";
    }
    // only reached for an invalid enum value
    return "unknown extraction status";
}

SiftExtractionResult extract_sift_descriptors(
    const std::filesystem::path& image_path) {
    // imread returns an empty matrix instead of throwing when loading fails
    cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);

    if(image.empty()) {
        return {{}, SiftExtractionStatus::ImageUnreadable};
    }

    // default SIFT parameters, which give 128 values per descriptor
    const auto sift = cv::SIFT::create();
    // the API requires the keypoints as output, but only the descriptors are used
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    // no mask, so keypoints are detected over the whole image
    sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

    // an image without keypoints, such as a blank one, yields no descriptors
    if(descriptors.empty()) {
        return {{}, SiftExtractionStatus::NoDescriptors};
    }
    if(descriptors.type() != CV_32F) {
        throw std::runtime_error("SIFT returned descriptors that are not 32-bit floats");
    }

    // copy into a DescriptorMatrix, so the rest of the program does not depend on OpenCV
    DescriptorMatrix result(static_cast<std::size_t>(descriptors.rows),
                            static_cast<std::size_t>(descriptors.cols));
    for(std::size_t r = 0; r < result.rows; ++r) {
        // row by row, since a cv::Mat is not guaranteed to be continuous
        const float* source = descriptors.ptr<float>(static_cast<int>(r));
        std::copy(source, source + result.cols, result.row(r));
    }

    return {std::move(result), SiftExtractionStatus::Success};
}

DescriptorMatrix sample_training_descriptors(
    const DescriptorMatrix& descriptors, int max_features, std::mt19937& rng) {
    if(max_features <= 0) {
        throw std::invalid_argument("max_features must be strictly positive");
    }
    // callers skip images without descriptors, so an empty input is a programming error
    if(descriptors.empty()) {
        throw std::invalid_argument("descriptors must not be empty");
    }

    const auto limit = static_cast<std::size_t>(max_features);
    if(descriptors.rows <= limit) {
        return descriptors;
    }

    // shuffling all row indices and taking the first limit of them samples without replacement
    std::vector<std::size_t> indices(descriptors.rows);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    DescriptorMatrix sampled(limit, descriptors.cols);
    for(std::size_t i = 0; i < limit; ++i) {
        const float* source = descriptors.row(indices[i]);
        std::copy(source, source + descriptors.cols, sampled.row(i));
    }

    return sampled;
}

}  // namespace image_search
