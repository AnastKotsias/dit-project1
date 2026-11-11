#include "image_features.hpp"
#include "local_features.hpp"

#include <numeric>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <utility>

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
    auto features = extract_sift_features(image_path);
    return {std::move(features.descriptors), features.status};
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
