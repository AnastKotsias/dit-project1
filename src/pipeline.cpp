#include "pipeline.hpp"

#include "image_features.hpp"
#include "vocabulary.hpp"

namespace image_search {

TrainingDescriptorResult extract_training_descriptors(
    const std::vector<std::filesystem::path>& image_paths,
    int max_features,
    std::mt19937& rng) {
    TrainingDescriptorResult result;
    result.sampled_descriptors.reserve(image_paths.size());

    for(const auto& path : image_paths) {
        const auto extraction = extract_sift_descriptors(path);
        if(extraction.status == SiftExtractionStatus::Success &&
           !extraction.descriptors.empty()) {
            result.sampled_descriptors.push_back(
                sample_training_descriptors(extraction.descriptors, max_features, rng));
            ++result.successful_images;
        } else {
            ++result.failed_images;
            result.failed_paths.push_back(path);
        }
    }

    return result;
}

BowVectorResult extract_bow_vectors(
    const std::vector<ImageRecord>& records,
    const cv::Mat& vocabulary,
    std::size_t successful_limit) {
    BowVectorResult result;
    result.vectors.reserve(records.size());
    result.identifiers.reserve(records.size());

    for(const auto& record : records) {
        if(successful_limit != 0 &&
           result.vectors.size() >= successful_limit) {
            break;
        }

        const auto extraction = extract_sift_descriptors(
            std::filesystem::path(record.filepath));
        if(extraction.status != SiftExtractionStatus::Success ||
           extraction.descriptors.empty()) {
            result.failed_records.push_back(record);
            continue;
        }

        result.vectors.push_back(
            compute_image_bow_histogram(extraction.descriptors, vocabulary));
        result.identifiers.push_back(record.identifier);
    }

    return result;
}

}  // namespace image_search
