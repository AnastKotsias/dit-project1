#include "pipeline.hpp"

#include "image_features.hpp"
#include "vocabulary.hpp"

namespace image_search {

// extracts and samples descriptors for every training image, recording failures instead of aborting
TrainingDescriptorResult extract_training_descriptors(
    const std::vector<std::filesystem::path>& image_paths,
    int max_features,
    std::mt19937& rng) {
    TrainingDescriptorResult result;
    // at most one matrix per image, so the vector never needs to reallocate
    result.sampled_descriptors.reserve(image_paths.size());

    for(const auto& path : image_paths) {
        // extract all SIFT descriptors of the current image
        const auto extraction = extract_sift_descriptors(path);
        // only images with at least one descriptor can contribute to k-means
        if(extraction.status == SiftExtractionStatus::Success &&
           !extraction.descriptors.empty()) {
            // keep a random subset of at most max_features descriptors
            result.sampled_descriptors.push_back(
                sample_training_descriptors(extraction.descriptors, max_features, rng));
            ++result.successful_images;
        } else {
            // remember which image failed and why, so the caller can report it
            ++result.failed_images;
            result.failed_paths.push_back(path);
            result.failed_statuses.push_back(extraction.status);
        }
    }

    return result;
}

// converts every record to its bag of visual words vector, skipping images that fail
BowVectorResult extract_bow_vectors(
    const std::vector<ImageRecord>& records,
    const cv::Mat& vocabulary,
    std::size_t successful_limit) {
    BowVectorResult result;
    // upper bound on the number of results, avoids reallocation
    result.vectors.reserve(records.size());
    result.identifiers.reserve(records.size());

    for(const auto& record : records) {
        // stop once enough images succeeded, which allows taking the first D distractors
        if(successful_limit != 0 &&
           result.vectors.size() >= successful_limit) {
            break;
        }

        // extract all SIFT descriptors of the current image
        const auto extraction = extract_sift_descriptors(
            std::filesystem::path(record.filepath));
        // failed images are recorded and skipped, so a replacement is taken from the next record
        if(extraction.status != SiftExtractionStatus::Success ||
           extraction.descriptors.empty()) {
            result.failed_records.push_back({record, extraction.status});
            continue;
        }

        // map the descriptors onto the vocabulary to get the image's histogram
        result.vectors.push_back(
            compute_image_bow_histogram(extraction.descriptors, vocabulary));
        // keep the identifier aligned with the vector just added
        result.identifiers.push_back(record.identifier);
    }

    return result;
}

}  // namespace image_search
