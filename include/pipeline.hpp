#ifndef PIPELINE_HPP
#define PIPELINE_HPP

#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "dataset.hpp"
#include "image_features.hpp"

namespace image_search {

// collects the sampled training descriptors together with a report of the images that failed
struct TrainingDescriptorResult {
    // one matrix per successful image, holding at most S sampled descriptors
    std::vector<cv::Mat> sampled_descriptors;
    std::size_t successful_images = 0;
    std::size_t failed_images = 0;
    std::vector<std::filesystem::path> failed_paths;
    // reason each image in failed_paths failed
    std::vector<SiftExtractionStatus> failed_statuses;
};

// extracts SIFT descriptors from each training image and samples at most max_features of them
TrainingDescriptorResult extract_training_descriptors(
    const std::vector<std::filesystem::path>& image_paths,
    int max_features,
    std::mt19937& rng);

// an image record that could not be turned into a vector, with the reason
struct FailedImageRecord {
    ImageRecord record;
    SiftExtractionStatus status;
};

// bag of visual words vectors of the successfully processed images, plus the failures
struct BowVectorResult {
    // one normalized histogram per successful image
    std::vector<std::vector<double>> vectors;
    // identifier of each image, at the same position as its vector
    std::vector<std::string> identifiers;
    // images skipped because their descriptors could not be extracted
    std::vector<FailedImageRecord> failed_records;
};

// A non-zero limit stops after that many successful records; zero processes all.
BowVectorResult extract_bow_vectors(
    const std::vector<ImageRecord>& records,
    const cv::Mat& vocabulary,
    std::size_t successful_limit = 0);

}  // namespace image_search

#endif
