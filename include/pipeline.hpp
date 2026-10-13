#ifndef PIPELINE_HPP
#define PIPELINE_HPP

#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "dataset.hpp"

namespace image_search {

struct TrainingDescriptorResult {
    std::vector<cv::Mat> sampled_descriptors;
    std::size_t successful_images = 0;
    std::size_t failed_images = 0;
    std::vector<std::filesystem::path> failed_paths;
};

TrainingDescriptorResult extract_training_descriptors(
    const std::vector<std::filesystem::path>& image_paths,
    int max_features,
    std::mt19937& rng);

struct BowVectorResult {
    std::vector<std::vector<double>> vectors;
    std::vector<std::string> identifiers;
    std::vector<ImageRecord> failed_records;
};

// A non-zero limit stops after that many successful records; zero processes all.
BowVectorResult extract_bow_vectors(
    const std::vector<ImageRecord>& records,
    const cv::Mat& vocabulary,
    std::size_t successful_limit = 0);

}  // namespace image_search

#endif
