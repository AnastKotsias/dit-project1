#ifndef IMAGE_FEATURES_HPP
#define IMAGE_FEATURES_HPP

#include <filesystem>
#include <random>

#include "matrix.hpp"

namespace image_search {

// outcome of a SIFT extraction attempt, so callers can tell why an image produced no descriptors
enum class SiftExtractionStatus {
    // descriptors were extracted successfully
    Success,
    // the file is missing or could not be decoded as an image
    ImageUnreadable,
    // the image was read but SIFT found no keypoints in it
    NoDescriptors,
};

// bundles the extracted descriptors with the status of the extraction
struct SiftExtractionResult {
    // one 128-dimensional float row per keypoint, empty unless status is Success
    DescriptorMatrix descriptors;
    // tells whether extraction succeeded or why it failed
    SiftExtractionStatus status;
};

// reads an image in grayscale and extracts its SIFT descriptors
// this is the only place where OpenCV is used
SiftExtractionResult extract_sift_descriptors(
    const std::filesystem::path& image_path);

// converts a status value to a human readable string for warnings and error messages
const char* sift_extraction_status_name(SiftExtractionStatus status);

// sub-samples descriptors for training k-means using a shared prng state
DescriptorMatrix sample_training_descriptors(
    const DescriptorMatrix& descriptors,
    int max_features,
    std::mt19937& rng);

}  // namespace image_search

#endif
