#ifndef IMAGE_FEATURES_HPP
#define IMAGE_FEATURES_HPP

#include <filesystem>
#include <random>

#include "matrix.hpp"

namespace image_search {

// outcome of a SIFT extraction, so callers can tell why an image produced no descriptors
enum class SiftExtractionStatus {
    Success,
    // the file is missing or could not be decoded as an image
    ImageUnreadable,
    // the image was read but SIFT found no keypoints in it
    NoDescriptors,
};

struct SiftExtractionResult {
    // one 128-dimensional row per keypoint, empty unless status is Success
    DescriptorMatrix descriptors;
    SiftExtractionStatus status;
};

// reads an image in grayscale and extracts its SIFT descriptors
// this is the only function that uses OpenCV, the descriptors are copied out of its matrix
// an unreadable image or one without keypoints is reported through the status, not as an exception
SiftExtractionResult extract_sift_descriptors(
    const std::filesystem::path& image_path);

// description of a status for warnings and error messages
const char* sift_extraction_status_name(SiftExtractionStatus status);

// randomly keeps at most max_features rows of the descriptors, which limits the training
// descriptors taken from one image (the S of the vocabulary); an image with fewer rows keeps all of them
// the sample depends only on the state of rng, so it is reproducible for a fixed seed
// throws if max_features is not positive or descriptors is empty
DescriptorMatrix sample_training_descriptors(
    const DescriptorMatrix& descriptors,
    int max_features,
    std::mt19937& rng);

}  // namespace image_search

#endif
