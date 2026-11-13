#ifndef LOCAL_FEATURES_HPP
#define LOCAL_FEATURES_HPP

#include "image_features.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <vector>

namespace image_search {

using Point2 = std::array<double, 2>;

struct LocalFeatureSet {
    std::vector<Point2> points;
    DescriptorMatrix descriptors;
    SiftExtractionStatus status = SiftExtractionStatus::ImageUnreadable;
    // image size in pixels, zero when unreadable
    std::size_t width = 0;
    std::size_t height = 0;
};

// Reads an image in grayscale and extracts SIFT keypoint coordinates together
// with the descriptors in the same row order. The image size is kept for the
// corner error, also when no descriptors are found.
LocalFeatureSet extract_sift_features(const std::filesystem::path& image_path);

}  // namespace image_search

#endif
