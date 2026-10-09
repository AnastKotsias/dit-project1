#include "image_features.hpp"
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>

namespace image_search {

SiftExtractionResult extract_sift_descriptors(
    const std::filesystem::path& image_path,
    int max_features,
    int seed) {
    cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);

    if(image.empty()) {
        return {{}, SiftExtractionStatus::ImageUnreadable};
    }

    const auto sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

    if(descriptors.empty()) {
        return {{}, SiftExtractionStatus::NoDescriptors};
    }

    // randomly sub-sample descriptors if they exceed the training limit
    if(max_features > 0 && descriptors.rows > max_features) {
        std::vector<int> indices(descriptors.rows);
        std::iota(indices.begin(), indices.end(), 0);

        std::mt19937 rng(seed);
        std::shuffle(indices.begin(), indices.end(), rng);

        cv::Mat sampled_descriptors(max_features, descriptors.cols, descriptors.type());
        for(int i=0 ; i < max_features ; ++i) {
            descriptors.row(indices[i]).copyTo(sampled_descriptors.row(i));
        }
        descriptors = sampled_descriptors;
    }

    return {descriptors, SiftExtractionStatus::Success};
}

}  // namespace image_search