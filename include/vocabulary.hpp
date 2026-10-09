#ifndef VOCABULARY_HPP
#define VOCABULARY_HPP

#include <vector>
#include <string>
#include <opencv2/core.hpp>

namespace image_search {

// computes visual vocabulary centroids using k-means on training descriptors
cv::Mat compute_visual_vocabulary(
    const std::vector<cv::Mat>& training_descriptors,
    int vocab_size,
    int seed);

// computes bag of visual words normalized histogram for an image descriptor matrix
std::vector<double> compute_image_bow_histogram(
    const cv::Mat& descriptors,
    const cv::Mat& vocabulary);


}   // namespace image_search

#endif