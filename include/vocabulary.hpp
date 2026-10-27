#ifndef VOCABULARY_HPP
#define VOCABULARY_HPP

#include <vector>
#include <string>
#include <opencv2/core.hpp>

namespace image_search {

// runs k-means on the pooled training descriptors and returns the vocab_size centroids as matrix rows
cv::Mat compute_visual_vocabulary(
    const std::vector<cv::Mat>& training_descriptors,
    int vocab_size,
    int seed);

// assigns each descriptor to its nearest centroid and returns the histogram normalized to sum to 1
std::vector<double> compute_image_bow_histogram(
    const cv::Mat& descriptors,
    const cv::Mat& vocabulary);


}   // namespace image_search

#endif