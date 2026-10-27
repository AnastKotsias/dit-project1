#ifndef VOCABULARY_HPP
#define VOCABULARY_HPP

#include <vector>
#include <string>
#include "matrix.hpp"

namespace image_search {

// runs k-means on the pooled training descriptors and returns the vocab_size centroids as matrix rows
DescriptorMatrix compute_visual_vocabulary(
    const std::vector<DescriptorMatrix>& training_descriptors,
    int vocab_size,
    int seed);

// assigns each descriptor to its nearest centroid and returns the histogram normalized to sum to 1
std::vector<double> compute_image_bow_histogram(
    const DescriptorMatrix& descriptors,
    const DescriptorMatrix& vocabulary);


}   // namespace image_search

#endif