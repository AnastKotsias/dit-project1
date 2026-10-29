#ifndef VOCABULARY_HPP
#define VOCABULARY_HPP

#include <vector>
#include <string>
#include "matrix.hpp"

namespace image_search {

// builds the visual vocabulary: pools the sampled descriptors of all training images and
// clusters them with k-means, so the returned vocab_size centers (one per row) are the visual words
// images without descriptors (empty matrices) are skipped
// throws if there is nothing to cluster, vocab_size is not positive or exceeds the number of
// descriptors, or the matrices have different dimensions
DescriptorMatrix compute_visual_vocabulary(
    const std::vector<DescriptorMatrix>& training_descriptors,
    int vocab_size,
    int seed);

// Bag of Visual Words vector of one image: the fraction of its descriptors nearest (Euclidean
// distance) to each visual word, so the K entries are non-negative and sum to 1
// throws if descriptors or vocabulary is empty, or their dimensions differ
std::vector<double> compute_image_bow_histogram(
    const DescriptorMatrix& descriptors,
    const DescriptorMatrix& vocabulary);

}   // namespace image_search

#endif
