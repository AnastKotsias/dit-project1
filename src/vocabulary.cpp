#include "vocabulary.hpp"
#include "kmeans.hpp"
#include <algorithm>
#include <stdexcept>

namespace image_search {
// helpers below are private to this translation unit
namespace {

// upper bound on lloyd iterations for the SIFT vocabulary
// measured on the training descriptors (K = 64 and 256), 50 iterations get within 0.1% of the
// converged sum of squared distances, which takes 240-300 iterations and 2-4 times longer
constexpr int kVocabularyIterations = 50;

}  // namespace

// builds the visual vocabulary c_1..c_K by clustering the SIFT descriptors of the training images
DescriptorMatrix compute_visual_vocabulary(
    const std::vector<DescriptorMatrix>& training_descriptors,
    int vocab_size,
    int seed) {

    // there must be at least one image worth of descriptors to cluster
    if(training_descriptors.empty()) {
        throw std::invalid_argument("training descriptors list cannot be empty");
    }
    // k-means needs at least one cluster
    if(vocab_size <= 0) {
        throw std::invalid_argument("vocab_size must be strictly positive");
    }

    // find the first non-empty matrix as reference
    // pointer rather than a copy, so no descriptor data is duplicated
    const DescriptorMatrix* reference = nullptr;
    for(const auto& mat : training_descriptors) {
        if(!mat.empty()) {
            reference = &mat;
            break;
        }
    }

    // every image produced an empty matrix, so there is nothing to cluster
    if(!reference) {
        throw std::invalid_argument("training descriptors cannot be all empty");
    }
    // descriptor dimension that every other matrix must match, 128 for SIFT
    const std::size_t cols = reference->cols;

    // count total rows across all descriptors matrices
    std::size_t total_rows = 0;
    for(const auto& mat : training_descriptors) {
        // images without descriptors are skipped
        if(mat.empty()) continue;
        // mixing descriptor sizes would make the clustering meaningless
        if(mat.cols != cols) {
            throw std::invalid_argument("all training descriptors must have identical dimensions");
        }
        total_rows += mat.rows;
    }

    // k-means cannot form more clusters than there are samples
    if(total_rows < static_cast<std::size_t>(vocab_size)) {
        throw std::invalid_argument("total training descriptors are fewer than vocabulary size");
    }

    // concatenating all training descriptors into a single matrix
    DescriptorMatrix all_descriptors(total_rows, cols);
    // next free row in the combined matrix
    std::size_t current_row = 0;
    for(const auto& mat : training_descriptors) {
        if(mat.empty()) continue;
        // copy this image's block of descriptors into its slice of the combined matrix
        std::copy(mat.data.begin(), mat.data.end(), all_descriptors.row(current_row));
        current_row += mat.rows;
    }

    // lloyd's algorithm with k-means++ initialization, implemented in kmeans.cpp
    return kmeans(all_descriptors, static_cast<std::size_t>(vocab_size), seed,
                  kVocabularyIterations).centers;
}

// builds the normalized bag of visual words vector x_I = (h_1/n_I, ..., h_K/n_I) for one image
std::vector<double> compute_image_bow_histogram(
    const DescriptorMatrix& descriptors,
    const DescriptorMatrix& vocabulary) {

    // normalizing by n_I would divide by zero for an image without descriptors
    if(descriptors.empty()) {
        throw std::invalid_argument(
            "cannot compute BoW histogram for empty descriptors");
    }

    // validate vocabulary state and dimensions
    if(vocabulary.empty()) {
        throw std::invalid_argument("vocabulary must not be empty");
    }
    // K, the number of centroids and therefore the histogram length
    const std::size_t vocab_size = vocabulary.rows;
    // h_j counters, one per visual word, all starting at zero
    std::vector<double> histogram(vocab_size, 0.0);

    // distances are only defined between vectors of equal length
    if(descriptors.cols != vocabulary.cols) {
        throw std::invalid_argument("descriptor dimension does not match vocabulary dimension");
    }

    // n_I, the total number of descriptors of the image
    const std::size_t n_descriptors = descriptors.rows;
    // increment the counter h_j of the centroid c_j nearest to each descriptor, under euclidean distance
    for(std::size_t r = 0; r < n_descriptors; ++r) {
        histogram[nearest_center(descriptors.row(r), vocabulary)] += 1.0;
    }

    // normalizing histogram by total descriptors count
    for(double& val : histogram) {
        val /= static_cast<double>(n_descriptors);
    }

    return histogram;
}

} // namespace image_search
