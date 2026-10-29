#include "vocabulary.hpp"
#include "kmeans.hpp"
#include <algorithm>
#include <stdexcept>

namespace image_search {
namespace {

// upper bound on Lloyd iterations for the SIFT vocabulary
// measured on the training descriptors (K = 64 and 256), 50 iterations get within 0.1% of the
// converged sum of squared distances, which takes 240-300 iterations and 2-4 times longer
constexpr int kVocabularyIterations = 50;

}  // namespace

DescriptorMatrix compute_visual_vocabulary(
    const std::vector<DescriptorMatrix>& training_descriptors,
    int vocab_size,
    int seed) {
    if(training_descriptors.empty()) {
        throw std::invalid_argument("training descriptors list cannot be empty");
    }
    if(vocab_size <= 0) {
        throw std::invalid_argument("vocab_size must be strictly positive");
    }

    // the first non-empty matrix fixes the descriptor dimension, a pointer avoids copying it
    const DescriptorMatrix* reference = nullptr;
    for(const auto& mat : training_descriptors) {
        if(!mat.empty()) {
            reference = &mat;
            break;
        }
    }
    if(!reference) {
        throw std::invalid_argument("training descriptors cannot be all empty");
    }
    const std::size_t cols = reference->cols;

    // total number of descriptors, checking that all matrices have the same dimension
    std::size_t total_rows = 0;
    for(const auto& mat : training_descriptors) {
        if(mat.empty()) continue;
        if(mat.cols != cols) {
            throw std::invalid_argument("all training descriptors must have identical dimensions");
        }
        total_rows += mat.rows;
    }

    // k-means cannot form more clusters than there are samples
    if(total_rows < static_cast<std::size_t>(vocab_size)) {
        throw std::invalid_argument("total training descriptors are fewer than vocabulary size");
    }

    // pool the descriptors of all images into one matrix
    DescriptorMatrix all_descriptors(total_rows, cols);
    std::size_t current_row = 0;
    for(const auto& mat : training_descriptors) {
        if(mat.empty()) continue;
        std::copy(mat.data.begin(), mat.data.end(), all_descriptors.row(current_row));
        current_row += mat.rows;
    }

    return kmeans(all_descriptors, static_cast<std::size_t>(vocab_size), seed,
                  kVocabularyIterations).centers;
}

std::vector<double> compute_image_bow_histogram(
    const DescriptorMatrix& descriptors,
    const DescriptorMatrix& vocabulary) {
    // an image without descriptors has no histogram, and normalizing would divide by zero
    if(descriptors.empty()) {
        throw std::invalid_argument(
            "cannot compute BoW histogram for empty descriptors");
    }
    if(vocabulary.empty()) {
        throw std::invalid_argument("vocabulary must not be empty");
    }
    if(descriptors.cols != vocabulary.cols) {
        throw std::invalid_argument("descriptor dimension does not match vocabulary dimension");
    }

    // histogram[j] counts the descriptors whose nearest visual word is j
    std::vector<double> histogram(vocabulary.rows, 0.0);
    const std::size_t n_descriptors = descriptors.rows;
    for(std::size_t r = 0; r < n_descriptors; ++r) {
        histogram[nearest_center(descriptors.row(r), vocabulary)] += 1.0;
    }

    // dividing by the descriptor count makes the entries sum to 1
    for(double& val : histogram) {
        val /= static_cast<double>(n_descriptors);
    }

    return histogram;
}

} // namespace image_search
