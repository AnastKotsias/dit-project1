#include "vocabulary.hpp"
#include <opencv2/ml.hpp>
#include <opencv2/features2d.hpp>
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace image_search {

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

    // concatenating all training descriptors into a single large mat
    DescriptorMatrix all_descriptors(total_rows, cols);
    // next free row in the combined matrix
    std::size_t current_row = 0;
    for(const auto& mat : training_descriptors) {
        if(mat.empty()) continue;
        // copy this image's block of descriptors into its slice of the combined matrix
        std::copy(mat.data.begin(), mat.data.end(), all_descriptors.row(current_row));
        current_row += mat.rows;
    }

    // TEMPORARY: cv::Mat view over our storage until compute_visual_vocabulary uses our own k-means
    cv::Mat all_view(static_cast<int>(total_rows), static_cast<int>(cols), CV_32F,
                     all_descriptors.data.data());

    // control randomness using the requested seed
    cv::theRNG().state = static_cast<uint64>(seed);

    // output matrix of centroids, one per row
    cv::Mat centers;
    // per-descriptor cluster assignment, required by the API but not used afterwards
    cv::Mat labels;
    // stop after 100 iterations or when centers move by less than 1e-4, whichever comes first
    cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 100, 1e-4);

    // running opencv kmeans with kmeanspp initialization
    cv::kmeans(
        all_view,
        vocab_size,
        labels,
        criteria,
        // number of independent runs, the best one in terms of compactness is kept
        3,
        cv::KMEANS_PP_CENTERS,
        centers
    );

    // copy the centroids back into our own matrix type
    DescriptorMatrix vocabulary(static_cast<std::size_t>(centers.rows), cols);
    for(std::size_t r = 0; r < vocabulary.rows; ++r) {
        const float* source = centers.ptr<float>(static_cast<int>(r));
        std::copy(source, source + cols, vocabulary.row(r));
    }
    return vocabulary;
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

    // TEMPORARY: cv::Mat views over our storage until the matcher is replaced by nearest_center
    const cv::Mat descriptors_view(static_cast<int>(descriptors.rows),
                                   static_cast<int>(descriptors.cols), CV_32F,
                                   const_cast<float*>(descriptors.data.data()));
    const cv::Mat vocabulary_view(static_cast<int>(vocabulary.rows),
                                  static_cast<int>(vocabulary.cols), CV_32F,
                                  const_cast<float*>(vocabulary.data.data()));
    // using opencv brute force matcher for optimized nearest neighbor search
    cv::BFMatcher matcher(cv::NORM_L2);
    // one match per descriptor, holding the index of its nearest centroid
    std::vector<cv::DMatch> matches;
    // finds for each descriptor the closest centroid under euclidean distance
    matcher.match(descriptors_view, vocabulary_view, matches);

    // n_I, the total number of descriptors of the image
    const std::size_t n_descriptors = descriptors.rows;
    // increment the counter h_j of the centroid c_j each descriptor was assigned to
    for(const auto& match : matches) {
        // trainIdx refers to a row of the vocabulary, so it must be a valid centroid index
        if(match.trainIdx < 0 || static_cast<std::size_t>(match.trainIdx) >= vocab_size) {
            throw std::runtime_error("nearest-word assignment returned an invalid index");
        }
        histogram[match.trainIdx] += 1.0;
    }

    // normalizing histogram by total descriptors count
    for(double& val : histogram) {
        val /= static_cast<double>(n_descriptors);
    }

    return histogram;
}

} // namespace image_search
