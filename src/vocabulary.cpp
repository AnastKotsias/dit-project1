#include "vocabulary.hpp"
#include <opencv2/ml.hpp>
#include <opencv2/features2d.hpp>
#include <stdexcept>
#include <cmath>

namespace image_search {

cv::Mat compute_visual_vocabulary(
    const std::vector<cv::Mat>& training_descriptors,
    int vocab_size,
    int seed) {

    if(training_descriptors.empty()) {
        throw std::invalid_argument("training descriptors list cannot be empty");
    }
    if(vocab_size <= 0) {
        throw std::invalid_argument("vocab_size must be strictly positive");
    }

    // find the first non-empty matrix as reference
    const cv::Mat* reference = nullptr;
    for(const auto& mat : training_descriptors) {
        if(!mat.empty()) {
            reference = &mat;
            break;
        }
    }

    if(!reference) {
        throw std::invalid_argument("training descriptors cannot be all empty");
    }

    const int cols = reference->cols;

    // count total rows across all descriptors matrices
    int total_rows = 0;
    for(const auto& mat : training_descriptors) {
        if(mat.empty()) continue;
        if(mat.cols != cols) {
            throw std::invalid_argument("all training descriptors must have identical dimensions");
        }
        if(mat.type() != CV_32F) {
            throw std::invalid_argument("all training descriptors must be of type CV_32F");
        }
        total_rows += mat.rows;
    }

    if(total_rows < vocab_size) {
        throw std::invalid_argument("total training descriptors are fewer than vocabulary size");
    }

    // concatenating all training descriptors into a single large mat
    cv::Mat all_descriptors(total_rows, cols, CV_32F);
    int current_row = 0;
    for(const auto& mat : training_descriptors) {
        if(mat.empty()) continue;
        mat.copyTo(all_descriptors.rowRange(current_row, current_row + mat.rows));
        current_row += mat.rows;
    }

    // control randomness using the requested seed
    cv::theRNG().state = static_cast<uint64>(seed);

    cv::Mat centers;
    cv::Mat labels;
    cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 100, 1e-4);

    // running opencv kmeans with kmeanspp initialization
    cv::kmeans(
        all_descriptors,
        vocab_size,
        labels,
        criteria,
        3,
        cv::KMEANS_PP_CENTERS,
        centers
    );

    return centers;
}

std::vector<double> compute_image_bow_histogram(
    const cv::Mat& descriptors,
    const cv::Mat& vocabulary) {

    // validate vocabulary state and dimensions
    if(vocabulary.empty() || vocabulary.rows == 0) {
        throw std::invalid_argument("vocabulary must not be empty");
    }
    if(vocabulary.type() != CV_32F) {
        throw std::invalid_argument("vocabulary must be of type CV_32F");
    }

    int vocab_size = vocabulary.rows;
    std::vector<double> histogram(vocab_size, 0.0);

    if(descriptors.empty()) {
        return histogram;
    }

    if(descriptors.type() != CV_32F) {
        throw std::invalid_argument("descriptors must be of type CV_32F");
    }
    if(descriptors.cols != vocabulary.cols) {
        throw std::invalid_argument("descriptor dimension does not match vocabulary dimension");
    }

    // using opencv brute force matcher for optimized nearest neighbor search
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors, vocabulary, matches);

    int n_descriptors = descriptors.rows;
    for(const auto& match : matches) {
        histogram[match.trainIdx] += 1.0;
    }

    // normalizing histogram by total descriptors count
    if(n_descriptors > 0) {
        for(double& val : histogram) {
            val /= static_cast<double>(n_descriptors);
        }
    }

    return histogram;
}

} // namespace image_search
