#include "vocabulary.hpp"
#include <opencv2/ml.hpp>
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

    // count total rows across all descriptors matrices
    int total_rows = 0;
    int cols = training_descriptors[0].cols;
    for(const auto& mat : training_descriptors) {
        if(!mat.empty()) {
            total_rows += mat.rows;
        }
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