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

std::vector<double> compute_image_bow_histogram(
    const cv::Mat& descriptors,
    const cv::Mat& vocabulary) {

    int vocab_size = vocabulary.rows;
    std::vector<double> histogram(vocab_size, 0.0);

    if(descriptors.empty()) {
        return histogram;
    }

    int n_descriptors = descriptors.rows;
    for(int i = 0 ; i < n_descriptors ; ++i) {
        cv::Mat desc = descriptors.row(i);
        double min_dist = -1.0;
        int best_cluster = 0;

        for(int j = 0 ; j < vocab_size ; ++j) {
            cv::Mat center = vocabulary.row(j);
            double dist = cv::norm(desc, center, cv::NORM_L2);
            if(min_dist < 0 || dist < min_dist) {
                min_dist = dist;
                best_cluster = j;
            }
        }
        histogram[best_cluster] += 1.0;
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
