#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cstddef>
#include <vector>

namespace image_search {

// dense row-major matrix, used instead of cv::Mat so OpenCV stays limited to SIFT extraction
template <typename T>
struct Matrix {
    std::size_t rows = 0;
    std::size_t cols = 0;
    // element (r, c) is stored at data[r * cols + c], so each row is contiguous
    std::vector<T> data;

    Matrix() = default;

    // allocates a zero-filled rows x cols matrix
    Matrix(std::size_t row_count, std::size_t col_count)
        : rows(row_count), cols(col_count), data(row_count * col_count) {}

    bool empty() const { return rows == 0 || cols == 0; }

    // pointer to the first element of row r, the row has cols elements
    T* row(std::size_t r) { return data.data() + r * cols; }
    const T* row(std::size_t r) const { return data.data() + r * cols; }
};

// SIFT descriptors or visual words, one 128-dimensional float row each
using DescriptorMatrix = Matrix<float>;

}  // namespace image_search

#endif
