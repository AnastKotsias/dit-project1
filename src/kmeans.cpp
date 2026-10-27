#include "kmeans.hpp"

#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>

namespace image_search {
// helpers below are private to this translation unit
namespace {

// copies row source_row of source into row target_row of target
template <typename T>
void copy_row(const Matrix<T>& source, std::size_t source_row,
              Matrix<T>& target, std::size_t target_row) {
    const T* begin = source.row(source_row);
    std::copy(begin, begin + source.cols, target.row(target_row));
}

// k-means++ seeding: the first center is a uniformly random point, every next one is
// chosen with probability proportional to its squared distance from the nearest chosen center
// also fills assignments and distances with the nearest initial center of every point
template <typename T>
Matrix<T> kmeanspp_centers(const Matrix<T>& points, std::size_t clusters, std::mt19937& rng,
                           std::vector<std::size_t>& assignments,
                           std::vector<double>& distances) {
    Matrix<T> centers(clusters, points.cols);

    std::uniform_int_distribution<std::size_t> pick_any(0, points.rows - 1);
    copy_row(points, pick_any(rng), centers, 0);
    for(std::size_t i = 0; i < points.rows; ++i) {
        distances[i] = squared_distance(points.row(i), centers.row(0), points.cols);
        assignments[i] = 0;
    }

    for(std::size_t c = 1; c < clusters; ++c) {
        // D(i)^2 summed over all points
        double total = 0.0;
        for(const double distance : distances)
            total += distance;

        std::size_t chosen = 0;
        if(total > 0.0) {
            // walk the cumulative weights until they pass a uniform draw in [0, total)
            std::uniform_real_distribution<double> draw(0.0, total);
            const double target = draw(rng);
            double cumulative = 0.0;
            // fallback for rounding, the last point with a non-zero weight
            std::size_t last_positive = 0;
            bool found = false;
            for(std::size_t i = 0; i < points.rows; ++i) {
                if(distances[i] <= 0.0) continue;
                last_positive = i;
                cumulative += distances[i];
                if(cumulative > target) {
                    chosen = i;
                    found = true;
                    break;
                }
            }
            if(!found) chosen = last_positive;
        } else {
            // every point coincides with a chosen center, so any choice is as good as another
            chosen = pick_any(rng);
        }
        copy_row(points, chosen, centers, c);

        // the new center can only bring points closer, so the nearest center is updated incrementally
        for(std::size_t i = 0; i < points.rows; ++i) {
            const double distance = squared_distance(points.row(i), centers.row(c), points.cols);
            if(distance < distances[i]) {
                distances[i] = distance;
                assignments[i] = c;
            }
        }
    }
    return centers;
}

// assignment step, moves every point to its nearest center and returns how many points moved
template <typename T>
std::size_t assign_points(const Matrix<T>& points, const Matrix<T>& centers,
                          std::vector<std::size_t>& assignments,
                          std::vector<double>& distances) {
    std::size_t changes = 0;
    for(std::size_t i = 0; i < points.rows; ++i) {
        const std::size_t nearest = nearest_center(points.row(i), centers);
        if(nearest != assignments[i]) {
            assignments[i] = nearest;
            ++changes;
        }
        distances[i] = squared_distance(points.row(i), centers.row(nearest), points.cols);
    }
    return changes;
}

// update step, moves every center to the mean of its points
// an empty cluster is moved onto the point that is farthest from its own center
template <typename T>
void update_centers(const Matrix<T>& points, const std::vector<std::size_t>& assignments,
                    std::vector<double>& distances, Matrix<T>& centers) {
    // sums are kept in double, so float descriptors do not lose precision while accumulating
    std::vector<double> sums(centers.rows * centers.cols, 0.0);
    std::vector<std::size_t> counts(centers.rows, 0);
    for(std::size_t i = 0; i < points.rows; ++i) {
        const T* point = points.row(i);
        double* sum = sums.data() + assignments[i] * centers.cols;
        for(std::size_t d = 0; d < points.cols; ++d)
            sum[d] += point[d];
        ++counts[assignments[i]];
    }

    for(std::size_t c = 0; c < centers.rows; ++c) {
        if(counts[c] == 0) {
            const auto farthest = static_cast<std::size_t>(
                std::max_element(distances.begin(), distances.end()) - distances.begin());
            copy_row(points, farthest, centers, c);
            // the point is now a center, so the next empty cluster takes a different one
            distances[farthest] = 0.0;
            continue;
        }
        const double* sum = sums.data() + c * centers.cols;
        T* center = centers.row(c);
        for(std::size_t d = 0; d < centers.cols; ++d)
            center[d] = static_cast<T>(sum[d] / static_cast<double>(counts[c]));
    }
}

}  // namespace

template <typename T>
double squared_distance(const T* a, const T* b, std::size_t dim) {
    // four independent partial sums let the compiler vectorize the loop
    // without reordering a single floating point accumulation
    T partial[4] = {0, 0, 0, 0};
    std::size_t d = 0;
    for(; d + 4 <= dim; d += 4) {
        for(std::size_t lane = 0; lane < 4; ++lane) {
            const T difference = a[d + lane] - b[d + lane];
            partial[lane] += difference * difference;
        }
    }
    for(; d < dim; ++d) {
        const T difference = a[d] - b[d];
        partial[0] += difference * difference;
    }
    return static_cast<double>(partial[0]) + static_cast<double>(partial[1]) +
           static_cast<double>(partial[2]) + static_cast<double>(partial[3]);
}

template <typename T>
std::size_t nearest_center(const T* point, const Matrix<T>& centers) {
    if(centers.empty()) {
        throw std::invalid_argument("nearest_center needs at least one center");
    }
    std::size_t best = 0;
    double best_distance = std::numeric_limits<double>::infinity();
    for(std::size_t c = 0; c < centers.rows; ++c) {
        const double distance = squared_distance(point, centers.row(c), centers.cols);
        // strict comparison keeps the lowest index on ties, which keeps results deterministic
        if(distance < best_distance) {
            best_distance = distance;
            best = c;
        }
    }
    return best;
}

template <typename T>
KMeansResult<T> kmeans(const Matrix<T>& points, std::size_t clusters, int seed,
                       int max_iterations) {
    if(points.empty()) {
        throw std::invalid_argument("k-means needs at least one point");
    }
    if(clusters == 0) {
        throw std::invalid_argument("k-means needs at least one cluster");
    }
    // each center starts on a distinct point, so there cannot be more centers than points
    if(clusters > points.rows) {
        throw std::invalid_argument("k-means cannot form more clusters than there are points");
    }
    if(max_iterations < 0) {
        throw std::invalid_argument("k-means max_iterations must not be negative");
    }

    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    KMeansResult<T> result;
    result.assignments.assign(points.rows, 0);
    // squared distance of every point to its assigned center
    std::vector<double> distances(points.rows, 0.0);
    result.centers = kmeanspp_centers(points, clusters, rng, result.assignments, distances);

    for(int iteration = 0; iteration < max_iterations; ++iteration) {
        update_centers(points, result.assignments, distances, result.centers);
        const std::size_t changes =
            assign_points(points, result.centers, result.assignments, distances);
        result.iterations = iteration + 1;
        // unchanged assignments mean the centers would not move either
        if(changes == 0) {
            result.converged = true;
            break;
        }
    }
    return result;
}

// the program clusters float SIFT descriptors and double BoW vectors
template double squared_distance<float>(const float*, const float*, std::size_t);
template double squared_distance<double>(const double*, const double*, std::size_t);
template std::size_t nearest_center<float>(const float*, const Matrix<float>&);
template std::size_t nearest_center<double>(const double*, const Matrix<double>&);
template KMeansResult<float> kmeans<float>(const Matrix<float>&, std::size_t, int, int);
template KMeansResult<double> kmeans<double>(const Matrix<double>&, std::size_t, int, int);

}  // namespace image_search
