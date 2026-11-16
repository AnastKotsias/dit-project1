#include "geometry/descriptor_matching.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace image_search {
namespace {

constexpr std::size_t kSiftDimension = 128;
// dimensions summed between two early-exit checks
constexpr std::size_t kDistanceBlock = 32;

void validate_features(const LocalFeatureSet& features, const char* name) {
    // point coordinates and descriptor rows must be aligned
    if(features.points.size() != features.descriptors.rows)
        throw std::invalid_argument(std::string(name) +
                                    " points and descriptors are misaligned");
    if(features.descriptors.cols != kSiftDimension)
        throw std::invalid_argument(std::string(name) +
                                    " descriptors must have 128 columns");
}

// squared distance, or a partial sum once it reaches bound
// squared distances preserve Lowe ordering, square root only for accepted matches
double squared_distance(const float* first, const float* second, double bound) {
    double result = 0.0;
    for(std::size_t start = 0 ; start < kSiftDimension ; start += kDistanceBlock) {
        // four independent sums, so the additions do not wait on each other
        double sums[4] = {0.0, 0.0, 0.0, 0.0};
        for(std::size_t i = start ; i < start + kDistanceBlock ; i += 4) {
            for(std::size_t lane = 0 ; lane < 4 ; ++lane) {
                const double difference =
                    static_cast<double>(first[i + lane]) - second[i + lane];
                sums[lane] += difference * difference;
            }
        }
        // SIFT values are integers 0..255, so the sum is exact in any order
        result += (sums[0] + sums[1]) + (sums[2] + sums[3]);
        // terms are nonnegative: partial >= bound fails D < best and D < second_best like the full D
        if(result >= bound)
            return result;
    }
    return result;
}

}  // namespace

std::vector<DescriptorMatch> match_descriptors(
    const LocalFeatureSet& query, const LocalFeatureSet& candidate,
    double rho) {
    if(!std::isfinite(rho) || rho <= 0.0 || rho >= 1.0)
        throw std::invalid_argument("rho must be finite and strictly between 0 and 1");
    validate_features(query, "query");
    validate_features(candidate, "candidate");

    // Lowe threshold for squared distances
    const double rho_squared = rho * rho;

    std::vector<DescriptorMatch> matches;
    // fewer than two candidates gives an empty result
    if(candidate.descriptors.rows < 2)
        return matches;
    matches.reserve(query.descriptors.rows);

    for(std::size_t query_index = 0 ; query_index < query.descriptors.rows ;
        ++query_index) {
        double best = std::numeric_limits<double>::infinity();
        double second_best = std::numeric_limits<double>::infinity();
        std::size_t best_index = 0;

        for(std::size_t candidate_index = 0 ;
            candidate_index < candidate.descriptors.rows ; ++candidate_index) {
            // only distances below second_best change best or second_best
            const double distance = squared_distance(
                query.descriptors.row(query_index),
                candidate.descriptors.row(candidate_index), second_best);
            // strict comparisons keep the lowest index on ties
            if(distance < best) {
                second_best = best;
                best = distance;
                best_index = candidate_index;
            } else if(distance < second_best) {
                second_best = distance;
            }
        }

        // zero second distance makes the ratio undefined
        // D1 = d1^2, D2 = d2^2
        // d1 < rho d2  <=>  D1 < rho^2 D2
        if(std::isfinite(second_best) && second_best > 0.0 &&
           best < rho_squared * second_best) {
            matches.push_back({
                query.points[query_index],
                candidate.points[best_index],
                std::sqrt(best),
                query_index,
                best_index});
        }
    }
    return matches;
}

}  // namespace image_search
