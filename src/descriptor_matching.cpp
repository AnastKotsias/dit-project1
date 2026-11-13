#include "descriptor_matching.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace image_search {
namespace {

void validate_features(const LocalFeatureSet& features, const char* name) {
    // point coordinates and descriptor rows must be aligned
    if(features.points.size() != features.descriptors.rows)
        throw std::invalid_argument(std::string(name) +
                                    " points and descriptors are misaligned");
    if(features.descriptors.cols != 128)
        throw std::invalid_argument(std::string(name) +
                                    " descriptors must have 128 columns");
}

double squared_distance(const float* first, const float* second,
                        std::size_t dimension) {
    // squared distances preserve Lowe ordering
    // square root only for accepted matches
    double result = 0.0;
    for(std::size_t i = 0 ; i < dimension ; ++i) {
        const double difference = static_cast<double>(first[i]) - second[i];
        result += difference * difference;
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
            const double distance = squared_distance(
                query.descriptors.row(query_index),
                candidate.descriptors.row(candidate_index),
                query.descriptors.cols);
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
        // best and second_best are squared distances D1 = d1^2, D2 = d2^2
        // d1 < rho d2 with d1, d2 >= 0 is equivalent to d1^2 < rho^2 d2^2
        // so the squared test must use rho squared, not rho
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
