#include "geometry/descriptor_matching.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace image_search {
namespace {

constexpr std::size_t kSiftDimension = 128;
// D <= 128 * 255^2 = 8,323,200 < 2^31
static_assert(kSiftDimension * 255 * 255 <
              static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()));

void validate_features(const LocalFeatureSet& features, const char* name) {
    // point coordinates and descriptor rows must be aligned
    if(features.points.size() != features.descriptors.rows)
        throw std::invalid_argument(std::string(name) +
                                    " points and descriptors are misaligned");
    if(features.descriptors.cols != kSiftDimension)
        throw std::invalid_argument(std::string(name) +
                                    " descriptors must have 128 columns");
}

// D = sum (a_i - b_i)^2 of two byte rows, exact in int32
// squared distances preserve Lowe ordering, square root only for accepted matches
// no early exit, so the compiler can vectorize the loop
std::int32_t squared_distance(const std::uint8_t* first, const std::uint8_t* second) {
    std::int32_t result = 0;
    for(std::size_t i = 0 ; i < kSiftDimension ; ++i) {
        // |a_i - b_i| <= 255
        const std::int32_t difference =
            static_cast<std::int32_t>(first[i]) - static_cast<std::int32_t>(second[i]);
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
        // above any D, replaced after two candidate rows
        std::int32_t best = std::numeric_limits<std::int32_t>::max();
        std::int32_t second_best = std::numeric_limits<std::int32_t>::max();
        std::size_t best_index = 0;

        for(std::size_t candidate_index = 0 ;
            candidate_index < candidate.descriptors.rows ; ++candidate_index) {
            const std::int32_t distance = squared_distance(
                query.descriptors.row(query_index),
                candidate.descriptors.row(candidate_index));
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
        // d1 < rho d2  <=>  D1 < rho^2 D2, in double
        const auto first_distance = static_cast<double>(best);
        const auto second_distance = static_cast<double>(second_best);
        if(second_best > 0 && first_distance < rho_squared * second_distance) {
            matches.push_back({
                query.points[query_index],
                candidate.points[best_index],
                std::sqrt(first_distance),
                query_index,
                best_index});
        }
    }
    return matches;
}

}  // namespace image_search
