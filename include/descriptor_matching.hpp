#ifndef DESCRIPTOR_MATCHING_HPP
#define DESCRIPTOR_MATCHING_HPP

#include "local_features.hpp"

#include <cstddef>
#include <vector>

namespace image_search {

struct DescriptorMatch {
    // coordinates aligned with descriptor rows
    Point2 query_point;
    Point2 candidate_point;
    // Euclidean distance; squared distances used internally
    double descriptor_distance = 0.0;
    std::size_t query_descriptor_index = 0;
    std::size_t candidate_descriptor_index = 0;
};

// two nearest candidate descriptors per query descriptor
// strict Lowe condition d1 < rho * d2
// evaluated on squared distances as D1 < rho^2 * D2
// one 128-dimensional descriptor row per point
// finite rho strictly between zero and one
std::vector<DescriptorMatch> match_descriptors(
    const LocalFeatureSet& query,
    const LocalFeatureSet& candidate,
    double rho);

}  // namespace image_search

#endif
