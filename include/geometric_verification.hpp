#ifndef GEOMETRIC_VERIFICATION_HPP
#define GEOMETRIC_VERIFICATION_HPP

#include "descriptor_matching.hpp"
#include "homography.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace image_search {

struct RansacConfig {
    // strict reprojection error threshold in pixels
    double tau = 5.0;
    // maximum number of random hypotheses
    std::size_t maximum_trials = 1000;
    // target probability of observing a valid sample
    double confidence = 0.99;
    // enable adaptive trial reduction
    bool adaptive = true;
    // deterministic random seed
    std::uint64_t seed = 1;
};

// estimates and refits a homography from descriptor matches
HomographyResult estimate_homography_ransac(
    const std::vector<DescriptorMatch>& matches,
    const RansacConfig& config);

}  // namespace image_search

#endif
