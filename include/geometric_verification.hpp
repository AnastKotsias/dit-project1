#ifndef GEOMETRIC_VERIFICATION_HPP
#define GEOMETRIC_VERIFICATION_HPP

#include "descriptor_matching.hpp"
#include "homography.hpp"
#include "search.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace image_search {

struct RansacConfig {
    // strict reprojection error threshold in pixels
    double tau = 3.0;
    // maximum number of random hypotheses
    std::size_t maximum_trials = 2000;
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

// RANSAC outcome of one TopC candidate
struct CandidateVerification {
    SearchCandidate candidate;
    // matches accepted by the Lowe test
    std::size_t match_count = 0;
    // invalid when RANSAC fails
    HomographyResult homography;
};

// S(I) = number of final inliers, zero on RANSAC failure
std::size_t final_inlier_count(const CandidateVerification& verification);

// more final inliers first, then smaller median error,
// then smaller original rank
std::vector<CandidateVerification> rerank_candidates(
    std::vector<CandidateVerification> verifications);

}  // namespace image_search

#endif
