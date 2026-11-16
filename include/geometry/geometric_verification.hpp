#ifndef GEOMETRIC_VERIFICATION_HPP
#define GEOMETRIC_VERIFICATION_HPP

#include "geometry/descriptor_matching.hpp"
#include "geometry/homography.hpp"
#include "search/search.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
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

// match_sift(Q, I, rho), then estimate_ransac on the accepted matches
// no HPatches ground truth used here
CandidateVerification verify_candidate(
    const SearchCandidate& candidate,
    const LocalFeatureSet& query_features,
    const LocalFeatureSet& candidate_features,
    double rho,
    const RansacConfig& config);

// S(I) = number of final inliers, zero on RANSAC failure
std::size_t final_inlier_count(const CandidateVerification& verification);

// more final inliers first, then smaller median error,
// then smaller original rank
std::vector<CandidateVerification> rerank_candidates(
    std::vector<CandidateVerification> verifications);

// comparison ranking without RANSAC: more Lowe-accepted matches first,
// then smaller original rank, the homography is ignored
std::vector<CandidateVerification> rank_by_match_count(
    std::vector<CandidateVerification> verifications);

// geometric verification outcome of one query
struct VerificationResult {
    // every TopC candidate after reranking, the first 10 are returned
    std::vector<CandidateVerification> ranking;
    // T_verify: matching, Lowe test, RANSAC and reranking
    double verification_time_ms = 0.0;
};

// load_features(image_id): stored features of a database index, no extraction
using FeatureLookup = std::function<const LocalFeatureSet&(std::size_t)>;

// verify_candidate for every TopC(Q) candidate, then rerank
VerificationResult verify_candidates(
    const std::vector<SearchCandidate>& candidates,
    const LocalFeatureSet& query_features,
    const FeatureLookup& load_features,
    double rho,
    const RansacConfig& config);

}  // namespace image_search

#endif
