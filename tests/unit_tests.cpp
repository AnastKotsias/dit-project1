#include "dataset.hpp"
#include "descriptor_matching.hpp"
#include "geometric_verification.hpp"
#include "homography.hpp"
#include "local_features.hpp"
#include "kmeans.hpp"
#include "search.hpp"

#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using image_search::ExactSearch;
using image_search::Matrix;

// Check that the given condition is true. If it is not, the test fails with the given message.
void check(bool condition, const std::string& message) {
    if(!condition)
        throw std::runtime_error(message);
}

// Check that the given operation throws an exception. If it does not, the test fails with the given message.
void checkThrows(const std::function<void()>& operation, const std::string& message) {
    bool threw = false;
    try {
        operation();
    } catch(const std::exception&) {
        threw = true;
    }
    check(threw, message);
}

// Check that H equals the expected matrix up to scale. The expected matrix has h33 = 1.
bool equalUpToScale(const image_search::Matrix3& estimated,
                    const image_search::Matrix3& expected) {
    // H ~ lambda H, fix lambda by h33 = 1
    if(std::abs(estimated[8]) < 1e-12)
        return false;
    for(std::size_t index = 0; index < 9; ++index) {
        if(std::abs(estimated[index] / estimated[8] - expected[index]) > 1e-6)
            return false;
    }
    return true;
}

// Check that H maps every source point onto its destination with near-zero error.
bool reprojectsExactly(
    const image_search::Matrix3& homography,
    const std::vector<image_search::PointCorrespondence>& correspondences) {
    for(const auto& correspondence : correspondences) {
        const auto projected =
            image_search::project_point(homography, correspondence.source);
        if(!projected.has_value() ||
           std::hypot((*projected)[0] - correspondence.destination[0],
                      (*projected)[1] - correspondence.destination[1]) > 1e-6)
            return false;
    }
    return true;
}

// Test the Matrix class and the squared_distance function.
void testMatrixAndDistance() {
    Matrix<double> matrix(2, 3);    // allocates a 2x3 matrix of zeros
    matrix.row(0)[1] = 2.0;
    matrix.row(1)[2] = 3.0;

    check(matrix.rows == 2 && matrix.cols == 3, "matrix dimensions are incorrect");
    check(matrix.row(0)[1] == 2.0 && matrix.row(1)[2] == 3.0, "matrix rows are not contiguous");   // check that the rows are contiguous
    check(image_search::squared_distance(matrix.row(0), matrix.row(1), 3) == 13.0, "squared distance is incorrect");  // check that the squared distance is correct
}

// Test the k-means clustering implementation.
void testKmeans() {
    Matrix<double> points(4, 2);
    points.data = {0.0, 0.0, 0.0, 2.0, 10.0, 10.0, 10.0, 12.0};     // four points in two clusters

    const auto result = image_search::kmeans(points, 2, 7);
    check(result.centers.rows == 2 && result.centers.cols == 2,
          "k-means returned incorrect center dimensions");
    check(result.assignments.size() == points.rows, "k-means returned incomplete assignments");
    check(result.converged, "k-means did not converge on separated points");

    bool found_low_center = false;
    bool found_high_center = false;
    for(std::size_t center = 0; center < result.centers.rows; ++center) {
        const double x = result.centers.row(center)[0];
        const double y = result.centers.row(center)[1];
        found_low_center = found_low_center || (x == 0.0 && y == 1.0);
        found_high_center = found_high_center || (x == 10.0 && y == 11.0);
    }
    check(found_low_center && found_high_center, "k-means centers are incorrect");
    checkThrows([&] { image_search::kmeans(points, 0, 1); }, "k-means accepted zero clusters");      // check that k-means throws an exception when the number of clusters is zero
    checkThrows([&] { image_search::kmeans(points, 5, 1); }, "k-means accepted too many clusters");  // check that k-means throws an exception when the number of clusters is greater than the number of points
}

// Test the ExactSearch class.
void testExactSearch() {
    const std::vector<std::vector<double>> database = {
        {1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0},
        {4.0, 0.0}, {5.0, 0.0}, {100.0, 0.0}
    };
    const std::vector<std::string> ids = {
        "sequence/2", "sequence/3", "sequence/4",
        "sequence/5", "sequence/6", "other/2"
    };

    const ExactSearch search(database, ids);
    const auto result = search.search({0.0, 0.0}, "query", "sequence");
    check(result.query_id == "query", "exact search returned the wrong query id");
    check(result.top_10_ids == ids, "exact search ranking is incorrect");
    check(result.recall_5 == 1.0 && result.recall_10 == 1.0, "exact search recall is incorrect");
    check(result.ap_10 == 1.0, "exact search average precision is incorrect");
    check(result.candidates_examined == database.size(), "exact search examined the wrong number of candidates");

    const auto candidates = search.candidates({0.0, 0.0}, "query", 3);
    check(candidates.query_id == "query" &&
              candidates.candidates.size() == 3 &&
              candidates.candidates[0].database_index == 0 &&
              candidates.candidates[0].original_rank == 0 &&
              candidates.candidates[0].id == "sequence/2" &&
              candidates.candidates[2].database_index == 2 &&
              candidates.candidates_examined == database.size(),
          "exact candidate retrieval returned the wrong ranking metadata");
    checkThrows([&] { search.candidates({0.0, 0.0}, "query", 0); },
                "exact candidate retrieval accepted zero candidates");
    checkThrows([&] { search.search({0.0}, "query", "sequence"); }, "exact search accepted a query with the wrong dimension");
    checkThrows([&] { ExactSearch({}, {}); }, "exact search accepted an empty database");
}

// Test the dataset path generation.
void testDatasetPaths() {
    const auto paths = image_search::get_training_image_paths("dataset", {"v_bird", "i_bike"});

    check(paths.size() == 12, "training path count is incorrect");
    check(paths.front() == std::filesystem::path("dataset") / "v_bird" / "1.ppm", "first training path is incorrect");
    check(paths.back() == std::filesystem::path("dataset") / "i_bike" / "6.ppm", "last training path is incorrect");
}

// Test the local-feature extraction status for an unreadable image.
void testLocalFeatures() {
    const auto result = image_search::extract_sift_features(
        std::filesystem::path("missing-image-for-unit-test.pgm"));

    check(result.status == image_search::SiftExtractionStatus::ImageUnreadable,
          "local feature extraction did not report an unreadable image");
    check(result.points.empty() && result.descriptors.empty(),
          "unreadable image returned local features");
}

// Test strict Lowe-ratio matching, tie handling and insufficient candidates.
void testDescriptorMatching() {
    image_search::LocalFeatureSet query;
    query.points = {{{1.0, 2.0}, {3.0, 4.0}}};
    query.descriptors = image_search::DescriptorMatrix(2, 128);
    query.descriptors.row(1)[0] = 10.0F;

    image_search::LocalFeatureSet candidate;
    candidate.points = {{{5.0, 6.0}, {7.0, 8.0}, {9.0, 10.0}}};
    candidate.descriptors = image_search::DescriptorMatrix(3, 128);
    candidate.descriptors.row(1)[0] = 1.0F;
    candidate.descriptors.row(2)[0] = 20.0F;

    const auto matches = image_search::match_descriptors(query, candidate, 0.5);
    check(matches.size() == 1, "descriptor ratio matching accepted the wrong matches");
    check(matches.front().query_descriptor_index == 0 &&
          matches.front().candidate_descriptor_index == 0,
          "descriptor matching selected the wrong nearest descriptors");

    candidate.descriptors.data[3] = 0.0F;
    const auto tied = image_search::match_descriptors(query, candidate, 0.5);
    check(tied.front().candidate_descriptor_index == 0,
          "descriptor matching did not use the lowest index for a tie");
    checkThrows([&] { image_search::match_descriptors(query, candidate, 0.0); },
                "descriptor matching accepted an invalid rho");

    candidate.points.resize(1);
    candidate.descriptors = image_search::DescriptorMatrix(1, 128);
    check(image_search::match_descriptors(query, candidate, 0.5).empty(),
          "descriptor matching accepted a candidate with one descriptor");

    // d1 / d2 = 0.9 fails rho = 0.85, though D1 / D2 = 0.81
    image_search::LocalFeatureSet single_query;
    single_query.points = {{{1.0, 2.0}}};
    single_query.descriptors = image_search::DescriptorMatrix(1, 128);
    image_search::LocalFeatureSet ratio_candidate;
    ratio_candidate.points = {{{5.0, 6.0}, {7.0, 8.0}}};
    ratio_candidate.descriptors = image_search::DescriptorMatrix(2, 128);
    ratio_candidate.descriptors.row(0)[0] = 9.0F;
    ratio_candidate.descriptors.row(1)[0] = 10.0F;
    check(image_search::match_descriptors(single_query, ratio_candidate, 0.85).empty(),
          "descriptor matching compared squared distances with rho instead of rho squared");

    // d1 / d2 = 0.8 passes rho = 0.85
    ratio_candidate.descriptors.row(0)[0] = 8.0F;
    const auto ratio_matches =
        image_search::match_descriptors(single_query, ratio_candidate, 0.85);
    check(ratio_matches.size() == 1 &&
              std::abs(ratio_matches.front().descriptor_distance - 8.0) < 1e-12,
          "descriptor matching rejected a match below the Lowe ratio");
}

// Test homography projection and invalid denominator handling.
void testHomographyProjection() {
    const image_search::Matrix3 translation = {
        1.0, 0.0, 4.0,
        0.0, 1.0, -2.0,
        0.0, 0.0, 1.0};
    const auto projected = image_search::project_point(translation, {3.0, 5.0});
    check(projected.has_value() && (*projected == image_search::Point2{7.0, 3.0}),
          "homography projection is incorrect");

    const image_search::Matrix3 zero_denominator = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0};
    check(!image_search::project_point(zero_denominator, {1.0, 2.0}).has_value(),
          "homography accepted a zero projection denominator");

    const image_search::Matrix3 scaled_translation = {
        1e-20, 0.0, 4e-20,
        0.0, 1e-20, -2e-20,
        0.0, 0.0, 1e-20};
    const auto scaled_projected =
        image_search::project_point(scaled_translation, {3.0, 5.0});
    check(scaled_projected.has_value() &&
              std::abs((*scaled_projected)[0] - 7.0) < 1e-12 &&
              std::abs((*scaled_projected)[1] - 3.0) < 1e-12,
          "homography projection is not scale invariant");
}

// Test normalized DLT estimation and geometric degeneracy rejection.
void testHomographyEstimation() {
    // tutorial check 1: p'_i = p_i gives H ~ I3
    const std::vector<image_search::PointCorrespondence> identity = {
        {{1.0, 2.0}, {1.0, 2.0}},
        {{40.0, 5.0}, {40.0, 5.0}},
        {{8.0, 30.0}, {8.0, 30.0}},
        {{35.0, 42.0}, {35.0, 42.0}}};
    const auto identity_result = image_search::estimate_homography(identity);
    check(identity_result.valid &&
              equalUpToScale(identity_result.matrix,
                             {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}),
          "normalized DLT did not estimate H ~ I3 for the identity");
    check(reprojectsExactly(identity_result.matrix, identity),
          "identity homography has non-zero reprojection error");

    // tutorial check 2: (x, y) <-> (x + 20, y + 10)
    const std::vector<image_search::PointCorrespondence> correspondences = {
        {{0.0, 0.0}, {20.0, 10.0}},
        {{10.0, 0.0}, {30.0, 10.0}},
        {{0.0, 10.0}, {20.0, 20.0}},
        {{10.0, 10.0}, {30.0, 20.0}}};
    const image_search::Matrix3 translation = {
        1.0, 0.0, 20.0,
        0.0, 1.0, 10.0,
        0.0, 0.0, 1.0};
    const auto result = image_search::estimate_homography(correspondences);
    check(result.valid && equalUpToScale(result.matrix, translation),
          "normalized DLT did not estimate the known translation");
    check(reprojectsExactly(result.matrix, correspondences),
          "translation homography has non-zero reprojection error");

    check(!image_search::is_degenerate_sample(correspondences),
          "a non-degenerate minimal sample was rejected");

    const std::vector<image_search::PointCorrespondence> collinear = {
        {{0.0, 0.0}, {0.0, 0.0}},
        {{1.0, 1.0}, {1.0, 1.0}},
        {{2.0, 2.0}, {2.0, 2.0}},
        {{3.0, 3.0}, {3.0, 3.0}}};
    check(image_search::is_degenerate_sample(collinear),
          "minimal sample check accepted collinear points");

    // repeated point in one image
    auto repeated_sample = correspondences;
    repeated_sample[3].destination = repeated_sample[0].destination;
    check(image_search::is_degenerate_sample(repeated_sample),
          "minimal sample check accepted a repeated point");

    // collinear triple in the destination only
    auto one_image_collinear = correspondences;
    one_image_collinear[2].destination = {25.0, 10.0};
    check(image_search::is_degenerate_sample(one_image_collinear),
          "minimal sample check accepted a collinear triple in one image");

    auto five_points = correspondences;
    five_points.push_back({{5.0, 3.0}, {25.0, 13.0}});
    checkThrows([&] { image_search::is_degenerate_sample(five_points); },
                "minimal sample check accepted more than four correspondences");

    const std::vector<image_search::PointCorrespondence> refit = {
        {{0.0, 0.0}, {4.0, -2.0}},
        {{1.0, 0.0}, {5.0, -2.0}},
        {{2.0, 0.0}, {6.0, -2.0}},
        {{0.0, 1.0}, {4.0, -1.0}},
        {{1.0, 1.0}, {5.0, -1.0}}};
    check(image_search::estimate_homography(refit).valid,
          "homography refit rejected valid collinear inliers");

    // rank-two map u = 0.1 x + 0.2 y, v = 0
    const std::vector<image_search::PointCorrespondence> onto_line = {
        {{0.0, 0.0}, {0.0, 0.0}},
        {{10.0, 0.0}, {1.0, 0.0}},
        {{0.0, 10.0}, {2.0, 0.0}},
        {{10.0, 10.0}, {3.0, 0.0}}};
    check(!image_search::estimate_homography(onto_line).valid,
          "homography accepted a non-invertible estimate");

    // one destination point, d' = 0
    auto one_destination = correspondences;
    for(auto& correspondence : one_destination)
        correspondence.destination = {7.0, 7.0};
    check(!image_search::estimate_homography(one_destination).valid,
          "homography accepted a zero normalization mean distance");

    // all-collinear matches, every sample degenerate
    std::vector<image_search::DescriptorMatch> collinear_matches;
    for(std::size_t index = 0; index < 5; ++index) {
        const double t = static_cast<double>(index);
        collinear_matches.push_back({{t, t}, {t + 4.0, t - 2.0}, 0.0, index, index});
    }
    check(!image_search::estimate_homography_ransac(collinear_matches, {}).valid,
          "RANSAC accepted all-collinear matches");

    // refit accepts repeated inlier points
    auto duplicate = refit;
    duplicate.push_back(duplicate.front());
    const auto duplicate_result = image_search::estimate_homography(duplicate);
    check(duplicate_result.valid,
          "homography refit rejected repeated inlier points");
    const auto duplicate_projected = image_search::project_point(
        duplicate_result.matrix, {1.0, 1.0});
    check(duplicate_projected.has_value() &&
              std::abs((*duplicate_projected)[0] - 5.0) < 1e-6 &&
              std::abs((*duplicate_projected)[1] + 1.0) < 1e-6,
          "homography refit with repeated points estimated the wrong homography");

    const std::vector<image_search::PointCorrespondence> nearly_collinear = {
        {{0.0, 0.0}, {0.0, 0.0}},
        {{1000000.0, 0.0}, {1000000.0, 0.0}},
        {{2000000.0, 1.0}, {2000000.0, 1.0}},
        {{0.0, 1000000.0}, {0.0, 1000000.0}}};
    check(image_search::is_degenerate_sample(nearly_collinear),
          "minimal sample check accepted a scale-normalized near-collinear sample");
}

// Test deterministic RANSAC rejection and refitting with synthetic matches
void testHomographyRansac() {
    std::vector<image_search::DescriptorMatch> matches;
    // tutorial check: 5 correct pairs (x, y) -> (x + 20, y + 10)
    // spread around the outliers, so no projective H fits both within tau
    const std::vector<image_search::Point2> source = {
        {0.0, 0.0}, {100.0, 0.0}, {0.0, 100.0}, {100.0, 100.0},
        {50.0, 30.0}};
    for(std::size_t index = 0; index < source.size(); ++index) {
        matches.push_back({source[index],
                           {source[index][0] + 20.0,
                            source[index][1] + 10.0},
                           0.0, index, index});
    }
    // tutorial outliers
    matches.push_back({{30.0, 30.0}, {200.0, 150.0}, 0.0, 5, 5});
    matches.push_back({{70.0, 50.0}, {15.0, 180.0}, 0.0, 6, 6});

    image_search::RansacConfig config;
    config.maximum_trials = 500;
    const auto first = image_search::estimate_homography_ransac(matches, config);
    const auto second = image_search::estimate_homography_ransac(matches, config);
    // 5 correct pairs are inliers, matches 5 and 6 are outliers
    const std::vector<std::size_t> expected_inliers = {0, 1, 2, 3, 4};
    check(first.valid && first.inlier_indices == expected_inliers,
          "RANSAC did not isolate the synthetic inliers");
    check(equalUpToScale(first.matrix,
                         {1.0, 0.0, 20.0, 0.0, 1.0, 10.0, 0.0, 0.0, 1.0}),
          "RANSAC did not estimate the expected translation");
    check(first.inlier_indices == second.inlier_indices &&
              first.matrix == second.matrix,
          "RANSAC is not deterministic for a fixed seed");
    check(first.median_error.has_value() && *first.median_error < config.tau,
          "RANSAC refit has unexpected reprojection error");
    checkThrows(
        [&] { image_search::estimate_homography_ransac(matches, {0.0}); },
        "RANSAC accepted an invalid reprojection threshold");

    // repeated inlier, as with multi-orientation SIFT keypoints
    auto repeated = matches;
    repeated.push_back(matches.front());
    const auto repeated_result =
        image_search::estimate_homography_ransac(repeated, config);
    check(repeated_result.valid && repeated_result.inlier_indices.size() == 6,
          "RANSAC failed when an inlier location was repeated");
}

}

int main() {
    try {
        // Each function tests one reusable part of the code. An exception stops the run and is converted into a non-zero test result for CTest.
        testMatrixAndDistance();
        testKmeans();
        testExactSearch();
        testDatasetPaths();
        testLocalFeatures();
        testDescriptorMatching();
        testHomographyProjection();
        testHomographyEstimation();
        testHomographyRansac();
    } catch(const std::exception& error) {
        std::cerr << "unit test failure: " << error.what() << '\n';
        return 1;
    }
    std::cout << "all unit tests passed\n";
    return 0;
}
