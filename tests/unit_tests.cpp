#include "dataset.hpp"
#include "descriptor_matching.hpp"
#include "geometric_verification.hpp"
#include "homography.hpp"
#include "local_features.hpp"
#include "kmeans.hpp"
#include "search.hpp"

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
}

// Test normalized DLT estimation and geometric degeneracy rejection.
void testHomographyEstimation() {
    const std::vector<image_search::PointCorrespondence> correspondences = {
        {{0.0, 0.0}, {4.0, -2.0}},
        {{10.0, 0.0}, {14.0, -2.0}},
        {{0.0, 10.0}, {4.0, 8.0}},
        {{10.0, 10.0}, {14.0, 8.0}}};
    const auto result = image_search::estimate_homography(correspondences);
    check(result.valid, "normalized DLT rejected a valid translation");
    for(const auto& correspondence : correspondences) {
        const auto projected = image_search::project_point(
            result.matrix, correspondence.source);
        check(projected.has_value(), "estimated homography did not project");
        check(std::abs((*projected)[0] - correspondence.destination[0]) < 1e-6 &&
              std::abs((*projected)[1] - correspondence.destination[1]) < 1e-6,
              "normalized DLT estimated the wrong homography");
    }

    const std::vector<image_search::PointCorrespondence> collinear = {
        {{0.0, 0.0}, {0.0, 0.0}},
        {{1.0, 1.0}, {1.0, 1.0}},
        {{2.0, 2.0}, {2.0, 2.0}},
        {{3.0, 3.0}, {3.0, 3.0}}};
    check(!image_search::estimate_homography(collinear).valid,
          "normalized DLT accepted collinear points");

    const std::vector<image_search::PointCorrespondence> refit = {
        {{0.0, 0.0}, {4.0, -2.0}},
        {{1.0, 0.0}, {5.0, -2.0}},
        {{2.0, 0.0}, {6.0, -2.0}},
        {{0.0, 1.0}, {4.0, -1.0}},
        {{1.0, 1.0}, {5.0, -1.0}}};
    check(image_search::estimate_homography(refit).valid,
          "homography refit rejected valid collinear inliers");

    const std::vector<image_search::PointCorrespondence> collinear_refit = {
        {{0.0, 0.0}, {4.0, -2.0}},
        {{1.0, 1.0}, {5.0, -1.0}},
        {{2.0, 2.0}, {6.0, 0.0}},
        {{3.0, 3.0}, {7.0, 1.0}},
        {{4.0, 4.0}, {8.0, 2.0}}};
    check(!image_search::estimate_homography(collinear_refit).valid,
          "homography accepted an all-collinear refit");

    auto duplicate = refit;
    duplicate.back().source = duplicate.front().source;
    check(!image_search::estimate_homography(duplicate).valid,
          "homography accepted duplicate refit points");
}

// Test deterministic RANSAC rejection and refitting with synthetic matches
void testHomographyRansac() {
    std::vector<image_search::DescriptorMatch> matches;
    const std::vector<image_search::Point2> source = {
        {0.0, 0.0}, {10.0, 0.0}, {0.0, 10.0}, {10.0, 10.0},
        {5.0, 3.0}};
    for(std::size_t index = 0; index < source.size(); ++index) {
        matches.push_back({source[index],
                           {source[index][0] + 20.0,
                            source[index][1] + 10.0},
                           0.0, index, index});
    }
    matches.push_back({{30.0, 30.0}, {2000.0, 1500.0}, 0.0, 5, 5});
    matches.push_back({{70.0, 50.0}, {150.0, 1800.0}, 0.0, 6, 6});

    image_search::RansacConfig config;
    config.maximum_trials = 500;
    const auto first = image_search::estimate_homography_ransac(matches, config);
    const auto second = image_search::estimate_homography_ransac(matches, config);
    check(first.valid && first.inlier_indices.size() == 5,
          "RANSAC did not isolate the synthetic inliers");
    check(first.inlier_indices == second.inlier_indices &&
              first.matrix == second.matrix,
          "RANSAC is not deterministic for a fixed seed");
    check(first.median_error.has_value() && *first.median_error < config.tau,
          "RANSAC refit has unexpected reprojection error");
    checkThrows(
        [&] { image_search::estimate_homography_ransac(matches, {0.0}); },
        "RANSAC accepted an invalid reprojection threshold");
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
