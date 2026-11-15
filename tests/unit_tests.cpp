#include "dataset.hpp"
#include "descriptor_matching.hpp"
#include "geometric_evaluation.hpp"
#include "geometric_verification.hpp"
#include "homography.hpp"
#include "local_features.hpp"
#include "pipeline.hpp"
#include "kmeans.hpp"
#include "search.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
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

// Check that TopC of one index agrees with its search() ranking.
template<typename Index>
void checkCandidateRetrieval(const Index& index, const std::vector<double>& query,
                             const std::string& name) {
    const auto top = index.search(query, "query", "sequence");
    const auto first_10 = index.candidates(query, "query", 10);
    const auto first_25 = index.candidates(query, "query", 25);
    check(!first_10.candidates.empty(), name + " returned no candidates");
    check(first_10.query_id == "query", name + " candidates have the wrong query id");
    check(first_25.candidates.size() <= 25, name + " returned more than C candidates");

    // the first 10 of TopC are the top 10 of search()
    std::vector<std::string> first_10_ids;
    for(const auto& candidate : first_10.candidates)
        first_10_ids.push_back(candidate.id);
    check(first_10_ids == top.top_10_ids,
          name + " TopC does not start with the search() ranking");
    // a larger C extends the same ranking
    for(std::size_t rank = 0; rank < first_25.candidates.size(); ++rank) {
        const auto& candidate = first_25.candidates[rank];
        check(candidate.original_rank == rank,
              name + " candidate has the wrong original rank");
        if(rank < first_10.candidates.size())
            check(candidate.database_index == first_10.candidates[rank].database_index,
                  name + " TopC changes with C");
    }
    checkThrows([&] { index.candidates(query, "query", 0); },
                name + " accepted zero candidates");
}

// Test TopC retrieval of the approximate methods
void testCandidateRetrieval() {
    // 40 seeded points in 4 dimensions
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> coordinate(0.0, 1.0);
    std::vector<std::vector<double>> database(40, std::vector<double>(4));
    std::vector<std::string> ids;
    for(std::size_t i = 0; i < database.size(); ++i) {
        for(double& value : database[i])
            value = coordinate(rng);
        ids.push_back("image" + std::to_string(i));
    }
    const std::vector<double> query = {0.5, 0.5, 0.5, 0.5};

    // parameters that leave at least ten candidates
    checkCandidateRetrieval(image_search::LSHSearch(database, ids, 2, 3, 10.0), query, "LSH");
    checkCandidateRetrieval(
        image_search::HypercubeSearch(database, ids, 3, 10.0, 40, 8), query, "Hypercube");
    checkCandidateRetrieval(image_search::IVFFlatSearch(database, ids, 2, 2), query, "IVFFlat");
    checkCandidateRetrieval(
        image_search::IVFPQSearch(database, ids, 2, 2, 2, 2), query, "IVFPQ");

    // M = 5 gives fewer than C candidates
    const image_search::HypercubeSearch limited(database, ids, 3, 10.0, 5, 8);
    check(limited.candidates(query, "query", 10).candidates.size() == 5,
          "Hypercube TopC exceeded the candidate budget M");
}

// Test the dataset path generation.
void testDatasetPaths() {
    const auto paths = image_search::get_training_image_paths("dataset", {"v_bird", "i_bike"});

    check(paths.size() == 12, "training path count is incorrect");
    check(paths.front() == std::filesystem::path("dataset") / "v_bird" / "1.ppm", "first training path is incorrect");
    check(paths.back() == std::filesystem::path("dataset") / "i_bike" / "6.ppm", "last training path is incorrect");
}

// Write text to a temporary file and return its path.
std::filesystem::path writeTemporaryFile(const std::string& name, const std::string& text) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream file(path);
    file << text;
    check(static_cast<bool>(file), "could not write temporary file " + path.string());
    return path;
}

// Test the HPatches ground-truth homography path and file parsing
void testGroundTruthHomography() {
    check(image_search::ground_truth_homography_path("hp", "v_bird", 2) ==
              std::filesystem::path("hp") / "v_bird" / "H_1_2",
          "ground truth homography path is incorrect");
    checkThrows([] { image_search::ground_truth_homography_path("hp", "v_bird", 1); },
                "ground truth path accepted the query image");
    checkThrows([] { image_search::ground_truth_homography_path("hp", "v_bird", 7); },
                "ground truth path accepted a missing image");

    // i_leuven layout: several spaces, upper-case exponent, trailing blank line
    const auto valid = writeTemporaryFile("image_search_H_valid",
        "0.5  -1E-5  20 \n0  1  10 \n-9.5E-6  0  1 \n\n");
    const image_search::Matrix3 expected = {0.5, -1e-5, 20.0, 0.0, 1.0, 10.0,
                                            -9.5e-6, 0.0, 1.0};
    check(image_search::load_ground_truth_homography(valid) == expected,
          "ground truth homography was not read row by row");

    const auto short_file = writeTemporaryFile("image_search_H_short", "1 0 0\n0 1 0\n0 0\n");
    const auto long_file = writeTemporaryFile("image_search_H_long", "1 0 0\n0 1 0\n0 0 1\n7\n");
    const auto text_file = writeTemporaryFile("image_search_H_text", "1 0 0\n0 x 0\n0 0 1\n");
    for(const auto& path : {short_file, long_file, text_file})
        checkThrows([&] { image_search::load_ground_truth_homography(path); },
                    "malformed ground truth homography was accepted: " + path.string());
    checkThrows([] {
        image_search::load_ground_truth_homography(
            std::filesystem::temp_directory_path() / "image_search_H_missing");
    }, "missing ground truth homography was accepted");

    for(const auto& path : {valid, short_file, long_file, text_file})
        std::filesystem::remove(path);
}

// Test the local-feature extraction status and the stored image size.
void testLocalFeatures() {
    const auto result = image_search::extract_sift_features(
        std::filesystem::path("missing-image-for-unit-test.pgm"));

    check(result.status == image_search::SiftExtractionStatus::ImageUnreadable,
          "local feature extraction did not report an unreadable image");
    check(result.points.empty() && result.descriptors.empty(),
          "unreadable image returned local features");
    check(result.width == 0 && result.height == 0,
          "unreadable image returned a size");

    // blank image: size kept without descriptors
    const auto blank_path =
        std::filesystem::temp_directory_path() / "image_search_blank_test.pgm";
    check(cv::imwrite(blank_path.string(), cv::Mat(48, 64, CV_8UC1, cv::Scalar(128))),
          "could not write the blank test image");
    const auto blank = image_search::extract_sift_features(blank_path);
    std::filesystem::remove(blank_path);
    check(blank.status == image_search::SiftExtractionStatus::NoDescriptors &&
              blank.width == 64 && blank.height == 48,
          "image without descriptors lost its size");

    // checkerboard: keypoints inside a 160 x 120 image
    cv::Mat board(120, 160, CV_8UC1);
    for(int row = 0; row < board.rows; ++row) {
        for(int column = 0; column < board.cols; ++column)
            board.at<unsigned char>(row, column) =
                ((row / 20 + column / 20) % 2 == 0) ? 30 : 220;
    }
    const auto board_path =
        std::filesystem::temp_directory_path() / "image_search_board_test.pgm";
    check(cv::imwrite(board_path.string(), board),
          "could not write the checkerboard test image");
    const auto features = image_search::extract_sift_features(board_path);
    std::filesystem::remove(board_path);
    check(features.status == image_search::SiftExtractionStatus::Success &&
              features.width == 160 && features.height == 120,
          "local feature extraction stored the wrong image size");
    check(features.points.size() == features.descriptors.rows,
          "keypoints and descriptors are misaligned");
    for(const auto& point : features.points) {
        check(point[0] >= 0.0 && point[0] < 160.0 &&
                  point[1] >= 0.0 && point[1] < 120.0,
              "keypoint lies outside the stored image size");
    }
}

// Test that local features are extracted once and only read afterwards
void testLocalFeatureStore() {
    // two checkerboards of different square size and a blank image
    const auto directory = std::filesystem::temp_directory_path();
    std::vector<std::filesystem::path> paths;
    for(const int square : {20, 16}) {
        cv::Mat board(120, 160, CV_8UC1);
        for(int row = 0; row < board.rows; ++row) {
            for(int column = 0; column < board.cols; ++column)
                board.at<unsigned char>(row, column) =
                    ((row / square + column / square) % 2 == 0) ? 30 : 220;
        }
        paths.push_back(directory /
                        ("image_search_store_" + std::to_string(square) + ".pgm"));
        check(cv::imwrite(paths.back().string(), board),
              "could not write the feature store test image");
    }
    paths.push_back(directory / "image_search_store_blank.pgm");
    check(cv::imwrite(paths.back().string(), cv::Mat(48, 64, CV_8UC1, cv::Scalar(128))),
          "could not write the blank feature store test image");

    image_search::LocalFeatureStore store(paths);
    checkThrows([&] { store.features(0); },
                "feature store returned features before loading them");

    // a repeated index is extracted once
    store.load({1, 0, 1});
    check(store.loaded_count() == 2, "feature store loaded the wrong number of images");
    const auto direct = image_search::extract_sift_features(paths[0]);
    const auto& stored = store.features(0);
    check(stored.points == direct.points &&
              stored.descriptors.data == direct.descriptors.data &&
              stored.width == 160 && stored.height == 120,
          "feature store returned different features than direct extraction");
    // already stored, nothing new is extracted
    store.load({0});
    check(store.loaded_count() == 2, "feature store extracted an image twice");

    // a failed load stores nothing
    checkThrows([&] { store.load({2}); },
                "feature store accepted an image without descriptors");
    checkThrows([&] { store.features(2); },
                "feature store kept a failed image");
    checkThrows([&] { store.load({3}); },
                "feature store accepted an out-of-range index");
    check(store.loaded_count() == 2, "failed loads changed the feature store");

    for(const auto& path : paths)
        std::filesystem::remove(path);
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

// Build features where point i has its own descriptor, so it only matches point i of an image built the same way.
image_search::LocalFeatureSet distinctFeatures(const std::vector<image_search::Point2>& points) {
    image_search::LocalFeatureSet features;
    features.points = points;
    features.descriptors = image_search::DescriptorMatrix(points.size(), 128);
    // D1 = 0 for the same row, D2 = 2 * 100^2 for any other row
    for(std::size_t row = 0; row < points.size(); ++row)
        features.descriptors.row(row)[row] = 100.0F;
    features.status = image_search::SiftExtractionStatus::Success;
    return features;
}

// Test matching followed by RANSAC for one candidate
void testCandidateVerification() {
    // tutorial check: 5 correct pairs (x, y) -> (x + 20, y + 10) and 2 wrong matches
    const std::vector<image_search::Point2> query_points = {
        {0.0, 0.0}, {100.0, 0.0}, {0.0, 100.0}, {100.0, 100.0},
        {50.0, 30.0}, {30.0, 30.0}, {70.0, 50.0}};
    std::vector<image_search::Point2> candidate_points;
    for(std::size_t index = 0; index < 5; ++index)
        candidate_points.push_back({query_points[index][0] + 20.0,
                                    query_points[index][1] + 10.0});
    candidate_points.push_back({200.0, 150.0});
    candidate_points.push_back({15.0, 180.0});
    const auto query = distinctFeatures(query_points);
    const auto candidate_features = distinctFeatures(candidate_points);

    const image_search::SearchCandidate candidate{42, 3, "v_bird/2"};
    image_search::RansacConfig config;
    config.maximum_trials = 500;
    const auto verification = image_search::verify_candidate(
        candidate, query, candidate_features, 0.8, config);
    check(verification.candidate.database_index == 42 &&
              verification.candidate.original_rank == 3 &&
              verification.candidate.id == "v_bird/2",
          "candidate verification lost the candidate identity");
    // all 7 descriptor matches pass Lowe, RANSAC keeps the 5 correct ones
    check(verification.match_count == 7,
          "candidate verification did not count the Lowe matches");
    const std::vector<std::size_t> expected_inliers = {0, 1, 2, 3, 4};
    check(verification.homography.valid &&
              verification.homography.inlier_indices == expected_inliers,
          "candidate verification did not isolate the synthetic inliers");
    check(equalUpToScale(verification.homography.matrix,
                         {1.0, 0.0, 20.0, 0.0, 1.0, 10.0, 0.0, 0.0, 1.0}),
          "candidate verification did not estimate the expected translation");

    // M = 3 < 4 matches: RANSAC failure, S(I) = 0
    const auto few = distinctFeatures({{0.0, 0.0}, {100.0, 0.0}, {0.0, 100.0}});
    const auto failed = image_search::verify_candidate(
        candidate, few, candidate_features, 0.8, config);
    check(failed.match_count == 3 && !failed.homography.valid &&
              image_search::final_inlier_count(failed) == 0,
          "candidate verification accepted fewer than four matches");
    checkThrows([&] {
        image_search::verify_candidate(candidate, query, candidate_features, 1.0, config);
    }, "candidate verification accepted an invalid rho");
}

// Test verification of a whole TopC list followed by reranking
void testCandidateListVerification() {
    const std::vector<image_search::Point2> query_points = {
        {0.0, 0.0}, {100.0, 0.0}, {0.0, 100.0}, {100.0, 100.0},
        {50.0, 30.0}, {30.0, 30.0}, {70.0, 50.0}};
    const auto query = distinctFeatures(query_points);

    // database index 10: rows 3 to 6 tie at D1 = D2, so M = 3 and RANSAC fails
    // database index 11: tutorial case, 5 inliers and 2 outliers
    // database index 12: all 7 points translated, 7 inliers
    std::vector<image_search::Point2> translated;
    for(const auto& point : query_points)
        translated.push_back({point[0] + 20.0, point[1] + 10.0});
    auto with_outliers = translated;
    with_outliers[5] = {200.0, 150.0};
    with_outliers[6] = {15.0, 180.0};
    const std::vector<image_search::LocalFeatureSet> database = {
        distinctFeatures({translated.begin(), translated.begin() + 3}),
        distinctFeatures(with_outliers),
        distinctFeatures(translated)};

    std::vector<std::size_t> requested;
    const image_search::FeatureLookup lookup =
        [&](std::size_t index) -> const image_search::LocalFeatureSet& {
            requested.push_back(index);
            return database.at(index - 10);
        };
    const std::vector<image_search::SearchCandidate> candidates = {
        {10, 0, "first"}, {11, 1, "second"}, {12, 2, "third"}};
    image_search::RansacConfig config;
    config.maximum_trials = 500;
    const auto result = image_search::verify_candidates(
        candidates, query, lookup, 0.8, config);

    check(requested == std::vector<std::size_t>({10, 11, 12}),
          "candidate list verification looked up the wrong database indices");
    check(result.ranking.size() == 3 &&
              result.ranking[0].candidate.id == "third" &&
              result.ranking[1].candidate.id == "second" &&
              result.ranking[2].candidate.id == "first",
          "candidate list verification did not rerank by final inliers");
    check(image_search::final_inlier_count(result.ranking[0]) == 7 &&
              image_search::final_inlier_count(result.ranking[1]) == 5 &&
              image_search::final_inlier_count(result.ranking[2]) == 0,
          "candidate list verification returned wrong inlier counts");
    check(result.verification_time_ms >= 0.0,
          "candidate list verification returned a negative time");

    // fewer returned candidates verify only those, none here
    const auto empty = image_search::verify_candidates({}, query, lookup, 0.8, config);
    check(empty.ranking.empty(), "candidate list verification invented candidates");
}

// Build a verified candidate with the given original rank, inlier count and median error.
image_search::CandidateVerification verifiedCandidate(
    std::size_t original_rank, std::size_t inliers, double median) {
    image_search::CandidateVerification verification;
    verification.candidate = {original_rank, original_rank,
                              "candidate" + std::to_string(original_rank)};
    verification.homography.valid = true;
    verification.homography.inlier_indices.resize(inliers);
    verification.homography.median_error = median;
    return verification;
}

// Build a candidate for which RANSAC failed.
image_search::CandidateVerification failedCandidate(std::size_t original_rank) {
    image_search::CandidateVerification verification;
    verification.candidate = {original_rank, original_rank,
                              "candidate" + std::to_string(original_rank)};
    return verification;
}

// Test geometric reranking by inliers, median error and original rank
void testGeometricReranking() {
    // input in original order
    const std::vector<image_search::CandidateVerification> verifications = {
        failedCandidate(0),
        verifiedCandidate(1, 6, 1.0),
        verifiedCandidate(2, 8, 2.0),
        verifiedCandidate(3, 6, 0.5),
        failedCandidate(4),
        // same inliers and median as rank 3
        verifiedCandidate(5, 6, 0.5)};

    // failure counts as zero inliers
    check(image_search::final_inlier_count(verifications[0]) == 0 &&
              image_search::final_inlier_count(verifications[2]) == 8,
          "final inlier count is incorrect");

    // 8 inliers first, then 6 inliers by median, then rank, failures last
    const std::vector<std::size_t> expected = {2, 3, 5, 1, 0, 4};
    const auto reranked = image_search::rerank_candidates(verifications);
    check(reranked.size() == expected.size(),
          "reranking changed the number of candidates");
    for(std::size_t position = 0; position < expected.size(); ++position) {
        check(reranked[position].candidate.original_rank == expected[position],
              "geometric reranking order is incorrect");
    }

    // failed candidates keep their original relative order
    std::vector<image_search::CandidateVerification> all_failed = {
        failedCandidate(2), failedCandidate(0), failedCandidate(1)};
    const auto failed_reranked = image_search::rerank_candidates(all_failed);
    check(failed_reranked[0].candidate.original_rank == 0 &&
              failed_reranked[1].candidate.original_rank == 1 &&
              failed_reranked[2].candidate.original_rank == 2,
          "failed candidates did not keep the original order");
}

}

int main() {
    try {
        // Each function tests one reusable part of the code. An exception stops the run and is converted into a non-zero test result for CTest.
        testMatrixAndDistance();
        testKmeans();
        testExactSearch();
        testCandidateRetrieval();
        testDatasetPaths();
        testGroundTruthHomography();
        testLocalFeatures();
        testLocalFeatureStore();
        testDescriptorMatching();
        testHomographyProjection();
        testHomographyEstimation();
        testHomographyRansac();
        testCandidateVerification();
        testGeometricReranking();
        testCandidateListVerification();
    } catch(const std::exception& error) {
        std::cerr << "unit test failure: " << error.what() << '\n';
        return 1;
    }
    std::cout << "all unit tests passed\n";
    return 0;
}
