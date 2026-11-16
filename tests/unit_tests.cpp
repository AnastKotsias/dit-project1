#include "app/command_line.hpp"
#include "data/dataset.hpp"
#include "geometry/descriptor_matching.hpp"
#include "evaluation/geometric_evaluation.hpp"
#include "geometry/geometric_verification.hpp"
#include "geometry/homography.hpp"
#include "geometry/local_features.hpp"
#include "data/pipeline.hpp"
#include "bovw/kmeans.hpp"
#include "search/search.hpp"
#include "evaluation/verification_output.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <optional>
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

// Test Recall@5, Recall@10 and AP@10 of a ranking before and after reranking
void testRetrievalMetrics() {
    const auto close = [](double first, double second) {
        return std::abs(first - second) < 1e-12;
    };
    // relevant at ranks 1, 3, 6, 10; v_b/2 other sequence; rank 11 ignored
    const std::vector<std::string> before = {
        "v_a/2", "im1.jpg", "v_a/3", "im2.jpg", "im3.jpg", "v_a/4",
        "v_b/2", "im4.jpg", "im5.jpg", "v_a/5", "v_a/6"};
    const auto original = image_search::computeRetrievalMetrics(before, "v_a");
    // AP@10 = (1/1 + 2/3 + 3/6 + 4/10) / 5
    check(close(original.recall_5, 0.4) && close(original.recall_10, 0.8) &&
              close(original.ap_10, (1.0 + 2.0 / 3.0 + 0.5 + 0.4) / 5.0),
          "retrieval metrics of the original ranking are incorrect");

    // same candidates, relevant first: Recall@10 unchanged, AP@10 = 4/5
    const std::vector<std::string> after = {
        "v_a/2", "v_a/3", "v_a/4", "v_a/5", "im1.jpg", "im2.jpg",
        "im3.jpg", "v_b/2", "im4.jpg", "im5.jpg", "v_a/6"};
    const auto reranked = image_search::computeRetrievalMetrics(after, "v_a");
    check(close(reranked.recall_5, 0.8) && close(reranked.recall_10, 0.8) &&
              close(reranked.ap_10, 0.8),
          "retrieval metrics of the reranked ranking are incorrect");

    // fewer than 10 ranks: denominators stay 5
    const auto few = image_search::computeRetrievalMetrics({"im1.jpg", "v_a/2"}, "v_a");
    check(close(few.recall_5, 0.2) && close(few.recall_10, 0.2) && close(few.ap_10, 0.1),
          "retrieval metrics of a short ranking are incorrect");
    const auto none = image_search::computeRetrievalMetrics({}, "v_a");
    check(none.recall_5 == 0.0 && none.recall_10 == 0.0 && none.ap_10 == 0.0,
          "retrieval metrics of an empty ranking are not zero");
}

// Test the search command line: defaults, method options and rejected input
void testSearchArgs() {
    // existing directories for -hp and -mir
    const std::string directory = std::filesystem::temp_directory_path().string();
    const std::vector<std::string> base = {"-hp", directory, "-mir", directory,
                                           "-split", "split.txt", "-vocab", "256"};
    const auto with = [&](const std::vector<std::string>& extra) {
        std::vector<std::string> arguments = base;
        arguments.insert(arguments.end(), extra.begin(), extra.end());
        return arguments;
    };

    const auto exact = image_search::parse_search_args(with({"-exact"}));
    check(exact.method == "exact" && exact.vocab == 256 && exact.s_max == 500 &&
              exact.d_mir == 1000 && exact.set_type == "validation" &&
              exact.out_file == "output.txt" && exact.seed == 1,
          "search arguments did not keep the defaults");
    // -M defaults to 16 for IVFPQ, -kclusters stays unset
    const auto ivfpq = image_search::parse_search_args(with({"-ivfpq", "-nprobe", "3"}));
    check(ivfpq.m_param == 16 && ivfpq.ivf_nprobe == 3 && ivfpq.ivf_kclusters == -1,
          "IVFPQ arguments are incorrect");

    const std::vector<std::vector<std::string>> invalid = {
        with({}),                                // no method
        with({"-exact", "-lsh"}),                // two methods
        with({"-exact", "-vocab", "128"}),       // duplicate option
        with({"-exact", "-unknown", "1"}),       // unknown option
        with({"-exact", "-k", "4"}),             // option of another method
        with({"-exact", "-D", "2000"}),          // D not in {1000, 5000, 10000, 25000}
        with({"-exact", "-S", "50x"}),           // trailing characters
        with({"-lsh", "-w", "nan"}),             // non-finite width
        with({"-ivfpq", "-M", "7"}),             // 256 not divisible by 7
        with({"-exact", "-set"}),                // missing value
        {"-hp", directory, "-mir", directory, "-split", "split.txt", "-exact"}};  // no -vocab
    for(const auto& arguments : invalid)
        checkThrows([&] { image_search::parse_search_args(arguments); },
                    "invalid search arguments were accepted");
    checkThrows([&] {
        image_search::parse_search_args({"-hp", directory + "/image_search_missing", "-mir",
                                         directory, "-split", "split.txt", "-vocab", "256",
                                         "-exact"});
    }, "search arguments accepted a missing -hp directory");
}

// Test the verify command line: defaults, verification options and rejected input
void testVerifyArgs() {
    const std::string directory = std::filesystem::temp_directory_path().string();
    const std::vector<std::string> base = {"-hp", directory, "-mir", directory,
                                           "-split", "split.txt", "-vocab", "256"};
    const auto with = [&](const std::vector<std::string>& extra) {
        std::vector<std::string> arguments = base;
        arguments.insert(arguments.end(), extra.begin(), extra.end());
        return arguments;
    };

    // defaults of the assignment: C 50, rho 0.8, tau 3, T 2000, p 0.99, adaptive 1
    const auto defaults = image_search::parse_verify_args(with({"-exact"}));
    check(defaults.search.method == "exact" && defaults.candidate_count == 50 &&
              defaults.rho == 0.8 && defaults.ransac.tau == 3.0 &&
              defaults.ransac.maximum_trials == 2000 && defaults.ransac.confidence == 0.99 &&
              defaults.ransac.adaptive && defaults.ransac.seed == 1,
          "verify arguments did not keep the defaults");

    // the example command of the assignment, options mixed with search options
    const auto example = image_search::parse_verify_args(with({
        "-set", "validation", "-ivfflat", "-kclusters", "16", "-nprobe", "5", "-C", "20",
        "-rho", "0.7", "-tau", "5", "-T", "500", "-p", "0.95", "-adaptive", "0",
        "-seed", "7", "-o", "-C"}));
    check(example.search.method == "ivfflat" && example.search.ivf_kclusters == 16 &&
              example.search.ivf_nprobe == 5 && example.candidate_count == 20 &&
              example.rho == 0.7 && example.ransac.tau == 5.0 &&
              example.ransac.maximum_trials == 500 && example.ransac.confidence == 0.95 &&
              !example.ransac.adaptive && example.ransac.seed == 7,
          "verify arguments were not parsed");
    // -o -C: a file name, not the -C option
    check(example.search.out_file == "-C", "verify arguments took an option value as an option");

    const std::vector<std::vector<std::string>> invalid = {
        with({"-exact", "-C", "9"}),              // C < 10
        with({"-exact", "-C", "50x"}),            // trailing characters
        with({"-exact", "-rho", "1"}),            // rho not below 1
        with({"-exact", "-rho", "nan"}),          // non-finite rho
        with({"-exact", "-tau", "0"}),            // tau not positive
        with({"-exact", "-tau", "inf"}),          // non-finite tau
        with({"-exact", "-T", "0"}),              // no trials
        with({"-exact", "-p", "1"}),              // log(1 - p) undefined
        with({"-exact", "-adaptive", "2"}),       // not 0 or 1
        with({"-exact", "-C", "20", "-C", "30"}), // duplicate option
        with({"-exact", "-C"}),                   // missing value
        with({"-ivfflat", "-nprobe", "5"}),       // -kclusters required
        with({"-ivfpq", "-nprobe", "5"}),         // -kclusters required
        with({"-experiment"}),                    // no experiment mode
        with({"-exact", "-k", "4"})};             // search check still applied
    for(const auto& arguments : invalid)
        checkThrows([&] { image_search::parse_verify_args(arguments); },
                    "invalid verify arguments were accepted");
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

// Test the corners a_1, ..., a_4 and the corner error of an estimated homography
void testCornerError() {
    const std::array<image_search::Point2, 4> corners = {
        {{0.0, 0.0}, {159.0, 0.0}, {159.0, 119.0}, {0.0, 119.0}}};
    check(image_search::image_corners(160, 120) == corners,
          "image corners are not the corner pixel centers");
    checkThrows([] { image_search::image_corners(0, 120); },
                "image corners accepted an empty image");

    const image_search::Matrix3 identity = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const image_search::Matrix3 translation = {1, 0, 20, 0, 1, 10, 0, 0, 1};
    // H = H_GT, and lambda H ~ H gives the same projections
    const image_search::Matrix3 scaled = {2, 0, 40, 0, 2, 20, 0, 0, 2};
    const auto exact = image_search::corner_error(translation, translation, 160, 120);
    const auto rescaled = image_search::corner_error(scaled, translation, 160, 120);
    check(exact.has_value() && std::abs(*exact) < 1e-12 &&
              rescaled.has_value() && std::abs(*rescaled) < 1e-12,
          "corner error is not zero for the ground truth homography");

    // every corner off by (20, 10)
    const auto shifted = image_search::corner_error(identity, translation, 160, 120);
    check(shifted.has_value() && std::abs(*shifted - std::sqrt(500.0)) < 1e-12,
          "corner error of a missing translation is incorrect");

    // H_GT = 2 I on a 3 x 2 image: corner errors 0, 2, sqrt(5), 1
    const image_search::Matrix3 doubling = {2, 0, 0, 0, 2, 0, 0, 0, 1};
    const auto mean = image_search::corner_error(identity, doubling, 3, 2);
    check(mean.has_value() &&
              std::abs(*mean - (0.0 + 2.0 + std::sqrt(5.0) + 1.0) / 4.0) < 1e-12,
          "corner error is not the mean over the four corners");

    // c = x is zero at corner (0, 0): no valid estimate
    const image_search::Matrix3 at_infinity = {1, 0, 0, 0, 1, 0, 1, 0, 0};
    check(!image_search::corner_error(at_infinity, translation, 160, 120).has_value(),
          "corner error accepted a corner projected to infinity");
    checkThrows([&] {
        image_search::corner_error(identity, at_infinity, 160, 120);
    }, "corner error accepted a ground truth corner at infinity");
}

// Test the corner error and failure cases of the 5 relevant pairs of a query
void testRelevantPairEvaluation() {
    // <hp>/v_test/H_1_j, only for the pairs with a valid estimate
    const auto hp_dir = std::filesystem::temp_directory_path() / "image_search_hp";
    std::filesystem::create_directories(hp_dir / "v_test");
    for(const std::string name : {"H_1_2", "H_1_3", "H_1_5"})
        writeTemporaryFile("image_search_hp/v_test/" + name, "1 0 20\n0 1 10\n0 0 1\n");

    const image_search::Matrix3 translation = {1, 0, 20, 0, 1, 10, 0, 0, 1};
    const image_search::Matrix3 identity = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const image_search::Matrix3 at_infinity = {1, 0, 0, 0, 1, 0, 1, 0, 0};
    const auto verified = [](std::size_t rank, const std::string& id, bool valid,
                             const image_search::Matrix3& matrix) {
        image_search::CandidateVerification verification;
        verification.candidate = {rank, rank, id};
        verification.homography.valid = valid;
        verification.homography.matrix = matrix;
        return verification;
    };
    // v_test/6 not in TopC, non-relevant candidates ignored
    const std::vector<image_search::CandidateVerification> ranking = {
        verified(0, "im7.jpg", true, identity),
        verified(1, "v_test/5", true, at_infinity),
        verified(2, "v_test/3", true, identity),
        verified(3, "v_other/2", true, identity),
        verified(4, "v_test/4", false, identity),
        verified(5, "v_test/2", true, translation)};
    const auto pairs = image_search::evaluate_relevant_pairs(
        ranking, hp_dir.string(), "v_test", 160, 120);

    for(int j = 2; j <= 6; ++j)
        check(pairs[static_cast<std::size_t>(j - 2)].image_number == j,
              "relevant pairs are not ordered by image number");
    // H = H_GT: zero corner error
    check(pairs[0].retrieved && pairs[0].corner_error.has_value() &&
              std::abs(*pairs[0].corner_error) < 1e-12,
          "relevant pair with the ground truth homography has a corner error");
    // every corner off by (20, 10)
    check(pairs[1].retrieved && pairs[1].corner_error.has_value() &&
              std::abs(*pairs[1].corner_error - std::sqrt(500.0)) < 1e-12,
          "relevant pair corner error is incorrect");
    // RANSAC failure, no H_1_4 needed
    check(pairs[2].retrieved && !pairs[2].corner_error.has_value(),
          "failed RANSAC pair received a corner error");
    // corner (0, 0) projected to infinity
    check(pairs[3].retrieved && !pairs[3].corner_error.has_value(),
          "pair with a corner at infinity received a corner error");
    check(!pairs[4].retrieved && !pairs[4].corner_error.has_value(),
          "relevant image outside TopC was evaluated");

    // valid estimate without H_1_j^GT: error, not a silent skip
    checkThrows([&] {
        image_search::evaluate_relevant_pairs(
            {verified(0, "v_test/6", true, identity)}, hp_dir.string(), "v_test", 160, 120);
    }, "missing ground truth homography was skipped");
    std::filesystem::remove_all(hp_dir);
}

// Test the before and after rankings, metrics, times and pairs of one query
void testQueryEvaluation() {
    const auto hp_dir = std::filesystem::temp_directory_path() / "image_search_hp_query";
    std::filesystem::create_directories(hp_dir / "v_q");
    for(const std::string name : {"H_1_2", "H_1_3"})
        writeTemporaryFile("image_search_hp_query/v_q/" + name, "1 0 20\n0 1 10\n0 0 1\n");

    const image_search::Matrix3 translation = {1, 0, 20, 0, 1, 10, 0, 0, 1};
    const image_search::Matrix3 identity = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    // zero inliers: RANSAC failure
    const auto verified = [](std::size_t rank, const std::string& id, std::size_t inliers,
                             const image_search::Matrix3& matrix) {
        image_search::CandidateVerification verification;
        verification.candidate = {rank, rank, id};
        verification.homography.valid = inliers > 0;
        verification.homography.matrix = matrix;
        verification.homography.inlier_indices.resize(inliers);
        if(inliers > 0)
            verification.homography.median_error = 1.0;
        return verification;
    };

    // TopC, C = 12: v_q/2 at rank 2, v_q/3 at rank 11
    image_search::CandidateSearchResult candidates;
    candidates.query_time_ms = 2.0;
    std::vector<image_search::CandidateVerification> verifications;
    for(std::size_t rank = 0; rank < 12; ++rank) {
        std::string id = "im" + std::to_string(rank) + ".jpg";
        std::size_t inliers = 0;
        image_search::Matrix3 matrix = identity;
        if(rank == 1) {
            id = "v_q/2";
            inliers = 30;
            matrix = translation;
        } else if(rank == 10) {
            id = "v_q/3";
            inliers = 20;
        }
        candidates.candidates.push_back({rank, rank, id});
        verifications.push_back(verified(rank, id, inliers, matrix));
    }
    image_search::VerificationResult verification;
    verification.ranking = image_search::rerank_candidates(verifications);
    verification.verification_time_ms = 3.5;

    const auto query = image_search::evaluate_query(
        candidates, verification, hp_dir.string(), "v_q", 160, 120);
    check(query.sequence == "v_q" && query.candidate_count == 12 &&
              query.before_ids.size() == 10 && query.after.size() == 10,
          "query evaluation did not keep the first 10 of TopC");
    check(query.before_ids[1] == "v_q/2" && query.after[0].candidate.id == "v_q/2" &&
              query.after[1].candidate.id == "v_q/3",
          "query evaluation did not keep both rankings");
    // before: relevant at rank 2; after: ranks 1 and 2
    check(std::abs(query.before_metrics.recall_10 - 0.2) < 1e-12 &&
              std::abs(query.before_metrics.ap_10 - 0.1) < 1e-12 &&
              std::abs(query.after_metrics.recall_5 - 0.4) < 1e-12 &&
              std::abs(query.after_metrics.ap_10 - 0.4) < 1e-12,
          "query evaluation metrics are incorrect");
    check(query.candidate_time_ms == 2.0 && query.verification_time_ms == 3.5 &&
              query.total_time_ms == 5.5,
          "query total time is not T_candidates + T_verify");
    // rank 11 of TopC still evaluated
    check(query.pairs[0].corner_error.has_value() &&
              std::abs(*query.pairs[0].corner_error) < 1e-12 &&
              query.pairs[1].corner_error.has_value() &&
              std::abs(*query.pairs[1].corner_error - std::sqrt(500.0)) < 1e-12 &&
              !query.pairs[2].retrieved,
          "query evaluation did not evaluate relevant pairs below rank 10");

    // fewer than 10 candidates: all of them shown
    image_search::CandidateSearchResult few;
    few.candidates.assign(candidates.candidates.begin(), candidates.candidates.begin() + 3);
    image_search::VerificationResult few_verified;
    few_verified.ranking = {verified(0, "im0.jpg", 0, identity),
                            verified(1, "im1.jpg", 0, identity),
                            verified(2, "im2.jpg", 0, identity)};
    const auto short_query = image_search::evaluate_query(
        few, few_verified, hp_dir.string(), "v_q", 160, 120);
    check(short_query.before_ids.size() == 3 && short_query.after.size() == 3,
          "query evaluation padded a short TopC");

    // verification must cover every candidate
    few_verified.ranking.pop_back();
    checkThrows([&] {
        image_search::evaluate_query(few, few_verified, hp_dir.string(), "v_q", 160, 120);
    }, "query evaluation accepted a verification of fewer candidates");
    std::filesystem::remove_all(hp_dir);
}

// Test the verify output format, missing ranks, NA values and the means
void testVerificationOutput() {
    image_search::CandidateVerification valid;
    valid.candidate = {0, 0, "v_a/2"};
    valid.homography.valid = true;
    valid.homography.inlier_indices = {0, 1, 2, 3, 4};
    valid.homography.median_error = 0.5;
    image_search::CandidateVerification failed;
    failed.candidate = {1, 1, "im1.jpg"};

    // TopC of 2: ranks 3 to 10 missing, one corner error of 2 px
    image_search::QueryEvaluation first;
    first.sequence = "v_a";
    first.candidate_count = 2;
    first.before_ids = {"im1.jpg", "v_a/2"};
    first.after = {valid, failed};
    first.before_metrics = {0.2, 0.2, 0.1};
    first.after_metrics = {0.2, 0.2, 0.2};
    first.candidate_time_ms = 2.0;
    first.verification_time_ms = 3.0;
    first.total_time_ms = 5.0;
    first.pairs[0].retrieved = true;
    first.pairs[0].corner_error = 2.0;
    // TopC of 1, no relevant image, no valid estimate
    image_search::QueryEvaluation second;
    second.sequence = "i_b";
    second.candidate_count = 1;
    second.before_ids = {"im1.jpg"};
    second.after = {failed};
    second.candidate_time_ms = 4.0;
    second.verification_time_ms = 1.0;
    second.total_time_ms = 5.0;

    const auto read_output = [](const std::filesystem::path& path) {
        std::ifstream file(path);
        return std::string(std::istreambuf_iterator<char>(file),
                           std::istreambuf_iterator<char>());
    };
    const auto path = std::filesystem::temp_directory_path() / "image_search_verify_output";
    image_search::write_verification_results(path.string(), "Exact", {first, second});
    const std::string text = read_output(path);

    check(text.rfind("Exact\nQuery: v_a\nCandidates: 2\n"
                     "Before-Rank-1: im1.jpg\nBefore-Rank-2: v_a/2\nBefore-Rank-3: -\n", 0) == 0,
          "verify output does not start with the method and first query");
    // inliers 0 and median NA on RANSAC failure
    check(text.find("After-Rank-1: v_a/2\nAfter-Inliers-1: 5\nAfter-MedianError-1: 0.5000\n"
                    "After-Rank-2: im1.jpg\nAfter-Inliers-2: 0\nAfter-MedianError-2: NA\n"
                    "After-Rank-3: -\nAfter-Inliers-3: -\nAfter-MedianError-3: -\n") !=
              std::string::npos,
          "verify output after-ranking lines are incorrect");
    check(text.find("After-MedianError-10: -\nBefore-Recall@5: 0.2000\nBefore-Recall@10: 0.2000\n"
                    "Before-AP@10: 0.1000\nAfter-Recall@5: 0.2000\nAfter-Recall@10: 0.2000\n"
                    "After-AP@10: 0.2000\n\nQuery: i_b\n") != std::string::npos,
          "verify output query metrics are incorrect");
    // 10 pairs, one valid with E_corner = 2 px
    const std::string summary =
        "MeanBeforeRecall@5: 0.1000\nMeanBeforeRecall@10: 0.1000\nMeanBeforeMAP@10: 0.0500\n"
        "MeanAfterRecall@5: 0.1000\nMeanAfterRecall@10: 0.1000\nMeanAfterMAP@10: 0.1000\n"
        "AverageCandidateTimeMs: 3.0000\nAverageVerificationTimeMs: 2.0000\n"
        "AverageTotalTimeMs: 5.0000\nValidHomographyRate: 0.1000\n"
        "AccurateHomographyRate5px: 0.1000\nMeanCornerErrorPx: 2.0000\n"
        "AccurateHomographyRate1px: 0.0000\nAccurateHomographyRate3px: 0.1000\n";
    check(text.size() >= summary.size() &&
              text.compare(text.size() - summary.size(), summary.size(), summary) == 0,
          "verify output summary is incorrect");

    // no valid estimate: mean corner error NA
    image_search::write_verification_results(path.string(), "Exact", {second});
    check(read_output(path).find("MeanCornerErrorPx: NA\n") != std::string::npos,
          "verify output printed a mean corner error without a valid estimate");
    checkThrows([&] { image_search::write_verification_results(path.string(), "Exact", {}); },
                "verify output accepted no queries");
    std::filesystem::remove(path);
}

// Test the valid and accurate rates and the mean corner error over all relevant pairs
void testHomographyAccuracy() {
    const auto pair = [](bool retrieved, std::optional<double> error) {
        image_search::PairEvaluation evaluation;
        evaluation.retrieved = retrieved;
        evaluation.corner_error = error;
        return evaluation;
    };
    // 2 queries, 10 pairs: 4 valid, 1 RANSAC failure, 5 outside TopC
    std::vector<image_search::PairEvaluation> pairs = {
        pair(true, 0.5), pair(true, 2.0), pair(true, 4.0), pair(true, 5.0),
        pair(true, std::nullopt)};
    for(int i = 0; i < 5; ++i)
        pairs.push_back(pair(false, std::nullopt));

    const auto accuracy = image_search::summarize_homography_accuracy(pairs);
    check(std::abs(accuracy.valid_rate - 0.4) < 1e-12,
          "valid homography rate does not divide by all relevant pairs");
    // E_corner = 5 is not below 5 px
    check(std::abs(accuracy.accurate_rates[0] - 0.1) < 1e-12 &&
              std::abs(accuracy.accurate_rates[1] - 0.2) < 1e-12 &&
              std::abs(accuracy.accurate_rates[2] - 0.3) < 1e-12,
          "accurate homography rates at 1, 3, 5 px are incorrect");
    // (0.5 + 2 + 4 + 5) / 4, invalid pairs excluded
    check(accuracy.mean_corner_error.has_value() &&
              std::abs(*accuracy.mean_corner_error - 2.875) < 1e-12,
          "mean corner error is not over the valid pairs only");

    // no valid estimate: rates 0, mean NA, not 0
    const auto none = image_search::summarize_homography_accuracy(
        {pair(true, std::nullopt), pair(false, std::nullopt)});
    check(none.valid_rate == 0.0 && none.accurate_rates[2] == 0.0 &&
              !none.mean_corner_error.has_value(),
          "mean corner error was set without a valid estimate");
    checkThrows([] { image_search::summarize_homography_accuracy({}); },
                "homography accuracy accepted no relevant pairs");
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

// Test that BoVW extraction reports the input record of every successful image
void testBowRecordIndices() {
    const auto directory = std::filesystem::temp_directory_path();
    const auto board_path = directory / "image_search_bow_board.pgm";
    const auto blank_path = directory / "image_search_bow_blank.pgm";
    cv::Mat board(120, 160, CV_8UC1);
    for(int row = 0; row < board.rows; ++row) {
        for(int column = 0; column < board.cols; ++column)
            board.at<unsigned char>(row, column) =
                ((row / 20 + column / 20) % 2 == 0) ? 30 : 220;
    }
    check(cv::imwrite(board_path.string(), board) &&
              cv::imwrite(blank_path.string(), cv::Mat(48, 64, CV_8UC1, cv::Scalar(128))),
          "could not write the BoVW test images");

    // blank and missing images fail, the boards succeed
    const std::vector<image_search::ImageRecord> records = {
        {blank_path.string(), "blank"}, {board_path.string(), "first"},
        {(directory / "image_search_bow_missing.pgm").string(), "missing"},
        {board_path.string(), "second"}};
    const image_search::DescriptorMatrix vocabulary(2, 128);
    const auto all = image_search::extract_bow_vectors(records, vocabulary);
    check(all.record_indices == std::vector<std::size_t>{1, 3} &&
              all.identifiers == std::vector<std::string>{"first", "second"} &&
              all.failed_records.size() == 2,
          "BoVW extraction did not report the records of the successful images");
    // first D = 1 usable image
    const auto limited = image_search::extract_bow_vectors(records, vocabulary, 1);
    check(limited.record_indices == std::vector<std::size_t>{1},
          "BoVW extraction with a limit reported the wrong record");

    std::filesystem::remove(board_path);
    std::filesystem::remove(blank_path);
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
        testRetrievalMetrics();
        testSearchArgs();
        testVerifyArgs();
        testCandidateRetrieval();
        testDatasetPaths();
        testGroundTruthHomography();
        testCornerError();
        testRelevantPairEvaluation();
        testQueryEvaluation();
        testHomographyAccuracy();
        testVerificationOutput();
        testLocalFeatures();
        testLocalFeatureStore();
        testBowRecordIndices();
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
