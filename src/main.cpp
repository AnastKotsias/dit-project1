#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <random>
#include <filesystem>
#include <fstream>

#include "dataset.hpp"
#include "pipeline.hpp"
#include "vocabulary.hpp"
#include "search.hpp"
#include "experiment.hpp"
#include "command_line.hpp"

int main(int argc, char** argv) {
    image_search::SearchArgs args;
    // parse errors print the message and stop before any image is processed
    try {
        args = image_search::parse_search_args(std::vector<std::string>(argv + 1, argv + argc));
    } catch(const std::invalid_argument& error) {
        std::cerr << "error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }

    if(args.experiment) {
        try {
            image_search::ExperimentRunner runner(
                args.hp_path, args.mir_path, args.split_file,
                args.vocab, args.s_max, args.seed);
            runner.run(args.out_file);
            std::cout << "wrote experiment results to " << args.out_file << "\n";
            return EXIT_SUCCESS;
        } catch(const std::exception& error) {
            std::cerr << "error: experiment failed: " << error.what() << "\n";
            return EXIT_FAILURE;
        }
    }

    // checked before the long preprocessing, in append mode so an existing result is not
    // truncated if the run fails later
    if(!std::ofstream(args.out_file, std::ios::app)) {
        std::cerr << "error: could not open output file: " << args.out_file << "\n";
        return EXIT_FAILURE;
    }

    std::map<std::string, std::vector<std::string>> splits;
    try {
        splits = image_search::load_dataset_splits(args.split_file);
    } catch(const std::exception& error) {
        std::cerr << "error: failed to load split file: "
                  << error.what() << "\n";
        return EXIT_FAILURE;
    }
    std::cout << "loaded " << splits["train"].size() << " training sequences\n";

    auto train_paths = image_search::get_training_image_paths(args.hp_path, splits["train"]);
    std::cout << "found " << train_paths.size() << " training images\n";

    // vocabulary: sample SIFT descriptors of the training images and cluster them
    std::mt19937 rng(args.seed);
    std::cout << "extracting and sampling sift descriptors for vocabulary...\n";
    const auto training_result = image_search::extract_training_descriptors(
        train_paths, args.s_max, rng);
    // failed training images are only warnings, the vocabulary is built from the rest
    for(std::size_t i = 0; i < training_result.failed_paths.size(); ++i) {
        std::cerr << "warning: failed to extract descriptors from training image: "
                  << training_result.failed_paths[i] << " ("
                  << image_search::sift_extraction_status_name(
                         training_result.failed_statuses[i])
                  << ")\n";
    }

    if(training_result.successful_images == 0) {
        std::cerr << "error: all training images failed descriptor extraction\n";
        return EXIT_FAILURE;
    }

    std::cout << "computing visual vocabulary with k=" << args.vocab << "...\n";
    image_search::DescriptorMatrix vocabulary;
    try {
        vocabulary = image_search::compute_visual_vocabulary(
            training_result.sampled_descriptors, args.vocab, args.seed);
    } catch(const std::exception& error) {
        std::cerr << "error: failed to compute vocabulary: "
                  << error.what() << "\n";
        return EXIT_FAILURE;
    }

    if(vocabulary.empty()) {
        std::cerr << "error: failed to compute vocabulary\n";
        return EXIT_FAILURE;
    }
    std::cout << "vocabulary computed successfully using "
    << training_result.successful_images << " images ("
    << training_result.failed_images << " failed)\n";

    // database and queries of the chosen split
    image_search::EvaluationDataset evaluation_dataset;
    try {
        evaluation_dataset = image_search::get_evaluation_paths(
            args.hp_path, args.mir_path, args.split_file, args.set_type);
    } catch(const std::exception& error) {
        std::cerr << "error: failed to construct evaluation dataset: "
                  << error.what() << "\n";
        return EXIT_FAILURE;
    }

    // every HPatches database image must be usable, otherwise its sequence would
    // lose one of its five relevant images
    const auto hpatches_database = image_search::extract_bow_vectors(
        evaluation_dataset.database, vocabulary);
    if(!hpatches_database.failed_records.empty()) {
        const auto& failed = hpatches_database.failed_records.front();
        std::cerr << "error: an HPatches database image failed descriptor extraction: "
                  << failed.record.filepath << " ("
                  << image_search::sift_extraction_status_name(failed.status)
                  << ")\n";
        return EXIT_FAILURE;
    }

    // the first D usable MIRFlickr images are the distractors
    const auto distractors = image_search::extract_bow_vectors(
        evaluation_dataset.distractor_pool, vocabulary, args.d_mir);
    // fewer than D means the pool ran out of usable images, so all of them are used
    const auto distractors_added = distractors.vectors.size();
    if(distractors_added < static_cast<std::size_t>(args.d_mir)) {
        std::cerr << "warning: only " << distractors_added
                  << " usable MIRFlickr images for -D=" << args.d_mir << " ("
                  << distractors.failed_records.size()
                  << " without SIFT descriptors), continuing with all of them\n";
    }
    std::cout << "using " << distractors_added << " MIRFlickr distractors ("
              << distractors.failed_records.size() << " skipped)\n";

    std::vector<std::vector<double>> database_bows = hpatches_database.vectors;
    std::vector<std::string> database_ids = hpatches_database.identifiers;
    database_bows.insert(database_bows.end(), distractors.vectors.begin(),
                          distractors.vectors.end());
    database_ids.insert(database_ids.end(), distractors.identifiers.begin(),
                        distractors.identifiers.end());

    const int n_database = static_cast<int>(database_bows.size());
    if(n_database <= 0) {
        std::cerr << "error: database contains no usable image vectors\n";
        return EXIT_FAILURE;
    }

    // IVF defaults depend on the database size, known only now
    if(args.method == "ivfflat" || args.method == "ivfpq") {
        // ceil(sqrt(n)) is the size of the k-means sample, not a cluster count
        if(args.ivf_kclusters == -1) {
            args.ivf_kclusters = std::max(5, static_cast<int>(
                std::lround(std::pow(static_cast<double>(n_database), 0.25))));
        }
        // k-means cannot form more clusters than its sample has points
        const int max_clusters = static_cast<int>(
            image_search::ivfTrainingSampleSize(static_cast<std::size_t>(n_database)));
        if(args.ivf_kclusters > max_clusters) {
            std::cerr << "warning: -kclusters " << args.ivf_kclusters
                      << " exceeds the IVF k-means sample size ceil(sqrt(" << n_database
                      << ")) = " << max_clusters << ", using " << max_clusters << " clusters\n";
            args.ivf_kclusters = max_clusters;
        }
        if(args.ivf_nprobe == -1) {
            args.ivf_nprobe = 5;
        }
        std::cout << "resolved IVF parameters: kclusters="
                  << args.ivf_kclusters
                  << ", nprobe=" << args.ivf_nprobe << "\n";
    }

    const auto queries = image_search::extract_bow_vectors(
        evaluation_dataset.queries, vocabulary);
    // a missing query cannot be evaluated
    if(!queries.failed_records.empty()) {
        std::cerr << "error: query image failed to produce descriptors: "
                  << queries.failed_records.front().record.filepath << " ("
                  << image_search::sift_extraction_status_name(
                         queries.failed_records.front().status)
                  << ")\n";
        return EXIT_FAILURE;
    }

    std::cout << "prepared " << database_bows.size()
              << " database vectors and " << queries.vectors.size()
              << " query vectors\n";

    // build the chosen index and run every query
    try {
        std::vector<image_search::SearchResult> all_results;
        all_results.reserve(queries.vectors.size());
        // method name printed at the top of the output file
        std::string output_name;

        // the sequence name of each query defines its relevant images
        const auto run_queries = [&](const auto& index) {
            for(std::size_t i = 0; i < queries.vectors.size(); ++i)
                all_results.push_back(index.search(
                    queries.vectors[i], queries.identifiers[i],
                    evaluation_dataset.queries[i].identifier));
        };

        if(args.method == "exact") {
            run_queries(image_search::ExactSearch(database_bows, database_ids));
            output_name = "EXACT";
        } else if(args.method == "lsh") {
            run_queries(image_search::LSHSearch(database_bows, database_ids, args.lsh_k,
                                                args.lsh_l, args.w, args.seed));
            output_name = "LSH";
        } else if(args.method == "hypercube") {
            run_queries(image_search::HypercubeSearch(
                database_bows, database_ids, args.hyper_kproj, args.w,
                args.m_param, args.hyper_probes, args.seed));
            output_name = "HYPERCUBE";
        } else if(args.method == "ivfflat") {
            run_queries(image_search::IVFFlatSearch(
                database_bows, database_ids, args.ivf_kclusters,
                args.ivf_nprobe, args.seed));
            output_name = "IVFFLAT";
        } else if(args.method == "ivfpq") {
            run_queries(image_search::IVFPQSearch(
                database_bows, database_ids, args.ivf_kclusters,
                args.ivf_nprobe, args.m_param, args.pq_nbits, args.seed));
            output_name = "IVFPQ";
        }

        image_search::evaluateAndWriteResults(args.out_file, output_name, all_results);
        std::cout << "wrote search results to " << args.out_file << "\n";
        // the output format has no field for it, but the experimental study reports this count
        std::size_t short_results = 0;
        for(const auto& result : all_results)
            if(result.top_10_ids.size() < 10)
                ++short_results;
        std::cout << short_results << " of " << all_results.size()
                  << " queries returned fewer than 10 results\n";
    } catch(const std::exception& error) {
        std::cerr << "error: search index failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
