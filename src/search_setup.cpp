#include "search_setup.hpp"

#include "pipeline.hpp"
#include "search.hpp"
#include "vocabulary.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>

namespace image_search {
namespace {

// "<path> (<status>)" of an image that failed
std::string failure_text(const FailedImageRecord& failed) {
    return failed.record.filepath + " (" + sift_extraction_status_name(failed.status) + ")";
}

}  // namespace

PreparedSearch prepare_search(const SearchArgs& args) {
    std::map<std::string, std::vector<std::string>> splits;
    try {
        splits = load_dataset_splits(args.split_file);
    } catch(const std::exception& error) {
        throw std::runtime_error(std::string("failed to load split file: ") + error.what());
    }
    std::cout << "loaded " << splits["train"].size() << " training sequences\n";

    auto train_paths = get_training_image_paths(args.hp_path, splits["train"]);
    std::cout << "found " << train_paths.size() << " training images\n";

    // vocabulary: sample SIFT descriptors of the training images and cluster them
    std::mt19937 rng(args.seed);
    std::cout << "extracting and sampling sift descriptors for vocabulary...\n";
    const auto training_result = extract_training_descriptors(train_paths, args.s_max, rng);
    // failed training images are only warnings, the vocabulary is built from the rest
    for(std::size_t i = 0; i < training_result.failed_paths.size(); ++i) {
        std::cerr << "warning: failed to extract descriptors from training image: "
                  << training_result.failed_paths[i] << " ("
                  << sift_extraction_status_name(training_result.failed_statuses[i])
                  << ")\n";
    }
    if(training_result.successful_images == 0)
        throw std::runtime_error("all training images failed descriptor extraction");

    std::cout << "computing visual vocabulary with k=" << args.vocab << "...\n";
    DescriptorMatrix vocabulary;
    try {
        vocabulary = compute_visual_vocabulary(training_result.sampled_descriptors,
                                               args.vocab, args.seed);
    } catch(const std::exception& error) {
        throw std::runtime_error(std::string("failed to compute vocabulary: ") + error.what());
    }
    if(vocabulary.empty())
        throw std::runtime_error("failed to compute vocabulary");
    std::cout << "vocabulary computed successfully using "
    << training_result.successful_images << " images ("
    << training_result.failed_images << " failed)\n";

    // database and queries of the chosen split
    EvaluationDataset evaluation_dataset;
    try {
        evaluation_dataset = get_evaluation_paths(args.hp_path, args.mir_path,
                                                  args.split_file, args.set_type);
    } catch(const std::exception& error) {
        throw std::runtime_error(std::string("failed to construct evaluation dataset: ") +
                                 error.what());
    }

    // every HPatches database image must be usable, otherwise its sequence would
    // lose one of its five relevant images
    const auto hpatches_database = extract_bow_vectors(evaluation_dataset.database, vocabulary);
    if(!hpatches_database.failed_records.empty())
        throw std::runtime_error("an HPatches database image failed descriptor extraction: " +
                                 failure_text(hpatches_database.failed_records.front()));

    // the first D usable MIRFlickr images are the distractors
    const auto distractors = extract_bow_vectors(evaluation_dataset.distractor_pool,
                                                 vocabulary, args.d_mir);
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

    PreparedSearch prepared;
    prepared.database_bows = hpatches_database.vectors;
    prepared.database_ids = hpatches_database.identifiers;
    prepared.database_bows.insert(prepared.database_bows.end(), distractors.vectors.begin(),
                                  distractors.vectors.end());
    prepared.database_ids.insert(prepared.database_ids.end(),
                                 distractors.identifiers.begin(), distractors.identifiers.end());
    // same order as the vectors: HPatches, then the distractors that were kept
    for(const std::size_t index : hpatches_database.record_indices)
        prepared.database_paths.emplace_back(evaluation_dataset.database[index].filepath);
    for(const std::size_t index : distractors.record_indices)
        prepared.database_paths.emplace_back(evaluation_dataset.distractor_pool[index].filepath);
    if(prepared.database_bows.empty())
        throw std::runtime_error("database contains no usable image vectors");

    const auto queries = extract_bow_vectors(evaluation_dataset.queries, vocabulary);
    // a missing query cannot be evaluated
    if(!queries.failed_records.empty())
        throw std::runtime_error("query image failed to produce descriptors: " +
                                 failure_text(queries.failed_records.front()));
    prepared.queries = evaluation_dataset.queries;
    prepared.query_bows = queries.vectors;

    std::cout << "prepared " << prepared.database_bows.size()
              << " database vectors and " << prepared.query_bows.size()
              << " query vectors\n";
    return prepared;
}

void resolve_ivf_parameters(SearchArgs& args, std::size_t database_size) {
    if(args.method != "ivfflat" && args.method != "ivfpq")
        return;
    const int n_database = static_cast<int>(database_size);
    // ceil(sqrt(n)) is the size of the k-means sample, not a cluster count
    if(args.ivf_kclusters == -1) {
        args.ivf_kclusters = std::max(5, static_cast<int>(
            std::lround(std::pow(static_cast<double>(n_database), 0.25))));
    }
    // k-means cannot form more clusters than its sample has points
    const int max_clusters = static_cast<int>(ivfTrainingSampleSize(database_size));
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

}  // namespace image_search
