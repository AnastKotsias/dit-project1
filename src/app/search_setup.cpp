#include "app/search_setup.hpp"

#include "data/pipeline.hpp"
#include "search/search.hpp"
#include "bovw/vocabulary.hpp"
#include "data/cache.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>

namespace image_search {
namespace {

// "<path> (<status>)" of an image that failed
std::string failure_text(const FailedImageRecord& failed) {
    return failed.record.filepath + " (" + sift_extraction_status_name(failed.status) + ")";
}

// vocabulary: sample SIFT descriptors of the training images and cluster them
// rng only used here, so a cached vocabulary leaves the rest of the run unchanged
DescriptorMatrix build_vocabulary(const SearchArgs& args,
                                  const std::vector<std::filesystem::path>& train_paths) {
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
    return vocabulary;
}

// "using N MIRFlickr distractors" line, warning first if fewer than D
void report_distractors(int d_mir, std::size_t added, std::size_t skipped) {
    // fewer than D means the pool ran out of usable images, so all of them are used
    if(added < static_cast<std::size_t>(d_mir)) {
        std::cerr << "warning: only " << added
                  << " usable MIRFlickr images for -D=" << d_mir << " ("
                  << skipped << " without SIFT descriptors), continuing with all of them\n";
    }
    std::cout << "using " << added << " MIRFlickr distractors (" << skipped << " skipped)\n";
}

// BoVW vectors of the database (HPatches, then distractors) and of the queries
BowCache encode_bow_vectors(const SearchArgs& args, const EvaluationDataset& dataset,
                            const DescriptorMatrix& vocabulary) {
    // every HPatches database image must be usable, otherwise its sequence would
    // lose one of its five relevant images
    const auto hpatches_database = extract_bow_vectors(dataset.database, vocabulary);
    if(!hpatches_database.failed_records.empty())
        throw std::runtime_error("an HPatches database image failed descriptor extraction: " +
                                 failure_text(hpatches_database.failed_records.front()));

    // the first D usable MIRFlickr images are the distractors
    const auto distractors = extract_bow_vectors(dataset.distractor_pool, vocabulary,
                                                 args.d_mir);
    report_distractors(args.d_mir, distractors.vectors.size(),
                       distractors.failed_records.size());

    BowCache result;
    result.distractors_skipped = distractors.failed_records.size();
    result.database_bows = hpatches_database.vectors;
    result.database_ids = hpatches_database.identifiers;
    result.database_bows.insert(result.database_bows.end(), distractors.vectors.begin(),
                                distractors.vectors.end());
    result.database_ids.insert(result.database_ids.end(),
                               distractors.identifiers.begin(), distractors.identifiers.end());
    // same order as the vectors: HPatches, then the distractors that were kept
    for(const std::size_t index : hpatches_database.record_indices)
        result.database_paths.emplace_back(dataset.database[index].filepath);
    for(const std::size_t index : distractors.record_indices)
        result.database_paths.emplace_back(dataset.distractor_pool[index].filepath);
    if(result.database_bows.empty())
        throw std::runtime_error("database contains no usable image vectors");

    const auto queries = extract_bow_vectors(dataset.queries, vocabulary);
    // a missing query cannot be evaluated
    if(!queries.failed_records.empty())
        throw std::runtime_error("query image failed to produce descriptors: " +
                                 failure_text(queries.failed_records.front()));
    result.queries = dataset.queries;
    result.query_bows = queries.vectors;
    return result;
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

    // -cache: the vocabulary of the same K, S, seed and training images, if stored
    const bool use_cache = !args.cache_dir.empty();
    std::string vocabulary_key;
    std::filesystem::path vocabulary_path;
    DescriptorMatrix vocabulary;
    if(use_cache) {
        vocabulary_key = vocabulary_cache_key(train_paths, args.vocab, args.s_max, args.seed);
        vocabulary_path = vocabulary_cache_path(args.cache_dir, args.vocab, args.s_max,
                                                args.seed);
        if(auto cached = load_vocabulary_cache(vocabulary_path, vocabulary_key)) {
            vocabulary = std::move(*cached);
            std::cout << "loaded vocabulary from cache " << vocabulary_path.string() << "\n";
        }
    }
    if(vocabulary.empty()) {
        vocabulary = build_vocabulary(args, train_paths);
        // the run goes on without the cache if it cannot be written
        if(use_cache) {
            try {
                save_vocabulary_cache(vocabulary_path, vocabulary_key, vocabulary);
                std::cout << "saved vocabulary to cache " << vocabulary_path.string() << "\n";
            } catch(const std::exception& error) {
                std::cerr << "warning: could not save vocabulary cache: " << error.what() << "\n";
            }
        }
    }

    // database and queries of the chosen split
    EvaluationDataset evaluation_dataset;
    try {
        evaluation_dataset = get_evaluation_paths(args.hp_path, args.mir_path,
                                                  args.split_file, args.set_type);
    } catch(const std::exception& error) {
        throw std::runtime_error(std::string("failed to construct evaluation dataset: ") +
                                 error.what());
    }

    // -cache: the BoVW vectors of the same vocabulary, set, D and images, if stored
    std::optional<BowCache> bows;
    std::string bow_key;
    std::filesystem::path bow_path;
    if(use_cache) {
        bow_key = bow_cache_key(vocabulary_key, args.set_type, args.d_mir, evaluation_dataset);
        bow_path = bow_cache_path(args.cache_dir, args.vocab, args.s_max, args.seed,
                                  args.set_type, args.d_mir);
        bows = load_bow_cache(bow_path, bow_key);
        // every HPatches image and query, as when encoded
        if(bows && (bows->database_bows.size() < evaluation_dataset.database.size() ||
                    bows->query_bows.size() != evaluation_dataset.queries.size()))
            bows.reset();
        if(bows) {
            std::cout << "loaded BoVW vectors from cache " << bow_path.string() << "\n";
            // distractors follow the HPatches images
            report_distractors(args.d_mir,
                               bows->database_bows.size() - evaluation_dataset.database.size(),
                               bows->distractors_skipped);
        }
    }
    if(!bows) {
        bows = encode_bow_vectors(args, evaluation_dataset, vocabulary);
        // the run goes on without the cache if it cannot be written
        if(use_cache) {
            try {
                save_bow_cache(bow_path, bow_key, *bows);
                std::cout << "saved BoVW vectors to cache " << bow_path.string() << "\n";
            } catch(const std::exception& error) {
                std::cerr << "warning: could not save BoVW cache: " << error.what() << "\n";
            }
        }
    }

    PreparedSearch prepared;
    prepared.database_bows = std::move(bows->database_bows);
    prepared.database_ids = std::move(bows->database_ids);
    prepared.database_paths = std::move(bows->database_paths);
    prepared.queries = std::move(bows->queries);
    prepared.query_bows = std::move(bows->query_bows);

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
