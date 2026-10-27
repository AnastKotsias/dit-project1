#include "experiment.hpp"

#include "dataset.hpp"
#include "pipeline.hpp"
#include "search.hpp"
#include "vocabulary.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace image_search {
// helpers below are private to this translation unit
namespace {

// one candidate parameter setting of a search method
// the meaning of the numbered fields depends on the method, see search_all
struct MethodConfig {
    // EXACT, LSH, HYPERCUBE, IVFFLAT or IVFPQ
    std::string method;
    int first = 0;
    int second = 0;
    // window widths are stored in thousandths so that every field can stay an int
    int third = 0;
    int fourth = 0;
};

// running sums of the metrics over a group of queries
struct Aggregate {
    double recall5 = 0.0;
    double recall10 = 0.0;
    double ap10 = 0.0;
    double time_ms = 0.0;
    // number of queries added so far
    std::size_t count = 0;
    // queries for which the index returned fewer than 10 results, reported by the assignment
    std::size_t short_results = 0;

    // accumulates the metrics of one query
    void add(const SearchResult& result) {
        recall5 += result.recall_5;
        recall10 += result.recall_10;
        ap10 += result.ap_10;
        time_ms += result.query_time_ms;
        ++count;
        if(result.top_10_ids.size() < 10)
            ++short_results;
    }

    // divides a running sum by the query count, giving 0 for an empty group
    double mean(double total) const {
        return count == 0 ? 0.0 : total / static_cast<double>(count);
    }
};

// metrics of one method, over all queries and separately for each sequence type
struct Summary {
    Aggregate overall;
    Aggregate illumination;
    Aggregate viewpoint;
};

// all vectors of one evaluation set, ready to be indexed and queried
struct PreparedSet {
    // BoW vector of each query image
    std::vector<std::vector<double>> queries;
    // identifier of each query, aligned with queries
    std::vector<std::string> query_ids;
    // sequence name of each query, used to decide which database images are relevant
    std::vector<std::string> sequence_ids;
    // database vectors, namely the HPatches images followed by the D distractors
    std::vector<std::vector<double>> hpatches_database;
    // identifier of each database vector, aligned with hpatches_database
    std::vector<std::string> hpatches_ids;
    // the distractor images that were actually used, fewer than D if the pool ran out
    std::vector<ImageRecord> distractors;
    // MIRFlickr images skipped because they produced no usable descriptors
    std::size_t failed_distractors = 0;
};

// the trained vocabulary together with the dataset split it was trained on
struct VocabularyData {
    DescriptorMatrix vocabulary;
    std::map<std::string, std::vector<std::string>> splits;
};

// converts the queries, the HPatches database images and D distractors into BoW vectors
PreparedSet prepare_set(const EvaluationDataset& dataset,
                        const DescriptorMatrix& vocabulary,
                        int distractor_count) {
    PreparedSet prepared;
    // every HPatches database image must be usable, otherwise its sequence would lose a relevant image
    const auto database = extract_bow_vectors(dataset.database, vocabulary);
    if(!database.failed_records.empty()) {
        throw std::runtime_error("an HPatches database image failed descriptor extraction: " +
                                 database.failed_records.front().record.filepath +
                                 " (" + sift_extraction_status_name(
                                     database.failed_records.front().status) + ")");
    }

    // take the first D usable MIRFlickr images, failed ones are replaced by the next in order
    const auto distractors = extract_bow_vectors(
        dataset.distractor_pool, vocabulary, static_cast<std::size_t>(distractor_count));
    // when the pool runs out, every usable image is kept and the actual number is reported
    if(distractors.vectors.size() < static_cast<std::size_t>(distractor_count)) {
        std::cerr << "warning: only " << distractors.vectors.size()
                  << " usable MIRFlickr images for D=" << distractor_count << " ("
                  << distractors.failed_records.size() << " without SIFT descriptors)\n";
    }

    // every query must be usable, otherwise it cannot be evaluated
    const auto queries = extract_bow_vectors(dataset.queries, vocabulary);
    if(!queries.failed_records.empty()) {
        throw std::runtime_error("a query image failed descriptor extraction: " +
                                 queries.failed_records.front().record.filepath +
                                 " (" + sift_extraction_status_name(
                                     queries.failed_records.front().status) + ")");
    }

    // copy the query and HPatches database data into the prepared set
    prepared.queries = queries.vectors;
    prepared.query_ids = queries.identifiers;
    prepared.hpatches_database = database.vectors;
    prepared.hpatches_ids = database.identifiers;
    // record which distractor files were used, pairing each with its identifier
    // failed images are skipped, so the usable files are found by identifier rather than position
    prepared.distractors.reserve(distractors.vectors.size());
    std::size_t pool_index = 0;
    for(const auto& identifier : distractors.identifiers) {
        while(dataset.distractor_pool[pool_index].identifier != identifier)
            ++pool_index;
        prepared.distractors.push_back(dataset.distractor_pool[pool_index]);
        ++pool_index;
    }
    prepared.failed_distractors = distractors.failed_records.size();
    // a query's identifier is its sequence name
    for(const auto& query : dataset.queries)
        prepared.sequence_ids.push_back(query.identifier);

    // append the distractors so the final database holds HPatches images plus D distractors
    prepared.hpatches_database.insert(prepared.hpatches_database.end(),
                                      distractors.vectors.begin(),
                                      distractors.vectors.end());
    prepared.hpatches_ids.insert(prepared.hpatches_ids.end(),
                                 distractors.identifiers.begin(),
                                 distractors.identifiers.end());
    return prepared;
}

// builds the index described by config and runs every query of the set against it
// field usage: LSH (k, L, w), HYPERCUBE (kproj, M, w, probes), IVFFLAT (clusters, probes), IVFPQ (clusters, probes, M, nbits)
std::vector<SearchResult> search_all(const MethodConfig& config,
                                     const PreparedSet& data,
                                     int seed) {
    std::vector<SearchResult> results;
    results.reserve(data.queries.size());

    // runs all queries on any index type, since every class exposes the same search method
    auto search = [&](auto& index) {
        for(std::size_t i = 0; i < data.queries.size(); ++i)
            results.push_back(index.search(data.queries[i], data.query_ids[i],
                                           data.sequence_ids[i]));
    };

    // pick the index class from the method name and construct it with the configured parameters
    if(config.method == "EXACT") {
        ExactSearch index(data.hpatches_database, data.hpatches_ids);
        search(index);
    } else if(config.method == "LSH") {
        LSHSearch index(data.hpatches_database, data.hpatches_ids,
                        config.first, config.second, config.third / 1000.0, seed);
        search(index);
    } else if(config.method == "HYPERCUBE") {
        HypercubeSearch index(data.hpatches_database, data.hpatches_ids,
                              config.first, config.third / 1000.0,
                              config.second, config.fourth, seed);
        search(index);
    } else if(config.method == "IVFFLAT") {
        IVFFlatSearch index(data.hpatches_database, data.hpatches_ids,
                            config.first, config.second, seed);
        search(index);
    } else if(config.method == "IVFPQ") {
        IVFPQSearch index(data.hpatches_database, data.hpatches_ids,
                          config.first, config.second, config.third,
                          config.fourth, seed);
        search(index);
    } else {
        // a misspelled method name is a programming error
        throw std::invalid_argument("unknown experiment method " + config.method);
    }
    return results;
}

// returns the mean of AP@10 over the results, which is the mAP@10 used to rank configurations
double mean_ap(const std::vector<SearchResult>& results) {
    if(results.empty())
        return 0.0;
    return std::accumulate(results.begin(), results.end(), 0.0,
        [](double total, const SearchResult& result) {
            return total + result.ap_10;
        }) / static_cast<double>(results.size());
}

// groups the results into overall, illumination and viewpoint metrics
Summary summarize(const std::vector<SearchResult>& results) {
    Summary summary;
    for(const auto& result : results) {
        summary.overall.add(result);
        // HPatches names illumination sequences i_* and viewpoint sequences v_*
        if(result.query_id.rfind("i_", 0) == 0)
            summary.illumination.add(result);
        else if(result.query_id.rfind("v_", 0) == 0)
            summary.viewpoint.add(result);
    }
    return summary;
}

// writes a markdown table with one row per category
void write_summary(std::ofstream& output, const Summary& summary) {
    output << "| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) "
              "| Queries with < 10 results |\n"
              "|---|---:|---:|---:|---:|---:|\n";
    // formats the mean metrics of one category as a row
    const auto write_row = [&](const char* name, const Aggregate& aggregate) {
        output << "| " << name << " | "
               << aggregate.mean(aggregate.recall5) << " | "
               << aggregate.mean(aggregate.recall10) << " | "
               << aggregate.mean(aggregate.ap10) << " | "
               << aggregate.mean(aggregate.time_ms) << " | "
               << aggregate.short_results << "/" << aggregate.count << " |\n";
    };
    write_row("Overall", summary.overall);
    write_row("Illumination", summary.illumination);
    write_row("Viewpoint", summary.viewpoint);
}

// lists the parameter grid to try on the validation set for every approximate method
std::vector<MethodConfig> validation_configs(int database_size) {
    // default number of clusters is ceil(sqrt(n)), where n is the database size
    const int clusters = static_cast<int>(
        std::ceil(std::sqrt(static_cast<double>(database_size))));
    std::vector<MethodConfig> configs;
    // LSH with k = 4, trying several window widths and table counts
    for(const int width : {10, 25, 50, 100}) // 0.01, 0.025, 0.05, 0.1
        for(const int tables : {3, 5, 7, 9})
            configs.push_back({"LSH", 4, tables, width, 0});
    // hypercube with kproj = 10 and w = 0.05, trying several candidate limits and probe counts
    for(const int max_candidates : {250, 500, 1000, 2000})
        for(const int probes : {5, 10, 20, 40})
            configs.push_back({"HYPERCUBE", 10, max_candidates, 50, probes});
    // IVFFlat trying several probe counts
    for(const int probes : {1, 2, 5, 10, 20})
        configs.push_back({"IVFFLAT", clusters, probes, 0, 0});
    // IVFPQ with M = 16 and nbits = 8, trying several probe counts
    for(const int probes : {1, 2, 5, 10, 20})
        configs.push_back({"IVFPQ", clusters, probes, 16, 8});
    return configs;
}

// returns the configuration with the highest mAP@10 on the given set
MethodConfig select_best(const std::vector<MethodConfig>& configs,
                         const PreparedSet& data, int seed) {
    MethodConfig best;
    // below any possible score, so the first configuration always becomes the best
    double best_score = -1.0;
    for(const auto& config : configs) {
        // evaluate this configuration on every query
        const auto score = mean_ap(search_all(config, data, seed));
        // keep the first configuration on ties
        if(score > best_score) {
            best = config;
            best_score = score;
        }
    }
    return best;
}

}  // namespace

// stores the settings, moving the strings to avoid copies
ExperimentRunner::ExperimentRunner(std::string hp_path, std::string mir_path,
                                   std::string split_file, int vocabulary_size,
                                   int max_training_descriptors, int seed)
    : hp_path_(std::move(hp_path)), mir_path_(std::move(mir_path)),
      split_file_(std::move(split_file)), vocabulary_size_(vocabulary_size),
      max_training_descriptors_(max_training_descriptors), seed_(seed) {}

void ExperimentRunner::run(const std::string& output_path) {
    // IVFPQ uses M = 16 subquantizers, which must divide the vector dimension K
    if(vocabulary_size_ % 16 != 0)
        throw std::invalid_argument(
            "experiment mode requires -vocab to be divisible by 16 for IVFPQ");

    // step 1, train the vocabulary on the training sequences only
    VocabularyData vocabulary_data;
    vocabulary_data.splits = load_dataset_splits(split_file_);
    const auto training_paths = get_training_image_paths(
        hp_path_, vocabulary_data.splits.at("train"));
    // seeded generator for the descriptor sampling
    std::mt19937 rng(seed_);
    const auto training = extract_training_descriptors(
        training_paths, max_training_descriptors_, rng);
    // k-means cannot run without any descriptors
    if(training.successful_images == 0)
        throw std::runtime_error("all training images failed descriptor extraction");
    vocabulary_data.vocabulary = compute_visual_vocabulary(
        training.sampled_descriptors, vocabulary_size_, seed_);

    // step 2, tune each approximate method on the validation set with D = 10000 distractors
    const auto validation_dataset = get_evaluation_paths(
        hp_path_, mir_path_, split_file_, "validation");
    const auto validation = prepare_set(validation_dataset,
                                        vocabulary_data.vocabulary, 10000);
    // best configuration found for each method
    std::map<std::string, MethodConfig> best;
    for(const auto& method : {"LSH", "HYPERCUBE", "IVFFLAT", "IVFPQ"}) {
        // keep only the grid entries of the current method
        std::vector<MethodConfig> candidates;
        for(const auto& config : validation_configs(validation.hpatches_database.size()))
            if(config.method == method)
                candidates.push_back(config);
        best[method] = select_best(candidates, validation, seed_);
    }

    // step 3, write the report starting with the selected parameters
    std::ofstream output(output_path);
    if(!output)
        throw std::runtime_error("could not open experiment output: " + output_path);
    output << std::fixed << std::setprecision(4);
    output << "# Image-search experiment\n\n";
    output << "Validation tuning uses D=10000 (" << validation.distractors.size()
           << " usable MIRFlickr images) and selects each approximate "
              "method by maximum mAP@10.\n\n";
    output << "## Selected validation parameters\n\n"
              "| Method | Parameters |\n|---|---|\n";
    // print each method's parameters using the names of the assignment's command line options
    for(const auto& [method, config] : best) {
        output << "| " << method << " | ";
        if(method == "LSH")
            output << "k=" << config.first << ", L=" << config.second
                   << ", w=" << config.third / 1000.0;
        else if(method == "HYPERCUBE")
            output << "kproj=" << config.first << ", M=" << config.second
                   << ", w=" << config.third / 1000.0
                   << ", probes=" << config.fourth;
        else if(method == "IVFFLAT")
            output << "kclusters=" << config.first << ", nprobe=" << config.second;
        else
            output << "kclusters=" << config.first << ", nprobe=" << config.second
                   << ", M=" << config.third << ", nbits=" << config.fourth;
        output << " |\n";
    }

    // step 4, evaluate on the test set for each database size D, reusing the tuned parameters
    for(const int distractor_count : {1000, 5000, 10000, 25000}) {
        const auto test_dataset = get_evaluation_paths(
            hp_path_, mir_path_, split_file_, "test");
        const auto test = prepare_set(test_dataset, vocabulary_data.vocabulary,
                                      distractor_count);
        output << "\n## D = " << distractor_count << "\n\n";
        // the actual number differs from D only if MIRFlickr ran out of usable images
        output << "Usable MIRFlickr distractors: " << test.distractors.size()
               << " (" << test.failed_distractors
               << " skipped without SIFT descriptors), database size: "
               << test.hpatches_database.size() << "\n";

        // exact search is the baseline and needs no tuning
        const MethodConfig exact{"EXACT"};
        const auto exact_results = search_all(exact, test, seed_);
        output << "\n### EXACT\n\n";
        write_summary(output, summarize(exact_results));
        // evaluate each approximate method with its validation-selected parameters
        for(const auto& method : {"LSH", "HYPERCUBE", "IVFFLAT", "IVFPQ"}) {
            const auto results = search_all(best.at(method), test, seed_);
            output << "\n### " << method << "\n\n";
            write_summary(output, summarize(results));
        }
    }
}

}  // namespace image_search
