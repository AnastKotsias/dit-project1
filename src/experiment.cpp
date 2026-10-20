#include "experiment.hpp"

#include "dataset.hpp"
#include "pipeline.hpp"
#include "search.hpp"
#include "vocabulary.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace image_search {
namespace {

struct MethodConfig {
    std::string method;
    int first = 0;
    int second = 0;
    int third = 0;
    int fourth = 0;
};

struct Aggregate {
    double recall5 = 0.0;
    double recall10 = 0.0;
    double ap10 = 0.0;
    double time_ms = 0.0;
    std::size_t count = 0;

    void add(const SearchResult& result) {
        recall5 += result.recall_5;
        recall10 += result.recall_10;
        ap10 += result.ap_10;
        time_ms += result.query_time_ms;
        ++count;
    }

    double mean(double total) const {
        return count == 0 ? 0.0 : total / static_cast<double>(count);
    }
};

struct Summary {
    Aggregate overall;
    Aggregate illumination;
    Aggregate viewpoint;
};

struct PreparedSet {
    std::vector<std::vector<double>> queries;
    std::vector<std::string> query_ids;
    std::vector<std::string> sequence_ids;
    std::vector<std::vector<double>> hpatches_database;
    std::vector<std::string> hpatches_ids;
    std::vector<ImageRecord> distractors;
};

struct VocabularyData {
    cv::Mat vocabulary;
    std::map<std::string, std::vector<std::string>> splits;
};

PreparedSet prepare_set(const EvaluationDataset& dataset,
                        const cv::Mat& vocabulary,
                        int distractor_count) {
    PreparedSet prepared;
    const auto database = extract_bow_vectors(dataset.database, vocabulary);
    if(!database.failed_records.empty()) {
        throw std::runtime_error("an HPatches database image failed descriptor extraction: " +
                                 database.failed_records.front().filepath);
    }

    const auto distractors = extract_bow_vectors(
        dataset.distractor_pool, vocabulary, static_cast<std::size_t>(distractor_count));
    if(distractors.vectors.size() != static_cast<std::size_t>(distractor_count)) {
        throw std::runtime_error("not enough usable MIRFlickr images for D=" +
                                 std::to_string(distractor_count));
    }

    const auto queries = extract_bow_vectors(dataset.queries, vocabulary);
    if(!queries.failed_records.empty()) {
        throw std::runtime_error("a query image failed descriptor extraction: " +
                                 queries.failed_records.front().filepath);
    }

    prepared.queries = queries.vectors;
    prepared.query_ids = queries.identifiers;
    prepared.hpatches_database = database.vectors;
    prepared.hpatches_ids = database.identifiers;
    prepared.distractors.reserve(distractors.vectors.size());
    for(std::size_t i = 0; i < distractors.vectors.size(); ++i) {
        prepared.distractors.push_back(
            {dataset.distractor_pool[i].filepath, distractors.identifiers[i]});
    }
    for(const auto& query : dataset.queries)
        prepared.sequence_ids.push_back(query.identifier);

    prepared.hpatches_database.insert(prepared.hpatches_database.end(),
                                      distractors.vectors.begin(),
                                      distractors.vectors.end());
    prepared.hpatches_ids.insert(prepared.hpatches_ids.end(),
                                 distractors.identifiers.begin(),
                                 distractors.identifiers.end());
    return prepared;
}

std::vector<SearchResult> search_all(const MethodConfig& config,
                                     const PreparedSet& data,
                                     int seed) {
    std::vector<SearchResult> results;
    results.reserve(data.queries.size());

    auto search = [&](auto& index) {
        for(std::size_t i = 0; i < data.queries.size(); ++i)
            results.push_back(index.search(data.queries[i], data.query_ids[i],
                                           data.sequence_ids[i]));
    };

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
        throw std::invalid_argument("unknown experiment method " + config.method);
    }
    return results;
}

double mean_ap(const std::vector<SearchResult>& results) {
    if(results.empty())
        return 0.0;
    return std::accumulate(results.begin(), results.end(), 0.0,
        [](double total, const SearchResult& result) {
            return total + result.ap_10;
        }) / static_cast<double>(results.size());
}

Summary summarize(const std::vector<SearchResult>& results) {
    Summary summary;
    for(const auto& result : results) {
        summary.overall.add(result);
        if(result.query_id.rfind("i_", 0) == 0)
            summary.illumination.add(result);
        else if(result.query_id.rfind("v_", 0) == 0)
            summary.viewpoint.add(result);
    }
    return summary;
}

void write_summary(std::ofstream& output, const Summary& summary) {
    const auto write_row = [&](const char* name, const Aggregate& aggregate) {
        output << "| " << name << " | "
               << aggregate.mean(aggregate.recall5) << " | "
               << aggregate.mean(aggregate.recall10) << " | "
               << aggregate.mean(aggregate.ap10) << " | "
               << aggregate.mean(aggregate.time_ms) << " |\n";
    };
    write_row("Overall", summary.overall);
    write_row("Illumination", summary.illumination);
    write_row("Viewpoint", summary.viewpoint);
}

std::vector<MethodConfig> validation_configs(int database_size) {
    const int clusters = static_cast<int>(
        std::ceil(std::sqrt(static_cast<double>(database_size))));
    std::vector<MethodConfig> configs;
    for(const int width : {10, 25, 50, 100}) // 0.01, 0.025, 0.05, 0.1
        for(const int tables : {3, 5, 7, 9})
            configs.push_back({"LSH", 4, tables, width, 0});
    for(const int max_candidates : {250, 500, 1000, 2000})
        for(const int probes : {5, 10, 20, 40})
            configs.push_back({"HYPERCUBE", 10, max_candidates, 50, probes});
    for(const int probes : {1, 2, 5, 10, 20})
        configs.push_back({"IVFFLAT", clusters, probes, 0, 0});
    for(const int probes : {1, 2, 5, 10, 20})
        configs.push_back({"IVFPQ", clusters, probes, 16, 8});
    return configs;
}

MethodConfig select_best(const std::vector<MethodConfig>& configs,
                         const PreparedSet& data, int seed) {
    MethodConfig best;
    double best_score = -1.0;
    for(const auto& config : configs) {
        const auto score = mean_ap(search_all(config, data, seed));
        if(score > best_score) {
            best = config;
            best_score = score;
        }
    }
    return best;
}

}  // namespace

ExperimentRunner::ExperimentRunner(std::string hp_path, std::string mir_path,
                                   std::string split_file, int vocabulary_size,
                                   int max_training_descriptors, int seed)
    : hp_path_(std::move(hp_path)), mir_path_(std::move(mir_path)),
      split_file_(std::move(split_file)), vocabulary_size_(vocabulary_size),
      max_training_descriptors_(max_training_descriptors), seed_(seed) {}

void ExperimentRunner::run(const std::string& output_path) {
    if(vocabulary_size_ % 16 != 0)
        throw std::invalid_argument(
            "experiment mode requires -vocab to be divisible by 16 for IVFPQ");

    VocabularyData vocabulary_data;
    vocabulary_data.splits = load_dataset_splits(split_file_);
    const auto training_paths = get_training_image_paths(
        hp_path_, vocabulary_data.splits.at("train"));
    std::mt19937 rng(seed_);
    const auto training = extract_training_descriptors(
        training_paths, max_training_descriptors_, rng);
    if(training.successful_images == 0)
        throw std::runtime_error("all training images failed descriptor extraction");
    vocabulary_data.vocabulary = compute_visual_vocabulary(
        training.sampled_descriptors, vocabulary_size_, seed_);

    const auto validation_dataset = get_evaluation_paths(
        hp_path_, mir_path_, split_file_, "validation");
    const auto validation = prepare_set(validation_dataset,
                                        vocabulary_data.vocabulary, 10000);
    std::map<std::string, MethodConfig> best;
    for(const auto& method : {"LSH", "HYPERCUBE", "IVFFLAT", "IVFPQ"}) {
        std::vector<MethodConfig> candidates;
        for(const auto& config : validation_configs(validation.hpatches_database.size()))
            if(config.method == method)
                candidates.push_back(config);
        best[method] = select_best(candidates, validation, seed_);
    }

    std::ofstream output(output_path);
    if(!output)
        throw std::runtime_error("could not open experiment output: " + output_path);
    output << std::fixed << std::setprecision(4);
    output << "# Image-search experiment\n\n";
    output << "Validation tuning uses D=10000 and selects each approximate "
              "method by maximum mAP@10.\n\n";
    output << "## Selected validation parameters\n\n"
              "| Method | Parameters |\n|---|---|\n";
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

    for(const int distractor_count : {1000, 5000, 10000, 25000}) {
        const auto test_dataset = get_evaluation_paths(
            hp_path_, mir_path_, split_file_, "test");
        const auto test = prepare_set(test_dataset, vocabulary_data.vocabulary,
                                      distractor_count);
        output << "\n## D = " << distractor_count << "\n\n";
        output << "| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) |\n"
                  "|---|---:|---:|---:|---:|\n";

        const MethodConfig exact{"EXACT"};
        const auto exact_results = search_all(exact, test, seed_);
        output << "\n### EXACT\n\n";
        write_summary(output, summarize(exact_results));
        for(const auto& method : {"LSH", "HYPERCUBE", "IVFFLAT", "IVFPQ"}) {
            const auto results = search_all(best.at(method), test, seed_);
            output << "\n### " << method << "\n\n";
            write_summary(output, summarize(results));
        }
    }
}

}  // namespace image_search
