#include "experiment.hpp"

#include "dataset.hpp"
#include "pipeline.hpp"
#include "search.hpp"
#include "vocabulary.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <optional>
#include <ostream>
#include <sstream>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace image_search {
// helpers below are private to this translation unit
namespace {

// vocabulary sizes compared when experiment mode is run without -vocab
constexpr std::array<int, 4> kVocabularySizes = {128, 256, 512, 1024};
// number of MIRFlickr distractors used to tune the methods on the validation set
constexpr int kValidationDistractors = 10000;
// database sizes D evaluated on the test set
constexpr std::array<int, 4> kTestDistractors = {1000, 5000, 10000, 25000};
// approximate methods, tuned on validation and evaluated on test
constexpr std::array<const char*, 4> kApproximateMethods = {"LSH", "HYPERCUBE", "IVFFLAT", "IVFPQ"};
// timed passes over the query set after one untimed warm-up pass, each query reports its median time
constexpr std::size_t kTimedPasses = 5;

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
    // distances computed per query, the deterministic cost used to compare configurations
    double candidates = 0.0;
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
        candidates += static_cast<double>(result.candidates_examined);
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

// BoW vectors of the HPatches images of one evaluation set, under every vocabulary
struct EncodedSet {
    MultiBowVectorResult queries;
    MultiBowVectorResult database;
    // sequence name of each query, used to decide which database images are relevant
    std::vector<std::string> sequence_ids;
};

// all vectors of one evaluation set under one vocabulary, ready to be indexed and queried
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
    // number of distractors actually used, fewer than D if the pool ran out
    std::size_t distractor_count = 0;
    // MIRFlickr images skipped because they produced no usable descriptors
    std::size_t failed_distractors = 0;
};

// encodes the queries and HPatches database images of one set, every one of which must be usable
EncodedSet encode_set(const EvaluationDataset& dataset,
                      const std::vector<DescriptorMatrix>& vocabularies) {
    EncodedSet encoded;
    // every HPatches database image must be usable, otherwise its sequence would lose a relevant image
    encoded.database = extract_bow_vector_sets(dataset.database, vocabularies);
    if(!encoded.database.failed_records.empty()) {
        const auto& failed = encoded.database.failed_records.front();
        throw std::runtime_error("an HPatches database image failed descriptor extraction: " +
                                 failed.record.filepath + " (" +
                                 sift_extraction_status_name(failed.status) + ")");
    }
    // every query must be usable, otherwise it cannot be evaluated
    encoded.queries = extract_bow_vector_sets(dataset.queries, vocabularies);
    if(!encoded.queries.failed_records.empty()) {
        const auto& failed = encoded.queries.failed_records.front();
        throw std::runtime_error("a query image failed descriptor extraction: " +
                                 failed.record.filepath + " (" +
                                 sift_extraction_status_name(failed.status) + ")");
    }
    // a query's identifier is its sequence name
    for(const auto& query : dataset.queries)
        encoded.sequence_ids.push_back(query.identifier);
    return encoded;
}

// assembles the set under vocabulary v: its HPatches database images followed by the first D usable
// MIRFlickr images of the shared pool, or all usable ones when the pool has fewer
PreparedSet prepare_set(const EncodedSet& set, const MultiBowVectorResult& pool,
                        std::size_t vocabulary, int distractor_count) {
    PreparedSet prepared;
    prepared.queries = set.queries.vectors[vocabulary];
    prepared.query_ids = set.queries.identifiers;
    prepared.sequence_ids = set.sequence_ids;
    prepared.hpatches_database = set.database.vectors[vocabulary];
    prepared.hpatches_ids = set.database.identifiers;

    const auto requested = static_cast<std::size_t>(distractor_count);
    const std::size_t used = std::min(requested, pool.identifiers.size());
    prepared.distractor_count = used;
    if(used < requested) {
        // the pool ran out, so every image was scanned and every failure counts,
        // and all usable images are kept
        prepared.failed_distractors = pool.failed_records.size();
        std::cerr << "warning: only " << used << " usable MIRFlickr images for D="
                  << distractor_count << " (" << prepared.failed_distractors
                  << " without SIFT descriptors)\n";
    } else {
        // only the failures before the last used image were skipped to reach D
        prepared.failed_distractors = used == 0 ? 0 : pool.record_indices[used - 1] + 1 - used;
    }

    // append the distractors so the final database holds HPatches images plus D distractors
    const auto& pool_vectors = pool.vectors[vocabulary];
    prepared.hpatches_database.insert(prepared.hpatches_database.end(), pool_vectors.begin(),
                                      pool_vectors.begin() + static_cast<std::ptrdiff_t>(used));
    prepared.hpatches_ids.insert(prepared.hpatches_ids.end(), pool.identifiers.begin(),
                                 pool.identifiers.begin() + static_cast<std::ptrdiff_t>(used));
    return prepared;
}

// IVF indexes trained on one prepared set and reused across nprobe values, since nprobe only
// changes the search and not the training; a cache must only be used with the set it was built on
struct IndexCache {
    std::optional<IVFFlatSearch> ivfflat;
    int ivfflat_clusters = 0;
    std::optional<IVFPQSearch> ivfpq;
    MethodConfig ivfpq_config;
};

// builds the index described by config, or reuses a cached IVF index, and runs every query of the set
// field usage: LSH (k, L, w), HYPERCUBE (kproj, M, w, probes), IVFFLAT (clusters, probes), IVFPQ (clusters, probes, M, nbits)
std::vector<SearchResult> search_all(const MethodConfig& config,
                                     const PreparedSet& data,
                                     int seed, IndexCache& cache) {
    std::vector<SearchResult> results;
    results.reserve(data.queries.size());

    // runs all queries on any index type, since every class exposes the same search method
    // a forwarding reference accepts both named indexes and the copies made by withProbes
    auto search = [&](auto&& index) {
        const auto run = [&](std::size_t i) {
            return index.search(data.queries[i], data.query_ids[i], data.sequence_ids[i]);
        };
        // untimed warm-up pass, so the first queries do not pay for cold caches and lazy allocations
        for(std::size_t i = 0; i < data.queries.size(); ++i)
            (void)run(i);
        // the first timed pass keeps the results, the searches are deterministic so later passes
        // return the same rankings and only contribute their times
        // whole passes are repeated rather than one query several times in a row, so a query does
        // not find its own candidates still in the cache
        std::vector<std::array<double, kTimedPasses>> times(data.queries.size());
        for(std::size_t i = 0; i < data.queries.size(); ++i) {
            results.push_back(run(i));
            times[i][0] = results.back().query_time_ms;
        }
        for(std::size_t pass = 1; pass < kTimedPasses; ++pass)
            for(std::size_t i = 0; i < data.queries.size(); ++i)
                times[i][pass] = run(i).query_time_ms;
        // the median ignores single runs slowed down by the operating system
        for(std::size_t i = 0; i < data.queries.size(); ++i) {
            auto& query_times = times[i];
            std::nth_element(query_times.begin(), query_times.begin() + kTimedPasses / 2, query_times.end());
            results[i].query_time_ms = query_times[kTimedPasses / 2];
        }
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
        // train only if no index with the same number of clusters exists yet
        if(!cache.ivfflat || cache.ivfflat_clusters != config.first) {
            cache.ivfflat.emplace(data.hpatches_database, data.hpatches_ids,
                                  config.first, config.second, seed);
            cache.ivfflat_clusters = config.first;
        }
        search(cache.ivfflat->withProbes(config.second));
    } else if(config.method == "IVFPQ") {
        // train only if no index with the same clusters, M and nbits exists yet
        const auto& cached = cache.ivfpq_config;
        if(!cache.ivfpq || cached.first != config.first || cached.third != config.third ||
           cached.fourth != config.fourth) {
            cache.ivfpq.emplace(data.hpatches_database, data.hpatches_ids,
                                config.first, config.second, config.third,
                                config.fourth, seed);
            cache.ivfpq_config = config;
        }
        search(cache.ivfpq->withProbes(config.second));
    } else {
        // a misspelled method name is a programming error
        throw std::invalid_argument("unknown experiment method " + config.method);
    }
    return results;
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
void write_summary(std::ostream& output, const Summary& summary) {
    output << "| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) "
              "| Mean candidates | Queries with < 10 results |\n"
              "|---|---:|---:|---:|---:|---:|---:|\n";
    // formats the mean metrics of one category as a row
    const auto write_row = [&](const char* name, const Aggregate& aggregate) {
        output << "| " << name << " | "
               << aggregate.mean(aggregate.recall5) << " | "
               << aggregate.mean(aggregate.recall10) << " | "
               << aggregate.mean(aggregate.ap10) << " | "
               << aggregate.mean(aggregate.time_ms) << " | "
               << std::setprecision(1) << aggregate.mean(aggregate.candidates)
               << std::setprecision(4) << " | "
               << aggregate.short_results << "/" << aggregate.count << " |\n";
    };
    write_row("Overall", summary.overall);
    write_row("Illumination", summary.illumination);
    write_row("Viewpoint", summary.viewpoint);
}

// lists the parameter grid to try on the validation set for every approximate method
std::vector<MethodConfig> validation_configs() {
    std::vector<MethodConfig> configs;
    // LSH with k = 4, trying several window widths and table counts
    for(const int width : {10, 25, 50, 100}) // 0.01, 0.025, 0.05, 0.1
        for(const int tables : {3, 5, 7, 9})
            configs.push_back({"LSH", 4, tables, width, 0});
    // hypercube with kproj = 10 and w = 0.05, trying several candidate limits and probe counts
    for(const int max_candidates : {250, 500, 1000, 2000})
        for(const int probes : {5, 10, 20, 40})
            configs.push_back({"HYPERCUBE", 10, max_candidates, 50, probes});
    // IVF cluster counts from the professor's range of 5 to n^(1/4), about 10 for the validation database,
    // every nprobe is at most the smallest cluster count, and kclusters=5 with nprobe=5 probes every list
    // kclusters is the outer loop, so each IVF index is trained once and reused for its nprobe values
    // IVFFlat trying several cluster and probe counts
    for(const int clusters : {5, 6, 8, 10})
        for(const int probes : {1, 2, 3, 5})
            configs.push_back({"IVFFLAT", clusters, probes, 0, 0});
    // IVFPQ with M = 16 and nbits = 8, trying several cluster and probe counts
    for(const int clusters : {5, 6, 8, 10})
        for(const int probes : {1, 2, 3, 5})
            configs.push_back({"IVFPQ", clusters, probes, 16, 8});
    return configs;
}

// formats the parameters of a configuration with the names of the assignment's command line options
std::string describe(const MethodConfig& config) {
    std::ostringstream text;
    if(config.method == "LSH")
        text << "k=" << config.first << ", L=" << config.second
             << ", w=" << config.third / 1000.0;
    else if(config.method == "HYPERCUBE")
        text << "kproj=" << config.first << ", M=" << config.second
             << ", w=" << config.third / 1000.0 << ", probes=" << config.fourth;
    else if(config.method == "IVFFLAT")
        text << "kclusters=" << config.first << ", nprobe=" << config.second;
    else if(config.method == "IVFPQ")
        text << "kclusters=" << config.first << ", nprobe=" << config.second
             << ", M=" << config.third << ", nbits=" << config.fourth;
    else
        text << "-";
    return text.str();
}

// overall validation metrics of one configuration
struct ValidationRow {
    MethodConfig config;
    Aggregate overall;
};

// configurations whose validation mAP@10 is at most this far below the best of their method count as
// equally accurate, so the cheapest of them is selected
// with the 23 validation queries, one query's AP changing by 0.23 already moves mAP@10 by 0.01
constexpr double kMapTolerance = 0.01;

// evaluates every configuration on the validation set, training each IVF index only once
std::vector<ValidationRow> evaluate_configs(const std::vector<MethodConfig>& configs,
                                            const PreparedSet& data, int seed) {
    std::vector<ValidationRow> rows;
    rows.reserve(configs.size());
    IndexCache cache;
    for(const auto& config : configs)
        rows.push_back({config, summarize(search_all(config, data, seed, cache)).overall});
    return rows;
}

// returns the position of the row with the lowest cost among the rows within kMapTolerance of the
// best mAP@10, keeping the first in order on ties
// the cost is deterministic rather than a measured time, so the choice is reproducible
template <typename Cost>
std::size_t select_row(const std::vector<ValidationRow>& rows, const Cost& cost) {
    if(rows.empty())
        throw std::invalid_argument("no validation configurations to select from");
    double best_map = 0.0;
    for(const auto& row : rows)
        best_map = std::max(best_map, row.overall.mean(row.overall.ap10));

    std::size_t selected = rows.size();
    for(std::size_t i = 0; i < rows.size(); ++i) {
        const auto& overall = rows[i].overall;
        // the small slack keeps rows exactly at the tolerance despite rounding
        if(overall.mean(overall.ap10) < best_map - kMapTolerance - 1e-12)
            continue;
        if(selected == rows.size() || cost(rows[i]) < cost(rows[selected]))
            selected = i;
    }
    return selected;
}

// selects a method configuration, the cost being the mean number of candidates per query
std::size_t select_config(const std::vector<ValidationRow>& rows) {
    return select_row(rows, [](const ValidationRow& row) {
        return row.overall.mean(row.overall.candidates);
    });
}

// writes the validation results of one method, marking the selected configuration
void write_validation_table(std::ostream& output, const std::vector<ValidationRow>& rows,
                            std::size_t selected) {
    output << "| Parameters | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) "
              "| Mean candidates | Queries with < 10 results | Selected |\n"
              "|---|---:|---:|---:|---:|---:|---:|:---:|\n";
    for(std::size_t i = 0; i < rows.size(); ++i) {
        const auto& overall = rows[i].overall;
        output << "| " << describe(rows[i].config) << " | "
               << overall.mean(overall.recall5) << " | "
               << overall.mean(overall.recall10) << " | "
               << overall.mean(overall.ap10) << " | "
               << overall.mean(overall.time_ms) << " | "
               << std::setprecision(1) << overall.mean(overall.candidates)
               << std::setprecision(4) << " | "
               << overall.short_results << "/" << overall.count << " | "
               << (i == selected ? "yes" : "") << " |\n";
    }
}

// tunes every approximate method on validation and evaluates all methods on test for one vocabulary,
// writing the section of the report, and returns the exact-search validation metrics of the vocabulary
Aggregate study_vocabulary(std::ostream& output, int vocabulary_size, std::size_t vocabulary,
                           const EncodedSet& validation_set, const EncodedSet& test_set,
                           const MultiBowVectorResult& pool, int seed) {
    output << "\n## K = " << vocabulary_size << "\n\n";

    // tune each approximate method on the validation set
    const auto validation = prepare_set(validation_set, pool, vocabulary, kValidationDistractors);
    // exact search on validation is the reference the approximate methods are compared against
    const auto exact_validation = evaluate_configs({MethodConfig{"EXACT"}}, validation, seed);
    // validation results and selected configuration of each approximate method
    std::map<std::string, std::vector<ValidationRow>> validation_rows;
    std::map<std::string, std::size_t> selected_row;
    for(const auto& method : kApproximateMethods) {
        // keep only the grid entries of the current method
        std::vector<MethodConfig> candidates;
        for(const auto& config : validation_configs())
            if(config.method == method)
                candidates.push_back(config);
        std::cout << "K=" << vocabulary_size << ": tuning " << method << " on "
                  << candidates.size() << " validation configurations...\n";
        validation_rows[method] = evaluate_configs(candidates, validation, seed);
        selected_row[method] = select_config(validation_rows[method]);
    }

    output << "### Selected validation parameters\n\n"
              "| Method | Parameters | Validation mAP@10 | Mean candidates |\n"
              "|---|---|---:|---:|\n";
    for(const auto& method : kApproximateMethods) {
        const auto& row = validation_rows[method][selected_row[method]];
        output << "| " << method << " | " << describe(row.config) << " | "
               << row.overall.mean(row.overall.ap10) << " | " << std::setprecision(1)
               << row.overall.mean(row.overall.candidates) << std::setprecision(4) << " |\n";
    }

    output << "\n### Validation results\n\n"
           << "Database: " << validation.hpatches_database.size() << " vectors ("
           << validation.distractor_count << " MIRFlickr distractors).\n\n";
    output << "#### EXACT (reference)\n\n";
    write_validation_table(output, exact_validation, exact_validation.size());
    for(const auto& method : kApproximateMethods) {
        output << "\n#### " << method << "\n\n";
        write_validation_table(output, validation_rows[method], selected_row[method]);
    }

    // evaluate on the test set for each database size D, reusing the tuned parameters
    output << "\n### Test results\n";
    for(const int distractor_count : kTestDistractors) {
        std::cout << "K=" << vocabulary_size << ": testing D=" << distractor_count << "...\n";
        const auto test = prepare_set(test_set, pool, vocabulary, distractor_count);
        output << "\n#### D = " << distractor_count << "\n\n";
        // the actual number differs from D only if MIRFlickr ran out of usable images
        output << "Usable MIRFlickr distractors: " << test.distractor_count
               << " (" << test.failed_distractors
               << " skipped without SIFT descriptors), database size: "
               << test.hpatches_database.size() << "\n";

        IndexCache cache;
        // exact search is the baseline and needs no tuning
        output << "\n##### EXACT\n\n";
        write_summary(output, summarize(search_all(MethodConfig{"EXACT"}, test, seed, cache)));
        // evaluate each approximate method with its validation-selected parameters
        for(const auto& method : kApproximateMethods) {
            const auto& config = validation_rows[method][selected_row[method]].config;
            output << "\n##### " << method << "\n\n";
            write_summary(output, summarize(search_all(config, test, seed, cache)));
        }
    }
    return exact_validation.front().overall;
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
    // a single vocabulary if -vocab was given, otherwise the vocabulary sizes are compared
    std::vector<int> vocabulary_sizes;
    if(vocabulary_size_ > 0)
        vocabulary_sizes.push_back(vocabulary_size_);
    else
        vocabulary_sizes.assign(kVocabularySizes.begin(), kVocabularySizes.end());
    // IVFPQ uses M = 16 subquantizers, which must divide the vector dimension K
    for(const int size : vocabulary_sizes) {
        if(size % 16 != 0)
            throw std::invalid_argument(
                "experiment mode requires -vocab to be divisible by 16 for IVFPQ");
    }
    // opened before the long computation, so an invalid path fails immediately
    std::ofstream output(output_path);
    if(!output)
        throw std::runtime_error("could not open experiment output: " + output_path);

    // step 1, sample the training descriptors once and train every vocabulary on the same sample
    const auto splits = load_dataset_splits(split_file_);
    const auto training_paths = get_training_image_paths(hp_path_, splits.at("train"));
    // seeded generator for the descriptor sampling
    std::mt19937 rng(seed_);
    std::cout << "extracting and sampling SIFT descriptors of " << training_paths.size()
              << " training images...\n";
    const auto training = extract_training_descriptors(
        training_paths, max_training_descriptors_, rng);
    // k-means cannot run without any descriptors
    if(training.successful_images == 0)
        throw std::runtime_error("all training images failed descriptor extraction");
    std::vector<DescriptorMatrix> vocabularies;
    for(const int size : vocabulary_sizes) {
        std::cout << "training vocabulary K=" << size << "...\n";
        vocabularies.push_back(compute_visual_vocabulary(
            training.sampled_descriptors, size, seed_));
    }

    // step 2, extract the SIFT descriptors of every image once and encode them with every vocabulary
    const auto validation_dataset = get_evaluation_paths(
        hp_path_, mir_path_, split_file_, "validation");
    const auto test_dataset = get_evaluation_paths(hp_path_, mir_path_, split_file_, "test");
    std::cout << "encoding validation and test HPatches images...\n";
    const auto validation_set = encode_set(validation_dataset, vocabularies);
    const auto test_set = encode_set(test_dataset, vocabularies);
    // both sets use the first D usable MIRFlickr images, so the largest D needed is encoded once
    // and every database takes a prefix of it
    const int pool_size = std::max(kValidationDistractors,
        *std::max_element(kTestDistractors.begin(), kTestDistractors.end()));
    std::cout << "encoding up to " << pool_size << " MIRFlickr images...\n";
    const auto pool = extract_bow_vector_sets(validation_dataset.distractor_pool, vocabularies,
                                              static_cast<std::size_t>(pool_size));

    // step 3, study every vocabulary, the sections are buffered so the comparison can come first
    std::ostringstream sections;
    sections << std::fixed << std::setprecision(4);
    std::vector<ValidationRow> vocabulary_rows;
    for(std::size_t v = 0; v < vocabulary_sizes.size(); ++v) {
        const auto exact = study_vocabulary(sections, vocabulary_sizes[v], v, validation_set,
                                            test_set, pool, seed_);
        vocabulary_rows.push_back({MethodConfig{"EXACT", vocabulary_sizes[v]}, exact});
    }
    // the smallest vocabulary within the tolerance of the best exact validation mAP@10 is selected,
    // since a smaller K gives shorter vectors and therefore cheaper searches
    const std::size_t selected_vocabulary = select_row(vocabulary_rows,
        [](const ValidationRow& row) { return row.config.first; });

    // step 4, write the report
    output << std::fixed << std::setprecision(4);
    output << "# Image-search experiment\n\n";
    output << "S=" << max_training_descriptors_ << ", seed=" << seed_
           << ". Every vocabulary is trained on the same sampled training descriptors.\n\n";
    output << "Validation uses D=" << kValidationDistractors << " and "
           << validation_set.queries.identifiers.size()
           << " queries. For each approximate method, the configurations whose mAP@10 is "
              "within " << std::setprecision(2) << kMapTolerance << std::setprecision(4)
           << " of the method's best are treated as equally accurate, and the one with the "
              "fewest mean candidates (distance computations per query) is selected. The "
              "candidate count is used instead of the measured time so that the selection is "
              "reproducible.\n\n";
    output << "Query times are single-threaded. Every configuration first runs the whole query "
              "set once untimed as a warm-up, then " << kTimedPasses << " timed passes over the "
              "query set; each query's time is the median of its " << kTimedPasses
           << " runs, and the reported time is the mean of these medians.\n\n";
    output << "## Vocabulary comparison\n\n"
              "Exact search on the validation set. The smallest K within "
           << std::setprecision(2) << kMapTolerance << std::setprecision(4)
           << " of the best mAP@10 is selected.\n\n"
              "| K | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Selected |\n"
              "|---:|---:|---:|---:|---:|:---:|\n";
    for(std::size_t v = 0; v < vocabulary_rows.size(); ++v) {
        const auto& overall = vocabulary_rows[v].overall;
        output << "| " << vocabulary_sizes[v] << " | " << overall.mean(overall.recall5) << " | "
               << overall.mean(overall.recall10) << " | " << overall.mean(overall.ap10) << " | "
               << overall.mean(overall.time_ms) << " | "
               << (v == selected_vocabulary ? "yes" : "") << " |\n";
    }
    output << sections.str();
}

}  // namespace image_search
