#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "command_line.hpp"
#include "geometric_evaluation.hpp"
#include "geometric_verification.hpp"
#include "pipeline.hpp"
#include "search.hpp"
#include "search_setup.hpp"
#include "verification_output.hpp"

namespace {

// TopC(Q), load_features, match_sift + estimate_ransac + rerank, then evaluation, per query
// only candidates() and verify_candidates() are timed, SIFT extraction and ground truth are not
template<class Index>
std::vector<image_search::QueryEvaluation> verify_queries(
    const Index& index,
    const image_search::VerifyArgs& args,
    const image_search::PreparedSearch& prepared,
    const image_search::LocalFeatureStore& query_features,
    image_search::LocalFeatureStore& database_features) {
    // load_features(image_id) reads the store, it never extracts
    const image_search::FeatureLookup load_features =
        [&](std::size_t database_index) -> const image_search::LocalFeatureSet& {
            return database_features.features(database_index);
        };

    std::vector<image_search::QueryEvaluation> evaluations;
    evaluations.reserve(prepared.queries.size());
    for(std::size_t i = 0; i < prepared.queries.size(); ++i) {
        const std::string& sequence = prepared.queries[i].identifier;
        // T_candidates, fewer than C if the method returns fewer
        const auto top_candidates = index.candidates(prepared.query_bows[i], sequence,
                                                     args.candidate_count);

        // untimed: each image is extracted once and kept for later queries
        std::vector<std::size_t> candidate_indices;
        candidate_indices.reserve(top_candidates.candidates.size());
        for(const auto& candidate : top_candidates.candidates)
            candidate_indices.push_back(candidate.database_index);
        database_features.load(candidate_indices);

        // T_verify
        const auto& query = query_features.features(i);
        const auto verification = image_search::verify_candidates(
            top_candidates.candidates, query, load_features, args.rho, args.ransac);

        // HPatches ground truth enters only here, after both timers
        evaluations.push_back(image_search::evaluate_query(
            top_candidates, verification, args.search.hp_path, sequence,
            query.width, query.height));
    }
    return evaluations;
}

}  // namespace

int main(int argc, char** argv) {
    image_search::VerifyArgs args;
    // parse errors print the message and stop before any image is processed
    try {
        args = image_search::parse_verify_args(std::vector<std::string>(argv + 1, argv + argc));
    } catch(const std::invalid_argument& error) {
        std::cerr << "error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
    const auto& search_args = args.search;

    // checked before the long preprocessing, in append mode so an existing result is not
    // truncated if the run fails later
    if(!std::ofstream(search_args.out_file, std::ios::app)) {
        std::cerr << "error: could not open output file: " << search_args.out_file << "\n";
        return EXIT_FAILURE;
    }

    // same vocabulary, BoVW database and queries as search
    image_search::PreparedSearch prepared;
    try {
        prepared = image_search::prepare_search(search_args);
    } catch(const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
    const auto& database_bows = prepared.database_bows;
    const auto& database_ids = prepared.database_ids;
    // -kclusters is required here, only the ceil(sqrt(n)) cap and nprobe default apply
    image_search::resolve_ivf_parameters(args.search, database_bows.size());

    try {
        // query features up front, candidate features per query
        std::vector<std::filesystem::path> query_paths;
        for(const auto& query : prepared.queries)
            query_paths.emplace_back(query.filepath);
        image_search::LocalFeatureStore query_features(std::move(query_paths));
        std::vector<std::size_t> query_indices(prepared.queries.size());
        for(std::size_t i = 0; i < query_indices.size(); ++i)
            query_indices[i] = i;
        query_features.load(query_indices);
        image_search::LocalFeatureStore database_features(prepared.database_paths);

        std::vector<image_search::QueryEvaluation> evaluations;
        // method name printed at the top of the output file
        std::string output_name;
        const auto run_queries = [&](const auto& index) {
            evaluations = verify_queries(index, args, prepared, query_features,
                                         database_features);
        };

        if(search_args.method == "exact") {
            run_queries(image_search::ExactSearch(database_bows, database_ids));
            output_name = "EXACT";
        } else if(search_args.method == "lsh") {
            run_queries(image_search::LSHSearch(database_bows, database_ids, search_args.lsh_k,
                                                search_args.lsh_l, search_args.w,
                                                search_args.seed));
            output_name = "LSH";
        } else if(search_args.method == "hypercube") {
            run_queries(image_search::HypercubeSearch(
                database_bows, database_ids, search_args.hyper_kproj, search_args.w,
                search_args.m_param, search_args.hyper_probes, search_args.seed));
            output_name = "HYPERCUBE";
        } else if(search_args.method == "ivfflat") {
            run_queries(image_search::IVFFlatSearch(
                database_bows, database_ids, search_args.ivf_kclusters,
                search_args.ivf_nprobe, search_args.seed));
            output_name = "IVFFLAT";
        } else if(search_args.method == "ivfpq") {
            run_queries(image_search::IVFPQSearch(
                database_bows, database_ids, search_args.ivf_kclusters,
                search_args.ivf_nprobe, search_args.m_param, search_args.pq_nbits,
                search_args.seed));
            output_name = "IVFPQ";
        }

        image_search::write_verification_results(search_args.out_file, output_name,
                                                 evaluations);
        std::cout << "wrote verification results to " << search_args.out_file << "\n";
        // only the returned candidates are verified, the study reports how often C is not reached
        std::size_t short_results = 0;
        for(const auto& evaluation : evaluations)
            if(evaluation.candidate_count < args.candidate_count)
                ++short_results;
        std::cout << short_results << " of " << evaluations.size()
                  << " queries returned fewer than C=" << args.candidate_count
                  << " candidates\n";
        std::cout << "extracted local features of " << database_features.loaded_count()
                  << " candidate images\n";
    } catch(const std::exception& error) {
        std::cerr << "error: verification failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
