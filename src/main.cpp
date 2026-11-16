#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "data/dataset.hpp"
#include "search/search.hpp"
#include "app/experiment.hpp"
#include "app/command_line.hpp"
#include "app/search_setup.hpp"

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

    // vocabulary, BoVW database and queries of the chosen split
    image_search::PreparedSearch prepared;
    try {
        prepared = image_search::prepare_search(args);
    } catch(const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
    const auto& database_bows = prepared.database_bows;
    const auto& database_ids = prepared.database_ids;
    // IVF defaults depend on the database size, known only now
    image_search::resolve_ivf_parameters(args, database_bows.size());

    // build the chosen index and run every query
    try {
        std::vector<image_search::SearchResult> all_results;
        all_results.reserve(prepared.query_bows.size());
        // method name printed at the top of the output file
        std::string output_name;

        // the sequence name of each query defines its relevant images
        const auto run_queries = [&](const auto& index) {
            for(std::size_t i = 0; i < prepared.query_bows.size(); ++i)
                all_results.push_back(index.search(
                    prepared.query_bows[i], prepared.queries[i].identifier,
                    prepared.queries[i].identifier));
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
