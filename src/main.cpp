#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <cstdlib>
#include <cmath>
#include <set>
#include <optional>
#include <random>
#include <filesystem>

#include "dataset.hpp"
#include "pipeline.hpp"
#include "vocabulary.hpp"
#include "search.hpp"
#include "experiment.hpp"

// holds parsed arguments
struct Args {
    std::string hp_path;
    std::string mir_path;
    std::string split_file;
    // number K of visual words (-vocab), zero means not given
    int vocab = 0;
    int s_max = 500;
    int d_mir = 1000;
    std::string set_type = "validation";
    std::string out_file = "output.txt";
    int seed = 1;
    // chosen search method without the leading dash, empty if none
    std::string method;

    // method specific parameters
    // LSH hash functions per table (-k)
    int lsh_k = 4;
    // LSH number of tables (-L)
    int lsh_l = 5;
    // window width shared by LSH and hypercube (-w)
    double w = 0.05;
    // hypercube dimension d' (-kproj)
    int hyper_kproj = 10;
    // hypercube maximum number of vertices probed (-probes)
    int hyper_probes = 10;
    int ivf_kclusters = -1;  // to dynamically calculate sqrt(n) later
    int ivf_nprobe = -1;     // resolved after the database size is known
    int m_param = -1;   // to handle default 500 for hypercube or 16 for pq later
    // bits per PQ code, so 2^nbits codebook entries (-nbits)
    int pq_nbits = 8;
    // true when the full experimental study is requested (-experiment)
    bool experiment = false;
};

// helper to safely parse ints and avoid stoi exceptions
int parse_int(const std::string& arg_name, const std::string& val_str) {
    try {
        // number of characters consumed by the conversion
        size_t pos;
        int val = std::stoi(val_str, &pos);
        // rejecting partial matches like 256abc
        if(pos != val_str.length()) {
            throw std::invalid_argument("trailing characters");
        }
        return val;
    } catch(const std::exception&) {
        // covers non-numeric text, out-of-range values and trailing characters
        std::cerr << "error: invalid integer value for " << arg_name << "\n";
        std::exit(EXIT_FAILURE);
    }
}

// helper to safely parse doubles
double parse_double(const std::string& arg_name, const std::string& val_str) {
    try {
        size_t pos;
        // stod accepts a numeric prefix, so the consumed length is checked below
        double val = std::stod(val_str, &pos);
        // rejecting partial matches
        if(pos != val_str.length()) {
            throw std::invalid_argument("trailing characters");
        }
        return val;
    } catch(const std::exception&) {
        std::cerr << "error: invalid double value for " << arg_name << "\n";
        std::exit(EXIT_FAILURE);
    }
}

// parsing arguments from command line
Args parse_args(int argc, char** argv) {
    Args args;
    // options seen so far, used to detect duplicates and invalid combinations
    std::set<std::string> parsed_options;

    for(int i = 1 ; i < argc ; ++i) {
        // current token, expected to be an option name
        std::string arg = argv[i];

        // checking for duplicate arguments to prevent silent overwriting
        if(parsed_options.count(arg)) {
            std::cerr << "error: duplicate argument " << arg << "\n";
            std::exit(EXIT_FAILURE);
        }
        
        // method flags are switches that take no value
        bool is_method = (arg == "-exact" || arg == "-lsh" || arg == "-hypercube" || arg == "-ivfflat" || arg == "-ivfpq");
        // -experiment is also a switch without a value
        if(arg == "-experiment") {
            if(args.experiment) {
                std::cerr << "error: duplicate argument -experiment\n";
                std::exit(EXIT_FAILURE);
            }
            args.experiment = true;
            continue;
        }
        if(is_method) {
            if(!args.method.empty()) {
                std::cerr << "error: multiple methods flags provided\n";
                std::exit(EXIT_FAILURE);
            }
            // store the name without the leading dash
            args.method = arg.substr(1);
            continue;
        }
        // every remaining option takes a value
        parsed_options.insert(arg);

        if(i + 1 >= argc) {
            std::cerr << "error: missing value for argument " << arg << "\n";
            std::exit(EXIT_FAILURE);
        }

        // core arguments
        if(arg == "-hp") args.hp_path = argv[++i];
        else if(arg == "-mir") args.mir_path = argv[++i];
        else if(arg == "-split") args.split_file = argv[++i];
        else if(arg == "-vocab") args.vocab = parse_int(arg, argv[++i]);
        else if(arg == "-S") args.s_max = parse_int(arg, argv[++i]);
        else if(arg == "-D") args.d_mir = parse_int(arg, argv[++i]);
        else if(arg == "-set") args.set_type = argv[++i];
        else if(arg == "-o") args.out_file = argv[++i];
        else if(arg == "-seed") args.seed = parse_int(arg, argv[++i]);

        // method specific arguments
        else if(arg == "-k") args.lsh_k = parse_int(arg, argv[++i]);
        else if(arg == "-L") args.lsh_l = parse_int(arg, argv[++i]);
        else if(arg == "-w") args.w = parse_double(arg, argv[++i]);
        else if(arg == "-kproj") args.hyper_kproj = parse_int(arg, argv[++i]);
        else if(arg == "-probes") args.hyper_probes = parse_int(arg, argv[++i]);
        else if(arg == "-kclusters") args.ivf_kclusters = parse_int(arg, argv[++i]);
        else if(arg == "-nprobe") args.ivf_nprobe = parse_int(arg, argv[++i]);
        else if(arg == "-M") args.m_param = parse_int(arg, argv[++i]);
        else if(arg == "-nbits") args.pq_nbits = parse_int(arg, argv[++i]);

        else {
            std::cerr << "error: unknown argument " << arg << "\n";
            std::exit(EXIT_FAILURE);
        }
    }

    // validating required arguments
    if(args.hp_path.empty() || args.mir_path.empty() || args.split_file.empty() || args.out_file.empty() || args.vocab == 0) {
        std::cerr << "error: missing required arguments (-hp, -mir, -split, -o, -vocab)\n";
        std::exit(EXIT_FAILURE);
    }
    if(args.method.empty() && !args.experiment) {
        std::cerr << "error: exactly one search method flag is required\n";
        std::exit(EXIT_FAILURE);
    }

    if(args.experiment && !args.method.empty()) {
        std::cerr << "error: -experiment cannot be combined with a search method flag\n";
        std::exit(EXIT_FAILURE);
    }

    // reject options that do not belong to the chosen method
    if(args.method == "exact" && (parsed_options.count("-k") || parsed_options.count("-L") || parsed_options.count("-w") ||
                                parsed_options.count("-kproj") || parsed_options.count("-probes") || parsed_options.count("-kclusters") ||
                                parsed_options.count("-nprobe") || parsed_options.count("-M") || parsed_options.count("-nbits"))) {
        std::cerr << "error: exact search does not accept method-specific parameters\n";
        std::exit(EXIT_FAILURE);
    } else if(args.method == "lsh" && (parsed_options.count("-kproj") || parsed_options.count("-probes") || parsed_options.count("-kclusters") ||
                                parsed_options.count("-nprobe") || parsed_options.count("-M") || parsed_options.count("-nbits"))) {
        std::cerr << "error: invalid parameters for lsh search\n";
        std::exit(EXIT_FAILURE);
    } else if(args.method == "hypercube" && (parsed_options.count("-k") || parsed_options.count("-L") ||
                                parsed_options.count("-kclusters") ||
                                parsed_options.count("-nprobe") || parsed_options.count("-nbits"))) {
        std::cerr << "error: invalid parameters for hypercube search\n";
        std::exit(EXIT_FAILURE);
    } else if(args.method == "ivfflat" && (parsed_options.count("-k") || parsed_options.count("-L") || parsed_options.count("-w") ||
                                parsed_options.count("-kproj") || parsed_options.count("-probes") || parsed_options.count("-M") ||
                                parsed_options.count("-nbits"))) {
        std::cerr << "error: invalid parameters for ivfflat search\n";
        std::exit(EXIT_FAILURE);
    } else if(args.method == "ivfpq" && (parsed_options.count("-k") || parsed_options.count("-L") || parsed_options.count("-w") ||
                                parsed_options.count("-kproj") || parsed_options.count("-probes"))) {
        std::cerr << "error: invalid parameters for ivfpq search\n";
        std::exit(EXIT_FAILURE);
    }

    // validating numeric ranges and set types
    if(args.vocab <= 0 || args.s_max <= 0 || args.d_mir <= 0) {
        std::cerr << "error: -vocab, -S, and -D must be strictly positive\n";
        std::exit(EXIT_FAILURE);
    }
    if(args.d_mir != 1000 && args.d_mir != 5000 &&
       args.d_mir != 10000 && args.d_mir != 25000) {
        std::cerr << "error: -D must be 1000, 5000, 10000, or 25000\n";
        std::exit(EXIT_FAILURE);
    }
    if(args.set_type != "validation" && args.set_type != "test") {
        std::cerr << "error: -set must be either validation or test\n";
        std::exit(EXIT_FAILURE);
    }

    // validating method specific parameter constraints
    // -k, -L, -w, -kproj, -probes and -nbits have defaults, so they must be positive
    if(args.lsh_k <= 0 || args.lsh_l <= 0 || args.w <= 0.0 || args.hyper_kproj <= 0 ||
        args.hyper_probes <= 0 || args.pq_nbits <= 0) {
        std::cerr << "error: method parameters (k, L, w, kproj, probes, nbits) must be strictly positive\n";
        std::exit(EXIT_FAILURE);
    }
    // rejects nan and inf, which pass the positivity check above
    if(!std::isfinite(args.w) || args.w <= 0.0) {
        std::cerr << "error: -w must be finite and strictly positive\n";
        std::exit(EXIT_FAILURE);
    }

    // checking optional parameters that have dynamic default calculation flags
    if(args.ivf_kclusters != -1 && args.ivf_kclusters <= 0) {
        std::cerr << "error: -kclusters must be strictly positive\n";
        std::exit(EXIT_FAILURE);
    }
    if(args.ivf_nprobe != -1 && args.ivf_nprobe <= 0) {
        std::cerr << "error: -nprobe must be strictly positive\n";
        std::exit(EXIT_FAILURE);
    }
    if(args.m_param != -1 && args.m_param <= 0) {
        std::cerr << "error: -M must be strictly positive\n";
        std::exit(EXIT_FAILURE);
    }

    // experiment mode sets its own parameters, so method defaults are not needed
    if(args.experiment)
        return args;

    // resolving method specific defaults for M
    if(args.m_param == -1) {
        if(args.method == "hypercube") args.m_param = 500;
        else if(args.method == "ivfpq") args.m_param = 16;
    }

    // ensuring vocabulary size is divisible by M for IVFPQ
    if(args.method == "ivfpq" && args.vocab % args.m_param != 0) {
        std::cerr << "error: -vocab must be divisible by -M for IVFPQ\n";
        std::exit(EXIT_FAILURE);
    }

    return args;
}

// entry point that trains the vocabulary, builds the database, runs the chosen method and writes the results
int main(int argc, char** argv) {
    // parsing exits the program with a message on invalid input
    Args args = parse_args(argc, argv);

    // experiment mode runs the whole study and ignores the single-run flow below
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
    
    // step 1, read the split file
    std::map<std::string, std::vector<std::string>> splits;
    try {
        splits = image_search::load_dataset_splits(args.split_file);
    } catch(const std::exception& error) {
        std::cerr << "error: failed to load split file: "
                  << error.what() << "\n";
        return EXIT_FAILURE;
    }
    // splits["train"] holds the sequences used for the vocabulary
    std::cout << "loaded " << splits["train"].size() << " training sequences\n";
    
    // resolving paths for training images
    auto train_paths = image_search::get_training_image_paths(args.hp_path, splits["train"]);
    std::cout << "found " << train_paths.size() << " training images\n";

    // setting up random number generator for sampling
    std::mt19937 rng(args.seed);
    // step 2, extract and sample SIFT descriptors of the training images
    std::cout << "extracting and sampling sift descriptors for vocabulary...\n";
    const auto training_result = image_search::extract_training_descriptors(
        train_paths, args.s_max, rng);
    // failed training images only produce warnings, the vocabulary can still be built from the rest
    for(std::size_t i = 0; i < training_result.failed_paths.size(); ++i) {
        std::cerr << "warning: failed to extract descriptors from training image: "
                  << training_result.failed_paths[i] << " ("
                  << image_search::sift_extraction_status_name(
                         training_result.failed_statuses[i])
                  << ")\n";
    }

    // nothing to cluster if every image failed
    if(training_result.successful_images == 0) {
        std::cerr << "error: all training images failed descriptor extraction\n";
        return EXIT_FAILURE;
    }

    // step 3, cluster the sampled descriptors into K visual words
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

    // defensive check that k-means returned centroids
    if(vocabulary.empty()) {
        std::cerr << "error: failed to compute vocabulary\n";
        return EXIT_FAILURE;
    }
    std::cout << "vocabulary computed successfully using "
    << training_result.successful_images << " images ("
    << training_result.failed_images << " failed)\n";

    // step 4, collect the queries, database images and distractor pool of the chosen split
    image_search::EvaluationDataset evaluation_dataset;
    try {
        evaluation_dataset = image_search::get_evaluation_paths(
            args.hp_path, args.mir_path, args.split_file, args.set_type);
    } catch(const std::exception& error) {
        std::cerr << "error: failed to construct evaluation dataset: "
                  << error.what() << "\n";
        return EXIT_FAILURE;
    }

    // convert the HPatches database images to BoW vectors, skipping unusable ones with a warning
    const auto hpatches_database = image_search::extract_bow_vectors(
        evaluation_dataset.database, vocabulary);
    for(const auto& record : hpatches_database.failed_records) {
        std::cerr << "warning: skipping invalid HPatches database image: "
                  << record.record.filepath << " ("
                  << image_search::sift_extraction_status_name(record.status)
                  << ")\n";
    }

    // take the first D usable MIRFlickr images as distractors
    const auto distractors = image_search::extract_bow_vectors(
        evaluation_dataset.distractor_pool, vocabulary, args.d_mir);
    // fewer than D means the pool ran out of usable images
    const auto distractors_added = distractors.vectors.size();
    if(distractors_added < static_cast<std::size_t>(args.d_mir)) {
        std::cerr << "error: ran out of MIRFlickr images before reaching -D="
                  << args.d_mir << "\n";
        return EXIT_FAILURE;
    }

    // final database is the HPatches vectors followed by the distractors
    std::vector<std::vector<double>> database_bows = hpatches_database.vectors;
    std::vector<std::string> database_ids = hpatches_database.identifiers;
    database_bows.insert(database_bows.end(), distractors.vectors.begin(),
                          distractors.vectors.end());
    database_ids.insert(database_ids.end(), distractors.identifiers.begin(),
                        distractors.identifiers.end());

    // database size n, needed for the default number of IVF clusters
    const int n_database = static_cast<int>(database_bows.size());
    if(n_database <= 0) {
        std::cerr << "error: database contains no usable image vectors\n";
        return EXIT_FAILURE;
    }

    // fill in IVF defaults that depend on the database size, which was unknown during parsing
    if(args.method == "ivfflat" || args.method == "ivfpq") {
        // default number of clusters is ceil(sqrt(n))
        if(args.ivf_kclusters == -1) {
            args.ivf_kclusters = static_cast<int>(
                std::ceil(std::sqrt(static_cast<double>(n_database))));
        }
        // default number of probed clusters is 5
        if(args.ivf_nprobe == -1) {
            args.ivf_nprobe = 5;
        }
        std::cout << "resolved IVF parameters: kclusters="
                  << args.ivf_kclusters
                  << ", nprobe=" << args.ivf_nprobe << "\n";
    }

    // step 5, convert the query images, one per sequence, to BoW vectors
    const auto queries = image_search::extract_bow_vectors(
        evaluation_dataset.queries, vocabulary);
    // a missing query cannot be evaluated, so this is treated as an error
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

    // step 6, build the chosen index and run every query
    try {
        // one result per query
        std::vector<image_search::SearchResult> all_results;
        all_results.reserve(queries.vectors.size());
        // method label printed at the top of each result block
        std::string output_name;

        // each branch builds its index and searches with every query, passing the query's sequence name for relevance

        if(args.method == "exact") {
            image_search::ExactSearch index(database_bows, database_ids);
            for(std::size_t i = 0; i < queries.vectors.size(); ++i)
                all_results.push_back(index.search(
                    queries.vectors[i], queries.identifiers[i],
                    evaluation_dataset.queries[i].identifier));
            output_name = "EXACT";
        } else if(args.method == "lsh") {
            image_search::LSHSearch index(database_bows, database_ids, args.lsh_k,
                                          args.lsh_l, args.w, args.seed);
            for(std::size_t i = 0; i < queries.vectors.size(); ++i)
                all_results.push_back(index.search(
                    queries.vectors[i], queries.identifiers[i],
                    evaluation_dataset.queries[i].identifier));
            output_name = "LSH";
        } else if(args.method == "hypercube") {
            image_search::HypercubeSearch index(
                database_bows, database_ids, args.hyper_kproj, args.w,
                args.m_param, args.hyper_probes, args.seed);
            for(std::size_t i = 0; i < queries.vectors.size(); ++i)
                all_results.push_back(index.search(
                    queries.vectors[i], queries.identifiers[i],
                    evaluation_dataset.queries[i].identifier));
            output_name = "HYPERCUBE";
        } else if(args.method == "ivfflat") {
            image_search::IVFFlatSearch index(
                database_bows, database_ids, args.ivf_kclusters,
                args.ivf_nprobe, args.seed);
            for(std::size_t i = 0; i < queries.vectors.size(); ++i)
                all_results.push_back(index.search(
                    queries.vectors[i], queries.identifiers[i],
                    evaluation_dataset.queries[i].identifier));
            output_name = "IVFFLAT";
        } else if(args.method == "ivfpq") {
            image_search::IVFPQSearch index(
                database_bows, database_ids, args.ivf_kclusters,
                args.ivf_nprobe, args.m_param, args.pq_nbits, args.seed);
            for(std::size_t i = 0; i < queries.vectors.size(); ++i)
                all_results.push_back(index.search(
                    queries.vectors[i], queries.identifiers[i],
                    evaluation_dataset.queries[i].identifier));
            output_name = "IVFPQ";
        }

        // step 7, write the rankings and metrics to the output file
        image_search::evaluateAndWriteResults(args.out_file, output_name, all_results);
        std::cout << "wrote search results to " << args.out_file << "\n";
    } catch(const std::exception& error) {
        std::cerr << "error: search index failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}