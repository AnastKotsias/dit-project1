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

// holds parsed arguments
struct Args {
    std::string hp_path;
    std::string mir_path;
    std::string split_file;
    int vocab = 0;
    int s_max = 500;
    int d_mir = 1000;
    std::string set_type = "validation";
    std::string out_file = "output.txt";
    int seed = 1;
    std::string method;

    // method specific parameters
    int lsh_k = 4;
    int lsh_l = 5;
    double w = 0.05;
    int hyper_kproj = 10;
    int hyper_probes = 10;
    int ivf_kclusters = -1;  // to dynamically calculate sqrt(n) later
    int ivf_nprobe = -1;     // resolved after the database size is known
    int m_param = -1;   // to handle default 500 for hypercube or 16 for pq later
    int pq_nbits = 8;
};

// helper to safely parse ints and avoid stoi exceptions
int parse_int(const std::string& arg_name, const std::string& val_str) {
    try {
        size_t pos;
        int val = std::stoi(val_str, &pos);
        // rejecting partial matches like 256abc
        if(pos != val_str.length()) {
            throw std::invalid_argument("trailing characters");
        }
        return val;
    } catch(const std::exception&) {
        std::cerr << "error: invalid integer value for " << arg_name << "\n";
        std::exit(EXIT_FAILURE);
    }
}

// helper to safely parse doubles
double parse_double(const std::string& arg_name, const std::string& val_str) {
    try {
        size_t pos;
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
    std::set<std::string> parsed_options;

    for(int i = 1 ; i < argc ; ++i) {
        std::string arg = argv[i];

        // checking for duplicate arguments to prevent silent overwriting
        if(parsed_options.count(arg)) {
            std::cerr << "error: duplicate argument " << arg << "\n";
            std::exit(EXIT_FAILURE);
        }
        
        bool is_method = (arg == "-exact" || arg == "-lsh" || arg == "-hypercube" || arg == "-ivfflat" || arg == "-ivfpq");
        if(is_method) {
            if(!args.method.empty()) {
                std::cerr << "error: multiple methods flags provided\n";
                std::exit(EXIT_FAILURE);
            }
            args.method = arg.substr(1);
            continue;
        }
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
    if(args.method.empty()) {
        std::cerr << "error: exactly one search method flag is required\n";
        std::exit(EXIT_FAILURE);
    }

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
    if(args.lsh_k <= 0 || args.lsh_l <= 0 || args.w <= 0.0 || args.hyper_kproj <= 0 ||
        args.hyper_probes <= 0 || args.pq_nbits <= 0) {
        std::cerr << "error: method parameters (k, L, w, kproj, probes, nbits) must be strictly positive\n";
        std::exit(EXIT_FAILURE);
    }
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

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    
    std::map<std::string, std::vector<std::string>> splits;
    try {
        splits = image_search::load_dataset_splits(args.split_file);
    } catch(const std::exception& error) {
        std::cerr << "error: failed to load split file: "
                  << error.what() << "\n";
        return EXIT_FAILURE;
    }
    std::cout << "loaded " << splits["train"].size() << " training sequences\n";
    
    // resolving paths for training images
    auto train_paths = image_search::get_training_image_paths(args.hp_path, splits["train"]);
    std::cout << "found " << train_paths.size() << " training images\n";

    // setting up random number generator for sampling
    std::mt19937 rng(args.seed);
    std::cout << "extracting and sampling sift descriptors for vocabulary...\n";
    const auto training_result = image_search::extract_training_descriptors(
        train_paths, args.s_max, rng);
    for(const auto& path : training_result.failed_paths) {
        std::cerr << "warning: failed to extract descriptors from training image: "
                  << path << "\n";
    }

    if(training_result.successful_images == 0) {
        std::cerr << "error: all training images failed descriptor extraction\n";
        return EXIT_FAILURE;
    }

    std::cout << "computing visual vocabulary with k=" << args.vocab << "...\n";
    cv::Mat vocabulary;
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

    image_search::EvaluationDataset evaluation_dataset;
    try {
        evaluation_dataset = image_search::get_evaluation_paths(
            args.hp_path, args.mir_path, args.split_file, args.set_type);
    } catch(const std::exception& error) {
        std::cerr << "error: failed to construct evaluation dataset: "
                  << error.what() << "\n";
        return EXIT_FAILURE;
    }

    const auto hpatches_database = image_search::extract_bow_vectors(
        evaluation_dataset.database, vocabulary);
    for(const auto& record : hpatches_database.failed_records) {
        std::cerr << "warning: skipping invalid HPatches database image: "
                  << record.filepath << "\n";
    }

    const auto distractors = image_search::extract_bow_vectors(
        evaluation_dataset.distractor_pool, vocabulary, args.d_mir);
    const auto distractors_added = distractors.vectors.size();
    if(distractors_added < static_cast<std::size_t>(args.d_mir)) {
        std::cerr << "error: ran out of MIRFlickr images before reaching -D="
                  << args.d_mir << "\n";
        return EXIT_FAILURE;
    }

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

    if(args.method == "ivfflat" || args.method == "ivfpq") {
        if(args.ivf_kclusters == -1) {
            args.ivf_kclusters = static_cast<int>(
                std::ceil(std::sqrt(static_cast<double>(n_database))));
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
    if(!queries.failed_records.empty()) {
        std::cerr << "error: query image failed to produce descriptors: "
                  << queries.failed_records.front().filepath << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "prepared " << database_bows.size()
              << " database vectors and " << queries.vectors.size()
              << " query vectors\n";

    // Exact search implementation
    if (args.method == "exact") {
        image_search::ExactSearch exactSearch(database_bows, database_ids);
        std::vector<image_search::SearchResult> all_results;
        all_results.reserve(queries.vectors.size());

        for (size_t i = 0; i < queries.vectors.size(); ++i) {
            // Extracting query ID and sequence ID for the current query
            std::string q_id = queries.identifiers[i];
            std::string seq_id = evaluation_dataset.queries[i].identifier;
            
            // Performing the exact search for the current query and storing the result
            image_search::SearchResult res = exactSearch.search(queries.vectors[i], q_id, seq_id);
            all_results.push_back(res);
        }

        // Writing the results and metrics to the output file
        image_search::evaluateAndWriteResults(args.output_path, "EXACT", all_results);
    }

    return EXIT_SUCCESS;
}