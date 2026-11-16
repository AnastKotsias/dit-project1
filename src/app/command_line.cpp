#include "app/command_line.hpp"

#include "search/search.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <system_error>

namespace image_search {
namespace {

// parses an integer, rejects text like "256abc"
int parse_int(const std::string& arg_name, const std::string& val_str) {
    try {
        size_t pos;
        int val = std::stoi(val_str, &pos);
        if(pos == val_str.length())
            return val;
    } catch(const std::exception&) {
    }
    throw std::invalid_argument("invalid integer value for " + arg_name);
}

// parses a double, rejects text like "0.05x"
double parse_double(const std::string& arg_name, const std::string& val_str) {
    try {
        size_t pos;
        double val = std::stod(val_str, &pos);
        if(pos == val_str.length())
            return val;
    } catch(const std::exception&) {
    }
    throw std::invalid_argument("invalid double value for " + arg_name);
}

}  // namespace

SearchArgs parse_search_args(const std::vector<std::string>& arguments) {
    SearchArgs args;
    // options that take a value and were seen so far
    std::set<std::string> parsed_options;

    for(std::size_t i = 0 ; i < arguments.size() ; ++i) {
        const std::string& arg = arguments[i];

        if(parsed_options.count(arg))
            throw std::invalid_argument("duplicate argument " + arg);

        // -experiment and the method flags are switches without a value
        if(arg == "-experiment") {
            if(args.experiment)
                throw std::invalid_argument("duplicate argument -experiment");
            args.experiment = true;
            continue;
        }
        const bool is_method = (arg == "-exact" || arg == "-lsh" || arg == "-hypercube" || arg == "-ivfflat" || arg == "-ivfpq");
        if(is_method) {
            if(!args.method.empty())
                throw std::invalid_argument("multiple methods flags provided");
            args.method = arg.substr(1);
            continue;
        }
        parsed_options.insert(arg);

        if(i + 1 >= arguments.size())
            throw std::invalid_argument("missing value for argument " + arg);
        const std::string& value = arguments[++i];

        if(arg == "-hp") args.hp_path = value;
        else if(arg == "-mir") args.mir_path = value;
        else if(arg == "-split") args.split_file = value;
        else if(arg == "-vocab") args.vocab = parse_int(arg, value);
        else if(arg == "-S") args.s_max = parse_int(arg, value);
        else if(arg == "-D") args.d_mir = parse_int(arg, value);
        else if(arg == "-set") args.set_type = value;
        else if(arg == "-o") args.out_file = value;
        else if(arg == "-seed") args.seed = parse_int(arg, value);

        else if(arg == "-k") args.lsh_k = parse_int(arg, value);
        else if(arg == "-L") args.lsh_l = parse_int(arg, value);
        else if(arg == "-w") args.w = parse_double(arg, value);
        else if(arg == "-kproj") args.hyper_kproj = parse_int(arg, value);
        else if(arg == "-probes") args.hyper_probes = parse_int(arg, value);
        else if(arg == "-kclusters") args.ivf_kclusters = parse_int(arg, value);
        else if(arg == "-nprobe") args.ivf_nprobe = parse_int(arg, value);
        else if(arg == "-M") args.m_param = parse_int(arg, value);
        else if(arg == "-nbits") args.pq_nbits = parse_int(arg, value);

        else
            throw std::invalid_argument("unknown argument " + arg);
    }

    if(args.hp_path.empty() || args.mir_path.empty() || args.split_file.empty() || args.out_file.empty())
        throw std::invalid_argument("missing required arguments (-hp, -mir, -split, -o, -vocab)");
    // experiment mode without -vocab compares several vocabulary sizes, a single run needs one
    if(!args.experiment && !parsed_options.count("-vocab"))
        throw std::invalid_argument("missing required argument -vocab");
    if(args.method.empty() && !args.experiment)
        throw std::invalid_argument("exactly one search method flag is required");

    if(args.experiment && !args.method.empty())
        throw std::invalid_argument("-experiment cannot be combined with a search method flag");
    // experiment mode chooses these itself, so they would be silently ignored
    if(args.experiment) {
        for(const char* option : {"-D", "-set", "-k", "-L", "-w", "-kproj", "-probes",
                                  "-kclusters", "-nprobe", "-M", "-nbits"}) {
            if(parsed_options.count(option))
                throw std::invalid_argument(std::string(option) +
                                            " cannot be used with -experiment, which sets it itself");
        }
    }

    // checked now, otherwise a wrong path is only noticed after the training images are processed
    std::error_code directory_error;
    if(!std::filesystem::is_directory(args.hp_path, directory_error))
        throw std::invalid_argument("-hp is not a directory: " + args.hp_path);
    if(!std::filesystem::is_directory(args.mir_path, directory_error))
        throw std::invalid_argument("-mir is not a directory: " + args.mir_path);

    // options of another method are rejected
    if(args.method == "exact" && (parsed_options.count("-k") || parsed_options.count("-L") || parsed_options.count("-w") ||
                                parsed_options.count("-kproj") || parsed_options.count("-probes") || parsed_options.count("-kclusters") ||
                                parsed_options.count("-nprobe") || parsed_options.count("-M") || parsed_options.count("-nbits"))) {
        throw std::invalid_argument("exact search does not accept method-specific parameters");
    } else if(args.method == "lsh" && (parsed_options.count("-kproj") || parsed_options.count("-probes") || parsed_options.count("-kclusters") ||
                                parsed_options.count("-nprobe") || parsed_options.count("-M") || parsed_options.count("-nbits"))) {
        throw std::invalid_argument("invalid parameters for lsh search");
    } else if(args.method == "hypercube" && (parsed_options.count("-k") || parsed_options.count("-L") ||
                                parsed_options.count("-kclusters") ||
                                parsed_options.count("-nprobe") || parsed_options.count("-nbits"))) {
        throw std::invalid_argument("invalid parameters for hypercube search");
    } else if(args.method == "ivfflat" && (parsed_options.count("-k") || parsed_options.count("-L") || parsed_options.count("-w") ||
                                parsed_options.count("-kproj") || parsed_options.count("-probes") || parsed_options.count("-M") ||
                                parsed_options.count("-nbits"))) {
        throw std::invalid_argument("invalid parameters for ivfflat search");
    } else if(args.method == "ivfpq" && (parsed_options.count("-k") || parsed_options.count("-L") || parsed_options.count("-w") ||
                                parsed_options.count("-kproj") || parsed_options.count("-probes"))) {
        throw std::invalid_argument("invalid parameters for ivfpq search");
    }

    if((parsed_options.count("-vocab") && args.vocab <= 0) || args.s_max <= 0 || args.d_mir <= 0)
        throw std::invalid_argument("-vocab, -S, and -D must be strictly positive");
    if(args.d_mir != 1000 && args.d_mir != 5000 &&
       args.d_mir != 10000 && args.d_mir != 25000)
        throw std::invalid_argument("-D must be 1000, 5000, 10000, or 25000");
    if(args.set_type != "validation" && args.set_type != "test")
        throw std::invalid_argument("-set must be either validation or test");

    if(args.lsh_k <= 0 || args.lsh_l <= 0 || args.w <= 0.0 || args.hyper_kproj <= 0 ||
        args.hyper_probes <= 0 || args.pq_nbits <= 0)
        throw std::invalid_argument("method parameters (k, L, w, kproj, probes, nbits) must be strictly positive");
    // upper limits of the index constructors, checked here so they fail before any image is processed
    if(args.hyper_kproj > kMaxHypercubeProjections)
        throw std::invalid_argument("-kproj must be at most " +
                                    std::to_string(kMaxHypercubeProjections));
    if(args.pq_nbits > kMaxPQBits)
        throw std::invalid_argument("-nbits must be at most " + std::to_string(kMaxPQBits));
    // nan and inf pass the positivity check above
    if(!std::isfinite(args.w) || args.w <= 0.0)
        throw std::invalid_argument("-w must be finite and strictly positive");

    if(args.ivf_kclusters != -1 && args.ivf_kclusters <= 0)
        throw std::invalid_argument("-kclusters must be strictly positive");
    if(args.ivf_nprobe != -1 && args.ivf_nprobe <= 0)
        throw std::invalid_argument("-nprobe must be strictly positive");
    if(args.m_param != -1 && args.m_param <= 0)
        throw std::invalid_argument("-M must be strictly positive");

    // experiment mode sets its own method parameters
    if(args.experiment)
        return args;

    if(args.m_param == -1) {
        if(args.method == "hypercube") args.m_param = 500;
        else if(args.method == "ivfpq") args.m_param = 16;
    }

    if(args.method == "ivfpq" && args.vocab % args.m_param != 0)
        throw std::invalid_argument("-vocab must be divisible by -M for IVFPQ");

    return args;
}

VerifyArgs parse_verify_args(const std::vector<std::string>& arguments) {
    VerifyArgs args;
    // verify options seen so far
    std::set<std::string> parsed_options;
    // everything else, for parse_search_args
    std::vector<std::string> search_arguments;

    for(std::size_t i = 0 ; i < arguments.size() ; ++i) {
        const std::string& arg = arguments[i];
        // single run per call, no automatic tuning
        if(arg == "-experiment")
            throw std::invalid_argument("verify does not accept -experiment");

        const bool is_verify_option = (arg == "-C" || arg == "-rho" || arg == "-tau" ||
                                       arg == "-T" || arg == "-p" || arg == "-adaptive");
        if(!is_verify_option) {
            search_arguments.push_back(arg);
            const bool is_method = (arg == "-exact" || arg == "-lsh" || arg == "-hypercube" || arg == "-ivfflat" || arg == "-ivfpq");
            // value kept with its option, e.g. -o -C is a file name
            if(!is_method && i + 1 < arguments.size())
                search_arguments.push_back(arguments[++i]);
            continue;
        }

        if(parsed_options.count(arg))
            throw std::invalid_argument("duplicate argument " + arg);
        parsed_options.insert(arg);
        if(i + 1 >= arguments.size())
            throw std::invalid_argument("missing value for argument " + arg);
        const std::string& value = arguments[++i];

        if(arg == "-C") {
            // C >= 10: Before/After rankings of 10 images
            const int count = parse_int(arg, value);
            if(count < 10)
                throw std::invalid_argument("-C must be at least 10");
            args.candidate_count = static_cast<std::size_t>(count);
        } else if(arg == "-rho") {
            // 0 < rho < 1, nan fails both comparisons
            args.rho = parse_double(arg, value);
            if(!(args.rho > 0.0 && args.rho < 1.0))
                throw std::invalid_argument("-rho must be strictly between 0 and 1");
        } else if(arg == "-tau") {
            // e_i(H) < tau pixels
            args.ransac.tau = parse_double(arg, value);
            if(!std::isfinite(args.ransac.tau) || args.ransac.tau <= 0.0)
                throw std::invalid_argument("-tau must be finite and strictly positive");
        } else if(arg == "-T") {
            // Tmax >= 1 trials
            const int trials = parse_int(arg, value);
            if(trials <= 0)
                throw std::invalid_argument("-T must be strictly positive");
            args.ransac.maximum_trials = static_cast<std::size_t>(trials);
        } else if(arg == "-p") {
            // 0 < p < 1, log(1 - p) finite and negative
            args.ransac.confidence = parse_double(arg, value);
            if(!(args.ransac.confidence > 0.0 && args.ransac.confidence < 1.0))
                throw std::invalid_argument("-p must be strictly between 0 and 1");
        } else {
            const int adaptive = parse_int(arg, value);
            if(adaptive != 0 && adaptive != 1)
                throw std::invalid_argument("-adaptive must be 0 or 1");
            args.ransac.adaptive = adaptive == 1;
        }
    }

    args.search = parse_search_args(search_arguments);
    // no default cluster count in assignment 2
    if((args.search.method == "ivfflat" || args.search.method == "ivfpq") &&
       args.search.ivf_kclusters == -1)
        throw std::invalid_argument("-kclusters is required for ivfflat and ivfpq");
    // same seed as the BoVW/ANN stage
    args.ransac.seed = static_cast<std::uint64_t>(args.search.seed);
    return args;
}

}  // namespace image_search
