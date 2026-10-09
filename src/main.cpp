#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <cstdlib>
#include <cmath>
#include <set>
#include <optional>
#include <sstream>

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
    int ivf_nprobe = 5;
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
    } else if(args.method == "hypercube" && (parsed_options.count("-k") || parsed_options.count("-L") || parsed_options.count("-w") ||
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
    if(args.set_type != "validation" && args.set_type != "test") {
        std::cerr << "error: -set must be either validation or test\n";
        std::exit(EXIT_FAILURE);
    }

    // validating method specific parameter constraints
    if(args.lsh_k <= 0 || args.lsh_l <= 0 || args.w <= 0.0 || args.hyper_kproj <= 0 ||
        args.hyper_probes <= 0 || args.ivf_nprobe <= 0 || args.pq_nbits <= 0) {
        std::cerr << "error: method parameters (k, L, w, kproj, probes, nprobe, nbits) must be strictly positive\n";
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

// loading train validation test splits
std::map<std::string, std::vector<std::string>> load_splits(const std::string& filename) {
    std::map<std::string, std::vector<std::string>> splits;
    std::set<std::string> seen_seqs;
    std::ifstream file(filename);

    // checking explicitly if file exists and is open
    if(!file.is_open()) {
        std::cerr << "error: could not open split file " << filename <<"\n";
        std::exit(EXIT_FAILURE);
    }

    std::string line;
    while(std::getline(file, line)) {
        if(line.empty()) continue;

        std::istringstream iss(line);
        std::string sequence, subset, extra;

        if(!(iss >> sequence >> subset) || (iss >> extra)) {
            std::cerr << "error: malformed line in split file: " << line << "\n";
            std::exit(EXIT_FAILURE);
        }

        if(subset != "train" && subset != "validation" && subset != "test") {
            std::cerr << "error: invalid subset name " << subset << " in split file\n";
            std::exit(EXIT_FAILURE);
        }

        if(!seen_seqs.insert(sequence).second) {
            std::cerr << "error: duplicate sequence " << sequence << " in split file\n";
            std::exit(EXIT_FAILURE);
        }

        splits[subset].push_back(sequence);
    }

    if(splits["train"].size() != 70 || splits["validation"].size() != 23 || splits["test"].size() != 23) {
        std::cerr << "error: incorrect sequence counts per split requirements\n";
        std::exit(EXIT_FAILURE);
    }

    return splits;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    
    auto splits = load_splits(args.split_file);
    std::cout << "loaded " << splits["train"].size() << " training sequences\n";
    
    return 0;
}