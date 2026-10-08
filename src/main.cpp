#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <cstdlib>

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
    std::string method = "exact";
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

// parsing arguments from command line
Args parse_args(int argc, char** argv) {
    Args args;
    for(int i=1 ; i<argc ; ++i) {
        std::string arg = argv[i];

        // checking if the argument needs a value
        bool needs_val = (arg == "-hp" || arg == "-mir" || arg == "-split" ||
                        arg == "-vocab" || arg == "-S" || arg == "-D" ||
                        arg == "-set" || arg == "-o" || arg == "-seed");

        // preventing out of bounds if the value is missing
        if(needs_val && i + 1 >= argc) {
            std::cerr << "error: missing value for argument " << arg << "\n";
            std::exit(EXIT_FAILURE);
        }

        if(arg == "-hp" && i + 1 < argc) args.hp_path = argv[++i];
        else if(arg == "-mir" && i + 1 < argc) args.mir_path = argv[++i];
        else if(arg == "-split" && i + 1 < argc) args.split_file = argv[++i];
        else if(arg == "-vocab" && i + 1 < argc) args.vocab = parse_int(arg, argv[++i]);
        else if(arg == "-S" && i + 1 < argc) args.s_max = parse_int(arg, argv[++i]);
        else if(arg == "-D" && i + 1 < argc) args.d_mir = parse_int(arg, argv[++i]);
        else if(arg == "-set" && i + 1 < argc) args.set_type = argv[++i];
        else if(arg == "-o" && i + 1 < argc) args.out_file = argv[++i];
        else if(arg == "-seed" && i + 1 < argc) args.seed = parse_int(arg, argv[++i]);
        else if(arg == "-exact") args.method = "exact";
        // adding lsh, hypercube, ivfflat, ivfpq parsing later
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

    // validating numeric ranges and set types
    if(args.vocab <= 0) {
        std::cerr << "error: -vocab must be strictly positive\n";
        std::exit(EXIT_FAILURE);
    }
    if(args.s_max <= 0) {
        std::cerr << "error: -S must be strictly positive\n";
        std::exit(EXIT_FAILURE);
    }
    if(args.d_mir < 0) {
        std::cerr << "error: -D cannot be negative\n";
        std::exit(EXIT_FAILURE);
    }
    if(args.set_type != "validation" && args.set_type != "test") {
        std::cerr << "error: -set must be either validation or test\n";
        std::exit(EXIT_FAILURE);
    }

    return args;
}

// loading train validation test splits
std::map<std::string, std::vector<std::string>> load_splits(const std::string& filename) {
    std::map<std::string, std::vector<std::string>> splits;
    std::ifstream file(filename);

    // checking explicitly if file exists and is open
    if(!file.is_open()) {
        std::cerr << "error: could not open split file " << filename <<"\n";
        std::exit(EXIT_FAILURE);
    }

    std::string sequence, subset;
    while(file >> sequence >> subset) {
        splits[subset].push_back(sequence);
    }
    return splits;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    auto splits = load_splits(args.split_file);

    std::cout << "loaded " << splits["train"].size() << " training sequences\n";
    return 0;
}