#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

// holds parsed arguments
struct Args {
    std::string hp_path;
    std::string mir_path;
    std::string split_file;
    int vocab = 256;
    int s_max = 500;
    int d_mir = 1000;
    std::string set_type = "validation";
    std::string out_file = "output.txt";
    int seed = 1;
    std::string method = "exact";
};

Args parse_args(int argc, char** argv) {
    Args args;
    for(int i=1 ; i<argc ; ++i) {
        std::string arg = argv[i];
        if(arg == "-hp" && i + 1 < argc) {
            args.hp_path = argv[++i];
        } else if(arg == "-mir" && i + 1 < argc) {
            args.mir_path = argv[++i];
        } else if(arg == "-split" && i + 1 < argc) {
            args.split_file = argv[++i];
        } else if(arg == "-vocab" && i + 1 < argc) {
            args.vocab = std::stoi(argv[++i]);
        } else if(arg == "-S" && i + 1 < argc) {
            args.s_max = std::stoi(argv[++i]);
        } else if(arg == "-D" && i + 1 < argc) {
            args.d_mir = std::stoi(argv[++i]);
        } else if(arg == "-set" && i + 1 < argc) {
            args.set_type = argv[++i];
        } else if(arg == "-o" && i + 1 < argc) {
            args.out_file = argv[++i];
        } else if(arg == "-seed" && i + 1 < argc) {
            args.seed = std::stoi(argv[++i]);
        } else if(arg == "-exact") {
            args.method = "exact";
        }
        // adding lsh, hypercube, ivfflat, ivfpq parsing later
    }
    return args;
}

// loading train validation test splits
std::map<std::string, std::vector<std::string>> load_splits(const std::string& filename) {
    std::map<std::string, std::vector<std::string>> splits;
    std::ifstream file(filename);
    std::string sequence, subset;
    while(file >> sequence >> subset) {
        splits[subset].push_back(sequence);
    }
    return splits;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    auto splits = load_splits(args.split_file);

    std::cout << "loaded" << splits["train"].size() << "training sequences\n";
    return 0;
}