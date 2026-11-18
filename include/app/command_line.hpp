#ifndef COMMAND_LINE_HPP
#define COMMAND_LINE_HPP

#include "geometry/geometric_verification.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace image_search {

// command-line options of search, also accepted by verify
struct SearchArgs {
    std::string hp_path;
    std::string mir_path;
    std::string split_file;
    // number K of visual words (-vocab), zero means not given
    int vocab = 0;
    // maximum training descriptors per image (-S)
    int s_max = 500;
    // number of MIRFlickr distractors (-D)
    int d_mir = 1000;
    std::string set_type = "validation";
    std::string out_file = "output.txt";
    int seed = 1;
    // -cache: directory of the preprocessing cache, empty means no cache
    std::string cache_dir;
    // search method without the leading dash, empty if none
    std::string method;

    // LSH: hash functions per table (-k) and number of tables (-L)
    int lsh_k = 4;
    int lsh_l = 5;
    // window width of LSH and Hypercube (-w)
    double w = 0.05;
    // Hypercube: dimension d' (-kproj) and vertices probed (-probes)
    int hyper_kproj = 10;
    int hyper_probes = 10;
    // IVF parameters, -1 means not given: their defaults depend on the database size
    int ivf_kclusters = -1;
    int ivf_nprobe = -1;
    // -M, the Hypercube candidate limit or the number of PQ subvectors, -1 means not given
    int m_param = -1;
    // bits per PQ code (-nbits)
    int pq_nbits = 8;
    // run the experimental study (-experiment)
    bool experiment = false;
};

// parses and validates the arguments of search, without the program name
// throws std::invalid_argument with the error message on invalid input
SearchArgs parse_search_args(const std::vector<std::string>& arguments);

// command-line options of verify
struct VerifyArgs {
    // options of search and of the selected method
    SearchArgs search;
    // candidate images C (-C), at least 10
    std::size_t candidate_count = 50;
    // Lowe ratio rho (-rho), d1 < rho d2
    double rho = 0.8;
    // -tau, -T, -p, -adaptive; seed from -seed
    RansacConfig ransac;
};

// parses and validates the arguments of verify, without the program name
// -C, -rho, -tau, -T, -p, -adaptive here, the rest by parse_search_args
// -kclusters required for IVFFlat and IVFPQ, no -experiment
// throws std::invalid_argument with the error message on invalid input
VerifyArgs parse_verify_args(const std::vector<std::string>& arguments);

}  // namespace image_search

#endif
