#ifndef SEARCH_SETUP_HPP
#define SEARCH_SETUP_HPP

#include "command_line.hpp"
#include "dataset.hpp"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace image_search {

// BoVW vectors of one split, built once before any timed query
struct PreparedSearch {
    // HPatches images 2 to 6, then the first D usable MIRFlickr images
    std::vector<std::vector<double>> database_bows;
    std::vector<std::string> database_ids;
    // image file of each database vector, for the local features of verify
    std::vector<std::filesystem::path> database_paths;
    // image 1 of every sequence, identifier = sequence name
    std::vector<ImageRecord> queries;
    std::vector<std::vector<double>> query_bows;
};

// trains the vocabulary on the training split and builds the BoVW vectors of the -set split,
// printing progress like search
// failed training images and missing distractors are warnings
// throws std::runtime_error if a split, the vocabulary, an HPatches image or a query fails
PreparedSearch prepare_search(const SearchArgs& args);

// IVF defaults depend on the database size, known only after prepare_search:
// kclusters max(5, round(n^(1/4))) if not given, capped at ceil(sqrt(n)), nprobe 5 if not given
// does nothing for the other methods
void resolve_ivf_parameters(SearchArgs& args, std::size_t database_size);

}  // namespace image_search

#endif
