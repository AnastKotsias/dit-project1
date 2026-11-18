#ifndef CACHE_HPP
#define CACHE_HPP

#include "data/dataset.hpp"
#include "matrix.hpp"

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace image_search {

// optional -cache of the preprocessing, the programs give the same results without it
// a file holds its key, a file with another key is never used

// text that identifies a vocabulary: K, S, seed and every training image with its size
// a missing training image is part of the key too
std::string vocabulary_cache_key(const std::vector<std::filesystem::path>& training_paths,
                                 int vocab, int s_max, int seed);

// <cache_dir>/vocabulary_K<K>_S<S>_seed<seed>.bin
std::filesystem::path vocabulary_cache_path(const std::string& cache_dir,
                                            int vocab, int s_max, int seed);

// the stored vocabulary, bit for bit as saved
// nullopt if the file is missing, unreadable, truncated or holds another key
std::optional<DescriptorMatrix> load_vocabulary_cache(const std::filesystem::path& path,
                                                      const std::string& key);

// writes a temporary file, then renames it, so a crash never leaves a partial file
// creates the cache directory, throws if the file cannot be written
void save_vocabulary_cache(const std::filesystem::path& path, const std::string& key,
                           const DescriptorMatrix& vocabulary);

// BoVW vectors of one set, as prepared before the queries
struct BowCache {
    // HPatches images 2 to 6, then the MIRFlickr distractors
    std::vector<std::vector<double>> database_bows;
    std::vector<std::string> database_ids;
    std::vector<std::filesystem::path> database_paths;
    std::vector<ImageRecord> queries;
    std::vector<std::vector<double>> query_bows;
    // MIRFlickr images skipped for having no SIFT descriptors
    std::size_t distractors_skipped = 0;
};

// the vocabulary key, the set, D and every query, database and MIRFlickr image with its size
std::string bow_cache_key(const std::string& vocabulary_key, const std::string& set_type,
                          int d_mir, const EvaluationDataset& dataset);

// <cache_dir>/bovw_K<K>_S<S>_seed<seed>_<set>_D<D>.bin
std::filesystem::path bow_cache_path(const std::string& cache_dir, int vocab, int s_max,
                                     int seed, const std::string& set_type, int d_mir);

// the stored vectors, bit for bit as saved
// nullopt if the file is missing, unreadable, truncated or holds another key
std::optional<BowCache> load_bow_cache(const std::filesystem::path& path,
                                       const std::string& key);

// temporary file, then rename, as the vocabulary
// throws if the cache is empty, its vectors differ in size or the file cannot be written
void save_bow_cache(const std::filesystem::path& path, const std::string& key,
                    const BowCache& cache);

}  // namespace image_search

#endif
