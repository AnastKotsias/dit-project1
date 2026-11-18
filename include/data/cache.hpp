#ifndef CACHE_HPP
#define CACHE_HPP

#include "matrix.hpp"

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

}  // namespace image_search

#endif
