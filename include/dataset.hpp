#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>
#include <string>
#include <filesystem>

namespace image_search {

// retrieves paths for all training images in the given sequences
std::vector<std::filesystem::path> get_training_image_paths(
    const std::string& hp_path,
    const std::vector<std::string>& train_sequences);

// retrieves paths for query images and their corresponding database images
void get_evaluation_paths(
    const std::string& hp_path,
    const std::vector<std::string>& eval_sequences,
    std::vector<std::filesystem::path>& queries,
    std::vector<std::filesystem::path>& database);
    
} // namespace image_search

#endif