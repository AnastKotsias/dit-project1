#ifndef DATASET_HPP
#define DATASET_HPP

#include <string>
#include <filesystem>
#include <vector>

namespace image_search {

struct ImageRecord {
    std::string filepath;
    std::string identifier;
};

struct EvaluationDataset {
    std::vector<ImageRecord> queries;
    std::vector<ImageRecord> database;        // HPatches images I_2 ... I_6
    std::vector<ImageRecord> distractor_pool; // Ordered MIRFlickr image pool
};

// retrieves paths for all training images in the given sequences
std::vector<std::filesystem::path> get_training_image_paths(
    const std::string& hp_path,
    const std::vector<std::string>& train_sequences);

// Builds query and database records for one evaluation split.
EvaluationDataset get_evaluation_paths(
    const std::string& hp_dir,
    const std::string& mir_dir,
    const std::string& split_file,
    const std::string& set_type);
    
} // namespace image_search

#endif