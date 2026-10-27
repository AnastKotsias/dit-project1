#ifndef DATASET_HPP
#define DATASET_HPP

#include <string>
#include <filesystem>
#include <map>
#include <vector>

namespace image_search {

// describes a single image on disk together with the label used to identify it in results
struct ImageRecord {
    // location of the image file on disk
    std::string filepath;
    // query: sequence name, HPatches database image: "sequence/number", distractor: file name
    std::string identifier;
};

// holds everything needed to evaluate retrieval on one split (validation or test)
struct EvaluationDataset {
    // one query image per sequence, namely image 1 of that sequence
    std::vector<ImageRecord> queries;
    std::vector<ImageRecord> database;        // HPatches images I_2 ... I_6
    std::vector<ImageRecord> distractor_pool; // Ordered MIRFlickr image pool
};

// reads and validates the split file, returning a map from subset name to its sequence names
std::map<std::string, std::vector<std::string>> load_dataset_splits(
    const std::string& split_file);

// retrieves paths for all training images in the given sequences
std::vector<std::filesystem::path> get_training_image_paths(
    const std::string& hp_path,
    const std::vector<std::string>& train_sequences);

// builds query, database and distractor records for the "validation" or "test" split
EvaluationDataset get_evaluation_paths(
    const std::string& hp_dir,
    const std::string& mir_dir,
    const std::string& split_file,
    const std::string& set_type);

} // namespace image_search

#endif
