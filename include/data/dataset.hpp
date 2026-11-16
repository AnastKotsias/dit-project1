#ifndef DATASET_HPP
#define DATASET_HPP

#include <string>
#include <filesystem>
#include <map>
#include <vector>

namespace image_search {

// an image on disk together with the name under which it appears in the results
struct ImageRecord {
    std::string filepath;
    // query: sequence name (v_bird), HPatches database image: "sequence/number" (v_bird/2),
    // MIRFlickr distractor: file name (im123.jpg)
    std::string identifier;
};

// the images needed to evaluate one split (validation or test)
struct EvaluationDataset {
    // image 1 of every sequence of the split
    std::vector<ImageRecord> queries;
    // images 2 to 6 of every sequence of the split, the five relevant images of each query
    std::vector<ImageRecord> database;
    // every MIRFlickr image in ascending numeric order, not yet filtered for usable SIFT descriptors
    std::vector<ImageRecord> distractor_pool;
};

// reads the split file, whose lines are "<sequence> <train|validation|test>", and returns the
// sequence names of each subset in file order
// throws if the file cannot be opened, a line is malformed, a sequence is repeated,
// or the subsets do not hold 70, 23 and 23 sequences
std::map<std::string, std::vector<std::string>> load_dataset_splits(
    const std::string& split_file);

// paths of images 1.ppm to 6.ppm of every training sequence, without opening the files
std::vector<std::filesystem::path> get_training_image_paths(
    const std::string& hp_path,
    const std::vector<std::string>& train_sequences);

// builds the queries, database and distractor pool of the "validation" or "test" split
// throws for any other set type, an invalid split file or a missing MIRFlickr directory
EvaluationDataset get_evaluation_paths(
    const std::string& hp_dir,
    const std::string& mir_dir,
    const std::string& split_file,
    const std::string& set_type);

} // namespace image_search

#endif
