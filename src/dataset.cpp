#include "dataset.hpp"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>

namespace image_search {

// helpers below are private to this translation unit
namespace {

// extracts the number from a file name of the form "im<number>", or nothing if it does not match
std::optional<int> extract_image_number(const std::filesystem::path& path) {
    // file name without directory and extension, e.g. "im123"
    const std::string stem = path.stem().string();
    constexpr char prefix[] = "im";
    constexpr std::size_t prefix_length = sizeof(prefix) - 1;
    // reject names that lack the prefix or have nothing after it
    if(stem.size() <= prefix_length ||
       stem.compare(0, prefix_length, prefix) != 0) {
        return std::nullopt;
    }

    // everything after the prefix must be a plain integer
    const auto digits = stem.substr(prefix_length);
    int number = 0;
    const auto result = std::from_chars(
        digits.data(), digits.data() + digits.size(), number);
    // reject conversion errors and trailing non-digit characters
    if(result.ec != std::errc{} || result.ptr != digits.data() + digits.size()) {
        return std::nullopt;
    }
    return number;
}

// lists all jpeg files of the MIRFlickr directory in a deterministic, numerically sorted order
std::vector<std::filesystem::path> list_mirflickr_images(
    const std::string& mir_dir) {
    namespace fs = std::filesystem;
    const fs::path directory(mir_dir);
    // fail early with a clear message when the directory is missing
    if(!fs::is_directory(directory)) {
        throw std::runtime_error("MIRFlickr directory does not exist: " + mir_dir);
    }

    std::vector<fs::path> images;
    // collect only regular files with a jpeg extension
    for(const auto& entry : fs::directory_iterator(directory)) {
        if(!entry.is_regular_file()) {
            continue;
        }
        const auto extension = entry.path().extension().string();
        if(extension == ".jpg" || extension == ".jpeg" ||
           extension == ".JPG" || extension == ".JPEG") {
            images.push_back(entry.path());
        }
    }

    // sort by image number when both names carry one, otherwise fall back to alphabetical order
    std::sort(images.begin(), images.end(),
        [](const fs::path& left, const fs::path& right) {
            const auto left_number = extract_image_number(left);
            const auto right_number = extract_image_number(right);
            // compare numerically so that im2 comes before im10
            if(left_number && right_number && *left_number != *right_number) {
                return *left_number < *right_number;
            }
            // tie-breaker that keeps the order deterministic
            return left.filename().string() < right.filename().string();
        });
    return images;
}

}  // namespace

// reads the split file where each line holds a sequence name and its subset, and validates it
std::map<std::string, std::vector<std::string>> load_dataset_splits(
    const std::string& split_file) {
    // subset name -> sequence names, in file order
    std::map<std::string, std::vector<std::string>> splits;
    std::ifstream file(split_file);
    if(!file.is_open()) {
        throw std::runtime_error("could not open split file " + split_file);
    }

    // used to detect a sequence that appears more than once
    std::set<std::string> seen_sequences;
    std::string line;

    while(std::getline(file, line)) {
        // blank lines are allowed and ignored
        if(line.empty()) {
            continue;
        }

        // split the line into whitespace separated fields
        std::istringstream fields(line);
        std::string sequence;
        std::string subset;
        std::string extra;
        // each line must contain exactly two fields
        if(!(fields >> sequence >> subset) || (fields >> extra)) {
            throw std::invalid_argument("malformed split file line: " + line);
        }
        // only the three known subset names are accepted
        if(subset != "train" && subset != "validation" && subset != "test") {
            throw std::invalid_argument("invalid subset name in split file");
        }
        // insert fails when the sequence was already seen, which means a duplicate
        if(!seen_sequences.insert(sequence).second) {
            throw std::invalid_argument("duplicate sequence in split file");
        }
        // record the sequence under its subset
        splits[subset].push_back(sequence);
    }

    // the HPatches split must contain exactly 70 train, 23 validation and 23 test sequences
    if(splits["train"].size() != 70 ||
       splits["validation"].size() != 23 ||
       splits["test"].size() != 23) {
        throw std::invalid_argument(
            "split file must contain 70 train, 23 validation, and 23 test sequences");
    }

    return splits;
}

// returns the paths of all six images of every given training sequence
std::vector<std::filesystem::path> get_training_image_paths(
    const std::string& hp_path,
    const std::vector<std::string>& train_sequences) {

    // result list, in sequence order and then image order
    std::vector<std::filesystem::path> paths;
    // root directory of the HPatches dataset
    std::filesystem::path base(hp_path);

    // hpatches sequences contain images named 1.ppm to 6.ppm
    for(const auto& seq : train_sequences) {
        for(int i = 1 ; i <= 6 ; ++i) {
            paths.push_back(base / seq / (std::to_string(i) + ".ppm"));
        }
    }

    return paths;
}

// builds the queries, database and distractor pool for either the validation or the test split
EvaluationDataset get_evaluation_paths(
    const std::string& hp_dir,
    const std::string& mir_dir,
    const std::string& split_file,
    const std::string& set_type) {
    EvaluationDataset dataset;
    // training data is not used for evaluation, so only these two splits are valid
    if(set_type != "validation" && set_type != "test") {
        throw std::invalid_argument("set_type must be validation or test");
    }
    // load and validate the split file
    const auto splits = load_dataset_splits(split_file);
    // sequences belonging to the requested split
    const auto& sequences = splits.at(set_type);
    // root directory of the HPatches dataset
    const std::filesystem::path hp_base(hp_dir);

    for(const auto& sequence : sequences) {
        // image 1 of each sequence is the query and is labelled with the sequence name
        dataset.queries.push_back({
            (hp_base / sequence / "1.ppm").string(),
            sequence
        });
        // images 2 to 6 are the relevant database images for that query
        for(int image_number = 2 ; image_number <= 6 ; ++image_number) {
            dataset.database.push_back({
                (hp_base / sequence /
                 (std::to_string(image_number) + ".ppm")).string(),
                sequence + "/" + std::to_string(image_number)
            });
        }
    }

    // MIRFlickr images serve as distractors and keep their numeric order
    const auto mirflickr_images = list_mirflickr_images(mir_dir);
    for(const auto& path : mirflickr_images) {
        dataset.distractor_pool.push_back({
            path.string(),
            path.filename().string()
        });
    }

    return dataset;
}

} // namespace image_search
