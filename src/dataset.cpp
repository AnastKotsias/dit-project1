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
#include <tuple>

namespace image_search {

namespace {

// number in a file name of the form "im<number>", e.g. 123 for "im123.jpg", or nothing if it does not match
std::optional<int> extract_image_number(const std::filesystem::path& path) {
    // file name without directory and extension
    const std::string stem = path.stem().string();
    constexpr char prefix[] = "im";
    constexpr std::size_t prefix_length = sizeof(prefix) - 1;
    // the name must start with the prefix and have at least one character after it
    if(stem.size() <= prefix_length ||
       stem.compare(0, prefix_length, prefix) != 0) {
        return std::nullopt;
    }

    // everything after the prefix must be a plain integer
    const auto digits = stem.substr(prefix_length);
    int number = 0;
    const auto result = std::from_chars(
        digits.data(), digits.data() + digits.size(), number);
    // reject conversion errors and trailing characters such as the "a" in "im12a"
    if(result.ec != std::errc{} || result.ptr != digits.data() + digits.size()) {
        return std::nullopt;
    }
    return number;
}

// all jpeg files of the MIRFlickr directory in ascending numeric order of their image number
std::vector<std::filesystem::path> list_mirflickr_images(
    const std::string& mir_dir) {
    namespace fs = std::filesystem;
    const fs::path directory(mir_dir);
    if(!fs::is_directory(directory)) {
        throw std::runtime_error("MIRFlickr directory does not exist: " + mir_dir);
    }

    std::vector<fs::path> images;
    // directory_iterator has no defined order, so the order is fixed by the sort below
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

    // sorts by number, so im2 comes before im10, with unnumbered names last in alphabetical order
    std::sort(images.begin(), images.end(),
        [](const fs::path& left, const fs::path& right) {
            const auto left_number = extract_image_number(left);
            const auto right_number = extract_image_number(right);
            return std::make_tuple(!left_number, left_number.value_or(0), left.filename().string()) <
                   std::make_tuple(!right_number, right_number.value_or(0), right.filename().string());
        });
    return images;
}

}  // namespace

std::map<std::string, std::vector<std::string>> load_dataset_splits(
    const std::string& split_file) {
    std::map<std::string, std::vector<std::string>> splits;
    std::ifstream file(split_file);
    if(!file.is_open()) {
        throw std::runtime_error("could not open split file " + split_file);
    }

    std::set<std::string> seen_sequences;
    std::string line;

    while(std::getline(file, line)) {
        // blank lines are ignored
        if(line.empty()) {
            continue;
        }

        // a line must hold exactly two whitespace separated fields
        std::istringstream fields(line);
        std::string sequence;
        std::string subset;
        std::string extra;
        if(!(fields >> sequence >> subset) || (fields >> extra)) {
            throw std::invalid_argument("malformed split file line: " + line);
        }
        // only the three known subsets are accepted
        if(subset != "train"&& subset != "validation" && subset != "test") {
            throw std::invalid_argument("invalid subset name in split file");
        }
        // insert reports false for a sequence that was already seen
        if(!seen_sequences.insert(sequence).second) {
            throw std::invalid_argument("duplicate sequence in split file");
        }
        splits[subset].push_back(sequence);
    }

    // the HPatches split of the dataset
    if(splits["train"].size() != 70 ||
       splits["validation"].size() != 23 ||
       splits["test"].size() != 23) {
        throw std::invalid_argument(
            "split file must contain 70 train, 23 validation, and 23 test sequences");
    }

    return splits;
}

std::vector<std::filesystem::path> get_training_image_paths(
    const std::string& hp_path,
    const std::vector<std::string>& train_sequences) {

    std::vector<std::filesystem::path> paths;
    const std::filesystem::path base(hp_path);

    for(const auto& seq : train_sequences) {
        for(int i = 1 ; i <= 6 ; ++i) {
            paths.push_back(base / seq / (std::to_string(i) + ".ppm"));
        }
    }

    return paths;
}

EvaluationDataset get_evaluation_paths(
    const std::string& hp_dir,
    const std::string& mir_dir,
    const std::string& split_file,
    const std::string& set_type) {
    EvaluationDataset dataset;
    // training sequences are never evaluated
    if(set_type != "validation" && set_type != "test") {
        throw std::invalid_argument("set_type must be validation or test");
    }
    const auto splits = load_dataset_splits(split_file);
    const auto& sequences = splits.at(set_type);
    const std::filesystem::path hp_base(hp_dir);

    for(const auto& sequence : sequences) {
        // image 1 is the query and is identified by the sequence name
        dataset.queries.push_back({
            (hp_base / sequence / "1.ppm").string(),
            sequence
        });
        // images 2 to 6 are the five relevant database images, identified as "<sequence>/<number>"
        for(int image_number = 2 ; image_number <= 6 ; ++image_number) {
            dataset.database.push_back({
                (hp_base / sequence /
                 (std::to_string(image_number) + ".ppm")).string(),
                sequence + "/" + std::to_string(image_number)
            });
        }
    }

    // every image is kept, since whether it is usable is only known after SIFT extraction
    for(const auto& path : list_mirflickr_images(mir_dir)) {
        dataset.distractor_pool.push_back({
            path.string(),
            path.filename().string()
        });
    }

    return dataset;
}

} // namespace image_search
