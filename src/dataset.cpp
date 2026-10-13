#include "dataset.hpp"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>

namespace image_search {

namespace {

std::optional<int> extract_image_number(const std::filesystem::path& path) {
    const std::string stem = path.stem().string();
    constexpr char prefix[] = "im";
    constexpr std::size_t prefix_length = sizeof(prefix) - 1;
    if(stem.size() <= prefix_length ||
       stem.compare(0, prefix_length, prefix) != 0) {
        return std::nullopt;
    }

    const auto digits = stem.substr(prefix_length);
    int number = 0;
    const auto result = std::from_chars(
        digits.data(), digits.data() + digits.size(), number);
    if(result.ec != std::errc{} || result.ptr != digits.data() + digits.size()) {
        return std::nullopt;
    }
    return number;
}

std::vector<std::string> read_sequences(
    const std::string& split_file,
    const std::string& requested_set) {
    if(requested_set != "validation" && requested_set != "test") {
        throw std::invalid_argument("set_type must be validation or test");
    }

    std::ifstream file(split_file);
    if(!file.is_open()) {
        throw std::runtime_error("could not open split file " + split_file);
    }

    std::vector<std::string> sequences;
    std::set<std::string> seen_sequences;
    std::string sequence;
    std::string subset;
    std::string extra;

    while(std::getline(file, sequence)) {
        if(sequence.empty()) {
            continue;
        }

        std::istringstream line(sequence);
        if(!(line >> sequence >> subset) || (line >> extra)) {
            throw std::invalid_argument("malformed split file line");
        }
        if(subset != "train" && subset != "validation" && subset != "test") {
            throw std::invalid_argument("invalid subset name in split file");
        }
        if(!seen_sequences.insert(sequence).second) {
            throw std::invalid_argument("duplicate sequence in split file");
        }
        if(subset == requested_set) {
            sequences.push_back(sequence);
        }
    }

    return sequences;
}

std::vector<std::filesystem::path> list_mirflickr_images(
    const std::string& mir_dir) {
    namespace fs = std::filesystem;
    const fs::path directory(mir_dir);
    if(!fs::is_directory(directory)) {
        throw std::runtime_error("MIRFlickr directory does not exist: " + mir_dir);
    }

    std::vector<fs::path> images;
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

    std::sort(images.begin(), images.end(),
        [](const fs::path& left, const fs::path& right) {
            const auto left_number = extract_image_number(left);
            const auto right_number = extract_image_number(right);
            if(left_number && right_number && *left_number != *right_number) {
                return *left_number < *right_number;
            }
            return left.filename().string() < right.filename().string();
        });
    return images;
}

}  // namespace

std::vector<std::filesystem::path> get_training_image_paths(
    const std::string& hp_path,
    const std::vector<std::string>& train_sequences) {

    std::vector<std::filesystem::path> paths;
    std::filesystem::path base(hp_path);

    // hpatches sequences contain images named 1.ppm to 6.ppm
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
    const auto sequences = read_sequences(split_file, set_type);
    const std::filesystem::path hp_base(hp_dir);

    for(const auto& sequence : sequences) {
        dataset.queries.push_back({
            (hp_base / sequence / "1.ppm").string(),
            sequence
        });
        for(int image_number = 2 ; image_number <= 6 ; ++image_number) {
            dataset.database.push_back({
                (hp_base / sequence /
                 (std::to_string(image_number) + ".ppm")).string(),
                sequence + "/" + std::to_string(image_number)
            });
        }
    }

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
