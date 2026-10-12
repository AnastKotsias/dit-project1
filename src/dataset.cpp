#include "dataset.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace image_search {

namespace {

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

    std::sort(images.begin(), images.end());
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
    const std::string& set_type,
    int distractor_count) {
    if(distractor_count <= 0) {
        throw std::invalid_argument("distractor_count must be strictly positive");
    }

    EvaluationDataset dataset;
    const auto sequences = read_sequences(split_file, set_type);
    const std::filesystem::path hp_base(hp_dir);

    for(const auto& sequence : sequences) {
        dataset.queries.push_back({
            (hp_base / sequence / "1.ppm").string(),
            sequence + "/1"
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
    if(static_cast<int>(mirflickr_images.size()) < distractor_count) {
        throw std::runtime_error("MIRFlickr contains fewer images than requested");
    }
    for(int i = 0 ; i < distractor_count ; ++i) {
        dataset.database.push_back({
            mirflickr_images[i].string(),
            mirflickr_images[i].filename().string()
        });
    }

    return dataset;
}

} // namespace image_search
