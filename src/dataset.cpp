#include "dataset.hpp"

namespace image_search {

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

void get_evaluation_paths(
    const std::string& hp_path,
    const std::vector<std::string>& eval_sequences,
    std::vector<std::filesystem::path>& queries,
    std::vector<std::filesystem::path>& database) {

    std::filesystem::path base(hp_path);

    for(const auto& seq : eval_sequences) {
        // image 1 is the query
        queries.push_back(base / seq / "1.ppm");
        // images 2-6 are the database distractors
        for(int i = 2 ; i <= 6 ; ++i) {
            database.push_back(base / seq / (std::to_string(i) + ".ppm"));
        }
    }

}

} // namespace image_search