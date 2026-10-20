#ifndef EXPERIMENT_HPP
#define EXPERIMENT_HPP

#include <string>

namespace image_search {

class ExperimentRunner {
public:
    ExperimentRunner(std::string hp_path, std::string mir_path,
                     std::string split_file, int vocabulary_size,
                     int max_training_descriptors, int seed);

    void run(const std::string& output_path);

private:
    std::string hp_path_;
    std::string mir_path_;
    std::string split_file_;
    int vocabulary_size_;
    int max_training_descriptors_;
    int seed_;
};

}  // namespace image_search

#endif
