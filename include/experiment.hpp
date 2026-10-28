#ifndef EXPERIMENT_HPP
#define EXPERIMENT_HPP

#include <string>

namespace image_search {

// runs the full experimental study: vocabulary training, parameter tuning on validation and evaluation on test
// the SIFT descriptors of every image are extracted once and encoded with every vocabulary
class ExperimentRunner {
public:
    // stores the dataset locations and the vocabulary and sampling settings, without doing any work yet
    ExperimentRunner(std::string hp_path, std::string mir_path,
                     std::string split_file, int vocabulary_size,
                     int max_training_descriptors, int seed);

    // executes the whole study and writes the results as markdown tables to output_path
    void run(const std::string& output_path);

private:
    // directory of the HPatches dataset
    std::string hp_path_;
    // directory of the MIRFlickr-25K dataset
    std::string mir_path_;
    // file assigning each sequence to train, validation or test
    std::string split_file_;
    // number K of k-means centroids, or 0 to compare the vocabulary sizes 128, 256, 512 and 1024
    int vocabulary_size_;
    // limit S of SIFT descriptors sampled per training image
    int max_training_descriptors_;
    // seed shared by every random procedure so runs are reproducible
    int seed_;
};

}  // namespace image_search

#endif
