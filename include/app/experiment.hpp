#ifndef EXPERIMENT_HPP
#define EXPERIMENT_HPP

#include <string>

namespace image_search {

// runs the experimental study: trains the vocabularies, tunes the approximate methods on the
// validation set and evaluates every method on the test set
// the SIFT descriptors of every image are extracted once and encoded with every vocabulary
class ExperimentRunner {
public:
    // vocabulary_size 0 compares the sizes 128, 256, 512 and 1024
    // nothing is computed until run
    ExperimentRunner(std::string hp_path, std::string mir_path,
                     std::string split_file, int vocabulary_size,
                     int max_training_descriptors, int seed);

    // executes the study and writes the results as Markdown tables to output_path
    // throws if the vocabulary size is not divisible by 16 (the IVFPQ subvector count), the output
    // cannot be opened, or a training image set, HPatches image or query cannot be processed
    void run(const std::string& output_path);

private:
    std::string hp_path_;
    std::string mir_path_;
    std::string split_file_;
    // number K of visual words, or 0 to compare several sizes
    int vocabulary_size_;
    // limit S of SIFT descriptors sampled per training image
    int max_training_descriptors_;
    int seed_;
};

}  // namespace image_search

#endif
