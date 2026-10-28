#include "pipeline.hpp"

#include "image_features.hpp"
#include "vocabulary.hpp"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <utility>

namespace image_search {
// helpers below are private to this translation unit
namespace {

// images processed in parallel at a time, their results are consumed in input order after each batch
constexpr std::size_t kBatchSize = 64;

// runs work(i) for every i in [begin, end), in parallel when the program is built with OpenMP
// only image preprocessing runs here, the search itself is always single-threaded
// exceptions cannot leave an OpenMP loop, so they are stored and the one with the lowest index is rethrown
template <typename Work>
void parallel_for_images(std::size_t begin, std::size_t end, const Work& work) {
    const auto count = static_cast<std::ptrdiff_t>(end - begin);
    std::vector<std::exception_ptr> errors(end - begin);
#ifdef _OPENMP
    // dynamic scheduling, since images differ in size and number of keypoints
#pragma omp parallel for schedule(dynamic)
#endif
    for(std::ptrdiff_t offset = 0; offset < count; ++offset) {
        try {
            work(begin + static_cast<std::size_t>(offset));
        } catch(...) {
            errors[static_cast<std::size_t>(offset)] = std::current_exception();
        }
    }
    for(const auto& error : errors) {
        if(error)
            std::rethrow_exception(error);
    }
}

// BoW vector of one image, or the reason it could not be computed
struct ImageVector {
    SiftExtractionStatus status = SiftExtractionStatus::Success;
    // empty if the image failed
    std::vector<double> histogram;
};

}  // namespace

// extracts and samples descriptors for every training image, recording failures instead of aborting
TrainingDescriptorResult extract_training_descriptors(
    const std::vector<std::filesystem::path>& image_paths,
    int max_features,
    std::mt19937& rng) {
    TrainingDescriptorResult result;
    // at most one matrix per image, so the vector never needs to reallocate
    result.sampled_descriptors.reserve(image_paths.size());

    for(std::size_t begin = 0; begin < image_paths.size(); begin += kBatchSize) {
        const std::size_t end = std::min(begin + kBatchSize, image_paths.size());
        // extract all SIFT descriptors of the images of this batch in parallel
        std::vector<SiftExtractionResult> extractions(end - begin);
        parallel_for_images(begin, end, [&](std::size_t i) {
            extractions[i - begin] = extract_sift_descriptors(image_paths[i]);
        });

        // sampling stays sequential and in image order, so the seeded rng draws exactly
        // the same samples as a serial run
        for(std::size_t i = begin; i < end; ++i) {
            const auto& extraction = extractions[i - begin];
            // only images with at least one descriptor can contribute to k-means
            if(extraction.status == SiftExtractionStatus::Success &&
               !extraction.descriptors.empty()) {
                // keep a random subset of at most max_features descriptors
                result.sampled_descriptors.push_back(
                    sample_training_descriptors(extraction.descriptors, max_features, rng));
                ++result.successful_images;
            } else {
                // remember which image failed and why, so the caller can report it
                ++result.failed_images;
                result.failed_paths.push_back(image_paths[i]);
                result.failed_statuses.push_back(extraction.status);
            }
        }
    }

    return result;
}

// converts every record to its bag of visual words vector, skipping images that fail
BowVectorResult extract_bow_vectors(
    const std::vector<ImageRecord>& records,
    const DescriptorMatrix& vocabulary,
    std::size_t successful_limit) {
    BowVectorResult result;
    // upper bound on the number of results, avoids reallocation
    result.vectors.reserve(records.size());
    result.identifiers.reserve(records.size());

    std::size_t begin = 0;
    while(begin < records.size()) {
        // stop once enough images succeeded, which allows taking the first D distractors
        if(successful_limit != 0 && result.vectors.size() >= successful_limit) {
            break;
        }
        // with a limit, a batch never exceeds the number of images still needed, so no work is wasted
        std::size_t batch = kBatchSize;
        if(successful_limit != 0)
            batch = std::min(batch, successful_limit - result.vectors.size());
        const std::size_t end = std::min(begin + batch, records.size());

        // extract the SIFT descriptors and build the histogram of every image of the batch in parallel
        std::vector<ImageVector> computed(end - begin);
        parallel_for_images(begin, end, [&](std::size_t i) {
            const auto extraction = extract_sift_descriptors(
                std::filesystem::path(records[i].filepath));
            auto& image = computed[i - begin];
            image.status = extraction.status;
            if(extraction.status == SiftExtractionStatus::Success &&
               !extraction.descriptors.empty()) {
                // map the descriptors onto the vocabulary to get the image's histogram
                image.histogram = compute_image_bow_histogram(extraction.descriptors, vocabulary);
            }
        });

        // results are appended in record order, exactly as a serial run would
        for(std::size_t i = begin; i < end; ++i) {
            auto& image = computed[i - begin];
            // failed images are recorded and skipped, so a replacement is taken from the next record
            if(image.histogram.empty()) {
                result.failed_records.push_back({records[i], image.status});
                continue;
            }
            result.vectors.push_back(std::move(image.histogram));
            // keep the identifier aligned with the vector just added
            result.identifiers.push_back(records[i].identifier);
        }
        begin = end;
    }

    return result;
}

}  // namespace image_search
