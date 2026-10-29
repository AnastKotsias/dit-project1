#include "pipeline.hpp"

#include "image_features.hpp"
#include "vocabulary.hpp"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <utility>

#ifdef _OPENMP
#include <cstdlib>
#include <omp.h>
#endif

namespace image_search {
namespace {

// images processed in parallel at a time, the results of a batch are consumed in input order
constexpr std::size_t kBatchSize = 64;

#ifdef _OPENMP
// default number of preprocessing threads when OMP_NUM_THREADS is not set
// every concurrent SIFT extraction of a large HPatches image needs about 0.65 GB, and OpenCV's SIFT
// already uses several cores per image, so more threads mostly add memory: measured on 12 cores,
// 4 threads give 89% of the 12-thread speedup with half the peak memory (4.5 GB instead of 8.6 GB)
constexpr int kDefaultPreprocessingThreads = 4;

// OMP_NUM_THREADS if the user set it, otherwise the capped default
int preprocessing_threads() {
    if(std::getenv("OMP_NUM_THREADS") != nullptr)
        return omp_get_max_threads();
    return std::min(kDefaultPreprocessingThreads, omp_get_max_threads());
}
#endif

// runs work(i) for every i in [begin, end), in parallel when built with OpenMP
// only image preprocessing runs here, searches are always single-threaded
// exceptions cannot leave an OpenMP loop, so they are stored and the one with the lowest index is rethrown
template <typename Work>
void parallel_for_images(std::size_t begin, std::size_t end, const Work& work) {
    const auto count = static_cast<std::ptrdiff_t>(end - begin);
    std::vector<std::exception_ptr> errors(end - begin);
#ifdef _OPENMP
    // dynamic scheduling, since images differ in size and number of keypoints
#pragma omp parallel for schedule(dynamic) num_threads(preprocessing_threads())
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

// BoVW vectors of one image, one per vocabulary, or the reason they could not be computed
struct ImageVectors {
    SiftExtractionStatus status = SiftExtractionStatus::Success;
    // empty if the image failed
    std::vector<std::vector<double>> histograms;
};

}  // namespace

TrainingDescriptorResult extract_training_descriptors(
    const std::vector<std::filesystem::path>& image_paths,
    int max_features,
    std::mt19937& rng) {
    TrainingDescriptorResult result;
    result.sampled_descriptors.reserve(image_paths.size());

    for(std::size_t begin = 0; begin < image_paths.size(); begin += kBatchSize) {
        const std::size_t end = std::min(begin + kBatchSize, image_paths.size());
        // SIFT extraction of the batch runs in parallel
        std::vector<SiftExtractionResult> extractions(end - begin);
        parallel_for_images(begin, end, [&](std::size_t i) {
            extractions[i - begin] = extract_sift_descriptors(image_paths[i]);
        });

        // sampling stays sequential and in image order, so the seeded rng draws the same samples as a serial run
        for(std::size_t i = begin; i < end; ++i) {
            const auto& extraction = extractions[i - begin];
            if(extraction.status == SiftExtractionStatus::Success &&
               !extraction.descriptors.empty()) {
                result.sampled_descriptors.push_back(
                    sample_training_descriptors(extraction.descriptors, max_features, rng));
                ++result.successful_images;
            } else {
                ++result.failed_images;
                result.failed_paths.push_back(image_paths[i]);
                result.failed_statuses.push_back(extraction.status);
            }
        }
    }

    return result;
}

BowVectorResult extract_bow_vectors(
    const std::vector<ImageRecord>& records,
    const DescriptorMatrix& vocabulary,
    std::size_t successful_limit) {
    auto sets = extract_bow_vector_sets(records, {vocabulary}, successful_limit);
    return {std::move(sets.vectors.front()), std::move(sets.identifiers),
            std::move(sets.failed_records)};
}

MultiBowVectorResult extract_bow_vector_sets(
    const std::vector<ImageRecord>& records,
    const std::vector<DescriptorMatrix>& vocabularies,
    std::size_t successful_limit) {
    if(vocabularies.empty()) {
        throw std::invalid_argument("at least one vocabulary is required");
    }
    MultiBowVectorResult result;
    result.vectors.resize(vocabularies.size());
    for(auto& vectors : result.vectors)
        vectors.reserve(records.size());
    result.identifiers.reserve(records.size());
    result.record_indices.reserve(records.size());

    std::size_t begin = 0;
    while(begin < records.size()) {
        // enough images succeeded
        if(successful_limit != 0 && result.identifiers.size() >= successful_limit) {
            break;
        }
        // with a limit, a batch never exceeds the number of images still needed, so no extraction is wasted
        std::size_t batch = kBatchSize;
        if(successful_limit != 0)
            batch = std::min(batch, successful_limit - result.identifiers.size());
        const std::size_t end = std::min(begin + batch, records.size());

        // SIFT extraction and histograms of the batch run in parallel
        std::vector<ImageVectors> computed(end - begin);
        parallel_for_images(begin, end, [&](std::size_t i) {
            const auto extraction = extract_sift_descriptors(
                std::filesystem::path(records[i].filepath));
            auto& image = computed[i - begin];
            image.status = extraction.status;
            if(extraction.status == SiftExtractionStatus::Success &&
               !extraction.descriptors.empty()) {
                for(const auto& vocabulary : vocabularies)
                    image.histograms.push_back(
                        compute_image_bow_histogram(extraction.descriptors, vocabulary));
            }
        });

        // results are appended in record order, exactly as a serial run would
        for(std::size_t i = begin; i < end; ++i) {
            auto& image = computed[i - begin];
            // a failed image is recorded and skipped, a replacement comes from the later records
            if(image.histograms.empty()) {
                result.failed_records.push_back({records[i], image.status});
                continue;
            }
            for(std::size_t v = 0; v < vocabularies.size(); ++v)
                result.vectors[v].push_back(std::move(image.histograms[v]));
            result.identifiers.push_back(records[i].identifier);
            result.record_indices.push_back(i);
        }
        begin = end;
    }

    return result;
}

}  // namespace image_search
