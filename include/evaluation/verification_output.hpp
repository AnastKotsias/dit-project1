#ifndef VERIFICATION_OUTPUT_HPP
#define VERIFICATION_OUTPUT_HPP

#include "evaluation/geometric_evaluation.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace image_search {

// writes the -o file of verify in the minimum output format of the assignment:
// method name, one block per query, then means, average times and homography rates
// ranks missing from a short TopC printed as "-"
// median error NA without a valid homography
// per query: metrics, then candidate, verification and total time
// appended after the required fields: AccurateHomographyRate1px/3px and the
// match-count ranking metrics, per query and as means
// each query block ends with its 5 relevant pairs, e.g. Pair-2-Inliers
// candidate_count: the requested C, each summary ends with QueriesBelowC
// then the summary again for the i_ and v_ sequences, e.g. Viewpoint-MeanAfterMAP@10
// throws if queries is empty, C is zero or the file cannot be opened
void write_verification_results(const std::string& output_path,
                                const std::string& method_name,
                                std::size_t candidate_count,
                                const std::vector<QueryEvaluation>& queries);

}  // namespace image_search

#endif
