#ifndef VERIFICATION_OUTPUT_HPP
#define VERIFICATION_OUTPUT_HPP

#include "geometric_evaluation.hpp"

#include <string>
#include <vector>

namespace image_search {

// writes the -o file of verify in the minimum output format of the assignment:
// method name, one block per query, then means, average times and homography rates
// ranks missing from a short TopC printed as "-"
// median error NA without a valid homography
// AccurateHomographyRate1px/3px appended after the required fields
// throws if queries is empty or the file cannot be opened
void write_verification_results(const std::string& output_path,
                                const std::string& method_name,
                                const std::vector<QueryEvaluation>& queries);

}  // namespace image_search

#endif
