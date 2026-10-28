#include "common.hpp"

#include <chrono>
#include <memory>
#include <numeric>

namespace image_search {
// shared helpers and aliases from common.hpp
using namespace detail;

// ExactSearch class implementation
// the exact search needs no structure beyond the validated database
struct ExactSearch::Impl : DatabaseIndex {
    Impl(const std::vector<Vector>& db, const std::vector<std::string>& names)
        : DatabaseIndex{db, names} {}
};

// builds the index by validating and storing the database
ExactSearch::ExactSearch(const std::vector<Vector>& db,
                         const std::vector<std::string>& ids)
    : impl(std::make_shared<Impl>(db, ids)) {}

// scores every database vector, so the candidate list is simply all indices
SearchResult ExactSearch::search(const Vector& query, const std::string& queryId, const std::string& sequenceId) const {
    // timing starts before candidate generation, measures the whole search
    const auto start = std::chrono::steady_clock::now();
    validateQuery(impl->database, query);
    // candidate list containing indices 0 .. n-1
    std::vector<std::size_t> all(impl->database.size());
    std::iota(all.begin(), all.end(), 0);
    return buildResult(
        impl->database, impl->ids, all, query, queryId, sequenceId, start);
}

} // namespace image_search
