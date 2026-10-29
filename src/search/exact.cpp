#include "common.hpp"

#include <chrono>
#include <memory>
#include <numeric>

namespace image_search {
using namespace detail;

// the exact search needs no structure beyond the validated database
struct ExactSearch::Impl : DatabaseIndex {
    Impl(const std::vector<Vector>& db, const std::vector<std::string>& names)
        : DatabaseIndex{db, names} {}
};

ExactSearch::ExactSearch(const std::vector<Vector>& db,
                         const std::vector<std::string>& ids)
    : impl(std::make_shared<Impl>(db, ids)) {}

SearchResult ExactSearch::search(const Vector& query, const std::string& queryId, const std::string& sequenceId) const {
    const auto start = std::chrono::steady_clock::now();
    validateQuery(impl->database, query);
    // every database vector is a candidate
    std::vector<std::size_t> all(impl->database.size());
    std::iota(all.begin(), all.end(), 0);
    return buildResult(
        impl->database, impl->ids, all, query, queryId, sequenceId, start);
}

} // namespace image_search
