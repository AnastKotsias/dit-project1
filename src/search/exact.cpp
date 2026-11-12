#include "common.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <numeric>
#include <stdexcept>

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

CandidateSearchResult ExactSearch::candidates(
    const Vector& query, const std::string& queryId,
    std::size_t candidateCount) const {
    if(candidateCount == 0)
        throw std::invalid_argument("candidate count must be positive");

    const auto start = std::chrono::steady_clock::now();
    validateQuery(impl->database, query);

    std::vector<ScoredId> scored;
    scored.reserve(impl->database.size());
    for(std::size_t index = 0; index < impl->database.size(); ++index) {
        scored.emplace_back(squaredEuclideanDistance(query, impl->database[index]),
                            index);
    }

    const std::size_t limit =
        std::min(candidateCount, scored.size());
    std::partial_sort(scored.begin(), scored.begin() + limit, scored.end());

    CandidateSearchResult result;
    result.query_id = queryId;
    result.candidates.reserve(limit);
    for(std::size_t rank = 0; rank < limit; ++rank) {
        const std::size_t databaseIndex = scored[rank].second;
        result.candidates.push_back(
            {databaseIndex, rank, impl->ids[databaseIndex]});
    }
    result.candidates_examined = scored.size();
    result.query_time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}

} // namespace image_search
