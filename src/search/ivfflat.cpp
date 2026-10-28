#include "common.hpp"

#include <chrono>
#include <memory>
#include <utility>

namespace image_search {
// shared helpers and aliases from common.hpp
using namespace detail;

// IVFFlatSearch class implementation
// IVFFlat index, which keeps full vectors in one inverted list per k-means cluster
struct IVFFlatSearch::Impl : DatabaseIndex {
    // cluster centers trained on a random sample of the database vectors
    std::vector<Vector> centers;
    // database indices belonging to each cluster
    std::vector<std::vector<std::size_t>> lists;

    // Constructor to initialize the IVFFlat index
    Impl(const std::vector<Vector>& db, const std::vector<std::string>& names, int clusterCount, int seed)
            : DatabaseIndex{db, names} {
        auto clustering = trainIVFClusters(database, clusterCount, seed);
        centers = std::move(clustering.centers);

        // Create inverted lists for each cluster center, every database vector was assigned to its nearest center
        lists.resize(centers.size());
        for(std::size_t i = 0; i < database.size(); ++i) {
            lists[clustering.assignments[i]].push_back(i);
        }
    }

    // Function to find candidate vectors for a given query
    std::vector<std::size_t> candidates(const Vector& query, int probes) const {
        // the probes closest clusters to the query
        const auto selected = nearestCenters(query, centers, probes);
        std::vector<std::size_t> result;
        // every vector of those clusters is a candidate
        for(const auto center : selected)
            result.insert(result.end(), lists[center].begin(), lists[center].end());
        return result;
    }
};

// IVFFlatSearch constructor
IVFFlatSearch::IVFFlatSearch(const std::vector<Vector>& db, const std::vector<std::string>& ids, int clusters, int probes, int seed)
    : impl(std::make_shared<Impl>(db, ids, clusters, seed)), probes(probes) {}

// copies the handle, so the trained index is shared rather than rebuilt
IVFFlatSearch IVFFlatSearch::withProbes(int probe_count) const {
    IVFFlatSearch copy = *this;
    copy.probes = probe_count;
    return copy;
}

// IVFFlatSearch search function
SearchResult IVFFlatSearch::search(const Vector& query, const std::string& queryId, const std::string& sequenceId) const {
    const auto start = std::chrono::steady_clock::now();
    return buildResult(impl->database, impl->ids, impl->candidates(query, probes), query, queryId, sequenceId, start);
}

} // namespace image_search
