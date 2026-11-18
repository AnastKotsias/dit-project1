#include "data/cache.hpp"

#include <cstdint>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <system_error>

namespace image_search {
namespace {

// first line of every file, changed when the format changes
const std::string kVocabularyMagic = "image_search vocabulary cache v1\n";
const std::string kBowMagic = "image_search bovw cache v1\n";
// longest identifier or path read back
constexpr std::uint64_t kMaximumTextSize = 4096;

void write_u64(std::ofstream& file, std::uint64_t value) {
    file.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

// false on a short read
bool read_u64(std::ifstream& file, std::uint64_t& value) {
    return static_cast<bool>(file.read(reinterpret_cast<char*>(&value), sizeof(value)));
}

// length, then the bytes
void write_string(std::ofstream& file, const std::string& text) {
    write_u64(file, text.size());
    file.write(text.data(), static_cast<std::streamsize>(text.size()));
}

// false on a short read or a length above limit
bool read_string(std::ifstream& file, std::string& text, std::uint64_t limit) {
    std::uint64_t size = 0;
    if(!read_u64(file, size) || size > limit)
        return false;
    text.assign(size, '\0');
    return static_cast<bool>(file.read(text.data(), static_cast<std::streamsize>(size)));
}

// raw doubles, so loading gives the same bits
void write_doubles(std::ofstream& file, const std::vector<double>& values) {
    file.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(double)));
}

bool read_doubles(std::ifstream& file, std::vector<double>& values, std::uint64_t count) {
    values.resize(count);
    return static_cast<bool>(file.read(reinterpret_cast<char*>(values.data()),
                                       static_cast<std::streamsize>(count * sizeof(double))));
}

// bytes left after the current position, 0 on an error
std::uint64_t remaining_bytes(std::ifstream& file, const std::filesystem::path& path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    const auto position = file.tellg();
    if(error || position < 0 || static_cast<std::uint64_t>(position) > size)
        return 0;
    return size - static_cast<std::uint64_t>(position);
}

// "<path> <size>\n", or "missing", so a replaced or missing image changes a key
void append_file(std::string& key, const std::filesystem::path& path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    key += path.string() + " " + (error ? "missing" : std::to_string(size)) + "\n";
}

// writes path.tmp, then renames it, so a crash never leaves a partial file
void write_atomically(const std::filesystem::path& path,
                      const std::function<void(std::ofstream&)>& write) {
    if(path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
    const auto temporary = std::filesystem::path(path.string() + ".tmp");
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if(!file)
            throw std::runtime_error("could not write cache file " + temporary.string());
        write(file);
        if(!file)
            throw std::runtime_error("could not write cache file " + temporary.string());
    }
    std::filesystem::rename(temporary, path);
}

// magic and key of a file, false if either differs
bool read_header(std::ifstream& file, const std::string& magic, const std::string& key) {
    std::string stored_magic(magic.size(), '\0');
    if(!file.read(stored_magic.data(), static_cast<std::streamsize>(stored_magic.size())) ||
       stored_magic != magic)
        return false;
    std::string stored_key;
    // a stored key longer than ours cannot match
    return read_string(file, stored_key, key.size()) && stored_key == key;
}

}  // namespace

std::string vocabulary_cache_key(const std::vector<std::filesystem::path>& training_paths,
                                 int vocab, int s_max, int seed) {
    std::string key = "K=" + std::to_string(vocab) + " S=" + std::to_string(s_max) +
                      " seed=" + std::to_string(seed) + "\n";
    for(const auto& path : training_paths)
        append_file(key, path);
    return key;
}

std::filesystem::path vocabulary_cache_path(const std::string& cache_dir,
                                            int vocab, int s_max, int seed) {
    return std::filesystem::path(cache_dir) /
           ("vocabulary_K" + std::to_string(vocab) + "_S" + std::to_string(s_max) +
            "_seed" + std::to_string(seed) + ".bin");
}

std::optional<DescriptorMatrix> load_vocabulary_cache(const std::filesystem::path& path,
                                                      const std::string& key) {
    std::ifstream file(path, std::ios::binary);
    if(!file || !read_header(file, kVocabularyMagic, key))
        return std::nullopt;

    // rows, cols, then rows * cols floats
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    if(!read_u64(file, rows) || !read_u64(file, cols) || rows == 0 || cols == 0)
        return std::nullopt;
    // the size check guards the allocation against a corrupt header
    if(cols > remaining_bytes(file, path) / sizeof(float) / rows)
        return std::nullopt;

    DescriptorMatrix vocabulary(rows, cols);
    if(!file.read(reinterpret_cast<char*>(vocabulary.data.data()),
                  static_cast<std::streamsize>(vocabulary.data.size() * sizeof(float))))
        return std::nullopt;
    // nothing may follow the last float
    if(file.peek() != std::ifstream::traits_type::eof())
        return std::nullopt;
    return vocabulary;
}

void save_vocabulary_cache(const std::filesystem::path& path, const std::string& key,
                           const DescriptorMatrix& vocabulary) {
    if(vocabulary.empty())
        throw std::invalid_argument("cannot cache an empty vocabulary");
    write_atomically(path, [&](std::ofstream& file) {
        file.write(kVocabularyMagic.data(),
                   static_cast<std::streamsize>(kVocabularyMagic.size()));
        write_string(file, key);
        write_u64(file, vocabulary.rows);
        write_u64(file, vocabulary.cols);
        // raw floats, so loading gives the same bits
        file.write(reinterpret_cast<const char*>(vocabulary.data.data()),
                   static_cast<std::streamsize>(vocabulary.data.size() * sizeof(float)));
    });
}

std::string bow_cache_key(const std::string& vocabulary_key, const std::string& set_type,
                          int d_mir, const EvaluationDataset& dataset) {
    // BoVW vectors depend on the vocabulary, so on its key
    std::string key = vocabulary_key + "set=" + set_type + " D=" + std::to_string(d_mir) + "\n";
    for(const auto& record : dataset.queries)
        append_file(key, record.filepath);
    for(const auto& record : dataset.database)
        append_file(key, record.filepath);
    // whole pool: the skipped images decide which D are used
    for(const auto& record : dataset.distractor_pool)
        append_file(key, record.filepath);
    return key;
}

std::filesystem::path bow_cache_path(const std::string& cache_dir, int vocab, int s_max,
                                     int seed, const std::string& set_type, int d_mir) {
    return std::filesystem::path(cache_dir) /
           ("bovw_K" + std::to_string(vocab) + "_S" + std::to_string(s_max) + "_seed" +
            std::to_string(seed) + "_" + set_type + "_D" + std::to_string(d_mir) + ".bin");
}

std::optional<BowCache> load_bow_cache(const std::filesystem::path& path,
                                       const std::string& key) {
    std::ifstream file(path, std::ios::binary);
    if(!file || !read_header(file, kBowMagic, key))
        return std::nullopt;

    // dimension K of every vector
    std::uint64_t dimension = 0;
    if(!read_u64(file, dimension) || dimension == 0 ||
       dimension > remaining_bytes(file, path) / sizeof(double))
        return std::nullopt;

    // count, then per image: identifier, path, K doubles
    // the size check guards the allocation against a corrupt count
    const auto read_count = [&](std::uint64_t& count) {
        return read_u64(file, count) &&
               count <= remaining_bytes(file, path) / (dimension * sizeof(double));
    };

    BowCache cache;
    std::uint64_t database_count = 0;
    if(!read_count(database_count))
        return std::nullopt;
    cache.database_bows.resize(database_count);
    cache.database_ids.resize(database_count);
    cache.database_paths.resize(database_count);
    for(std::uint64_t i = 0; i < database_count; ++i) {
        std::string path_text;
        if(!read_string(file, cache.database_ids[i], kMaximumTextSize) ||
           !read_string(file, path_text, kMaximumTextSize) ||
           !read_doubles(file, cache.database_bows[i], dimension))
            return std::nullopt;
        cache.database_paths[i] = path_text;
    }

    std::uint64_t skipped = 0;
    if(!read_u64(file, skipped))
        return std::nullopt;
    cache.distractors_skipped = skipped;

    std::uint64_t query_count = 0;
    if(!read_count(query_count))
        return std::nullopt;
    cache.queries.resize(query_count);
    cache.query_bows.resize(query_count);
    for(std::uint64_t i = 0; i < query_count; ++i) {
        if(!read_string(file, cache.queries[i].filepath, kMaximumTextSize) ||
           !read_string(file, cache.queries[i].identifier, kMaximumTextSize) ||
           !read_doubles(file, cache.query_bows[i], dimension))
            return std::nullopt;
    }

    // nothing may follow the last query
    if(file.peek() != std::ifstream::traits_type::eof())
        return std::nullopt;
    return cache;
}

void save_bow_cache(const std::filesystem::path& path, const std::string& key,
                    const BowCache& cache) {
    if(cache.database_bows.empty() || cache.query_bows.empty())
        throw std::invalid_argument("cannot cache an empty database or query set");
    if(cache.database_ids.size() != cache.database_bows.size() ||
       cache.database_paths.size() != cache.database_bows.size() ||
       cache.queries.size() != cache.query_bows.size())
        throw std::invalid_argument("BoVW cache vectors, identifiers and paths are misaligned");
    // one dimension K for every vector
    const std::size_t dimension = cache.database_bows.front().size();
    const auto same_size = [&](const std::vector<std::vector<double>>& vectors) {
        for(const auto& vector : vectors)
            if(vector.size() != dimension)
                return false;
        return true;
    };
    if(dimension == 0 || !same_size(cache.database_bows) || !same_size(cache.query_bows))
        throw std::invalid_argument("BoVW cache vectors differ in size");

    write_atomically(path, [&](std::ofstream& file) {
        file.write(kBowMagic.data(), static_cast<std::streamsize>(kBowMagic.size()));
        write_string(file, key);
        write_u64(file, dimension);
        write_u64(file, cache.database_bows.size());
        for(std::size_t i = 0; i < cache.database_bows.size(); ++i) {
            write_string(file, cache.database_ids[i]);
            write_string(file, cache.database_paths[i].string());
            write_doubles(file, cache.database_bows[i]);
        }
        write_u64(file, cache.distractors_skipped);
        write_u64(file, cache.queries.size());
        for(std::size_t i = 0; i < cache.queries.size(); ++i) {
            write_string(file, cache.queries[i].filepath);
            write_string(file, cache.queries[i].identifier);
            write_doubles(file, cache.query_bows[i]);
        }
    });
}

}  // namespace image_search
