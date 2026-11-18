#include "data/cache.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <system_error>

namespace image_search {
namespace {

// first line of every vocabulary file, changed when the format changes
const std::string kVocabularyMagic = "image_search vocabulary cache v1\n";

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

}  // namespace

std::string vocabulary_cache_key(const std::vector<std::filesystem::path>& training_paths,
                                 int vocab, int s_max, int seed) {
    std::string key = "K=" + std::to_string(vocab) + " S=" + std::to_string(s_max) +
                      " seed=" + std::to_string(seed) + "\n";
    // a replaced or missing image changes the key
    for(const auto& path : training_paths) {
        std::error_code error;
        const auto size = std::filesystem::file_size(path, error);
        key += path.string() + " " + (error ? "missing" : std::to_string(size)) + "\n";
    }
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
    if(!file)
        return std::nullopt;

    // magic, key, rows, cols, then rows * cols floats
    std::string magic(kVocabularyMagic.size(), '\0');
    if(!file.read(magic.data(), static_cast<std::streamsize>(magic.size())) ||
       magic != kVocabularyMagic)
        return std::nullopt;
    std::string stored_key;
    // a stored key longer than ours cannot match
    if(!read_string(file, stored_key, key.size()) || stored_key != key)
        return std::nullopt;
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    if(!read_u64(file, rows) || !read_u64(file, cols) || rows == 0 || cols == 0)
        return std::nullopt;

    // the size check guards the allocation against a corrupt header
    std::error_code error;
    const auto file_size = std::filesystem::file_size(path, error);
    const auto position = static_cast<std::uint64_t>(file.tellg());
    if(error || cols > (file_size - position) / sizeof(float) / rows)
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
    if(path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());

    // renamed only once complete
    const auto temporary = std::filesystem::path(path.string() + ".tmp");
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if(!file)
            throw std::runtime_error("could not write cache file " + temporary.string());
        file.write(kVocabularyMagic.data(),
                   static_cast<std::streamsize>(kVocabularyMagic.size()));
        write_string(file, key);
        write_u64(file, vocabulary.rows);
        write_u64(file, vocabulary.cols);
        // raw floats, so loading gives the same bits
        file.write(reinterpret_cast<const char*>(vocabulary.data.data()),
                   static_cast<std::streamsize>(vocabulary.data.size() * sizeof(float)));
        if(!file)
            throw std::runtime_error("could not write cache file " + temporary.string());
    }
    std::filesystem::rename(temporary, path);
}

}  // namespace image_search
