#include "mtmd-cache.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct temp_dir_guard {
    fs::path path;
    ~temp_dir_guard() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

int main() {
    const uint8_t abc[] = {'a', 'b', 'c'};
    assert(mtmd_cache_sha256(nullptr, 0) ==
           "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    assert(mtmd_cache_sha256(abc, sizeof(abc)) ==
           "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    const std::string padding_boundary =
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    assert(mtmd_cache_sha256(
               reinterpret_cast<const uint8_t *>(padding_boundary.data()),
               padding_boundary.size()) ==
           "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");

    const auto unique = std::to_string(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    temp_dir_guard root{fs::temp_directory_path() / ("llama-mtmd-cache-test-" + unique)};
    const fs::path model_path = root.path / "model.gguf";
    const fs::path cache_path = root.path / "nested" / "cache";

    fs::create_directories(root.path);
    {
        std::ofstream model(model_path, std::ios::binary);
        model.write(reinterpret_cast<const char *>(abc), sizeof(abc));
    }

    mtmd_cache cache;
    assert(cache.init(cache_path.u8string(), model_path.u8string()));
    assert(cache.mmproj_hash ==
           "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    assert(fs::is_directory(cache_path));

    const std::string image_hash = mtmd_cache_sha256(
        reinterpret_cast<const uint8_t *>("image"), 5);
    std::vector<float> first = {1, 2, 3, 4, 5, 6};
    assert(cache.store_by_hash(
        image_hash, 4, 5, mtmd_projector_type::UNKNOWN,
        first.data(), 2, 3, true));

    std::vector<float> loaded;
    uint32_t n_tokens = 0;
    uint32_t n_embd = 0;
    bool use_mrope = false;
    assert(cache.lookup_by_hash(
        image_hash, 4, 5, mtmd_projector_type::UNKNOWN,
        loaded, n_tokens, n_embd, use_mrope));
    assert(loaded == first);
    assert(n_tokens == 2 && n_embd == 3 && use_mrope);

    std::vector<float> replacement = {6, 5, 4, 3, 2, 1};
    assert(cache.store_by_hash(
        image_hash, 4, 5, mtmd_projector_type::UNKNOWN,
        replacement.data(), 2, 3, false));
    assert(cache.lookup_by_hash(
        image_hash, 4, 5, mtmd_projector_type::UNKNOWN,
        loaded, n_tokens, n_embd, use_mrope));
    assert(loaded == replacement && !use_mrope);

    std::vector<fs::path> cache_files;
    for (const auto & entry : fs::directory_iterator(cache_path)) {
        if (entry.path().extension() == ".bin") {
            cache_files.push_back(entry.path());
        }
    }
    assert(cache_files.size() == 1);

    // A malicious/corrupt header must be rejected before allocating its claimed payload.
    {
        std::fstream file(cache_files.front(), std::ios::binary | std::ios::in | std::ios::out);
        mtmd_cache_header header{};
        file.read(reinterpret_cast<char *>(&header), sizeof(header));
        header.n_tokens = UINT32_MAX;
        header.n_embd = UINT32_MAX;
        file.seekp(0);
        file.write(reinterpret_cast<const char *>(&header), sizeof(header));
    }
    assert(!cache.lookup_by_hash(
        image_hash, 4, 5, mtmd_projector_type::UNKNOWN,
        loaded, n_tokens, n_embd, use_mrope));
    assert(!fs::exists(cache_files.front()));

    assert(!cache.store_by_hash(
        image_hash, 4, 5, mtmd_projector_type::UNKNOWN,
        nullptr, 2, 3, false));
    return 0;
}
