#include "mtmd-cache.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <system_error>

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <process.h>
#else
#   include <fcntl.h>
#   include <unistd.h>
#endif

#define MTMD_CACHE_LOG_INF(...) fprintf(stderr, __VA_ARGS__)
#define MTMD_CACHE_LOG_WRN(...) fprintf(stderr, __VA_ARGS__)

namespace fs = std::filesystem;

namespace {

constexpr size_t SHA256_SIZE = 32;
constexpr size_t CACHE_MAX_EMBEDDING_BYTES = size_t(2) * 1024 * 1024 * 1024;

struct sha256_state {
    uint32_t h[8];
    uint64_t total_bytes;
    uint8_t block[64];
    size_t block_size;
};

constexpr uint32_t SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

static uint32_t rotr32(uint32_t value, unsigned bits) {
    return (value >> bits) | (value << (32 - bits));
}

static void sha256_init(sha256_state & state) {
    state.h[0] = 0x6a09e667;
    state.h[1] = 0xbb67ae85;
    state.h[2] = 0x3c6ef372;
    state.h[3] = 0xa54ff53a;
    state.h[4] = 0x510e527f;
    state.h[5] = 0x9b05688c;
    state.h[6] = 0x1f83d9ab;
    state.h[7] = 0x5be0cd19;
    state.total_bytes = 0;
    state.block_size = 0;
}

static void sha256_transform(sha256_state & state, const uint8_t block[64]) {
    uint32_t w[64];
    for (size_t i = 0; i < 16; ++i) {
        w[i] = (uint32_t(block[i * 4 + 0]) << 24) |
               (uint32_t(block[i * 4 + 1]) << 16) |
               (uint32_t(block[i * 4 + 2]) << 8) |
               uint32_t(block[i * 4 + 3]);
    }
    for (size_t i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = state.h[0];
    uint32_t b = state.h[1];
    uint32_t c = state.h[2];
    uint32_t d = state.h[3];
    uint32_t e = state.h[4];
    uint32_t f = state.h[5];
    uint32_t g = state.h[6];
    uint32_t h = state.h[7];

    for (size_t i = 0; i < 64; ++i) {
        const uint32_t s1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        const uint32_t ch = (e & f) ^ (~e & g);
        const uint32_t temp1 = h + s1 + ch + SHA256_K[i] + w[i];
        const uint32_t s0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temp2 = s0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    state.h[0] += a;
    state.h[1] += b;
    state.h[2] += c;
    state.h[3] += d;
    state.h[4] += e;
    state.h[5] += f;
    state.h[6] += g;
    state.h[7] += h;
}

static void sha256_update(sha256_state & state, const uint8_t * data, size_t size) {
    if (!data && size != 0) {
        return;
    }

    state.total_bytes += size;
    while (size > 0) {
        const size_t to_copy = std::min(size, sizeof(state.block) - state.block_size);
        memcpy(state.block + state.block_size, data, to_copy);
        state.block_size += to_copy;
        data += to_copy;
        size -= to_copy;

        if (state.block_size == sizeof(state.block)) {
            sha256_transform(state, state.block);
            state.block_size = 0;
        }
    }
}

static std::array<uint8_t, SHA256_SIZE> sha256_final(sha256_state & state) {
    const uint64_t total_bits = state.total_bytes * 8;
    state.block[state.block_size++] = 0x80;

    if (state.block_size > 56) {
        memset(state.block + state.block_size, 0, sizeof(state.block) - state.block_size);
        sha256_transform(state, state.block);
        state.block_size = 0;
    }

    memset(state.block + state.block_size, 0, 56 - state.block_size);
    for (size_t i = 0; i < 8; ++i) {
        state.block[56 + i] = uint8_t(total_bits >> (56 - 8 * i));
    }
    sha256_transform(state, state.block);

    std::array<uint8_t, SHA256_SIZE> digest{};
    for (size_t i = 0; i < 8; ++i) {
        digest[i * 4 + 0] = uint8_t(state.h[i] >> 24);
        digest[i * 4 + 1] = uint8_t(state.h[i] >> 16);
        digest[i * 4 + 2] = uint8_t(state.h[i] >> 8);
        digest[i * 4 + 3] = uint8_t(state.h[i]);
    }
    return digest;
}

static std::string digest_to_hex(const std::array<uint8_t, SHA256_SIZE> & digest) {
    static constexpr char HEX[] = "0123456789abcdef";
    std::string result;
    result.resize(SHA256_SIZE * 2);
    for (size_t i = 0; i < digest.size(); ++i) {
        result[i * 2 + 0] = HEX[digest[i] >> 4];
        result[i * 2 + 1] = HEX[digest[i] & 0x0f];
    }
    return result;
}

static fs::path utf8_path(const std::string & path) {
    return fs::u8path(path);
}

static void append_u32_le(std::string & out, uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) {
        out.push_back(char(value >> (8 * i)));
    }
}

static void append_u64_le(std::string & out, uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) {
        out.push_back(char(value >> (8 * i)));
    }
}

static void append_sized_string(std::string & out, const std::string & value) {
    append_u64_le(out, value.size());
    out.append(value);
}

static bool checked_embedding_size(
        uint32_t n_tokens,
        uint32_t n_embd,
        size_t & n_values,
        size_t & n_bytes) {
    if (n_tokens == 0 || n_embd == 0 ||
        size_t(n_tokens) > SIZE_MAX / size_t(n_embd)) {
        return false;
    }
    n_values = size_t(n_tokens) * size_t(n_embd);
    if (n_values > SIZE_MAX / sizeof(float)) {
        return false;
    }
    n_bytes = n_values * sizeof(float);
    return n_bytes <= CACHE_MAX_EMBEDDING_BYTES;
}

static uint64_t process_id() {
#ifdef _WIN32
    return uint64_t(_getpid());
#else
    return uint64_t(getpid());
#endif
}

} // namespace

std::string mtmd_cache_sha256(const uint8_t * data, size_t size) {
    if (!data && size != 0) {
        return {};
    }
    sha256_state state;
    sha256_init(state);
    sha256_update(state, data, size);
    return digest_to_hex(sha256_final(state));
}

std::string mtmd_cache::compute_file_sha256(const std::string & path) {
    std::ifstream file(utf8_path(path), std::ios::binary);
    if (!file.is_open()) {
        return {};
    }

    sha256_state state;
    sha256_init(state);
    std::array<uint8_t, 64 * 1024> chunk{};
    while (file) {
        file.read(reinterpret_cast<char *>(chunk.data()), chunk.size());
        const std::streamsize count = file.gcount();
        if (count > 0) {
            sha256_update(state, chunk.data(), size_t(count));
        }
    }
    if (!file.eof()) {
        return {};
    }
    return digest_to_hex(sha256_final(state));
}

std::string mtmd_cache::compute_cache_key(
        const uint8_t * image_data,
        size_t image_data_size,
        uint32_t nx,
        uint32_t ny,
        mtmd_projector_type proj_type) const {
    const std::string image_hash = mtmd_cache_sha256(image_data, image_data_size);
    if (image_hash.empty() && image_data_size != 0) {
        return {};
    }
    return compute_cache_key_from_hash(image_hash, nx, ny, proj_type);
}

std::string mtmd_cache::compute_cache_key_from_hash(
        const std::string & image_hash,
        uint32_t nx,
        uint32_t ny,
        mtmd_projector_type proj_type) const {
    std::string input("mtmd-cache-key-v1", 17);
    append_sized_string(input, mmproj_hash);
    append_sized_string(input, image_hash);
    append_u32_le(input, nx);
    append_u32_le(input, ny);
    append_u32_le(input, static_cast<uint32_t>(proj_type));
    return mtmd_cache_sha256(reinterpret_cast<const uint8_t *>(input.data()), input.size());
}

bool mtmd_cache::init(const std::string & cache_dir_in, const std::string & mmproj_path) {
    std::lock_guard<std::mutex> lock(mtx);
    enabled = false;
    cache_dir.clear();
    mmproj_hash.clear();

    if (cache_dir_in.empty() || mmproj_path.empty()) {
        return false;
    }

    const std::string hash = compute_file_sha256(mmproj_path);
    if (hash.empty()) {
        MTMD_CACHE_LOG_WRN("%s: failed to compute mmproj SHA-256\n", __func__);
        return false;
    }

    std::error_code ec;
    const fs::path dir = utf8_path(cache_dir_in);
    fs::create_directories(dir, ec);
    if (ec || !fs::is_directory(dir, ec) || ec) {
        MTMD_CACHE_LOG_WRN("%s: failed to create cache directory: %s\n", __func__, cache_dir_in.c_str());
        return false;
    }

    cache_dir = cache_dir_in;
    mmproj_hash = hash;
    enabled = true;
    MTMD_CACHE_LOG_INF("%s: cache enabled at %s (mmproj %s)\n",
        __func__, cache_dir.c_str(), mmproj_hash.substr(0, 12).c_str());
    return true;
}

bool mtmd_cache::lookup(
        const uint8_t * image_data,
        size_t image_data_size,
        uint32_t nx,
        uint32_t ny,
        mtmd_projector_type proj_type,
        std::vector<float> & embd,
        uint32_t & n_tokens,
        uint32_t & n_embd,
        bool & use_mrope_pos) {
    const std::string key = compute_cache_key(image_data, image_data_size, nx, ny, proj_type);
    if (key.empty()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mtx);
    return enabled && lookup_key(key, nx, ny, embd, n_tokens, n_embd, use_mrope_pos);
}

bool mtmd_cache::store(
        const uint8_t * image_data,
        size_t image_data_size,
        uint32_t nx,
        uint32_t ny,
        mtmd_projector_type proj_type,
        const float * embd,
        uint32_t n_tokens,
        uint32_t n_embd,
        bool use_mrope_pos) {
    const std::string key = compute_cache_key(image_data, image_data_size, nx, ny, proj_type);
    if (key.empty()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mtx);
    return enabled && store_key(key, nx, ny, embd, n_tokens, n_embd, use_mrope_pos);
}

bool mtmd_cache::lookup_by_hash(
        const std::string & image_hash,
        uint32_t nx,
        uint32_t ny,
        mtmd_projector_type proj_type,
        std::vector<float> & embd,
        uint32_t & n_tokens,
        uint32_t & n_embd,
        bool & use_mrope_pos) {
    if (image_hash.empty()) {
        return false;
    }
    const std::string key = compute_cache_key_from_hash(image_hash, nx, ny, proj_type);
    std::lock_guard<std::mutex> lock(mtx);
    return enabled && lookup_key(key, nx, ny, embd, n_tokens, n_embd, use_mrope_pos);
}

bool mtmd_cache::store_by_hash(
        const std::string & image_hash,
        uint32_t nx,
        uint32_t ny,
        mtmd_projector_type proj_type,
        const float * embd,
        uint32_t n_tokens,
        uint32_t n_embd,
        bool use_mrope_pos) {
    if (image_hash.empty()) {
        return false;
    }
    const std::string key = compute_cache_key_from_hash(image_hash, nx, ny, proj_type);
    std::lock_guard<std::mutex> lock(mtx);
    return enabled && store_key(key, nx, ny, embd, n_tokens, n_embd, use_mrope_pos);
}

bool mtmd_cache::lookup_key(
        const std::string & key,
        uint32_t nx,
        uint32_t ny,
        std::vector<float> & embd,
        uint32_t & n_tokens,
        uint32_t & n_embd,
        bool & use_mrope_pos) {
    const fs::path path = utf8_path(cache_dir) / (key + ".bin");
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    auto reject = [&](const char * reason) {
        MTMD_CACHE_LOG_WRN("%s: rejecting %s: %s\n", __func__, path.string().c_str(), reason);
        file.close();
        std::error_code remove_ec;
        fs::remove(path, remove_ec);
        embd.clear();
        return false;
    };

    mtmd_cache_header header{};
    if (!file.read(reinterpret_cast<char *>(&header), sizeof(header))) {
        return reject("truncated header");
    }
    if (header.magic != MTMD_CACHE_MAGIC || header.version != MTMD_CACHE_VERSION) {
        return reject("invalid magic or version");
    }
    if (header.nx != nx || header.ny != ny || header.use_mrope_pos > 1) {
        return reject("metadata mismatch");
    }

    size_t n_values = 0;
    size_t n_bytes = 0;
    if (!checked_embedding_size(header.n_tokens, header.n_embd, n_values, n_bytes)) {
        return reject("invalid embedding dimensions");
    }

    std::error_code size_ec;
    const uintmax_t file_size = fs::file_size(path, size_ec);
    if (size_ec || file_size != sizeof(header) + uintmax_t(n_bytes)) {
        return reject("unexpected file size");
    }

    embd.resize(n_values);
    if (!file.read(reinterpret_cast<char *>(embd.data()), std::streamsize(n_bytes))) {
        return reject("truncated embedding payload");
    }

    n_tokens = header.n_tokens;
    n_embd = header.n_embd;
    use_mrope_pos = header.use_mrope_pos != 0;
    MTMD_CACHE_LOG_INF("%s: cache hit %s (%u x %u)\n",
        __func__, key.substr(0, 12).c_str(), n_tokens, n_embd);
    return true;
}

bool mtmd_cache::store_key(
        const std::string & key,
        uint32_t nx,
        uint32_t ny,
        const float * embd,
        uint32_t n_tokens,
        uint32_t n_embd,
        bool use_mrope_pos) {
    size_t n_values = 0;
    size_t n_bytes = 0;
    if (!embd || !checked_embedding_size(n_tokens, n_embd, n_values, n_bytes)) {
        return false;
    }

    mtmd_cache_header header{};
    header.magic = MTMD_CACHE_MAGIC;
    header.version = MTMD_CACHE_VERSION;
    header.n_tokens = n_tokens;
    header.n_embd = n_embd;
    header.nx = nx;
    header.ny = ny;
    header.use_mrope_pos = use_mrope_pos ? 1 : 0;

    const fs::path path = utf8_path(cache_dir) / (key + ".bin");
    if (!atomic_write_file(path.u8string(), header, embd, n_bytes)) {
        MTMD_CACHE_LOG_WRN("%s: failed to store %s\n", __func__, path.string().c_str());
        return false;
    }

    MTMD_CACHE_LOG_INF("%s: cache stored %s (%u x %u, %.2f MiB)\n",
        __func__, key.substr(0, 12).c_str(), n_tokens, n_embd, n_bytes / 1024.0 / 1024.0);
    return true;
}

bool mtmd_cache::atomic_write_file(
        const std::string & path_string,
        const mtmd_cache_header & header,
        const float * embd,
        size_t embd_bytes) {
    static std::atomic<uint64_t> temp_counter{0};

    const fs::path destination = utf8_path(path_string);
    fs::path temporary = destination;
    temporary += ".tmp." + std::to_string(process_id()) + "." +
                 std::to_string(temp_counter.fetch_add(1, std::memory_order_relaxed));

    auto remove_temporary = [&]() {
        std::error_code ec;
        fs::remove(temporary, ec);
    };

    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) {
            return false;
        }
        file.write(reinterpret_cast<const char *>(&header), sizeof(header));
        file.write(reinterpret_cast<const char *>(embd), std::streamsize(embd_bytes));
        file.flush();
        if (!file.good()) {
            file.close();
            remove_temporary();
            return false;
        }
    }

#ifdef _WIN32
    HANDLE handle = CreateFileW(
        temporary.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        remove_temporary();
        return false;
    }
    const BOOL flushed = FlushFileBuffers(handle);
    CloseHandle(handle);
    if (!flushed || !MoveFileExW(
            temporary.c_str(), destination.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        remove_temporary();
        return false;
    }
#else
    const int fd = open(temporary.c_str(), O_RDONLY);
    if (fd < 0 || fsync(fd) != 0) {
        if (fd >= 0) {
            close(fd);
        }
        remove_temporary();
        return false;
    }
    close(fd);
    if (rename(temporary.c_str(), destination.c_str()) != 0) {
        remove_temporary();
        return false;
    }

    const int dir_fd = open(destination.parent_path().c_str(), O_RDONLY);
    if (dir_fd >= 0) {
        (void) fsync(dir_fd);
        close(dir_fd);
    }
#endif

    return true;
}
