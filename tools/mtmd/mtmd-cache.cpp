#include "mtmd-cache.h"

#include "ggml.h"

#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <sys/stat.h>

#ifdef _WIN32
#   include <windows.h>
#   include <direct.h>
#   include <io.h>
#else
#   include <unistd.h>
#endif

#define MTMD_CACHE_LOG_INF(...)  fprintf(stderr, __VA_ARGS__)
#define MTMD_CACHE_LOG_WRN(...)  fprintf(stderr, __VA_ARGS__)
#define MTMD_CACHE_LOG_ERR(...)  fprintf(stderr, __VA_ARGS__)
// Adapted from examples/gguf-hash/deps/sha256/ (Igor Pavlov, Public domain)
// with rotate-bits macros inlined to avoid external dependency
// ============================================================

#define SHA256_DIGEST_SIZE 32

struct sha256_t {
    uint32_t state[8];
    uint64_t count;
    uint8_t  buffer[64];
};

// Cross-platform bit rotation macros
#ifdef _MSC_VER
#   include <stdlib.h>
#   define SHA256_ROTR32(v, n) _rotr((v), (n))
#else
#   define SHA256_ROTR32(v, n) (((uint32_t)(v) >> (n)) | ((uint32_t)(v) << (32 - (n))))
#endif

static void sha256_init(sha256_t * p) {
    p->state[0] = 0x6a09e667;
    p->state[1] = 0xbb67ae85;
    p->state[2] = 0x3c6ef372;
    p->state[3] = 0xa54ff53a;
    p->state[4] = 0x510e527f;
    p->state[5] = 0x9b05688c;
    p->state[6] = 0x1f83d9ab;
    p->state[7] = 0x5be0cd19;
    p->count = 0;
}

#define SHA256_S0(x) (SHA256_ROTR32(x, 2)  ^ SHA256_ROTR32(x,13) ^ SHA256_ROTR32(x, 22))
#define SHA256_S1(x) (SHA256_ROTR32(x, 6)  ^ SHA256_ROTR32(x,11) ^ SHA256_ROTR32(x, 25))
#define SHA256_s0(x) (SHA256_ROTR32(x, 7)  ^ SHA256_ROTR32(x,18) ^ ((x) >> 3))
#define SHA256_s1(x) (SHA256_ROTR32(x,17)  ^ SHA256_ROTR32(x,19) ^ ((x) >> 10))

#define SHA256_Ch(x,y,z) (z^(x&(y^z)))
#define SHA256_Maj(x,y,z) ((x&y)|(z&(x|y)))

static const uint32_t sha256_K[64] = {
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
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static void sha256_transform(uint32_t * state, const uint32_t * data) {
    uint32_t W[16];
    uint32_t a, b, c, d, e, f, g, h;
    unsigned j;

    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];

    for (j = 0; j < 64; j += 16) {
        // Rounds 0-15
        W[0]  = data[0];  W[1]  = data[1];  W[2]  = data[2];  W[3]  = data[3];
        W[4]  = data[4];  W[5]  = data[5];  W[6]  = data[6];  W[7]  = data[7];
        W[8]  = data[8];  W[9]  = data[9];  W[10] = data[10]; W[11] = data[11];
        W[12] = data[12]; W[13] = data[13]; W[14] = data[14]; W[15] = data[15];

        for (unsigned i = 0; i < 16; i++) {
            uint32_t t = h + SHA256_S1(e) + SHA256_Ch(e,f,g) + sha256_K[j+i] + W[i&15];
            d += t; t += SHA256_S0(a) + SHA256_Maj(a,b,c);
            h = g; g = f; f = e; e = d; d = c; c = b; b = a; a = t;
        }
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

// Undefine internal macros to avoid polluting the namespace
#undef SHA256_S0
#undef SHA256_S1
#undef SHA256_s0
#undef SHA256_s1
#undef SHA256_Ch
#undef SHA256_Maj
#undef SHA256_ROTR32

static void sha256_write_byte_block(sha256_t * p) {
    uint32_t data32[16];
    for (unsigned i = 0; i < 16; i++) {
        data32[i] =
            ((uint32_t)(p->buffer[i * 4    ]) << 24) +
            ((uint32_t)(p->buffer[i * 4 + 1]) << 16) +
            ((uint32_t)(p->buffer[i * 4 + 2]) <<  8) +
            ((uint32_t)(p->buffer[i * 4 + 3]));
    }
    sha256_transform(p->state, data32);
}

static void sha256_update(sha256_t * p, const uint8_t * data, size_t size) {
    uint32_t curBufferPos = (uint32_t)(p->count & 0x3F);
    while (size > 0) {
        p->buffer[curBufferPos++] = *data++;
        p->count++;
        size--;
        if (curBufferPos == 64) {
            curBufferPos = 0;
            sha256_write_byte_block(p);
        }
    }
}

static void sha256_final(sha256_t * p, uint8_t * digest) {
    uint64_t lenInBits = (p->count << 3);
    uint32_t curBufferPos = (uint32_t)(p->count & 0x3F);

    p->buffer[curBufferPos++] = 0x80;
    while (curBufferPos != (64 - 8)) {
        curBufferPos &= 0x3F;
        if (curBufferPos == 0) {
            sha256_write_byte_block(p);
        }
        p->buffer[curBufferPos++] = 0;
    }
    for (unsigned i = 0; i < 8; i++) {
        p->buffer[curBufferPos++] = (uint8_t)(lenInBits >> 56);
        lenInBits <<= 8;
    }
    sha256_write_byte_block(p);

    for (unsigned i = 0; i < 8; i++) {
        *digest++ = (uint8_t)(p->state[i] >> 24);
        *digest++ = (uint8_t)(p->state[i] >> 16);
        *digest++ = (uint8_t)(p->state[i] >> 8);
        *digest++ = (uint8_t)(p->state[i]);
    }
    sha256_init(p);
}

static void sha256_hash(uint8_t * buf, const uint8_t * data, size_t size) {
    sha256_t hash;
    sha256_init(&hash);
    sha256_update(&hash, data, size);
    sha256_final(&hash, buf);
}

// ============================================================
// Utility: convert raw SHA-256 digest to hex string
// ============================================================

static std::string sha256_to_hex(const uint8_t * digest) {
    static const char hex_chars[] = "0123456789abcdef";
    std::string result;
    result.reserve(SHA256_DIGEST_SIZE * 2);
    for (int i = 0; i < SHA256_DIGEST_SIZE; i++) {
        result.push_back(hex_chars[(digest[i] >> 4) & 0x0F]);
        result.push_back(hex_chars[digest[i] & 0x0F]);
    }
    return result;
}

// ============================================================
// Utility: cross-platform directory creation
// Copied from tools/rpc/rpc-server.cpp (avoids dependency on libcommon)
// ============================================================

#ifdef _WIN32
// UTF-8 to wide string conversion for Windows API calls
static std::wstring mtmd_utf8_to_wstring(const std::string & str) {
    if (str.empty()) return std::wstring();
    int size = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), nullptr, 0);
    if (size <= 0) return std::wstring();
    std::wstring result(size, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &result[0], size);
    return result;
}
#endif

static bool mtmd_create_directory_with_parents(const std::string & path) {
#ifdef _WIN32
    std::wstring wpath = mtmd_utf8_to_wstring(path);

    // If the path already exists, check whether it's a directory
    const DWORD attributes = GetFileAttributesW(wpath.c_str());
    if ((attributes != INVALID_FILE_ATTRIBUTES) && (attributes & FILE_ATTRIBUTE_DIRECTORY)) {
        return true;
    }

    size_t pos_slash = 0;

    // Process path from front to back, procedurally creating directories
    // Handle both '\\' and '/' as path separators on Windows
    while (true) {
        size_t pos_bs = path.find('\\', pos_slash);
        size_t pos_fs = path.find('/', pos_slash);
        size_t next_sep;
        if (pos_bs == std::string::npos && pos_fs == std::string::npos) {
            break;
        } else if (pos_bs == std::string::npos) {
            next_sep = pos_fs;
        } else if (pos_fs == std::string::npos) {
            next_sep = pos_bs;
        } else {
            next_sep = (std::min)(pos_bs, pos_fs);
        }

        const std::wstring subpath = wpath.substr(0, next_sep);
        pos_slash = next_sep + 1;

        // Skip the drive letter, in some systems it can return an access denied error
        if (subpath.length() == 2 && subpath[1] == L':') {
            continue;
        }

        const BOOL success = CreateDirectoryW(subpath.c_str(), NULL);
        if (!success) {
            const DWORD error = GetLastError();
            if (error == ERROR_ALREADY_EXISTS) {
                const DWORD attr = GetFileAttributesW(subpath.c_str());
                if (attr == INVALID_FILE_ATTRIBUTES || !(attr & FILE_ATTRIBUTE_DIRECTORY)) {
                    return false;
                }
            } else {
                return false;
            }
        }
    }

    return true;
#else
    // If the path already exists, check whether it's a directory
    struct stat info;
    if (stat(path.c_str(), &info) == 0) {
        return S_ISDIR(info.st_mode);
    }

    size_t pos_slash = 1; // Skip leading slashes for directory creation

    // Process path from front to back, procedurally creating directories
    while ((pos_slash = path.find('/', pos_slash)) != std::string::npos) {
        const std::string subpath = path.substr(0, pos_slash);
        struct stat st_info;

        if (stat(subpath.c_str(), &st_info) == 0) {
            if (!S_ISDIR(st_info.st_mode)) {
                return false;
            }
        } else {
            const int ret = mkdir(subpath.c_str(), 0755);
            if (ret != 0) {
                return false;
            }
        }

        pos_slash += 1;
    }

    return true;
#endif
}

// ============================================================
// mtmd_cache implementation
// ============================================================

std::string mtmd_cache::compute_data_sha256(const uint8_t * data, size_t size) {
    uint8_t digest[SHA256_DIGEST_SIZE];
    sha256_hash(digest, data, size);
    return sha256_to_hex(digest);
}

std::string mtmd_cache::compute_file_sha256(const std::string & path) {
    // Read file in chunks to handle large mmproj models without excessive memory use
    static const size_t CHUNK_SIZE = 1024 * 1024; // 1 MiB

#ifdef _WIN32
    std::wstring wpath = mtmd_utf8_to_wstring(path);
    FILE * file = _wfopen(wpath.c_str(), L"rb");
#else
    FILE * file = fopen(path.c_str(), "rb");
#endif
    if (!file) {
        MTMD_CACHE_LOG_WRN("%s: failed to open file for SHA-256: %s\n", __func__, path.c_str());
        return "";
    }

    sha256_t ctx;
    sha256_init(&ctx);

    std::vector<uint8_t> chunk(CHUNK_SIZE);
    size_t bytes_read;
    while ((bytes_read = fread(chunk.data(), 1, CHUNK_SIZE, file)) > 0) {
        sha256_update(&ctx, chunk.data(), bytes_read);
    }
    fclose(file);

    uint8_t digest[SHA256_DIGEST_SIZE];
    sha256_final(&ctx, digest);
    return sha256_to_hex(digest);
}

std::string mtmd_cache::compute_cache_key(
    const uint8_t * image_data, size_t image_data_size,
    uint32_t nx, uint32_t ny,
    mtmd_projector_type proj_type) const {

    // Build composite input: mmproj_hash + image_hash + nx + ny + proj_type
    // This ensures cache invalidation when the mmproj model changes,
    // the image data changes, or tokenization parameters change
    std::string input;
    input += mmproj_hash;
    input += compute_data_sha256(image_data, image_data_size);
    input.append(reinterpret_cast<const char *>(&nx), sizeof(nx));
    input.append(reinterpret_cast<const char *>(&ny), sizeof(ny));
    int32_t pt = static_cast<int32_t>(proj_type);
    input.append(reinterpret_cast<const char *>(&pt), sizeof(pt));

    return compute_data_sha256(reinterpret_cast<const uint8_t *>(input.data()), input.size());
}

std::string mtmd_cache::compute_cache_key_from_hash(
    const std::string & image_hash,
    uint32_t nx, uint32_t ny,
    mtmd_projector_type proj_type) const {

    // Same as compute_cache_key but uses a pre-computed hash instead of raw image data
    // The image_hash is expected to be a unique identifier for the image content
    // (e.g., FNV hash of raw bitmap data computed during tokenization)
    std::string input;
    input += mmproj_hash;
    input += image_hash;
    input.append(reinterpret_cast<const char *>(&nx), sizeof(nx));
    input.append(reinterpret_cast<const char *>(&ny), sizeof(ny));
    int32_t pt = static_cast<int32_t>(proj_type);
    input.append(reinterpret_cast<const char *>(&pt), sizeof(pt));

    return compute_data_sha256(reinterpret_cast<const uint8_t *>(input.data()), input.size());
}

bool mtmd_cache::init(const std::string & cache_dir_, const std::string & mmproj_path) {
    enabled = false;

    // Compute mmproj file hash first — this is required regardless of directory creation
    mmproj_hash = compute_file_sha256(mmproj_path);
    if (mmproj_hash.empty()) {
        MTMD_CACHE_LOG_WRN("%s: failed to compute mmproj SHA-256, cache disabled\n", __func__);
        return false;
    }

    MTMD_CACHE_LOG_INF("%s: mmproj hash: %s\n", __func__, mmproj_hash.c_str());

    // Create cache directory (including parents) if it doesn't exist
    if (!mtmd_create_directory_with_parents(cache_dir_)) {
        MTMD_CACHE_LOG_WRN("%s: failed to create cache directory: %s\n", __func__, cache_dir_.c_str());
        return false;
    }

    cache_dir = cache_dir_;
    enabled = true;

    MTMD_CACHE_LOG_INF("%s: image tokenization cache enabled at: %s\n", __func__, cache_dir.c_str());
    return true;
}

bool mtmd_cache::lookup(const uint8_t * image_data, size_t image_data_size,
                         uint32_t nx, uint32_t ny,
                         mtmd_projector_type proj_type,
                         std::vector<float> & embd,
                         uint32_t & n_tokens, uint32_t & n_embd,
                         bool & use_mrope_pos) {
    if (!enabled) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mtx);

    std::string key = compute_cache_key(image_data, image_data_size, nx, ny, proj_type);
    std::string path = cache_dir + "/" + key + ".bin";

#ifdef _WIN32
    std::wstring wpath = mtmd_utf8_to_wstring(path);
    std::ifstream file(wpath, std::ios::binary);
#else
    std::ifstream file(path, std::ios::binary);
#endif
    if (!file.is_open()) {
        return false; // Cache miss — normal case, no warning needed
    }

    // Read and validate header
    mtmd_cache_header header;
    if (!file.read(reinterpret_cast<char *>(&header), sizeof(header))) {
        MTMD_CACHE_LOG_WRN("%s: failed to read cache header: %s\n", __func__, path.c_str());
        return false;
    }

    if (header.magic != MTMD_CACHE_MAGIC) {
        MTMD_CACHE_LOG_WRN("%s: invalid cache magic in %s\n", __func__, path.c_str());
        return false;
    }

    if (header.version != MTMD_CACHE_VERSION) {
        MTMD_CACHE_LOG_WRN("%s: cache version mismatch (expected %u, got %u) in %s\n",
                __func__, MTMD_CACHE_VERSION, header.version, path.c_str());
        return false;
    }

    // Validate dimensions to guard against corrupted cache files
    if (header.n_tokens == 0 || header.n_embd == 0) {
        MTMD_CACHE_LOG_WRN("%s: invalid cache dimensions (n_tokens=%u, n_embd=%u) in %s\n",
                __func__, header.n_tokens, header.n_embd, path.c_str());
        return false;
    }

    // Read embedding data
    size_t embd_size = (size_t)header.n_tokens * header.n_embd;
    embd.resize(embd_size);
    if (!file.read(reinterpret_cast<char *>(embd.data()), embd_size * sizeof(float))) {
        MTMD_CACHE_LOG_WRN("%s: failed to read cache embedding data: %s\n", __func__, path.c_str());
        return false;
    }

    n_tokens = header.n_tokens;
    n_embd = header.n_embd;
    use_mrope_pos = header.use_mrope_pos != 0;

    MTMD_CACHE_LOG_INF("%s: cache hit: %s (n_tokens=%u, n_embd=%u)\n",
            __func__, key.substr(0, 12).c_str(), n_tokens, n_embd);
    return true;
}

bool mtmd_cache::store(const uint8_t * image_data, size_t image_data_size,
                        uint32_t nx, uint32_t ny,
                        mtmd_projector_type proj_type,
                        const float * embd,
                        uint32_t n_tokens, uint32_t n_embd,
                        bool use_mrope_pos) {
    if (!enabled) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mtx);

    std::string key = compute_cache_key(image_data, image_data_size, nx, ny, proj_type);
    std::string path = cache_dir + "/" + key + ".bin";

    mtmd_cache_header header;
    header.magic = MTMD_CACHE_MAGIC;
    header.version = MTMD_CACHE_VERSION;
    header.n_tokens = n_tokens;
    header.n_embd = n_embd;
    header.nx = nx;
    header.ny = ny;
    header.use_mrope_pos = use_mrope_pos ? 1 : 0;
    header.reserved[0] = 0;
    header.reserved[1] = 0;
    header.reserved[2] = 0;

    size_t embd_size = (size_t)n_tokens * n_embd;

    if (!atomic_write_file(path, header, embd, embd_size * sizeof(float))) {
        // Store failure should not affect inference — just log a warning
        MTMD_CACHE_LOG_WRN("%s: failed to write cache file: %s\n", __func__, path.c_str());
        return false;
    }

    MTMD_CACHE_LOG_INF("%s: cache stored: %s (n_tokens=%u, n_embd=%u, %.2f MiB)\n",
            __func__, key.substr(0, 12).c_str(), n_tokens, n_embd,
            (double)(embd_size * sizeof(float)) / (1024.0 * 1024.0));
    return true;
}

bool mtmd_cache::atomic_write_file(const std::string & path,
                                    const mtmd_cache_header & header,
                                    const float * embd, size_t embd_size) {
    // Write to a temporary file first, then atomically rename.
    // This prevents readers from seeing a partially-written cache file.
    std::string temp_path = path + ".tmp";

    {
#ifdef _WIN32
        std::wstring wtemp_path = mtmd_utf8_to_wstring(temp_path);
        std::ofstream file(wtemp_path, std::ios::binary);
#else
        std::ofstream file(temp_path, std::ios::binary);
#endif
        if (!file.is_open()) {
            return false;
        }

        if (!file.write(reinterpret_cast<const char *>(&header), sizeof(header))) {
            return false;
        }

        if (embd_size > 0 && embd) {
            if (!file.write(reinterpret_cast<const char *>(embd), embd_size)) {
                return false;
            }
        }

        // Flush to disk before rename to ensure durability
        file.flush();
        if (!file.good()) {
            return false;
        }
    }

    // Atomic rename: temp_path -> path
    // On Windows, rename() cannot overwrite an existing file, so we must
    // remove the destination first. This is safe because concurrent access
    // is protected by the mutex in the calling lookup/store methods.
#ifdef _WIN32
    std::wstring wtemp = mtmd_utf8_to_wstring(temp_path);
    std::wstring wdest = mtmd_utf8_to_wstring(path);

    // If destination exists, remove it first
    DWORD attrs = GetFileAttributesW(wdest.c_str());
    if (attrs != INVALID_FILE_ATTRIBUTES) {
        if (!DeleteFileW(wdest.c_str())) {
            MTMD_CACHE_LOG_WRN("%s: failed to remove existing cache file: %s\n", __func__, path.c_str());
            // Continue anyway — MoveFileEx may succeed
        }
    }

    if (!MoveFileExW(wtemp.c_str(), wdest.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        MTMD_CACHE_LOG_WRN("%s: failed to rename temp cache file: %s\n", __func__, path.c_str());
        return false;
    }
#else
    if (rename(temp_path.c_str(), path.c_str()) != 0) {
        MTMD_CACHE_LOG_WRN("%s: failed to rename temp cache file: %s\n", __func__, path.c_str());
        return false;
    }
#endif

    return true;
}

// ============================================================
// Hash-based cache lookup/store (for use when raw image data is not available)
// ============================================================

bool mtmd_cache::lookup_by_hash(const std::string & image_hash,
                                 uint32_t nx, uint32_t ny,
                                 mtmd_projector_type proj_type,
                                 std::vector<float> & embd,
                                 uint32_t & n_tokens, uint32_t & n_embd,
                                 bool & use_mrope_pos) {
    if (!enabled) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mtx);

    std::string key = compute_cache_key_from_hash(image_hash, nx, ny, proj_type);
    std::string path = cache_dir + "/" + key + ".bin";

#ifdef _WIN32
    std::wstring wpath = mtmd_utf8_to_wstring(path);
    std::ifstream file(wpath, std::ios::binary);
#else
    std::ifstream file(path, std::ios::binary);
#endif
    if (!file.is_open()) {
        return false; // Cache miss
    }

    // Read and validate header
    mtmd_cache_header header;
    if (!file.read(reinterpret_cast<char *>(&header), sizeof(header))) {
        MTMD_CACHE_LOG_WRN("%s: failed to read cache header: %s\n", __func__, path.c_str());
        return false;
    }

    if (header.magic != MTMD_CACHE_MAGIC) {
        MTMD_CACHE_LOG_WRN("%s: invalid cache magic in %s\n", __func__, path.c_str());
        return false;
    }

    if (header.version != MTMD_CACHE_VERSION) {
        MTMD_CACHE_LOG_WRN("%s: cache version mismatch (expected %u, got %u) in %s\n",
                __func__, MTMD_CACHE_VERSION, header.version, path.c_str());
        return false;
    }

    if (header.n_tokens == 0 || header.n_embd == 0) {
        MTMD_CACHE_LOG_WRN("%s: invalid cache dimensions (n_tokens=%u, n_embd=%u) in %s\n",
                __func__, header.n_tokens, header.n_embd, path.c_str());
        return false;
    }

    // Read embedding data
    size_t embd_size = (size_t)header.n_tokens * header.n_embd;
    embd.resize(embd_size);
    if (!file.read(reinterpret_cast<char *>(embd.data()), embd_size * sizeof(float))) {
        MTMD_CACHE_LOG_WRN("%s: failed to read cache embedding data: %s\n", __func__, path.c_str());
        return false;
    }

    n_tokens = header.n_tokens;
    n_embd = header.n_embd;
    use_mrope_pos = header.use_mrope_pos != 0;

    MTMD_CACHE_LOG_INF("%s: cache hit (by hash): %s (n_tokens=%u, n_embd=%u)\n",
            __func__, key.substr(0, 12).c_str(), n_tokens, n_embd);
    return true;
}

bool mtmd_cache::store_by_hash(const std::string & image_hash,
                                uint32_t nx, uint32_t ny,
                                mtmd_projector_type proj_type,
                                const float * embd,
                                uint32_t n_tokens, uint32_t n_embd,
                                bool use_mrope_pos) {
    if (!enabled) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mtx);

    std::string key = compute_cache_key_from_hash(image_hash, nx, ny, proj_type);
    std::string path = cache_dir + "/" + key + ".bin";

    mtmd_cache_header header;
    header.magic = MTMD_CACHE_MAGIC;
    header.version = MTMD_CACHE_VERSION;
    header.n_tokens = n_tokens;
    header.n_embd = n_embd;
    header.nx = nx;
    header.ny = ny;
    header.use_mrope_pos = use_mrope_pos ? 1 : 0;
    header.reserved[0] = 0;
    header.reserved[1] = 0;
    header.reserved[2] = 0;

    size_t embd_size = (size_t)n_tokens * n_embd;

    if (!atomic_write_file(path, header, embd, embd_size * sizeof(float))) {
        MTMD_CACHE_LOG_WRN("%s: failed to write cache file: %s\n", __func__, path.c_str());
        return false;
    }

    MTMD_CACHE_LOG_INF("%s: cache stored (by hash): %s (n_tokens=%u, n_embd=%u, %.2f MiB)\n",
            __func__, key.substr(0, 12).c_str(), n_tokens, n_embd,
            (double)(embd_size * sizeof(float)) / (1024.0 * 1024.0));
    return true;
}
