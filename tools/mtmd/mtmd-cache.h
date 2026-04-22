#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <mutex>

// Cache file magic "MTC\0"
#define MTMD_CACHE_MAGIC 0x0043544D

// Cache file format version
#define MTMD_CACHE_VERSION 1

/**
 * Cache file header structure
 * Followed by n_tokens * n_embd * sizeof(float) embedding data
 */
struct mtmd_cache_header {
    uint32_t magic;         // MTMD_CACHE_MAGIC
    uint32_t version;       // MTMD_CACHE_VERSION
    uint32_t n_tokens;      // number of tokens
    uint32_t n_embd;        // embedding dimension
    uint32_t nx;            // image width in tokens
    uint32_t ny;            // image height in tokens
    uint8_t  use_mrope_pos; // whether M-RoPE is used (0/1)
    uint8_t  reserved[3];   // alignment padding
};

static_assert(sizeof(mtmd_cache_header) == 28, "mtmd_cache_header size mismatch");

/**
 * Projector type enum for cache key computation
 */
enum class mtmd_projector_type {
    UNKNOWN = 0,
    LLaVA,
    QWEN2VL,
    QWEN25VL,
    GEMMA3,
    MINICPMV,
    INTERNVL,
    LLAMA4,
    GLM4V,
    COGVLM,
    KIMIVL,
    KIMIK25,
    NEMOTRON_V2_VL,
};

/**
 * Image tokenization cache manager
 * Thread-safe: mutex protects file operations
 */
struct mtmd_cache {
    std::string cache_dir;        // cache directory path
    bool enabled = false;         // whether cache is enabled
    std::string mmproj_hash;      // mmproj model file SHA256

    std::mutex mtx;  // protect concurrent file access

    /**
     * Initialize cache manager
     * @param cache_dir cache directory path
     * @param mmproj_path mmproj model file path (for computing hash)
     * @return true on success, false on failure (disables cache)
     */
    bool init(const std::string & cache_dir, const std::string & mmproj_path);

    /**
     * Lookup cache
     * @param image_data raw image data
     * @param image_data_size image data size
     * @param nx image width in tokens
     * @param ny image height in tokens
     * @param proj_type projector type
     * @param[out] embd output embedding data (filled on hit)
     * @param[out] n_tokens token count (filled on hit)
     * @param[out] n_embd embedding dimension (filled on hit)
     * @param[out] use_mrope_pos whether M-RoPE is used (filled on hit)
     * @return true on cache hit, false on miss
     */
    bool lookup(const uint8_t * image_data, size_t image_data_size,
                uint32_t nx, uint32_t ny,
                mtmd_projector_type proj_type,
                std::vector<float> & embd,
                uint32_t & n_tokens, uint32_t & n_embd,
                bool & use_mrope_pos);

    /**
     * Store to cache
     * @param image_data raw image data
     * @param image_data_size image data size
     * @param nx image width in tokens
     * @param ny image height in tokens
     * @param proj_type projector type
     * @param embd embedding data
     * @param n_tokens token count
     * @param n_embd embedding dimension
     * @param use_mrope_pos whether M-RoPE is used
     * @return true on success, false on failure (doesn't affect inference)
     */
    bool store(const uint8_t * image_data, size_t image_data_size,
               uint32_t nx, uint32_t ny,
               mtmd_projector_type proj_type,
               const float * embd,
               uint32_t n_tokens, uint32_t n_embd,
               bool use_mrope_pos);

    /**
     * Lookup cache using a pre-computed image hash (e.g., from bitmap id)
     * This is useful when raw image data is not available at inference time
     * @param image_hash pre-computed hash string for the image
     * @param nx image width in tokens
     * @param ny image height in tokens
     * @param proj_type projector type
     * @param[out] embd output embedding data (filled on hit)
     * @param[out] n_tokens token count (filled on hit)
     * @param[out] n_embd embedding dimension (filled on hit)
     * @param[out] use_mrope_pos whether M-RoPE is used (filled on hit)
     * @return true on cache hit, false on miss
     */
    bool lookup_by_hash(const std::string & image_hash,
                        uint32_t nx, uint32_t ny,
                        mtmd_projector_type proj_type,
                        std::vector<float> & embd,
                        uint32_t & n_tokens, uint32_t & n_embd,
                        bool & use_mrope_pos);

    /**
     * Store to cache using a pre-computed image hash
     * @param image_hash pre-computed hash string for the image
     * @param nx image width in tokens
     * @param ny image height in tokens
     * @param proj_type projector type
     * @param embd embedding data
     * @param n_tokens token count
     * @param n_embd embedding dimension
     * @param use_mrope_pos whether M-RoPE is used
     * @return true on success, false on failure (doesn't affect inference)
     */
    bool store_by_hash(const std::string & image_hash,
                       uint32_t nx, uint32_t ny,
                       mtmd_projector_type proj_type,
                       const float * embd,
                       uint32_t n_tokens, uint32_t n_embd,
                       bool use_mrope_pos);

private:
    std::string compute_cache_key(const uint8_t * image_data, size_t image_data_size,
                                   uint32_t nx, uint32_t ny,
                                   mtmd_projector_type proj_type) const;

    // Overload that accepts a pre-computed image hash string instead of raw data
    std::string compute_cache_key_from_hash(const std::string & image_hash,
                                             uint32_t nx, uint32_t ny,
                                             mtmd_projector_type proj_type) const;

    static std::string compute_file_sha256(const std::string & path);
    static std::string compute_data_sha256(const uint8_t * data, size_t size);

    bool atomic_write_file(const std::string & path,
                           const mtmd_cache_header & header,
                           const float * embd, size_t embd_size);
};
