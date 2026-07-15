#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#define MTMD_CACHE_MAGIC 0x0043544D
#define MTMD_CACHE_VERSION 1

struct mtmd_cache_header {
    uint32_t magic;
    uint32_t version;
    uint32_t n_tokens;
    uint32_t n_embd;
    uint32_t nx;
    uint32_t ny;
    uint8_t use_mrope_pos;
    uint8_t reserved[3];
};

static_assert(sizeof(mtmd_cache_header) == 28, "mtmd_cache_header size mismatch");

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

// Exposed for server bitmap IDs and deterministic unit tests.
std::string mtmd_cache_sha256(const uint8_t * data, size_t size);

struct mtmd_cache {
    std::string cache_dir;
    bool enabled = false;
    std::string mmproj_hash;

    std::mutex mtx;

    bool init(const std::string & cache_dir, const std::string & mmproj_path);

    bool lookup(const uint8_t * image_data, size_t image_data_size,
                uint32_t nx, uint32_t ny,
                mtmd_projector_type proj_type,
                std::vector<float> & embd,
                uint32_t & n_tokens, uint32_t & n_embd,
                bool & use_mrope_pos);

    bool store(const uint8_t * image_data, size_t image_data_size,
               uint32_t nx, uint32_t ny,
               mtmd_projector_type proj_type,
               const float * embd,
               uint32_t n_tokens, uint32_t n_embd,
               bool use_mrope_pos);

    bool lookup_by_hash(const std::string & image_hash,
                        uint32_t nx, uint32_t ny,
                        mtmd_projector_type proj_type,
                        std::vector<float> & embd,
                        uint32_t & n_tokens, uint32_t & n_embd,
                        bool & use_mrope_pos);

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

    std::string compute_cache_key_from_hash(const std::string & image_hash,
                                            uint32_t nx, uint32_t ny,
                                            mtmd_projector_type proj_type) const;

    static std::string compute_file_sha256(const std::string & path);

    bool lookup_key(const std::string & key,
                    uint32_t nx, uint32_t ny,
                    std::vector<float> & embd,
                    uint32_t & n_tokens, uint32_t & n_embd,
                    bool & use_mrope_pos);

    bool store_key(const std::string & key,
                   uint32_t nx, uint32_t ny,
                   const float * embd,
                   uint32_t n_tokens, uint32_t n_embd,
                   bool use_mrope_pos);

    bool atomic_write_file(const std::string & path,
                           const mtmd_cache_header & header,
                           const float * embd, size_t embd_bytes);
};
