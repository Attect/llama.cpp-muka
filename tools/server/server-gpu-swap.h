#pragma once

#include "ggml.h"
#include "llama.h"
#include "mtmd.h"

#include <mutex>
#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>

enum class gpu_swap_state {
    MODEL_ON_GPU,    // Model on GPU (text inference ready)
    MMPROJ_ON_GPU,   // Visual projector on GPU (image inference ready)
    TRANSITIONING,   // Switching state
    ERROR,           // Error state: swap operation failed and cannot recover, prohibits further swap operations
};

struct gpu_swap_manager {
    gpu_swap_state state = gpu_swap_state::MODEL_ON_GPU;
    bool enabled = false;
    std::mutex mtx;

    // Model CPU backup data (saved when offloading model from GPU)
    struct model_cpu_backup {
        ggml_backend_buffer_t gpu_buf = nullptr;        // Original GPU buffer reference (freed after offload)
        ggml_backend_buffer_t cpu_buf = nullptr;        // CPU buffer allocated during offload (freed during reload)
        ggml_backend_buffer_type_t gpu_buft = nullptr;  // GPU buffer type (needed for reload)
        std::vector<uint8_t> data;                       // CPU data copy (safety backup for reload)
        struct tensor_info {
            std::string name;
            size_t offset;
            size_t size;
        };
        std::vector<tensor_info> tensors;
    };
    std::vector<model_cpu_backup> model_backups;

    // Performance stats
    struct stats {
        int n_swap_model_to_cpu = 0;
        int n_swap_mmproj_to_gpu = 0;
        int n_swap_mmproj_to_cpu = 0;
        int n_swap_model_to_gpu = 0;
        int64_t t_swap_model_to_cpu_us = 0;
        int64_t t_swap_mmproj_to_gpu_us = 0;
        int64_t t_swap_mmproj_to_cpu_us = 0;
        int64_t t_swap_model_to_gpu_us = 0;
    };
    stats swap_stats;

    // GPU backend for model reloading (set during initialization)
    ggml_backend_t model_gpu_backend = nullptr;
    ggml_backend_buffer_type_t model_gpu_buft = nullptr;

    bool swap_to_mmproj_gpu(struct mtmd_context * mctx,
                            struct llama_model * model,
                            struct llama_context * lctx);

    bool swap_to_model_gpu(struct mtmd_context * mctx,
                           struct llama_model * model,
                           struct llama_context * lctx);

    gpu_swap_state get_state() const;
    void print_stats() const;
};
