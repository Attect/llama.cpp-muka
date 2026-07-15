#pragma once

#include "llama.h"
#include "mtmd.h"

#include <cstdint>
#include <mutex>
#include <vector>

enum class gpu_swap_state {
    MODEL_ON_GPU,
    MMPROJ_ON_GPU,
    TRANSITIONING,
    ERROR,
};

struct gpu_swap_manager {
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

    bool configure(struct llama_model * model, const std::vector<struct llama_context *> & contexts);

    bool swap_to_mmproj_gpu(struct mtmd_context * mctx);
    bool swap_to_model_gpu(struct mtmd_context * mctx);

    // Best-effort recovery used before the server frees model/context objects.
    bool restore_for_shutdown(struct mtmd_context * mctx);

    bool is_enabled() const;
    gpu_swap_state get_state() const;
    void print_stats() const;

private:
    bool suspend_contexts_locked();
    bool resume_contexts_locked();

    mutable std::mutex mtx;
    gpu_swap_state state = gpu_swap_state::MODEL_ON_GPU;
    bool enabled = false;
    bool schedulers_suspended = false;

    struct llama_model * model = nullptr;
    std::vector<struct llama_context *> contexts;
    stats swap_stats;
};
