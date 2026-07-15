#include "server-gpu-swap.h"

#include "ggml.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>

bool gpu_swap_manager::configure(
        struct llama_model * model_in,
        const std::vector<struct llama_context *> & contexts_in) {
    std::lock_guard<std::mutex> lock(mtx);

    if (!model_in || !llama_model_gpu_swap_supported(model_in)) {
        fprintf(stderr, "%s: model does not support safe single-device CUDA weight migration\n", __func__);
        return false;
    }

    std::vector<llama_context *> unique_contexts;
    for (llama_context * ctx : contexts_in) {
        if (!ctx || llama_get_model(ctx) != model_in) {
            continue;
        }
        if (std::find(unique_contexts.begin(), unique_contexts.end(), ctx) == unique_contexts.end()) {
            unique_contexts.push_back(ctx);
        }
    }
    if (unique_contexts.empty()) {
        fprintf(stderr, "%s: no llama context references the target model\n", __func__);
        return false;
    }

    model = model_in;
    contexts = std::move(unique_contexts);
    state = gpu_swap_state::MODEL_ON_GPU;
    schedulers_suspended = false;
    enabled = true;
    return true;
}

bool gpu_swap_manager::suspend_contexts_locked() {
    if (schedulers_suspended) {
        return true;
    }

    size_t suspended = 0;
    for (llama_context * ctx : contexts) {
        if (!llama_context_sched_suspend(ctx)) {
            for (size_t i = 0; i < suspended; ++i) {
                (void) llama_context_sched_resume(contexts[i]);
            }
            return false;
        }
        suspended++;
    }

    schedulers_suspended = true;
    return true;
}

bool gpu_swap_manager::resume_contexts_locked() {
    if (!schedulers_suspended) {
        return true;
    }

    bool ok = true;
    for (llama_context * ctx : contexts) {
        ok = llama_context_sched_resume(ctx) && ok;
    }
    if (ok) {
        schedulers_suspended = false;
    }
    return ok;
}

bool gpu_swap_manager::swap_to_mmproj_gpu(struct mtmd_context * mctx) {
    std::lock_guard<std::mutex> lock(mtx);

    if (!enabled || !model || !mctx) {
        return false;
    }
    if (state == gpu_swap_state::MMPROJ_ON_GPU) {
        return true;
    }
    if (state != gpu_swap_state::MODEL_ON_GPU) {
        fprintf(stderr, "%s: wrong state (expected MODEL_ON_GPU)\n", __func__);
        return false;
    }

    state = gpu_swap_state::TRANSITIONING;
    const int64_t t_start = ggml_time_us();

    if (!suspend_contexts_locked()) {
        fprintf(stderr, "%s: failed to suspend llama schedulers\n", __func__);
        state = gpu_swap_state::ERROR;
        return false;
    }

    if (!llama_model_gpu_swap_to_cpu(model)) {
        fprintf(stderr, "%s: failed to migrate model weights to CPU\n", __func__);
        const bool resumed = resume_contexts_locked();
        state = !llama_model_gpu_swap_active(model) && resumed
            ? gpu_swap_state::MODEL_ON_GPU
            : gpu_swap_state::ERROR;
        return false;
    }

    const size_t swapped_bytes = llama_model_gpu_swap_size(model);
    if (!mtmd_gpu_swap_upload(mctx)) {
        fprintf(stderr, "%s: failed to upload mmproj weights to GPU; restoring model\n", __func__);
        const bool model_restored = llama_model_gpu_swap_to_gpu(model);
        const bool schedulers_restored = resume_contexts_locked();
        state = model_restored && schedulers_restored
            ? gpu_swap_state::MODEL_ON_GPU
            : gpu_swap_state::ERROR;
        return false;
    }

    state = gpu_swap_state::MMPROJ_ON_GPU;
    const int64_t elapsed = ggml_time_us() - t_start;
    swap_stats.n_swap_model_to_cpu++;
    swap_stats.n_swap_mmproj_to_gpu++;
    swap_stats.t_swap_model_to_cpu_us += elapsed;
    swap_stats.t_swap_mmproj_to_gpu_us += elapsed;

    fprintf(stderr, "%s: model -> CPU, mmproj -> GPU (%.2f MiB, %" PRId64 " ms)\n",
        __func__, swapped_bytes / 1024.0 / 1024.0, elapsed / 1000);
    return true;
}

bool gpu_swap_manager::swap_to_model_gpu(struct mtmd_context * mctx) {
    std::lock_guard<std::mutex> lock(mtx);

    if (!enabled || !model || !mctx) {
        return false;
    }
    if (state == gpu_swap_state::MODEL_ON_GPU && !llama_model_gpu_swap_active(model)) {
        return true;
    }
    if (state != gpu_swap_state::MMPROJ_ON_GPU && state != gpu_swap_state::ERROR) {
        fprintf(stderr, "%s: wrong state (expected MMPROJ_ON_GPU)\n", __func__);
        return false;
    }

    state = gpu_swap_state::TRANSITIONING;
    const int64_t t_start = ggml_time_us();

    if (!mtmd_gpu_swap_download(mctx)) {
        fprintf(stderr, "%s: failed to release mmproj GPU weights\n", __func__);
        state = gpu_swap_state::ERROR;
        return false;
    }

    if (!llama_model_gpu_swap_to_gpu(model)) {
        fprintf(stderr, "%s: failed to restore all model weights to GPU\n", __func__);
        (void) resume_contexts_locked();
        state = gpu_swap_state::ERROR;
        return false;
    }

    if (!resume_contexts_locked()) {
        fprintf(stderr, "%s: failed to rebuild llama schedulers after model restore\n", __func__);
        state = gpu_swap_state::ERROR;
        return false;
    }

    state = gpu_swap_state::MODEL_ON_GPU;
    const int64_t elapsed = ggml_time_us() - t_start;
    swap_stats.n_swap_mmproj_to_cpu++;
    swap_stats.n_swap_model_to_gpu++;
    swap_stats.t_swap_mmproj_to_cpu_us += elapsed;
    swap_stats.t_swap_model_to_gpu_us += elapsed;

    fprintf(stderr, "%s: mmproj -> CPU, model -> GPU (%" PRId64 " ms)\n", __func__, elapsed / 1000);
    return true;
}

bool gpu_swap_manager::restore_for_shutdown(struct mtmd_context * mctx) {
    std::lock_guard<std::mutex> lock(mtx);

    bool ok = true;
    if (mctx && state != gpu_swap_state::MODEL_ON_GPU) {
        ok = mtmd_gpu_swap_download(mctx) && ok;
    }
    if (model && llama_model_gpu_swap_active(model)) {
        ok = llama_model_gpu_swap_to_gpu(model) && ok;
    }
    ok = resume_contexts_locked() && ok;

    state = ok ? gpu_swap_state::MODEL_ON_GPU : gpu_swap_state::ERROR;
    enabled = false;
    return ok;
}

bool gpu_swap_manager::is_enabled() const {
    std::lock_guard<std::mutex> lock(mtx);
    return enabled;
}

gpu_swap_state gpu_swap_manager::get_state() const {
    std::lock_guard<std::mutex> lock(mtx);
    return state;
}

void gpu_swap_manager::print_stats() const {
    std::lock_guard<std::mutex> lock(mtx);
    if (!enabled && swap_stats.n_swap_model_to_cpu == 0) {
        return;
    }

    fprintf(stderr, "%s: GPU swap statistics:\n", __func__);
    fprintf(stderr, "%s:   model->CPU swaps:  %d (total %" PRId64 " us)\n",
        __func__, swap_stats.n_swap_model_to_cpu, swap_stats.t_swap_model_to_cpu_us);
    fprintf(stderr, "%s:   mmproj->GPU swaps: %d (total %" PRId64 " us)\n",
        __func__, swap_stats.n_swap_mmproj_to_gpu, swap_stats.t_swap_mmproj_to_gpu_us);
    fprintf(stderr, "%s:   mmproj->CPU swaps: %d (total %" PRId64 " us)\n",
        __func__, swap_stats.n_swap_mmproj_to_cpu, swap_stats.t_swap_mmproj_to_cpu_us);
    fprintf(stderr, "%s:   model->GPU swaps:  %d (total %" PRId64 " us)\n",
        __func__, swap_stats.n_swap_model_to_gpu, swap_stats.t_swap_model_to_gpu_us);
}
