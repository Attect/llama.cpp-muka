#include "server-gpu-swap.h"

#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <vector>

// Use the existing internal API to access model tensors
// Helper to iterate over model tensors using public API
// Returns a vector of (name, tensor*) pairs
static std::vector<std::pair<std::string, struct ggml_tensor *>> get_model_tensors(const struct llama_model * model) {
    std::vector<std::pair<std::string, struct ggml_tensor *>> result;
    size_t n = llama_model_n_tensors(model);
    for (size_t i = 0; i < n; i++) {
        const char * name = llama_model_get_tensor_name(model, i);
        if (!name) break;
        struct ggml_tensor * tensor = llama_model_get_tensor(model, name);
        if (tensor) {
            result.emplace_back(name, tensor);
        }
    }
    return result;
}

//
// gpu_swap_manager implementation
//

gpu_swap_state gpu_swap_manager::get_state() const {
    return state;
}

void gpu_swap_manager::print_stats() const {
    if (!enabled) return;

    fprintf(stderr, "%s: GPU swap statistics:\n", __func__);
    fprintf(stderr, "%s:   model→CPU swaps:  %d (total %" PRId64 " us)\n",
            __func__, swap_stats.n_swap_model_to_cpu, swap_stats.t_swap_model_to_cpu_us);
    fprintf(stderr, "%s:   mmproj→GPU swaps: %d (total %" PRId64 " us)\n",
            __func__, swap_stats.n_swap_mmproj_to_gpu, swap_stats.t_swap_mmproj_to_gpu_us);
    fprintf(stderr, "%s:   mmproj→CPU swaps: %d (total %" PRId64 " us)\n",
            __func__, swap_stats.n_swap_mmproj_to_cpu, swap_stats.t_swap_mmproj_to_cpu_us);
    fprintf(stderr, "%s:   model→GPU swaps:  %d (total %" PRId64 " us)\n",
            __func__, swap_stats.n_swap_model_to_gpu, swap_stats.t_swap_model_to_gpu_us);
}

bool gpu_swap_manager::swap_to_mmproj_gpu(struct mtmd_context * mctx,
                                           struct llama_model * model,
                                           struct llama_context * lctx) {
    std::lock_guard<std::mutex> lock(mtx);

    if (state == gpu_swap_state::ERROR) {
        fprintf(stderr, "%s: swap manager is in ERROR state, refusing operation\n", __func__);
        return false;
    }

    if (state != gpu_swap_state::MODEL_ON_GPU) {
        fprintf(stderr, "%s: wrong state (expected MODEL_ON_GPU)\n", __func__);
        return false;
    }

    state = gpu_swap_state::TRANSITIONING;

    const int64_t t_start = ggml_time_us();

    // ===== Step 1: Get all model tensors =====
    auto tensors = get_model_tensors(model);

    // ===== Step 2: Group GPU tensors by their buffer =====
    // We only offload tensors that are on non-host (GPU) buffers
    std::unordered_map<ggml_backend_buffer_t, std::vector<size_t>> gpu_buffer_groups;
    // Map: buffer pointer -> list of indices into tensors vector

    for (size_t i = 0; i < tensors.size(); i++) {
        struct ggml_tensor * tensor = tensors[i].second;
        if (!tensor || !tensor->buffer) continue;

        // Skip CPU (host) buffers - these don't need to be offloaded
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(tensor->buffer);
        if (ggml_backend_buft_is_host(buft)) continue;

        gpu_buffer_groups[tensor->buffer].push_back(i);
    }

    if (gpu_buffer_groups.empty()) {
        // No GPU tensors found - model is already on CPU
        fprintf(stderr, "%s: no GPU tensors found, model appears to be on CPU already\n", __func__);
        // Still proceed with clip upload
    }

    // ===== Step 3: Offload each GPU buffer group to CPU =====
    fprintf(stderr, "%s: found %zu GPU buffer groups to offload\n", __func__, gpu_buffer_groups.size());

    for (auto & [gpu_buf, tensor_indices] : gpu_buffer_groups) {
        model_cpu_backup backup;
        backup.gpu_buf = gpu_buf;

        // Save GPU buffer type for later reload
        ggml_backend_buffer_type_t gpu_buft = ggml_backend_buffer_get_type(gpu_buf);
        backup.gpu_buft = gpu_buft;

        // Log buffer info
        const char * buft_name = ggml_backend_buft_name(gpu_buft);
        size_t buf_size = ggml_backend_buffer_get_size(gpu_buf);
        fprintf(stderr, "%s: offloading GPU buffer: name=%s, size=%.2f MiB, n_tensors=%zu\n",
                __func__, buft_name ? buft_name : "unknown", buf_size / 1024.0 / 1024.0, tensor_indices.size());

        // Save the first encountered GPU buffer type as the model's GPU buffer type
        if (!model_gpu_buft) {
            model_gpu_buft = gpu_buft;
        }

        // Calculate total size with alignment for both CPU buffer and data backup
        size_t total_size = 0;
        size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());

        for (size_t idx : tensor_indices) {
            struct ggml_tensor * tensor = tensors[idx].second;
            size_t nbytes = ggml_nbytes(tensor);
            size_t aligned_offset = GGML_PAD(total_size, alignment);

            model_cpu_backup::tensor_info info;
            info.name = tensors[idx].first;
            info.offset = aligned_offset;
            info.size = nbytes;
            backup.tensors.push_back(info);

            total_size = aligned_offset + nbytes;
        }

        if (total_size == 0) continue;

        // Allocate CPU memory for backup data
        backup.data.resize(total_size, 0);

        // Copy GPU data to CPU backup memory
        for (size_t i = 0; i < tensor_indices.size(); i++) {
            struct ggml_tensor * tensor = tensors[tensor_indices[i]].second;
            const auto & info = backup.tensors[i];
            // Read from GPU buffer into our CPU backup
            ggml_backend_tensor_get(tensor, backup.data.data() + info.offset, 0, info.size);
        }

        // Detach all tensors from GPU buffer before freeing it
        // ggml_backend_tensor_alloc requires tensor->buffer == nullptr and tensor->data == nullptr
        for (size_t idx : tensor_indices) {
            struct ggml_tensor * tensor = tensors[idx].second;
            tensor->buffer = nullptr;
            tensor->data = nullptr;
        }

        // Free the GPU buffer - all data has been safely copied to backup.data
        ggml_backend_buffer_free(gpu_buf);

        // Allocate CPU buffer to host the offloaded tensors
        // This keeps the tensors in a valid buffer so the scheduler can handle them
        ggml_backend_buffer_t cpu_buf = ggml_backend_buft_alloc_buffer(
            ggml_backend_cpu_buffer_type(), total_size);
        if (!cpu_buf) {
            fprintf(stderr, "%s: failed to allocate CPU buffer (%zu bytes), attempting GPU buffer recovery\n", __func__, total_size);
            // Critical: GPU buffer is already freed but CPU buffer allocation failed
            // Attempt recovery: re-allocate GPU buffer and restore tensors from backup data
            size_t gpu_total_size = 0;
            size_t gpu_alignment = ggml_backend_buft_get_alignment(backup.gpu_buft);
            for (const auto & info : backup.tensors) {
                size_t aligned_offset = GGML_PAD(gpu_total_size, gpu_alignment);
                gpu_total_size = aligned_offset + info.size;
            }

            ggml_backend_buffer_t recovery_buf = ggml_backend_buft_alloc_buffer(backup.gpu_buft, gpu_total_size);
            if (!recovery_buf) {
                fprintf(stderr, "%s: CRITICAL: GPU buffer recovery also failed, entering ERROR state\n", __func__);
                // Both CPU and GPU allocation failed - tensors are orphaned with no valid buffer
                model_backups.push_back(std::move(backup));
                state = gpu_swap_state::ERROR;
                return false;
            }

            ggml_backend_buffer_set_usage(recovery_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            void * recovery_base = ggml_backend_buffer_get_base(recovery_buf);

            // Re-assign tensors to recovered GPU buffer with GPU-aligned offsets
            size_t cur_offset = 0;
            for (size_t i = 0; i < tensor_indices.size(); i++) {
                struct ggml_tensor * tensor = tensors[tensor_indices[i]].second;
                size_t aligned_pos = GGML_PAD(cur_offset, gpu_alignment);
                ggml_backend_tensor_alloc(recovery_buf, tensor, (uint8_t *)recovery_base + aligned_pos);
                cur_offset = aligned_pos + backup.tensors[i].size;
            }

            // Copy data from backup to recovered GPU buffer
            for (size_t i = 0; i < tensor_indices.size(); i++) {
                struct ggml_tensor * tensor = tensors[tensor_indices[i]].second;
                const auto & info = backup.tensors[i];
                ggml_backend_tensor_set(tensor, backup.data.data() + info.offset, 0, info.size);
            }

            fprintf(stderr, "%s: GPU buffer recovery succeeded, tensors restored to GPU\n", __func__);
            // Tensors are back on GPU, skip this backup group (don't push backup)
            continue;
        }

        ggml_backend_buffer_set_usage(cpu_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        // Assign tensors to CPU buffer
        void * cpu_base = ggml_backend_buffer_get_base(cpu_buf);
        for (size_t i = 0; i < tensor_indices.size(); i++) {
            struct ggml_tensor * tensor = tensors[tensor_indices[i]].second;
            const auto & info = backup.tensors[i];

            // Allocate tensor in CPU buffer at the correct offset
            ggml_backend_tensor_alloc(cpu_buf, tensor, (uint8_t *)cpu_base + info.offset);
        }

        // Copy data from backup to CPU buffer
        // For host (CPU) buffers, ggml_backend_tensor_set copies data into the buffer
        for (size_t i = 0; i < tensor_indices.size(); i++) {
            struct ggml_tensor * tensor = tensors[tensor_indices[i]].second;
            const auto & info = backup.tensors[i];
            ggml_backend_tensor_set(tensor, backup.data.data() + info.offset, 0, info.size);
        }

        // Save CPU buffer reference for cleanup during reload
        backup.cpu_buf = cpu_buf;

        // Save backup
        model_backups.push_back(std::move(backup));
    }

    // ===== Step 4: Upload clip model to GPU =====
    if (mctx) {
        if (!mtmd_gpu_swap_upload(mctx)) {
            fprintf(stderr, "%s: failed to upload clip to GPU, rolling back model offload\n", __func__);

            // Rollback: reload model from CPU back to GPU
            // Build a name -> tensor pointer map for quick lookup during rollback
            std::unordered_map<std::string, struct ggml_tensor *> rollback_tensor_map;
            for (const auto & [name, tensor] : tensors) {
                rollback_tensor_map[name] = tensor;
            }

            bool rollback_ok = true;
            for (auto & backup : model_backups) {
                ggml_backend_buffer_type_t rb_gpu_buft = backup.gpu_buft;
                if (!rb_gpu_buft) rb_gpu_buft = model_gpu_buft;
                if (!rb_gpu_buft) {
                    fprintf(stderr, "%s: no GPU buffer type available for rollback\n", __func__);
                    rollback_ok = false;
                    break;
                }

                // Calculate GPU-aligned total size for rollback buffer
                size_t rb_total_size = 0;
                size_t rb_alignment = ggml_backend_buft_get_alignment(rb_gpu_buft);
                for (const auto & info : backup.tensors) {
                    size_t aligned_offset = GGML_PAD(rb_total_size, rb_alignment);
                    rb_total_size = aligned_offset + info.size;
                }
                if (rb_total_size == 0) continue;

                ggml_backend_buffer_t rb_gpu_buf = ggml_backend_buft_alloc_buffer(rb_gpu_buft, rb_total_size);
                if (!rb_gpu_buf) {
                    fprintf(stderr, "%s: failed to allocate GPU buffer for rollback\n", __func__);
                    rollback_ok = false;
                    break;
                }

                ggml_backend_buffer_set_usage(rb_gpu_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                void * rb_base = ggml_backend_buffer_get_base(rb_gpu_buf);

                // Detach tensors from CPU buffer and assign to rollback GPU buffer
                size_t rb_offset = 0;
                for (const auto & info : backup.tensors) {
                    auto it = rollback_tensor_map.find(info.name);
                    if (it == rollback_tensor_map.end()) continue;

                    struct ggml_tensor * tensor = it->second;
                    tensor->buffer = nullptr;
                    tensor->data = nullptr;

                    size_t aligned_pos = GGML_PAD(rb_offset, rb_alignment);
                    ggml_backend_tensor_alloc(rb_gpu_buf, tensor, (uint8_t *)rb_base + aligned_pos);
                    rb_offset = aligned_pos + info.size;
                }

                // Copy data from backup to rollback GPU buffer
                for (const auto & info : backup.tensors) {
                    auto it = rollback_tensor_map.find(info.name);
                    if (it == rollback_tensor_map.end()) continue;

                    struct ggml_tensor * tensor = it->second;
                    ggml_backend_tensor_set(tensor, backup.data.data() + info.offset, 0, info.size);
                }

                // Free CPU buffer
                if (backup.cpu_buf) {
                    ggml_backend_buffer_free(backup.cpu_buf);
                    backup.cpu_buf = nullptr;
                }
            }

            if (rollback_ok) {
                model_backups.clear();
                state = gpu_swap_state::MODEL_ON_GPU;
                if (lctx) {
                    llama_context_sched_update(lctx);
                }
                fprintf(stderr, "%s: model rollback to GPU succeeded\n", __func__);
            } else {
                fprintf(stderr, "%s: CRITICAL: model rollback to GPU failed, entering ERROR state\n", __func__);
                state = gpu_swap_state::ERROR;
            }
            return false;
        }
    }

    // ===== Step 5: Reset the backend scheduler =====
    // After changing tensor locations, the scheduler needs to be reset
    // so it re-evaluates which backend handles each operation
    if (lctx) {
        llama_context_sched_update(lctx);
    }

    // ===== Step 6: Update state and stats =====
    state = gpu_swap_state::MMPROJ_ON_GPU;

    const int64_t t_end = ggml_time_us();
    swap_stats.n_swap_model_to_cpu++;
    swap_stats.n_swap_mmproj_to_gpu++;
    swap_stats.t_swap_model_to_cpu_us += (t_end - t_start);
    swap_stats.t_swap_mmproj_to_gpu_us += (t_end - t_start);

    fprintf(stderr, "%s: model offloaded to CPU, mmproj uploaded to GPU (%.2f MiB offloaded, %" PRId64 " ms)\n",
            __func__,
            (double)(model_backups.empty() ? 0 : model_backups.back().data.size()) / (1024.0 * 1024.0),
            (t_end - t_start) / 1000);

    return true;
}

bool gpu_swap_manager::swap_to_model_gpu(struct mtmd_context * mctx,
                                           struct llama_model * model,
                                           struct llama_context * lctx) {
    std::lock_guard<std::mutex> lock(mtx);

    if (state == gpu_swap_state::ERROR) {
        fprintf(stderr, "%s: swap manager is in ERROR state, refusing operation\n", __func__);
        return false;
    }

    if (state != gpu_swap_state::MMPROJ_ON_GPU) {
        fprintf(stderr, "%s: wrong state (expected MMPROJ_ON_GPU)\n", __func__);
        return false;
    }

    state = gpu_swap_state::TRANSITIONING;

    const int64_t t_start = ggml_time_us();

    // ===== Step 1: Download clip model from GPU =====
    if (mctx) {
        if (!mtmd_gpu_swap_download(mctx)) {
            fprintf(stderr, "%s: failed to download clip from GPU\n", __func__);
            // Continue anyway - we need to get the model back on GPU
        }
    }

    // ===== Step 2: Reload model from CPU backup to GPU =====
    auto tensors = get_model_tensors(model);

    // Build a name -> tensor pointer map for quick lookup
    std::unordered_map<std::string, struct ggml_tensor *> tensor_map;
    for (const auto & [name, tensor] : tensors) {
        tensor_map[name] = tensor;
    }

    // Sort backups by size (largest first) to minimize peak GPU memory usage
    // Process the largest buffer first, free its CPU backup, then process the next
    std::vector<size_t> backup_order(model_backups.size());
    for (size_t i = 0; i < model_backups.size(); i++) backup_order[i] = i;
    std::sort(backup_order.begin(), backup_order.end(), [this](size_t a, size_t b) {
        size_t size_a = 0, size_b = 0;
        for (const auto & info : model_backups[a].tensors) size_a += info.size;
        for (const auto & info : model_backups[b].tensors) size_b += info.size;
        return size_a > size_b; // largest first
    });

    for (size_t bi = 0; bi < backup_order.size(); bi++) {
        auto & backup = model_backups[backup_order[bi]];
        ggml_backend_buffer_type_t gpu_buft = backup.gpu_buft;
        if (!gpu_buft) {
            fprintf(stderr, "%s: no GPU buffer type saved, using model_gpu_buft\n", __func__);
            gpu_buft = model_gpu_buft;
        }
        if (!gpu_buft) {
            fprintf(stderr, "%s: no GPU buffer type available, cannot reload model\n", __func__);
            state = gpu_swap_state::MMPROJ_ON_GPU;
            return false;
        }

        // If the saved gpu_buft is a split buffer type, use the regular GPU buffer type instead.
        // Split buffers are only needed for multi-GPU row-split mode. When reloading,
        // using the regular CUDA buffer avoids the split buffer's extra data_device allocation
        // which can cause OOM during set_tensor.
        if (model_gpu_buft && gpu_buft != model_gpu_buft) {
            const char * buft_name = ggml_backend_buft_name(gpu_buft);
            const char * main_buft_name = ggml_backend_buft_name(model_gpu_buft);
            fprintf(stderr, "%s: replacing split buffer type '%s' with main GPU buffer type '%s' to avoid OOM\n",
                    __func__, buft_name ? buft_name : "unknown", main_buft_name ? main_buft_name : "unknown");
            gpu_buft = model_gpu_buft;
        }

        // Calculate total size needed for GPU buffer
        size_t total_size = 0;
        size_t alignment = ggml_backend_buft_get_alignment(gpu_buft);

        // Recalculate offsets with GPU alignment (may differ from CPU alignment)
        std::vector<size_t> gpu_offsets(backup.tensors.size());
        for (size_t i = 0; i < backup.tensors.size(); i++) {
            size_t aligned_offset = GGML_PAD(total_size, alignment);
            gpu_offsets[i] = aligned_offset;
            total_size = aligned_offset + backup.tensors[i].size;
        }

        if (total_size == 0) continue;

        // Allocate GPU buffer
        ggml_backend_buffer_t gpu_buf = ggml_backend_buft_alloc_buffer(gpu_buft, total_size);
        if (!gpu_buf) {
            fprintf(stderr, "%s: CRITICAL: failed to allocate GPU buffer (%zu bytes)\n", __func__, total_size);
            fprintf(stderr, "%s: model remains on CPU, entering ERROR state\n", __func__);
            // Model tensors are still on CPU buffers (valid), but cannot be moved to GPU
            // Enter ERROR state to prevent further swap operations that would be inconsistent
            state = gpu_swap_state::ERROR;
            return false;
        }

        ggml_backend_buffer_set_usage(gpu_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        void * gpu_base = ggml_backend_buffer_get_base(gpu_buf);

        // Detach tensors from CPU buffer and assign to GPU buffer
        for (size_t i = 0; i < backup.tensors.size(); i++) {
            const auto & info = backup.tensors[i];
            auto it = tensor_map.find(info.name);
            if (it == tensor_map.end()) {
                fprintf(stderr, "%s: tensor '%s' not found in model\n", __func__, info.name.c_str());
                continue;
            }

            struct ggml_tensor * tensor = it->second;

            // Detach from current (CPU) buffer
            // ggml_backend_tensor_alloc requires buffer == nullptr and data == nullptr
            tensor->buffer = nullptr;
            tensor->data = nullptr;

            // Assign to GPU buffer
            ggml_backend_tensor_alloc(gpu_buf, tensor, (uint8_t *)gpu_base + gpu_offsets[i]);
        }

        // Copy data from backup to GPU buffer
        for (size_t i = 0; i < backup.tensors.size(); i++) {
            const auto & info = backup.tensors[i];
            auto it = tensor_map.find(info.name);
            if (it == tensor_map.end()) continue;

            struct ggml_tensor * tensor = it->second;
            // Copy from CPU backup data to GPU buffer
            ggml_backend_tensor_set(tensor, backup.data.data() + info.offset, 0, info.size);
        }

        // Free CPU buffer (if it exists)
        if (backup.cpu_buf) {
            ggml_backend_buffer_free(backup.cpu_buf);
            backup.cpu_buf = nullptr;
        }

        // Clear backup data to free memory
        backup.data.clear();
        backup.data.shrink_to_fit();

        // After processing each backup, update the scheduler to free
        // any stale compute buffers and allow the next allocation
        if (bi + 1 < backup_order.size() && lctx) {
            llama_context_sched_update(lctx);
        }
    }

    // Clear all backups
    model_backups.clear();

    // ===== Step 3: Reset the backend scheduler =====
    if (lctx) {
        llama_context_sched_update(lctx);
    }

    // ===== Step 4: Update state and stats =====
    state = gpu_swap_state::MODEL_ON_GPU;

    const int64_t t_end = ggml_time_us();
    swap_stats.n_swap_mmproj_to_cpu++;
    swap_stats.n_swap_model_to_gpu++;
    swap_stats.t_swap_mmproj_to_cpu_us += (t_end - t_start);
    swap_stats.t_swap_model_to_gpu_us += (t_end - t_start);

    fprintf(stderr, "%s: model reloaded to GPU, mmproj offloaded (%" PRId64 " ms)\n",
            __func__, (t_end - t_start) / 1000);

    return true;
}
