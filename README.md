# llama.cpp (mmproj-gpu-swap fork)

> **本仓库是 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的功能增强分支，基于 `feat/mmproj-gpu-swap-cache` 分支。**
> 以下内容描述了本分支相对于官方版本的差异。官方版本的原始 README 见 [下方](#官方-readme)。

----

## 与官方版本的差异：mmproj GPU Swap 机制

### 概述

在单显卡环境下运行视觉语言模型（VLM）时，模型本体（~16GB）和 mmproj 视觉编码器（~1.7GB）通常无法同时放入显存。官方 llama.cpp 的处理方式是将 mmproj 放在 CPU 上推理，导致图像编码速度极慢（数十分钟 vs GPU 的数秒）。

本分支实现了 **mmproj GPU Swap** 机制：在文字推理时，模型本体占用 GPU；在图像推理时，将模型本体临时卸载到 CPU 内存，将 mmproj 加载到 GPU 进行推理；推理完成后，再换回模型本体。整个过程始终在 GPU 上执行，避免了 CPU 推理的性能惩罚。

### 工作原理

```
┌─────────────────────────────────────────────────────────┐
│                    GPU 显存 (24GB)                       │
│                                                         │
│  文字推理阶段:                                           │
│  ┌──────────────────────┐  ┌──────────────────────────┐ │
│  │   模型本体 (~16GB)    │  │ KV Cache + Compute (~7GB)│ │
│  └──────────────────────┘  └──────────────────────────┘ │
│                                                         │
│  图像推理阶段 (swap 后):                                  │
│  ┌──────────────────────┐  ┌──────────────────────────┐ │
│  │   mmproj (~1.7GB)    │  │ KV Cache + Compute (~7GB)│ │
│  └──────────────────────┘  └──────────────────────────┘ │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  CPU 内存                                │
│                                                         │
│  图像推理阶段:                                           │
│  ┌──────────────────────┐                               │
│  │  模型权重备份 (~16GB) │ ← 从显存卸载，推理完后换回     │
│  └──────────────────────┘                               │
└─────────────────────────────────────────────────────────┘
```

1. **文字推理**：模型本体在 GPU，mmproj 在 CPU 内存（不占显存）
2. **图像推理**：将模型权重从 GPU 卸载到 CPU → 将 mmproj 从 CPU 上传到 GPU → GPU 编码图像 → 将 mmproj 从 GPU 卸载 → 将模型权重从 CPU 加载回 GPU
3. **继续文字推理**：模型本体重新在 GPU 上运行

### 触发条件

GPU Swap 仅在显式传入 `--mmproj-gpu-swap` 时启用，不再根据 GGUF 文件大小或模型类型自动判断。该模式要求单个 CUDA 设备、单并发，以及可安全迁移的模型权重 buffer；条件不满足时服务端会拒绝启动，而不是静默退回不完整的 swap 状态。

对于模型和 mmproj 可以同时放入显存的场景，通常无需启用该参数；是否启用应由实际显存预算决定，文件大小不能准确代表运行时显存占用。

### 命令行参数

| 参数 | 说明 |
|------|------|
| `--mmproj-gpu-swap` | 启用 mmproj GPU swap 模式 |
| `--mmproj <path>` | 指定 mmproj 模型文件路径 |
| `--mmproj-cache-dir <dir>` | 图像 token 化缓存目录，避免重复编码同一图像 |

**注意**：启用 `--mmproj-gpu-swap` 时，`n_parallel` 会被自动强制为 1（单并发），以减少 KV Cache 显存占用并避免并发 GPU 状态冲突。

### 使用示例

```sh
llama-server \
  -m Qwen3.6-27B-UD-Q4_K_XL.gguf \
  --mmproj mmproj-F32.gguf \
  --mmproj-gpu-swap \
  -ngl 99 \
  -c 180000 \
  --mmproj-cache-dir ./image-cache
```

### 性能对比

| 场景 | 无 GPU Swap (CPU推理) | 有 GPU Swap (GPU推理) |
|------|----------------------|----------------------|
| 图像编码 (1472x1472) | ~30 分钟 | ~6 秒 |
| Swap 开销 | 无 | ~5 秒 (卸载+上传+回载) |
| 每次图像请求总耗时 | ~30 分钟 | ~11 秒 |

### 限制

- 仅支持单 GPU、单并发场景
- Swap 过程需要 ~5 秒的显存搬运开销
- 需要足够的 CPU 内存来缓存模型权重备份（约等于模型文件大小）
- 目前仅支持 NVIDIA CUDA GPU

### 实现细节

核心修改涉及以下文件：

- **`tools/server/server-context.cpp`**：模型加载时处理显式 swap 配置、校验运行条件，并在 context 创建前强制 `n_parallel=1`
- **`tools/server/server-gpu-swap.cpp`**：`swap_to_mmproj_gpu()` 和 `swap_to_model_gpu()` 实现模型权重与 mmproj 在 GPU/CPU 间的动态交换
- **`tools/mtmd/clip.cpp`**：`clip_gpu_upload()` / `clip_gpu_download()` 实现 mmproj 权重在 GPU/CPU 间的上传/下载；GPU swap 模式下初始化 CUDA backend 并将 mmproj 加载到 CPU
- **`tools/mtmd/mtmd.cpp`**：将 `gpu_swap_mode` 从 `mtmd_context_params` 正确传递给 `clip_context_params`

----

<h2 id="官方-readme">官方 README</h2>

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp)](https://github.com/ggml-org/llama.cpp/releases)
[![Server](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3A0cc4m%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [compile times](https://github.com/ggml-org/llama.cpp-dev/blob/master/README-compile-times.md) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
