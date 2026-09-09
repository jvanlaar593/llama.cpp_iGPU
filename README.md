# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## About this fork

This is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) focused on
integrated GPUs, where the CPU and the GPU share one pool of system memory. Changes on top
of upstream:

- Weights are imported straight from the mmap-ed model file into Vulkan buffers on an iGPU,
  instead of being copied into a device buffer that lives in the same RAM. Saves a copy at
  load time and lowers peak memory.
- The Intel float `mul_mat_vec` shaders run at the native SIMD width. Worth about 30% of
  matvec throughput on Xe1 class hardware.
- `scripts/gguf_align_check.py` and `scripts/gguf_realign.py`, to check and fix the tensor
  alignment that the import path needs.

Everything below was verified on an Intel Core Ultra 9 285 (Arrow Lake-S, Xe-LPG iGPU),
64 GB DDR5-5600, Windows 11, Intel driver 32.0.101.8331.

### Step 1: install the toolchain

The Vulkan backend compiles its shaders during the build, so the Vulkan SDK is required and
not just the headers. On Windows:

```bat
:: MSVC, CMake and Ninja (CMake and Ninja ship inside the C++ workload)
winget install --id Microsoft.VisualStudio.BuildTools --accept-package-agreements --accept-source-agreements ^
  --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --quiet --wait --norestart"

:: glslc and SPIRV-Headers
winget install --id KhronosGroup.VulkanSDK --accept-package-agreements --accept-source-agreements
```

Both installers need administrator rights, so accept the UAC prompt. If Build Tools is
already present without the C++ workload, `winget` will try to upgrade it and ignore the
`--override`. Add the workload to the existing install instead:

```bat
"C:\Program Files (x86)\Microsoft Visual Studio\Installer\setup.exe" modify ^
  --installPath "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools" ^
  --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --quiet --wait --norestart
```

Check that everything is present before continuing. `cl.exe` is not on `PATH` until
`vcvars64.bat` runs, so look for the files rather than calling them:

```bat
dir "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC"
dir "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
dir "C:\VulkanSDK"
```

Version numbers differ between installs. Note the MSVC and Vulkan SDK versions you see,
they are needed in the next step.

### Step 2: set up the build environment

None of these tools land on `PATH`, and CMake needs `VULKAN_SDK`. Save this as
`llama-vk-env.bat` outside the repository and adjust the three paths to match your machine:

```bat
@echo off
REM Keep CRLF line endings in this file or cmd.exe misparses it.
setlocal
set "REPO=C:\dev\llama.cpp"
set "BT=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
set "VULKAN_SDK=C:\VulkanSDK\1.4.357.0"

REM vcvars64 shells out to vswhere.exe, which lives in the installer directory
set "VSINSTALLER=C:\Program Files (x86)\Microsoft Visual Studio\Installer"
if exist "%VSINSTALLER%\vswhere.exe" set "PATH=%VSINSTALLER%;%PATH%"

if not exist "%BT%\VC\Auxiliary\Build\vcvars64.bat" goto no_msvc
call "%BT%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 goto no_msvc
if not exist "%VULKAN_SDK%\Bin\glslc.exe" goto no_vulkan

set "PATH=%BT%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%BT%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%VULKAN_SDK%\Bin;%PATH%"
cd /d "%REPO%"

if "%~1"=="" goto shell
%*
exit /b %ERRORLEVEL%

:shell
echo Vulkan build environment ready in %CD%
cmd /k
exit /b 0

:no_msvc
echo ERROR: MSVC not found under "%BT%".
exit /b 1

:no_vulkan
echo ERROR: glslc not found under "%VULKAN_SDK%".
exit /b 1
```

Avoid `if ... (` blocks in this file. A batch file saved with Unix line endings misparses
them and fails with `) was unexpected at this time`, whereas `goto` labels are parsed one
line at a time and survive it.

Use it either as a wrapper for one command or as an interactive shell:

```bat
.\llama-vk-env.bat cmake --version
.\llama-vk-env.bat
```

Relative paths in the command you pass are resolved against `%REPO%`, not against the
directory you started from.

### Step 3: configure and build

```bat
llama-vk-env.bat cmake -B build-vk -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DLLAMA_CURL=OFF
llama-vk-env.bat cmake --build build-vk --target llama-cli llama-bench llama-server -j 24
```

The first build takes several minutes because every Vulkan shader is compiled; later builds
are incremental. Binaries land in `build-vk\bin\`. Two notes:

- `-DLLAMA_CURL=OFF` drops the curl dependency, which also disables `-hf` model downloads.
  Fetch GGUF files yourself, or leave it on if you have curl.
- Linking fails with `LNK1104: cannot open file 'bin\ggml-vulkan.dll'` if a llama.cpp
  process is still running. Stop it and build again.

### Step 4: confirm the iGPU is detected

```bat
build-vk\bin\llama-cli.exe --list-devices
```

Expected output names the integrated GPU, for example:

```
Available devices:
  Vulkan0: Intel(R) Graphics (36974 MiB, 47715 MiB free)
```

The reported total is the share of system RAM the driver exposes to the GPU, not a fixed
VRAM size. On some drivers the free figure exceeds the total; that is a driver quirk.

### Step 5: run a model

```bat
:: interactive chat, all layers on the iGPU
build-vk\bin\llama-cli.exe -m models\model.gguf -ngl 99

:: single prompt, deterministic
build-vk\bin\llama-cli.exe -m models\model.gguf -ngl 99 -p "Name three primary colors." -n 128 --single-turn --temp 0

:: OpenAI compatible server
build-vk\bin\llama-server.exe -m models\model.gguf -ngl 99 -c 8192 --host 127.0.0.1 --port 8080
```

Useful flags on an iGPU:

| flag | purpose |
|---|---|
| `-ngl N` | layers on the GPU. `-ngl 0` keeps everything on the CPU |
| `-c N` | context length |
| `-t N` | CPU threads for any layers left on the CPU |
| `-b N` / `-ub N` | logical and physical batch size |
| `--load-mode mmap` | map the weights. Add `mmap+mlock` to also lock them in RAM |
| `--load-mode none` | read weights into a device buffer, disabling the import path |
| `-ncmoe N` | keep the expert weights of N layers on the CPU, for MoE models |
| `-ot "<regex>=CPU"` | pin matching tensors to the CPU, e.g. `-ot "token_embd\.weight=CPU"` |

`-kvu`, `-np` and other server options are rejected by `llama-cli`; they only apply to
`llama-server`, `llama-bench` and the other examples that register them. Reasoning models
need a generous `-n`, otherwise the whole budget goes into the thinking block.

### Step 6: decide between the iGPU and the CPU

On a shared memory system the CPU and the iGPU compete for the same bandwidth, and which
one wins depends on the workload. Measure instead of guessing:

```bat
llama-vk-env.bat build-vk\bin\llama-bench.exe -m models\model.gguf -ngl 99 -t 9 -r 3
llama-vk-env.bat build-vk\bin\llama-bench.exe -m models\model.gguf -ngl 0  -t 24 -r 3
```

On the reference machine, with a 26B A4B MoE model at Q4_0, the iGPU is clearly better at
prompt processing and clearly worse at generation:

| configuration | prompt (t/s) | generation (t/s) |
|---|---|---|
| `-ngl 30 -t 9` (all layers on the iGPU) | 133.0 | 11.0 |
| `-ngl 0 -t 24` (CPU only) | 78.8 | 24.6 |
| `-ngl 30 -ncmoe 30 -t 9` (experts on the CPU) | 90.9 | 10.0 |

Forcing experts onto the CPU with `-ncmoe` only pays off when the weights do not fit in the
memory the driver exposes to the GPU. Here they did fit, so it lost 32% of prompt throughput
and gained nothing.

### Step 7: zero-copy weight loading

On an iGPU this fork imports the mmap-ed file directly, which needs every tensor offset to
be a multiple of the Vulkan `minStorageBufferOffsetAlignment`, commonly 64. GGUF pads tensor
data to `general.alignment`, 32 by default, so many published models do not qualify. When
that happens the import is skipped and loading falls back to the copy path, with no error.

Check a model before running it:

```bat
python scripts\gguf_align_check.py models\model.gguf 64
```

Rewrite one that does not qualify. Tensor payloads are copied unchanged, only the padding
and the offsets differ:

```bat
python scripts\gguf_realign.py models\model.gguf models\model-a64.gguf 64
```

Both scripts read the GGUF header directly and need no `numpy` or `gguf-py`.

There is no log line announcing the import. Confirm it by running with `-v` and checking
that the load mode is `mmap` and that no `host pointer import failed` warning appears, or by
comparing peak memory against `--load-mode none`. On the reference machine a 379 MiB model
peaked at 477 MiB with the import against 620 MiB without it.

### Step 8: tuning knob

`GGML_VK_MMV_SUBGROUP_SIZE` overrides the subgroup size of the `mul_mat_vec` shaders. The
fork already defaults to the native SIMD width, which is the best value for most types, but
the optimum depends on the quantization:

```bat
set GGML_VK_MMV_SUBGROUP_SIZE=32
llama-vk-env.bat build-vk\bin\llama-bench.exe -m models\model.gguf -ngl 99 -r 3
```

Measured per kernel on the reference iGPU, time for one `m=4096 n=1 k=14336` matvec:

| type | subgroup 8 | subgroup 16 | subgroup 32 |
|---|---|---|---|
| q4_0 | 685 us | 991 us | 783 us |
| q4_K | 989 us | 996 us | 862 us |
| q6_K | 1411 us | 1412 us | 2190 us |

Whole model benchmarks on this hardware carry about 13% run to run variance, enough to hide
an effect this size. To measure a single kernel instead, which is far more stable, use:

```bat
build-vk\bin\test-backend-ops.exe perf -b Vulkan0 -o MUL_MAT -p "type_a=q4_0,type_b=f32,m=4096,n=1,k=14336"
```

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
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
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
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
