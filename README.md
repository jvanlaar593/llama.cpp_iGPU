# llama.cpp_iGPU

**A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) focused on integrated GPUs**, where the
CPU and the GPU share one pool of system memory.

This README covers only what is specific to the fork: how to build it, how to run it on an iGPU, and how to
configure and verify the changes made here. For everything else, including general usage, the other backends,
the server API and the model list, see upstream:

- [Upstream README](https://github.com/ggml-org/llama.cpp/blob/master/README.md)
- [Upstream build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [Upstream server docs](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [Upstream Vulkan notes](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#vulkan)

Upstream is MIT licensed and so is this fork. See [LICENSE](LICENSE) and upstream's acknowledgements for the
vendored third party libraries.

---

## What is different in this fork

| Change | Effect |
|---|---|
| Weights are imported directly from the mmap-ed model file into Vulkan buffers on an iGPU, rather than copied into a device buffer that lives in the same RAM | Removes one full copy at load time and lowers peak memory. Measured 477 MiB vs 620 MiB peak on a 379 MiB model |
| The Intel float `mul_mat_vec` shaders run at the native SIMD width instead of a fixed 16 | About 30% faster matvec on Xe1 class hardware. Measured +24% generation on a 26B A4B MoE model |
| `scripts/gguf_align_check.py` and `scripts/gguf_realign.py` | Check and fix the tensor alignment the import path requires |
| `GGML_VK_MMV_SUBGROUP_SIZE` environment variable | Override the `mul_mat_vec` subgroup size, since the best value depends on the quantization |

Backends other than Vulkan are untouched. Within Vulkan, discrete GPUs keep their existing behaviour; the import
path is gated on the device reporting itself as integrated.

### Reference hardware

Every number in this README was measured on:

- Intel Core Ultra 9 285 (Arrow Lake-S, 24 CPU cores), integrated Xe GPU, Vulkan device ID `0x7d67`
- 64 GB DDR5-5600 dual channel, 89.6 GB/s theoretical
- Windows 11, Intel Vulkan driver 32.0.101.8331, Vulkan 1.4.328
- 36 GB of system RAM exposed to the GPU by the driver

Results on other iGPUs will differ. Lunar Lake and later report a minimum subgroup size of 16 and are
unaffected by the subgroup change.

---

## Step 1: install the toolchain

The Vulkan backend compiles its compute shaders during the build, so the Vulkan SDK is required, not just the
headers. CMake and Ninja ship inside the Visual Studio C++ workload, so there is nothing else to install.

```bat
:: MSVC, CMake and Ninja
winget install --id Microsoft.VisualStudio.BuildTools --accept-package-agreements --accept-source-agreements ^
  --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --quiet --wait --norestart"

:: glslc and SPIRV-Headers
winget install --id KhronosGroup.VulkanSDK --accept-package-agreements --accept-source-agreements
```

Both installers need administrator rights, so accept the UAC prompt. A dismissed prompt shows up as
`Installer failed with exit code: 1602`.

If Visual Studio Build Tools is already installed but without the C++ workload, `winget` will try to *upgrade*
it and silently ignore the `--override`. Add the workload to the existing install instead:

```bat
"C:\Program Files (x86)\Microsoft Visual Studio\Installer\setup.exe" modify ^
  --installPath "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools" ^
  --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --quiet --wait --norestart
```

Confirm everything is present. `cl.exe` does not reach `PATH` until `vcvars64.bat` runs, so look for the files
rather than calling them:

```bat
dir "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC"
dir "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
dir "C:\VulkanSDK"
```

Note the MSVC and Vulkan SDK version numbers you see. They are needed in the next step.

On Linux, install a C++ compiler, CMake, Ninja and the Vulkan SDK through your package manager, then skip to
step 3. The Vulkan headers alone are not enough; `glslc` must be present.

## Step 2: set up the build environment

None of these tools land on `PATH`, and CMake needs `VULKAN_SDK` set. Save the following as `llama-vk-env.bat`
outside the repository, and adjust `REPO`, `BT` and `VULKAN_SDK` to match your machine.

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

Two things about this file. Keep the CRLF line endings: a batch file saved with Unix line endings fails with
`) was unexpected at this time`. And note it uses `goto` labels rather than `if ... (` blocks, because those
parenthesised blocks are exactly what breaks under Unix line endings.

Use it as a wrapper for a single command, or with no arguments for an interactive shell:

```bat
.\llama-vk-env.bat cmake --version
.\llama-vk-env.bat
```

Relative paths in the command you pass are resolved against `REPO`, not against the directory you ran it from.
Do not pass `%CD%` in arguments; it is expanded before the script changes directory.

## Step 3: configure and build

```bat
.\llama-vk-env.bat cmake -B build-vk -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DLLAMA_CURL=OFF
.\llama-vk-env.bat cmake --build build-vk --target llama-cli llama-bench llama-server -j 24
```

Adjust `-j` to your core count. The first build takes several minutes because every Vulkan shader is compiled;
later builds are incremental. Binaries land in `build-vk\bin\`.

A successful configure prints lines like these:

```
-- Found Vulkan: C:/VulkanSDK/1.4.357.0/Lib/vulkan-1.lib (found version "1.4.357") found components: glslc glslangValidator
-- Vulkan found
-- Including Vulkan backend
```

Notes:

- `-DLLAMA_CURL=OFF` drops the curl dependency, which also disables `-hf` model downloads. Leave it on if you
  have curl and want to pull models from Hugging Face directly.
- Add `-DLLAMA_BUILD_TESTS=ON` if you want `test-backend-ops`, used in step 7 and step 8.
- Linking fails with `LNK1104: cannot open file 'bin\ggml-vulkan.dll'` if a llama.cpp process is still running.
  Stop it and build again.

## Step 4: confirm the iGPU is detected

```bat
build-vk\bin\llama-cli.exe --list-devices
```

Expected output names the integrated GPU:

```
Available devices:
  Vulkan0: Intel(R) Graphics (36974 MiB, 47715 MiB free)
```

The total is the share of system RAM the driver exposes to the GPU, not a fixed VRAM size. On some drivers the
free figure exceeds the total; that is a driver reporting quirk, not a problem.

To see the capabilities the backend detected:

```bat
set GGML_VK_DEBUG=1
build-vk\bin\llama-bench.exe -m models\model.gguf -ngl 99 -p 0 -n 1 -r 1
```

```
ggml_vulkan: 0 = Intel(R) Graphics (Intel Corporation) | uma: 1 | fp16: 1 | bf16: 0 | fp4: 0 | warp size: 32 | shared memory: 32768 | int dot: 1 | matrix cores: none
```

`uma: 1` is what enables the shared memory paths in this fork. `matrix cores: none` means the driver exposes no
`VK_KHR_cooperative_matrix`, which is the case on this hardware and cannot be changed by configuration. Check
with `vulkaninfo | findstr cooperative_matrix` if you want to confirm it on your own device.

## Step 5: run a model

```bat
:: interactive chat, all layers on the iGPU
build-vk\bin\llama-cli.exe -m models\model.gguf -ngl 99

:: single prompt, deterministic
build-vk\bin\llama-cli.exe -m models\model.gguf -ngl 99 -p "Name three primary colors." -n 128 --single-turn --temp 0

:: OpenAI compatible server
build-vk\bin\llama-server.exe -m models\model.gguf -ngl 99 -c 8192 --host 127.0.0.1 --port 8080
```

Flags that matter on an iGPU:

| Flag | Purpose |
|---|---|
| `-ngl N` | layers on the GPU. `-ngl 0` keeps everything on the CPU |
| `-c N` | context length |
| `-t N` | CPU threads, for any layers left on the CPU |
| `-b N` / `-ub N` | logical and physical batch size |
| `--load-mode mmap` | map the weights, which is what the import path needs |
| `--load-mode mmap+mlock` | map and lock them in RAM |
| `--load-mode none` | read weights into a device buffer, disabling the import path |
| `-ncmoe N` | keep the expert weights of N layers on the CPU, for MoE models |
| `-ot "<regex>=CPU"` | pin matching tensors to the CPU, e.g. `-ot "token_embd\.weight=CPU"` |

Use `--load-mode mmap+mlock` rather than combining `--load-mode mmap` with `--mlock`. Both write the same
setting, so passing them separately silently loses one.

`-kvu`, `-np` and several other options are rejected by `llama-cli` with `invalid argument`; they are registered
only for `llama-server`, `llama-bench` and other examples. Reasoning models need a generous `-n`, otherwise the
whole token budget is consumed by the thinking block.

## Step 6: choose between the iGPU and the CPU

On a shared memory system both engines draw on the same bandwidth, and which one wins depends on the workload.
Measure rather than assume:

```bat
.\llama-vk-env.bat build-vk\bin\llama-bench.exe -m models\model.gguf -ngl 99 -t 9  -r 3
.\llama-vk-env.bat build-vk\bin\llama-bench.exe -m models\model.gguf -ngl 0  -t 24 -r 3
```

On the reference machine with a 26B A4B MoE model at Q4_0, 30 layers:

| Configuration | Prompt (t/s) | Generation (t/s) |
|---|---|---|
| `-ngl 30 -t 9`, all layers on the iGPU | **133.0** | 11.0 |
| `-ngl 0 -t 24`, CPU only | 78.8 | **24.6** |
| `-ngl 30 -ncmoe 30 -t 9`, experts on the CPU | 90.9 | 10.0 |
| `-ngl 30 -ncmoe 30 -t 24` | 84.7 | 10.7 |

The iGPU is 69% faster at prompt processing and the CPU is 2.2x faster at generation, so pick per workload:
the iGPU for long prompts, RAG and code review, the CPU for chat.

`-ncmoe` is a technique for when weights do not fit in the memory the driver exposes to the GPU. Here they did
fit, so forcing experts onto the CPU cost 32% of prompt throughput and gained nothing. Only reach for it if the
model does not otherwise fit.

Generation on this class of hardware is memory bandwidth bound, so more CPU threads help very little: 22.7 t/s
at 9 threads against 24.6 t/s at 24.

## Step 7: zero-copy weight loading

On an iGPU this fork imports the mmap-ed file directly as Vulkan buffers, using
`VK_EXT_external_memory_host`. Ranges larger than the device `maxBufferSize`, commonly 4 GiB, are imported as
overlapping chunks.

### The alignment requirement

Every tensor offset must be a multiple of the Vulkan `minStorageBufferOffsetAlignment`, which is 64 on the
reference hardware. GGUF pads tensor data to `general.alignment`, 32 by default, so **many published models do
not qualify**. When a model does not qualify the import is skipped and loading falls back to the copy path,
with no error and no log line.

Check a model before running it:

```bat
python scripts\gguf_align_check.py models\model.gguf 64
```

```
tensors         : 658
general.alignment: 32
misaligned      : 325 of 658

zero-copy import will NOT engage. First offenders:
  blk.0.post_attention_norm.weight   offset 1064755744 (mod 64 = 32)
  ...
```

Rewrite a model that does not qualify. Tensor payloads are copied unchanged; only the padding and the offsets
differ, and the file size is effectively identical:

```bat
python scripts\gguf_realign.py models\model.gguf models\model-a64.gguf 64
python scripts\gguf_align_check.py models\model-a64.gguf 64
```

Both scripts parse the GGUF header directly and need no `numpy` or `gguf-py`.

### Confirming the import engaged

There is no dedicated log line. Use any of these:

1. Run with `-v` and check the load mode is `mmap`. Without this fork an iGPU forces `none`.
2. Run with `-v` and check that no `ggml_vulkan: host pointer import failed` warning appears. A failed import
   aborts the load outright, so a model that loads at all took the path.
3. Compare peak memory against `--load-mode none`. On the reference machine a 379 MiB model peaked at 477 MiB
   with the import and 620 MiB without.

To disable the path entirely, pass `--load-mode none`.

### Note on mmap semantics

Importing requires a copy-on-write mapping, because drivers reject importing read-only pages. This fork maps
the file copy-on-write only when a device reports itself as an integrated GPU and supports host pointer import.
Metal, CUDA, CPU and BLAS keep read-only mappings.

## Step 8: tuning

### mul_mat_vec subgroup size

`GGML_VK_MMV_SUBGROUP_SIZE` overrides the subgroup size of the `mul_mat_vec` shaders. The fork already defaults
to the native SIMD width, which is the best value for most quantizations, but the optimum varies:

```bat
set GGML_VK_MMV_SUBGROUP_SIZE=32
.\llama-vk-env.bat build-vk\bin\llama-bench.exe -m models\model.gguf -ngl 99 -r 3
```

Time for one `m=4096 n=1 k=14336` matvec on the reference iGPU, lower is better:

| Type | subgroup 8 (default here) | subgroup 16 (upstream) | subgroup 32 |
|---|---|---|---|
| q4_0 | **685 us** | 991 us | 783 us |
| q4_K | 989 us | 996 us | **862 us** |
| q6_K | **1411 us** | 1412 us | 2190 us |
| f16 | 1654 us | 1622 us | 1626 us |

f16 is unaffected because it is bandwidth bound rather than dequantization bound, which is why it makes a good
control when testing.

### Measuring reliably

Whole model benchmarks on this hardware carry about 13% run to run variance, enough to hide an effect of this
size. Measure a single kernel instead:

```bat
build-vk\bin\test-backend-ops.exe perf -b Vulkan0 -o MUL_MAT -p "type_a=q4_0,type_b=f32,m=4096,n=1,k=14336"
```

That repeats one kernel about 1700 times and is stable to roughly 1 us. Requires
`-DLLAMA_BUILD_TESTS=ON` at configure time.

### Why quantized matvec is the bottleneck

For context on where the remaining headroom is: the reference iGPU reaches 69.8 GB/s on f16 matvec, 78% of the
89.6 GB/s theoretical peak and above what the CPU achieves. The quantized kernels reach only 26 to 33 GB/s
because they are dequantization ALU bound, not bandwidth bound. Time per kernel is nearly independent of bytes
read, which is the giveaway. The subgroup change above lifts q4_0 from 33 GB/s to 48 GB/s; closing the rest
would need shader work rather than parameter tuning.

## Step 9: verify a build

Correctness against the CPU backend:

```bat
build-vk\bin\test-backend-ops.exe test -b Vulkan0 -o MUL_MAT
build-vk\bin\test-backend-ops.exe test -b Vulkan0 -o MUL_MAT_ID
```

Both should report all tests passed. The full suite, `test-backend-ops test -b Vulkan0`, takes upwards of
15 minutes and occasionally reports a marginal `iq2_xxs` failure at roughly 0.0008 against a 0.0005 tolerance.
That is non-reproducible driver level numerical noise, present on unmodified upstream as well.

## Staying in sync with upstream

```bat
git remote add upstream https://github.com/ggml-org/llama.cpp.git
git fetch upstream
git rebase upstream/master
```

Rebuild and re-run step 9 afterwards. Upstream changes the Vulkan backend often, so a clean rebase is not by
itself evidence that the fork's changes still behave.

## Files changed by this fork

| File | Change |
|---|---|
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | host pointer import, chunking, registry integration, subgroup size, import error reporting |
| `src/llama-model.cpp` | alignment gate, copy-on-write mapping decision |
| `src/llama-model-loader.cpp` / `.h` | `mapping_is_aligned`, writable mapping flag |
| `src/llama-mmap.cpp` / `.h` | copy-on-write mapping support |
| `scripts/gguf_align_check.py` | report whether a GGUF meets a given tensor alignment |
| `scripts/gguf_realign.py` | rewrite a GGUF with a larger tensor alignment |
