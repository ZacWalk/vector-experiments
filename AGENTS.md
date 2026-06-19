# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository.

## What this project is

A small, self-contained benchmark that **contrasts a scalar baseline against
several vectorised implementations** of two kernels, plus a sorting experiment:

- **distance** – sum of absolute differences between two 64-byte vectors.
- **crc32c** – CRC-32C (Castagnoli) over a buffer.
- **sorting** – `std::sort` (quicksort) vs radix / bitonic-network / k-way-merge.
  Two AVX2 variants (runtime-gated by `cpu().avx2`) beat `std::sort`:
  `sort_bitonic_avx2` (8 compare-exchanges per instruction) and `sort_merge_avx2`
  (a vectorised bottom-up merge sort with a bitonic 2-way merge).

It is a teaching/demo repo (originally an internal SIMD course). The point is the
*comparison*, so when changing code keep every contestant honest and comparable.

## Layout

| Path | Purpose |
| ---- | ------- |
| `src/platform.hpp` | Arch/compiler detection, timing, micro-benchmark barriers, CPU feature detection, `VEXP_*` macros. |
| `src/distance.hpp` | All distance implementations (scalar, autovec, portable vector-extensions, SSE2/AVX2/AVX-512, NEON). |
| `src/crc32c.hpp`   | CRC32C implementations (table, SSE4.2, ARM hardware). |
| `src/sorting.hpp`  | Sort experiment: std::sort vs radix / bitonic (scalar + AVX2) / k-way merge / AVX2 merge sort. |
| `src/main.cpp`     | Benchmark harness + Markdown table output. |
| `CMakeLists.txt`   | Cross-platform build (C++20). |
| `run.ps1`          | Configure (Release) + build + run, any platform with `pwsh`. |
| `.github/workflows/ci.yml` | Builds/tests on MSVC, GCC, Clang, Linux arm64, macOS arm64. |

There is **no** Visual Studio solution/project any more — the build is CMake only.

## Build, run, test

```pwsh
# One-shot (Release) build + run:
./run.ps1                      # full benchmark
./run.ps1 -Iterations 5000000  # quick run
./run.ps1 -Clean               # wipe build dir first

# Manual:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/vector-experiments [iterations]
```

The executable prints a Markdown table and **returns a non-zero exit code if any
implementation computes a wrong answer** — that is the regression test used by CI.

Supported toolchains: **MSVC, GCC, Clang**. Standard: **C++20**.

### Testing ARM / other compilers without that hardware

- This machine is Windows + MSVC. Use **WSL (Ubuntu)** for GCC/Clang x64:
  `wsl -d Ubuntu -- bash -lc "cd /mnt/c/code/vector-experiments && cmake -S . -B build-linux -DCMAKE_BUILD_TYPE=Release && cmake --build build-linux -j && ./build-linux/vector-experiments 2000000"`
  Keep a **separate build dir** (`build-linux`) from the Windows `build` dir.
- **ARM (NEON / vector-ext / ARM-CRC) is validated in CI** on `ubuntu-24.04-arm`
  and `macos-latest` (Apple silicon). You do not need a local ARM machine.

## Design rules / gotchas (read before editing SIMD code)

- **Portable SIMD = GCC/Clang `__attribute__((vector_size))`** (`distance_vext`).
  This is the "platform independent" path: one source, lowered to SSE/AVX on x86
  and NEON on ARM. It is **not available on MSVC** (guarded by
  `VEXP_HAVE_VECTOR_EXTENSIONS`).
- **Do NOT wrap `#include <immintrin.h>` in `#pragma GCC target(...)`.** Modern
  GCC/Clang already declare every intrinsic (and `__m256i`/`__m512i`) tagged with
  its own ISA requirement. Wrapping the include over-decorates the *baseline* SSE2
  intrinsics with AVX-512 targets, which then fail to inline into the SSE2 path
  (`error: target specific option mismatch`). Use a plain include; gate code-gen
  per function with `VEXP_TARGET("avx2")` etc.
- **Runtime dispatch:** higher-ISA paths are compiled with `VEXP_TARGET(...)` and
  only entered after a runtime CPU check (`vexp::cpu()`), so one binary runs
  everywhere. Never raise the whole-program baseline to AVX2/AVX-512.
- **ARM CRC intrinsics** need `__ARM_FEATURE_CRC32`. CMake adds `-march=armv8-a+crc`
  on non-Apple AArch64; Apple silicon and MSVC have it already. Execution is still
  gated by a runtime check.
- **Keep the scalar baseline scalar.** `distance_scalar` disables
  auto-vectorisation (`VEXP_NO_VECTORIZE_*`); `distance_autovec` is the *same loop*
  with vectorisation enabled. The pair shows what the auto-vectoriser buys.
- **Benchmark integrity:** the timing loop uses `vexp::clobber_memory()` /
  `vexp::do_not_optimize()` so the optimiser can't hoist loop-invariant work out
  or delete results. Keep them when touching `time_calls`.
- Allocation is `alignas(64)` + `new` (no `_aligned_malloc`); timing is
  `std::chrono` (no `QueryPerformanceCounter`). Keep it portable.

## Conventions

- Headers are self-contained and guard all arch/compiler specifics behind `VEXP_*`
  macros from `platform.hpp`.
- Everything lives in namespace `vexp`.
- Don't reintroduce Windows-only APIs into the hot path; OS calls are confined to
  `platform.hpp` behind `#if defined(_WIN32)` / POSIX branches.
