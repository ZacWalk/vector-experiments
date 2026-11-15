# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository.
[README.md](README.md) explains the experiments and results, how the harness
measures, and how to build and run it. The point is the *comparison*, so keep
every contestant honest and comparable.

## Verify

- `./dd.ps1 run -- 50` builds Release and runs a quick benchmark, which exits
  non-zero if any implementation computes a wrong answer. That is the
  regression test. `./dd.ps1 test --label vector-experiments` only checks that
  the project builds.
- On Windows, use WSL (Ubuntu) for GCC and Clang on x64, in a separate build
  directory from the Windows `build`:

  ```pwsh
  wsl -d Ubuntu -- bash -lc 'cmake -S . -B build-linux -DCMAKE_BUILD_TYPE=Release && cmake --build build-linux -j && ./build-linux/vector-experiments 300'
  ```

- ARM (NEON, vector extensions, ARM CRC) is built and run in CI on
  `ubuntu-24.04-arm` and `macos-latest` (Apple silicon); no local ARM machine
  is needed.
- `dd.ps1` and `.dd/` are vendored [dd](https://github.com/ZacWalk/dd), with
  hashes in [docs/dd-upstream.json](docs/dd-upstream.json); do not edit them.
  Project settings live in `dd.psd1`. dd's
  [two modes](https://github.com/ZacWalk/dd/blob/main/docs/mcp.md#two-modes)
  and [commands](https://github.com/ZacWalk/dd/blob/main/docs/cli.md#commands)
  are documented in its own repository.

## SIMD rules

- **Runtime dispatch.** Higher-ISA paths compile with `VEXP_TARGET(...)` and
  run only after a runtime CPU check (`vexp::cpu()`), so one binary runs
  everywhere. Never raise the whole-program baseline to AVX2 or AVX-512.
- **Do not wrap `#include <immintrin.h>` in `#pragma GCC target(...)`.** GCC
  and Clang already tag every intrinsic with its own ISA requirement; the
  wrapper over-decorates the baseline SSE2 intrinsics, which then fail to
  inline into the SSE2 path (`error: target specific option mismatch`). Use a
  plain include and gate code generation per function with `VEXP_TARGET`.
- **Portable SIMD** is GCC/Clang `__attribute__((vector_size))`
  (`distance_vext`), guarded by `VEXP_HAVE_VECTOR_EXTENSIONS`: MSVC lacks it.
- **ARM CRC intrinsics** need `__ARM_FEATURE_CRC32`. CMake adds
  `-march=armv8-a+crc` on non-Apple AArch64; Apple silicon and MSVC have it
  already. Execution is still gated by a runtime check.
- **CRC-32C steps 8 bytes at a time** (`_mm_crc32_u64` / `__crc32cd`) on 64-bit
  targets. The CRC chain is strictly serial, so the 4-byte form takes twice the
  steps; do not "simplify" back to it.
- **Keep the scalar baseline scalar.** `distance_scalar` disables
  auto-vectorisation (`VEXP_NO_VECTORIZE_*`); `distance_autovec` is the *same
  loop* with it enabled.
- **`vector64_t` is plain `alignas(64)` bytes**, not a union of
  `__m128i`/`__m256i`: reading one union member after writing another is type
  punning, which C++ does not define. The SIMD paths load from `h` explicitly.

## Benchmark integrity

Read this before touching `main.cpp`. The three properties described in the
README's [How it is measured](README.md#how-it-is-measured) are load-bearing:

1. Each timing loop carries **its kernel's own `VEXP_TARGET`**, stamped out by
   `VEXP_DISTANCE_BENCH` and `VEXP_CRC_BENCH`. Never demote one to a plain
   template or a runtime function pointer.
2. The accumulator stays in a register and is passed to `do_not_optimize()`
   **once, after the loop**. `clobber_memory()` stays inside the loop, so
   loop-invariant loads cannot be hoisted.
3. `measure()` calibrates a batch size against the time budget and keeps the
   **fastest of five batches**.

**How to tell you broke the harness.** MSVC and GCC land within a few percent
of each other on every *hand-written* row (SSE2, AVX2, CRC SSE4.2), because
they emit the same instructions; a 2x disagreement means the harness is
measuring scaffolding, not the kernel. Confirm inlining with
`objdump -dC build-linux/vector-experiments | sed -n '/bench_distance_avx2_loop.*:$/,/^$/p'`
— expect `vpsadbw` in the loop body and **no `call`**.

## Conventions

- Format with `.clang-format` (tabs, Allman braces, 120 columns).
  `./dd.ps1 fmt` applies it to `include/`, `src/` and `tests/` only, and looks
  for `clang-format` only on `PATH`; Visual Studio installs one in
  `VC\Tools\Llvm\x64\bin`.
- Every target builds warning-free at `/W4` on MSVC and `-Wall -Wextra` on GCC
  and Clang; `.clang-tidy` holds the bug-finding and performance checks.
- Headers are self-contained, live in namespace `vexp`, and keep architecture
  and compiler specifics behind the `VEXP_*` macros of `platform.hpp`.
- OS calls are confined to `platform.hpp`, behind `#if defined(_WIN32)` and
  POSIX branches; keep Windows-only APIs out of the hot path.
- Time with `now_ns()`, which is `std::chrono` in nanoseconds; keep it
  portable, with no platform clock such as `QueryPerformanceCounter`.
