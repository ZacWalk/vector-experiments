# SIMD optimisation and performance

[![CI](https://github.com/ZacWalk/vector-experiments/actions/workflows/ci.yml/badge.svg)](https://github.com/ZacWalk/vector-experiments/actions/workflows/ci.yml)

A benchmark that contrasts a **scalar baseline against vectorised
implementations** of two kernels — sum-of-absolute-differences over 64-byte
vectors (**distance**) and **CRC-32C** — to show what SIMD actually buys you. A
second experiment pits `std::sort` against three novel integer-sort
architectures. Cross-platform CMake; builds with **MSVC, GCC and Clang** on
**x86-64 and ARM64**. Originally an internal SIMD course at Microsoft.

## Results

Milliseconds for 100,000,000 iterations, with the speed-up over each kernel's
scalar baseline (`distance scalar` and `crc32 table`). Lower ms / higher × is
better; numbers are illustrative and vary by CPU, compiler and run.

| Test | MSVC x64 (ms) | vs scalar | GCC 13 x64 (ms) | vs scalar |
| ---- | ------------: | --------: | --------------: | --------: |
| distance scalar     | 2289 |  1.0× | 3037 |  1.0× |
| distance autovec    |  169 | 13.5× |  973 |  3.1× |
| distance vector-ext |    — |    —  |  399 |  7.6× |
| distance SSE2       |  210 | 10.9× |  167 | 18.2× |
| distance AVX2       |  212 | 10.8× |  130 | 23.4× |
| crc32 table         | 6967 |  1.0× | 6555 |  1.0× |
| crc32 SSE4.2        | 1117 |  6.2× | 1188 |  5.5× |

**The headline:** vectorising the distance baseline is **~13× faster** (MSVC
auto-vectoriser, 2289 → 169 ms) and the hardware **CRC instruction is ~6×
faster** than a tuned table (6967 → 1117 ms). On GCC, where auto-vectorisation
helps less, the explicit SIMD paths show it even more starkly — scalar 3037 →
portable `vector-ext` 399 → hand-written AVX2 130 ms, a **~23× span**.

What the rows mean:

- **`scalar`** — plain byte loop, auto-vectorisation **disabled** (the baseline).
- **`autovec`** — the *same loop* with auto-vectorisation enabled.
- **`vector-ext`** — portable SIMD via GCC/Clang vector extensions; one source → SSE/AVX **and** NEON (absent on MSVC).
- **`SSE2` / `AVX2` / `AVX512` / `NEON`** — hand-written intrinsics, chosen at runtime per CPU.
- **crc32 `table` vs `SSE4.2` / `ARM`** — software table vs the hardware CRC instruction.

`AVX512` and `NEON` rows appear only on a capable CPU / architecture. ARM (NEON,
`vector-ext`, ARM-CRC) is built and run in CI on Linux-arm64 and macOS-arm64.

## Sorting experiment

A second experiment contrasts `std::sort` (the standard introsort — "quicksort")
with three novel sorting architectures, all sorting the same array of 65,536
random 32-bit integers. Their headline tricks are AVX-512 instructions this test
machine (a Ryzen 9 5900XT) lacks — `vcompress`, a runtime JIT, `gather` — so the
scalar rows measure each design's algorithmic core. Where the machine's **AVX2**
can carry the idea there are vectorised variants too: `sort_bitonic_avx2` and a
SIMD merge sort, `sort_merge_avx2`.

| Sort (65,536 × uint32) | MSVC x64 (ms) | vs std::sort | GCC 13 x64 (ms) | vs std::sort |
| ---------------------- | ------------: | -----------: | --------------: | -----------: |
| std::sort (quicksort)        |  868 | 1.0× |  712 | 1.0× |
| radix — VMCRS                |   95 | 9.1× |  156 | 4.6× |
| bitonic scalar — JIT-SN      | 1964 | 0.4× | 1539 | 0.5× |
| bitonic **AVX2** — JIT-SN    |  267 | **3.3×** |  221 | **3.2×** |
| k-way merge scalar — RP-KWM  | 1039 | 0.8× |  830 | 0.9× |
| merge **AVX2** — RP-KWM      |  217 | **4.0×** |  218 | **3.3×** |

- **radix (VMCRS)** is the fastest by a wide margin (MSVC ~9×, GCC ~4.5×): no
  comparisons, byte-wise counting. All four byte histograms are built in a single
  read pass — they don't depend on element order — then four stable scatter passes
  finish the sort, touching memory 5 times instead of the naive 8. What's left is
  the random scatter; an AVX-512 `vcompress` partition is what would attack that.
- **bitonic network (JIT-SN)**: *scalar* it is ~2× slower than quicksort — a
  sorting network does O(N log²N) compare-exchanges versus quicksort's
  O(N log N). But it is pure branchless min/max, so on this **AVX2** machine
  `sort_bitonic_avx2` runs eight compare-exchanges per instruction
  (`vpminud`/`vpmaxud` + lane permutes) and flips the result to **~3× faster**
  than quicksort — a ~6× jump over the scalar network.
- **k-way merge (RP-KWM)**: the *scalar* k-way merge is ~0.9× — the serial "scan
  K heads for the minimum" dominates. The SIMD answer is `sort_merge_avx2`, a
  bottom-up **vectorised merge sort**: 8-wide in-register block sort, then a
  bitonic 2-way merge that moves 8 elements per step. Being O(N log N) it does
  less work than the bitonic network and is the fastest comparison sort here at
  **~3.4–4.2×**. (The spec's K-way register reduction is the AVX-512 `gather`
  variant of the same merge idea.)

Takeaway: on this **AVX2** machine three of the variants beat quicksort — radix,
the vectorised bitonic network, and the SIMD merge sort (the fastest
comparison-based sort here) — while only the two deliberately scalar baselines
(the textbook bitonic network and the k-way min-scan) trail it. Absolute numbers
swing run to run as the CPU boosts, but the ordering is stable. Every run
validates each sort against `std::sort`, and the ARM / Clang builds are covered
in CI.

## Build & run

Needs CMake and a C++20 compiler (MSVC, GCC or Clang).

```pwsh
./run.ps1                       # configure (Release), build, run
./run.ps1 -Iterations 5000000   # quick run
```

`run.ps1` finds CMake by itself (including the copy bundled with Visual Studio).
The manual equivalent:

```pwsh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/vector-experiments      # optional: iteration count
```

The program exits non-zero if any implementation produces a wrong answer (the CI
regression gate).

## Implementation

Everything is in [`src/`](src/): `distance.hpp` and `crc32c.hpp` hold the
implementations, `main.cpp` is the harness, and `platform.hpp` has the
arch/compiler macros, timing and CPU detection. The portable `vector-ext` path
uses GCC/Clang vector extensions (`__attribute__((vector_size(16)))`); the
higher-ISA paths are gated by per-function target attributes plus a runtime CPU
check, so one baseline binary dispatches everywhere. Design notes are in
[AGENTS.md](AGENTS.md).

## Glossary

- **SIMD** — *Single Instruction, Multiple Data*. One instruction applied to several data elements packed into a wide register.
- **SAD** — *Sum of Absolute Differences*. The "distance" kernel here; `_mm_sad_epu8` / `vabdq_u8` compute it directly in hardware.
- **Auto-vectorisation** — the compiler turning a scalar loop into SIMD by itself at `-O2`/`-O3`. Free, but limited; it rarely synthesises specialised instructions like SAD or CRC.
- **Vector extensions** — GCC/Clang `__attribute__((vector_size(N)))` types. Portable, explicit SIMD: write `a - b` on whole vectors and the compiler emits SSE/AVX or NEON. Not supported by MSVC.
- **SSE / SSE2** — *Streaming SIMD Extensions*. x86 SIMD on 128-bit XMM registers (sixteen 8-bit lanes). SSE2 is baseline on all x64 CPUs.
- **AVX2** — *Advanced Vector Extensions 2*. 256-bit YMM registers, doubling integer SIMD throughput over SSE.
- **AVX-512** — 512-bit ZMM registers plus richer mask and reduction instructions (e.g. `_mm512_reduce_add_epi64`).
- **NEON** — Arm's SIMD instruction set, 128-bit Q-registers (e.g. `uint8x16_t`). Mandatory on AArch64; roughly the ARM equivalent of SSE2.
- **CRC32C** — Cyclic Redundancy Check using the Castagnoli (iSCSI) polynomial. Both x86 (`_mm_crc32_*`) and ARM64 (`__crc32c*`) provide hardware instructions for it.
