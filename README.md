# SIMD optimisation and performance

[![CI](https://github.com/ZacWalk/vector-experiments/actions/workflows/ci.yml/badge.svg)](https://github.com/ZacWalk/vector-experiments/actions/workflows/ci.yml)

A benchmark that contrasts a **scalar baseline against vectorised
implementations** of two kernels — sum-of-absolute-differences over 64-byte
vectors (**distance**) and **CRC-32C** — to show what SIMD actually buys you. A
further experiment pits `std::sort` against three integer-sort architectures.
Cross-platform CMake; builds with **MSVC, GCC and Clang** on **x86-64 and
ARM64**. Originally an internal SIMD course at Microsoft.

## Results

Nanoseconds per call, on a Ryzen 9 5900XT (median of three runs; lower is
better). Numbers vary by CPU, compiler and run, but the ordering is stable.

| Distance (64 bytes) | MSVC x64 | vs scalar | GCC 13 x64 | vs scalar |
| ------------------- | -------: | --------: | ---------: | --------: |
| scalar     | 19.99 |  1.0× | 30.13 |  1.0× |
| autovec    |  0.95 | 20.9× |  9.58 |  3.1× |
| vector-ext |     — |    —  |  3.79 |  8.0× |
| SSE2       |  0.91 | 21.9× |  0.91 | 33.1× |
| AVX2       |  0.59 | 33.9× |  0.60 | 50.4× |

| CRC-32C (129 bytes) | MSVC x64 | vs table | GCC 13 x64 | vs table |
| ------------------- | -------: | -------: | ---------: | -------: |
| table  | 66.88 |  1.0× | 61.26 |  1.0× |
| SSE4.2 |  4.98 | 13.4× |  4.76 | 12.9× |

**The headline:** hand-written AVX2 does the 64-byte distance in **0.6 ns, ~34×
faster than the scalar loop** (~50× on GCC), and the hardware **CRC instruction
is ~13× faster** than a tuned slice-by-4 table. Note that the two compilers agree
to within a couple of percent on every hand-written row — they are emitting the
same instructions — and disagree wildly everywhere else.

The most interesting row is `autovec`, which is *the same source* as `scalar`
with auto-vectorisation left enabled:

- **MSVC finds it.** Its auto-vectoriser emits four `psadbw` instructions, the
  same ones the hand-written SSE2 kernel uses, and lands within 5% of it. Free.
- **GCC does not.** At `-O3` it transliterates the loop instead: unpack bytes to
  words to dwords (`punpcklbw`/`punpckldq`), compare, subtract, `pmaxsw`,
  accumulate — hundreds of instructions where one would do, and 10× slower than
  the intrinsics.

That is the case for explicit SIMD in one line: auto-vectorisation will widen a
loop, but you cannot rely on it to *recognise* the specialised instruction that
makes the kernel fast. Note also that the portable `vector-ext` path (8×) is much
better than nothing but still 4× off the intrinsics, for the same reason — there
is no portable way to spell "sum of absolute differences".

CRC-32C makes a second point: *how* you feed a hardware instruction matters. The
CRC chain is strictly serial — every step depends on the previous result — so the
cost is the number of steps times the instruction's latency, and nothing overlaps.
Consuming 8 bytes per step (`_mm_crc32_u64`) instead of 4 therefore nearly doubles
throughput: the same kernel measures 13.4× on x86-64 but only 6.0× in a 32-bit
build, where only the 4-byte form exists.

What the rows mean:

- **`scalar`** — plain byte loop, auto-vectorisation **disabled** (the baseline).
- **`autovec`** — the *same loop* with auto-vectorisation enabled.
- **`vector-ext`** — portable SIMD via GCC/Clang vector extensions; one source → SSE/AVX **and** NEON (absent on MSVC).
- **`SSE2` / `AVX2` / `AVX512` / `NEON`** — hand-written intrinsics, chosen at runtime per CPU.
- **crc32 `table` vs `SSE4.2` / `ARM`** — software table vs the hardware CRC instruction.

`AVX512` and `NEON` rows appear only on a capable CPU / architecture. ARM (NEON,
`vector-ext`, ARM-CRC) is built and run in CI on Linux-arm64 and macOS-arm64.

## Sorting experiment

A further experiment contrasts `std::sort` (the standard introsort — "quicksort")
with three integer-sort architectures, all sorting the same array of 65,536
random 32-bit integers. Their headline tricks are AVX-512 instructions this test
machine lacks — `vcompress`, a runtime JIT, `gather` — so the scalar rows measure
each design's algorithmic core. Where **AVX2** can carry the idea there are
vectorised variants too: `sort_bitonic_avx2` and a SIMD merge sort,
`sort_merge_avx2`.

| Sort (65,536 × uint32) | MSVC x64 (ms) | vs std::sort | GCC 13 x64 (ms) | vs std::sort |
| ---------------------- | ------------: | -----------: | --------------: | -----------: |
| std::sort (quicksort)        | 2.74 | 1.0× | 2.43 | 1.0× |
| radix — VMCRS                | 0.50 | **5.5×** | 0.54 | **4.5×** |
| bitonic scalar — JIT-SN      | 5.15 | 0.5× | 5.13 | 0.5× |
| bitonic **AVX2** — JIT-SN    | 0.85 | **3.2×** | 0.75 | **3.2×** |
| k-way merge scalar — RP-KWM  | 3.20 | 0.9× | 2.86 | 0.9× |
| merge **AVX2** — RP-KWM      | 0.74 | **3.7×** | 0.75 | **3.2×** |

- **radix (VMCRS)** is the fastest by a wide margin (~5×): no comparisons,
  byte-wise counting. All four byte histograms are built in a single read pass —
  they don't depend on element order — then four stable scatter passes finish the
  sort, touching memory 5 times instead of the naive 8. What's left is the random
  scatter; an AVX-512 `vcompress` partition is what would attack that.
- **bitonic network (JIT-SN)**: *scalar* it is ~2× slower than quicksort — a
  sorting network does O(N log²N) compare-exchanges versus quicksort's
  O(N log N). But it is pure branchless min/max, so on **AVX2**
  `sort_bitonic_avx2` runs eight compare-exchanges per instruction
  (`vpminud`/`vpmaxud` + lane permutes) and flips the result to **3.2× faster**
  than quicksort — a ~6× jump over the scalar network.
- **k-way merge (RP-KWM)**: the *scalar* k-way merge is ~0.9× — the serial "scan
  K heads for the minimum" dominates. The SIMD answer is `sort_merge_avx2`, a
  bottom-up **vectorised merge sort**: 8-wide in-register block sort, then a
  bitonic 2-way merge that moves 8 elements per step. Being O(N log N) it does
  less work than the bitonic network and is the fastest comparison sort here at
  **3.2–3.7×**. (The spec's K-way register reduction is the AVX-512 `gather`
  variant of the same merge idea.)

Takeaway: three of the variants beat quicksort — radix, the vectorised bitonic
network, and the SIMD merge sort — while only the two deliberately scalar
baselines trail it. Every run validates each sort against `std::sort`, and the
ARM / Clang builds are covered in CI.

## Build & run

Needs CMake and a C++20 compiler (MSVC, GCC or Clang).

```pwsh
./dd.ps1 run                    # configure (Release), build, run
./dd.ps1 run 50                 # quick run (50 ms budget per row)
./dd.ps1 build                  # build only
./dd.ps1 clean                  # wipe the build dir
```

`dd.ps1` finds CMake by itself (including the copy bundled with Visual Studio).
The manual equivalent:

```pwsh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/vector-experiments      # MSVC: ./build/Release/vector-experiments.exe
```

The program exits non-zero if any implementation produces a wrong answer (the CI
regression gate).

## How it is measured

Timing a four-instruction kernel is easy to get wrong, so the harness is
deliberate about three things:

- **Self-calibrating.** The only argument is a budget in milliseconds per row
  (default 300). The harness grows the batch size until a batch fills that
  budget, so results never depend on the clock's resolution or on how fast the
  machine is. It then keeps the **fastest of five batches** — interference only
  ever adds time, so the minimum is closest to the kernel alone.
- **The kernel is inlined into the timing loop.** GCC and Clang will not inline a
  function carrying a higher `target(...)` attribute into a caller without one,
  and GCC appends a `vzeroupper` to every AVX function. Measured through that
  call, the AVX2 kernel came out *slower* than SSE2 — the dispatch, not the
  arithmetic. Each timing loop therefore carries its kernel's own ISA target.
- **The accumulator stays in a register**, made observable once after the loop.
  Forcing it to memory each iteration adds a store-forwarding stall longer than
  any of these kernels, which flattens every row to the same number.

## Implementation

Everything is in [`src/`](src/): `distance.hpp`, `crc32c.hpp` and `sorting.hpp`
hold the implementations, `main.cpp` is the harness, and `platform.hpp` has the
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
