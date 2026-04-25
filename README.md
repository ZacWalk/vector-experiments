# SIMD optimisation and performance

Some examples implementing vectorised algorithms using SSE2 and NEON intrinsics. These were used for an internal SIMD instructions course that I ran at Microsoft.

Implements a routine to calculate the distance (difference) between two 64-byte vectors (with benchmarking), using C, SSE2 and NEON intrinsic instructions. Also includes an implementation of CRC32.

Performance (milliseconds for 100,000,000 iterations):

The benchmark results below cover three configurations:

- An **x64 binary running natively** on x64 hardware.
- An **x64 binary running on ARM64** under Windows' x64 emulation. This is interesting because Windows on ARM64 emulates (or JITs) x64 SSE2 instructions into the equivalent NEON instructions, so the SSE path still benefits from hardware vectorisation.
- A **natively built ARM64 binary** running on ARM64 hardware, using NEON intrinsics directly.

| Host | Build | Test | Result | Time   |
| ---- | ----- | ---- | ------ | ------ |
| x64 | X64 | distance C | pass | 258 |
| x64 | X64 | distance SSE | pass | 196 |
| x64 | X64 | distance AVX2 | pass | 170 |
| x64 | X64 | distance AVX512 | pass | 167 |
| x64 | X64 | crc32 C | pass | 5868 |
| x64 | X64 | crc32 SSE | pass | 1685 |
| ARM64 | X64 | distance C | pass | 606 |
| ARM64 | X64 | distance SSE | pass | 497 |
| ARM64 | X64 | crc32 C | pass | 11506 |
| ARM64 | X64 | crc32 SSE | pass | 1941 |
| ARM64 | ARM64 | distance C | pass | 391 |
| ARM64 | ARM64 | distance NEON | pass | 288 |
| ARM64 | ARM64 | crc32 C | pass | 9338 |
| ARM64 | ARM64 | crc32 NEON | pass | 1604 |

## Glossary

- **SIMD** — *Single Instruction, Multiple Data*. A class of CPU instructions that apply the same operation to several data elements packed into one wide register.
- **SSE / SSE2** — *Streaming SIMD Extensions*. Intel/AMD x86 SIMD instruction sets operating on 128-bit XMM registers (e.g. sixteen 8-bit lanes per instruction). SSE2 is baseline on all x64 CPUs.
- **AVX2** — *Advanced Vector Extensions 2*. Extends SSE to 256-bit YMM registers, doubling per-instruction throughput for integer SIMD.
- **AVX-512** — Further extension to 512-bit ZMM registers, plus richer mask and reduction instructions (e.g. `_mm512_reduce_add_epi64`).
- **NEON** — Arm's SIMD instruction set, operating on 128-bit Q-registers (e.g. `uint8x16_t`). Roughly the ARM equivalent of SSE2.
- **CRC32C** — Cyclic Redundancy Check using the Castagnoli (iSCSI) polynomial. Both x86 (`_mm_crc32_*`) and ARM64 (`__crc32c*`) provide hardware instructions for it.
