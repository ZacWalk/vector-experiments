# SIMD optimisation and performance 

Some experiments implementing SIMD algorithms on SSE2 and NEON intrinsics.

Implements a routine to calculate the distance (difference) between two 64-byte vectors. Using C, SSE2 and NEON intrinsics.

Performance (Milliseconds for 100,000,000 iterations):

| Host | Build | Test | Result | Time   |
| ---- | ----- | ---- | ------ | ------ |
| x64 | X64 | distance C | pass | 258 |
| x64 | X64 | distance SSE | pass | 196 |
| x64 | X64 | distance AVX2 | pass | 170 |
| x64 | X64 | distance AVX512 | pass | 167 |
| x64 | X64 | crc32 C | pass | 5868 |
| x64 | X64 | crc32 SSE | pass | 1685 |