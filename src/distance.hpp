// distance.hpp
//
// Sum-of-absolute-differences ("distance") between two 64-byte vectors, in a
// range of styles so they can be benchmarked against each other:
//
//   * distance_scalar   - plain byte loop, auto-vectorisation suppressed
//   * distance_autovec  - portable loop written so every compiler (incl. MSVC)
//                         can auto-vectorise it
//   * distance_vext     - explicit, portable SIMD via GCC/Clang vector
//                         extensions; one source, lowered to SSE/AVX on x86 and
//                         to NEON on ARM
//   * distance_sse2     - x86 SSE2 intrinsics (`_mm_sad_epu8`)
//   * distance_avx2     - x86 AVX2 intrinsics
//   * distance_avx512   - x86 AVX-512 intrinsics
//   * distance_neon     - AArch64 NEON intrinsics
#pragma once

#include <cstring>

#include "platform.hpp"

#if defined(VEXP_ARCH_X86)
// Modern GCC and Clang declare every x86 intrinsic (and the __m256i / __m512i
// types) from this single umbrella header, each tagged with its own ISA
// requirement. The per-function VEXP_TARGET attributes below decide which paths
// actually emit those instructions, so one binary can dispatch at runtime.
#  include <immintrin.h>
#elif defined(VEXP_ARCH_ARM64)
#  include <arm_neon.h>
#endif

namespace vexp {

inline constexpr size_t hash_size = 64;

// 64-byte vector, over-aligned so the widest aligned SIMD load is always legal.
// Deliberately plain bytes: reading them back through a union of __m128i /
// __m256i members would be type punning, which C++ (unlike C) does not define.
// The SIMD paths below load from `h` explicitly instead, which is both well
// specified and exactly the same instruction.
struct alignas(64) vector64_t
{
    uint8_t h[hash_size];
};

inline vector64_t make_hash(const char* src)
{
    vector64_t v{};
    for (size_t i = 0; i < hash_size; ++i)
        v.h[i] = static_cast<uint8_t>(src[i]);
    return v;
}

// ----------------------------------------------------------------------------
// Scalar baseline (auto-vectorisation deliberately disabled)
// ----------------------------------------------------------------------------
VEXP_NO_VECTORIZE_FN
inline uint64_t distance_scalar(const vector64_t* v1, const vector64_t* v2)
{
    uint64_t distance = 0;
    VEXP_NO_VECTORIZE_LOOP
    for (size_t i = 0; i < hash_size; ++i)
    {
        const int d = static_cast<int>(v1->h[i]) - static_cast<int>(v2->h[i]);
        distance += static_cast<uint64_t>(d < 0 ? -d : d);
    }
    return distance;
}

// ----------------------------------------------------------------------------
// The exact same loop as distance_scalar, but with auto-vectorisation left
// enabled. The contrast between the two rows shows what the compiler's
// auto-vectoriser buys you for free on a given target (a lot on GCC/Clang at
// -O3, often little on MSVC, which is why explicit SIMD still matters).
// Works on every compiler, including MSVC.
// ----------------------------------------------------------------------------
inline uint64_t distance_autovec(const vector64_t* v1, const vector64_t* v2)
{
    uint64_t distance = 0;
    for (size_t i = 0; i < hash_size; ++i)
    {
        const int d = static_cast<int>(v1->h[i]) - static_cast<int>(v2->h[i]);
        distance += static_cast<uint64_t>(d < 0 ? -d : d);
    }
    return distance;
}

// ----------------------------------------------------------------------------
// Explicit portable SIMD via GCC/Clang vector extensions.
// The same source compiles to SSE/AVX on x86 and to NEON on ARM.
// ----------------------------------------------------------------------------
#if defined(VEXP_HAVE_VECTOR_EXTENSIONS)
typedef uint8_t  vexp_u8x16 __attribute__((vector_size(16)));
typedef uint16_t vexp_u16x8 __attribute__((vector_size(16)));

inline uint64_t distance_vext(const vector64_t* v1, const vector64_t* v2)
{
    vexp_u16x8 acc = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int chunk = 0; chunk < 4; ++chunk)
    {
        vexp_u8x16 va, vb;
        std::memcpy(&va, v1->h + chunk * 16, 16);
        std::memcpy(&vb, v2->h + chunk * 16, 16);

        // Per-byte absolute difference of unsigned lanes, using only operators
        // that the vector extensions guarantee: a>b yields a 0x00/0xFF mask.
        const vexp_u8x16 ab = va - vb;
        const vexp_u8x16 ba = vb - va;
        const vexp_u8x16 mask = (vexp_u8x16)(va > vb);
        const vexp_u8x16 diff = (ab & mask) | (ba & ~mask);

        // Widen and accumulate: reinterpret 16 bytes as 8x u16, then add the low
        // and high byte of each pair. Max per lane after 4 chunks is 4*510 < 2^16.
        const vexp_u16x8 pairs = (vexp_u16x8)diff;
        acc += (pairs & static_cast<uint16_t>(0x00FF)) + (pairs >> 8);
    }

    uint64_t sum = 0;
    for (int i = 0; i < 8; ++i)
        sum += acc[i];
    return sum;
}
#endif  // VEXP_HAVE_VECTOR_EXTENSIONS

// ----------------------------------------------------------------------------
// Native x86 intrinsics
// ----------------------------------------------------------------------------
#if defined(VEXP_ARCH_X86)
VEXP_TARGET("sse2")
inline uint64_t distance_sse2(const vector64_t* v1, const vector64_t* v2)
{
    const auto* a = reinterpret_cast<const __m128i*>(v1->h);
    const auto* b = reinterpret_cast<const __m128i*>(v2->h);

    // _mm_sad_epu8 is the whole kernel in one instruction: it sums the absolute
    // differences of 16 byte lanes into two 64-bit totals.
    const __m128i d0 = _mm_add_epi64(_mm_sad_epu8(_mm_load_si128(a + 0), _mm_load_si128(b + 0)),
                                     _mm_sad_epu8(_mm_load_si128(a + 1), _mm_load_si128(b + 1)));
    const __m128i d1 = _mm_add_epi64(_mm_sad_epu8(_mm_load_si128(a + 2), _mm_load_si128(b + 2)),
                                     _mm_sad_epu8(_mm_load_si128(a + 3), _mm_load_si128(b + 3)));

    const __m128i d = _mm_add_epi64(d0, d1);
    const __m128i r = _mm_add_epi64(d, _mm_unpackhi_epi64(d, d));

#if defined(VEXP_ARCH_X86_64)
    return static_cast<uint64_t>(_mm_cvtsi128_si64(r));
#else
    return static_cast<uint64_t>(static_cast<uint32_t>(_mm_cvtsi128_si32(r)));
#endif
}

VEXP_TARGET("avx2")
inline uint64_t distance_avx2(const vector64_t* v1, const vector64_t* v2)
{
    const auto* a = reinterpret_cast<const __m256i*>(v1->h);
    const auto* b = reinterpret_cast<const __m256i*>(v2->h);

    const __m256i d0 = _mm256_sad_epu8(_mm256_load_si256(a + 0), _mm256_load_si256(b + 0));
    const __m256i d1 = _mm256_sad_epu8(_mm256_load_si256(a + 1), _mm256_load_si256(b + 1));
    const __m256i d = _mm256_add_epi64(d0, d1);

    const __m128i x = _mm_add_epi64(_mm256_castsi256_si128(d), _mm256_extracti128_si256(d, 1));
    const __m128i r = _mm_add_epi64(x, _mm_unpackhi_epi64(x, x));

#if defined(VEXP_ARCH_X86_64)
    return static_cast<uint64_t>(_mm_cvtsi128_si64(r));
#else
    return static_cast<uint64_t>(static_cast<uint32_t>(_mm_cvtsi128_si32(r)));
#endif
}

VEXP_TARGET("avx512f,avx512bw")
inline uint64_t distance_avx512(const vector64_t* v1, const vector64_t* v2)
{
    // The whole 64-byte vector is one register, so this is a single SAD.
    const __m512i d = _mm512_sad_epu8(_mm512_load_si512(v1->h), _mm512_load_si512(v2->h));
    return static_cast<uint64_t>(_mm512_reduce_add_epi64(d));
}
#endif  // VEXP_ARCH_X86

// ----------------------------------------------------------------------------
// Native AArch64 NEON intrinsics
// ----------------------------------------------------------------------------
#if defined(VEXP_ARCH_ARM64)
inline uint64_t distance_neon(const vector64_t* v1, const vector64_t* v2)
{
    // Per-byte absolute difference, widened to 16-bit lane sums.
    // Max per u16 lane after summing all four chunks: 4 * 2 * 255 = 2040.
    const uint16x8_t s0 = vpaddlq_u8(vabdq_u8(vld1q_u8(v1->h + 0), vld1q_u8(v2->h + 0)));
    const uint16x8_t s1 = vpaddlq_u8(vabdq_u8(vld1q_u8(v1->h + 16), vld1q_u8(v2->h + 16)));
    const uint16x8_t s2 = vpaddlq_u8(vabdq_u8(vld1q_u8(v1->h + 32), vld1q_u8(v2->h + 32)));
    const uint16x8_t s3 = vpaddlq_u8(vabdq_u8(vld1q_u8(v1->h + 48), vld1q_u8(v2->h + 48)));
    const uint16x8_t sum = vaddq_u16(vaddq_u16(s0, s1), vaddq_u16(s2, s3));
    return vaddlvq_u16(sum);
}
#endif  // VEXP_ARCH_ARM64

}  // namespace vexp
