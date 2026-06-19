// crc32c.hpp
//
// CRC-32C (Castagnoli / iSCSI polynomial) in three styles for benchmarking:
//
//   * calc_crc32c_table - portable slice-by-4 table lookup (the scalar baseline)
//   * calc_crc32c_sse   - x86 SSE4.2 hardware CRC instruction (`_mm_crc32_*`)
//   * calc_crc32c_arm   - AArch64 hardware CRC instruction (`__crc32c*`)
//
// CRC32C is a scalar hardware instruction rather than a SIMD one, but it is a
// great companion demo: it shows a dedicated instruction beating a tuned table.
#pragma once

#include <array>

#include "platform.hpp"

#if defined(VEXP_ARCH_X86)
#  include <nmmintrin.h>  // _mm_crc32_* (SSE4.2); gated per-function via VEXP_TARGET
#endif

#if defined(VEXP_ARCH_ARM64)
#  if defined(_MSC_VER)
#    define VEXP_HAVE_ARM_CRC 1   // intrinsics always declared by <intrin.h>
#  elif defined(__ARM_FEATURE_CRC32)
#    include <arm_acle.h>
#    define VEXP_HAVE_ARM_CRC 1
#  endif
#endif

namespace vexp {

inline constexpr uint32_t crc_init = 0xFFFFFFFFu;

// ----------------------------------------------------------------------------
// Slice-by-4 lookup table (scalar baseline)
// ----------------------------------------------------------------------------
inline std::array<std::array<uint32_t, 256>, 4> create_crc32_precalc()
{
    // CRC-32C (iSCSI) polynomial in reversed bit order.
    static constexpr uint32_t poly = 0x82f63b78u;

    std::array<std::array<uint32_t, 256>, 4> table{};

    for (uint32_t i = 0; i <= 0xFF; ++i)
    {
        uint32_t x = i;
        for (uint32_t j = 0; j < 8; ++j)
            x = (x >> 1) ^ (poly & (~(x & 1u) + 1u));
        table[0][i] = x;
    }

    for (uint32_t i = 0; i <= 0xFF; ++i)
    {
        uint32_t c = table[0][i];
        for (uint32_t j = 1; j < 4; ++j)
        {
            c = table[0][c & 0xFF] ^ (c >> 8);
            table[j][i] = c;
        }
    }

    return table;
}

inline const std::array<std::array<uint32_t, 256>, 4>& crc_table()
{
    static const auto table = create_crc32_precalc();
    return table;
}

inline uint32_t calc_crc32c_table(uint32_t crc, const void* data, size_t len)
{
    const auto& table = crc_table();
    const auto* p = static_cast<const uint8_t*>(data);
    const auto* const end = p + len;

    while (p < end && (reinterpret_cast<uintptr_t>(p) & 3u))
        crc = table[0][(crc ^ *p++) & 0xFF] ^ (crc >> 8);

    while (p + sizeof(uint32_t) <= end)
    {
        uint32_t word;
        std::memcpy(&word, p, sizeof(word));
        crc ^= word;
        crc = table[3][crc & 0xFF] ^
              table[2][(crc >> 8) & 0xFF] ^
              table[1][(crc >> 16) & 0xFF] ^
              table[0][(crc >> 24) & 0xFF];
        p += sizeof(uint32_t);
    }

    while (p < end)
        crc = table[0][(crc ^ *p++) & 0xFF] ^ (crc >> 8);

    return crc;
}

// ----------------------------------------------------------------------------
// x86 SSE4.2 hardware CRC
// ----------------------------------------------------------------------------
#if defined(VEXP_ARCH_X86)
VEXP_TARGET("sse4.2")
inline uint32_t calc_crc32c_sse(uint32_t crc, const void* data, size_t len)
{
    const auto* p = static_cast<const uint8_t*>(data);
    const auto* const end = p + len;

    while (p + sizeof(uint32_t) <= end)
    {
        uint32_t word;
        std::memcpy(&word, p, sizeof(word));
        crc = _mm_crc32_u32(crc, word);
        p += sizeof(uint32_t);
    }

    while (p < end)
        crc = _mm_crc32_u8(crc, *p++);

    return crc;
}
#endif  // VEXP_ARCH_X86

// ----------------------------------------------------------------------------
// AArch64 hardware CRC
// ----------------------------------------------------------------------------
#if defined(VEXP_HAVE_ARM_CRC)
inline uint32_t calc_crc32c_arm(uint32_t crc, const void* data, size_t len)
{
    const auto* p = static_cast<const uint8_t*>(data);
    const auto* const end = p + len;

    while (p + sizeof(uint32_t) <= end)
    {
        uint32_t word;
        std::memcpy(&word, p, sizeof(word));
        crc = __crc32cw(crc, word);
        p += sizeof(uint32_t);
    }

    while (p < end)
        crc = __crc32cb(crc, *p++);

    return crc;
}
#endif  // VEXP_HAVE_ARM_CRC

}  // namespace vexp
