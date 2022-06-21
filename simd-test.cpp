// simd-test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <windows.h>
#include <iostream>
#include <intrin.h>

#include <functional>
#include <array>


#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#define COMPILE_SIMD_INTRINSIC
#endif

#if defined(_MSC_VER) && defined(_M_ARM64)
#define COMPILE_ARM_INTRINSIC
#endif

#ifdef _M_X64
const auto build_arch = "X64";
#endif

#ifdef _M_IX86 
const auto build_arch = "x86";
#endif

#ifdef _M_ARM64  
const auto build_arch = "ARM64";
#endif

static uint64_t now_ms()
{
	LARGE_INTEGER tps = { 0 };
	QueryPerformanceFrequency(&tps);

	LARGE_INTEGER pc = { 0 };
	QueryPerformanceCounter(&pc);
	return (pc.QuadPart * 1000ll) / tps.QuadPart;
}


static bool sse2_supported = ::IsProcessorFeaturePresent(PF_XMMI64_INSTRUCTIONS_AVAILABLE) != 0;
static bool avx2_supported = ::IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE) != 0;
static bool avx512_supported = ::IsProcessorFeaturePresent(PF_AVX512F_INSTRUCTIONS_AVAILABLE) != 0;
static bool neon_supported = ::IsProcessorFeaturePresent(PF_ARM_NEON_INSTRUCTIONS_AVAILABLE) != 0;

static constexpr size_t hash_size = 64;

struct vector64_t
{
	union
	{
#ifdef COMPILE_SIMD_INTRINSIC
		__m128i mm[hash_size / 16];
		__m256i a256[hash_size / 32];
		__m512i a512;
#endif
#ifdef COMPILE_ARM_INTRINSIC
		uint8x16_t nn[hash_size / 16];
#endif
		uint8_t h[hash_size];
	};
};

__forceinline uint64_t distance_c(const vector64_t* v1, const vector64_t* v2)
{
	uint64_t distance = 0;

	for (int i = 0; i < hash_size; i++)
	{
		const auto dd = v1->h[i] - v2->h[i];
		distance += abs(dd);
	}

	return distance;
}

__forceinline static uint64_t distance_sse(const vector64_t* v1, const vector64_t* v2)
{
#ifdef COMPILE_SIMD_INTRINSIC
	if (sse2_supported)
	{
		__m128i s0 = _mm_sad_epu8(v1->mm[0], v2->mm[0]);
		__m128i s1 = _mm_sad_epu8(v1->mm[1], v2->mm[1]);
		__m128i d0 = _mm_add_epi64(s0, s1);

		__m128i s2 = _mm_sad_epu8(v1->mm[2], v2->mm[2]);
		__m128i s3 = _mm_sad_epu8(v1->mm[3], v2->mm[3]);
		__m128i d1 = _mm_add_epi64(s2, s3);

		__m128i d = _mm_add_epi64(d0, d1);
		__m128i r = _mm_add_epi64(d, _mm_unpackhi_epi64(d, d));

#if defined(_M_X64)
		return _mm_cvtsi128_si64(r);
#else
		return _mm_cvtsi128_si32(r);
#endif

	}
#endif

	return 0;
}

__forceinline static uint64_t distance_avx2(const vector64_t* v1, const vector64_t* v2)
{
#ifdef COMPILE_SIMD_INTRINSIC
	if (avx2_supported)
	{
		__m256i d0 = _mm256_sad_epu8(v1->a256[0], v2->a256[0]);
		__m256i d1 = _mm256_sad_epu8(v1->a256[1], v2->a256[1]);
		__m256i d = _mm256_add_epi64(d0, d1);
		__m128i x = _mm_add_epi64(_mm256_castsi256_si128(d), _mm256_extracti128_si256(d, 1));
		__m128i r = _mm_add_epi64(x, _mm_unpackhi_epi64(x, x));

#if defined(_M_X64)
		return _mm_cvtsi128_si64(r);
#else
		return _mm_cvtsi128_si32(r);
#endif
	}
#endif

	return 0;
}

__forceinline static uint64_t distance_avx512(const vector64_t* v1, const vector64_t* v2)
{
#ifdef COMPILE_SIMD_INTRINSIC
	if (avx512_supported)
	{
		__m512i d = _mm512_sad_epu8(v1->a512, v2->a512);
		return _mm512_reduce_add_epi64(d);
	}
#endif

	return 0;
}

__forceinline static uint64_t distance_neon(const vector64_t* v1, const vector64_t* v2)
{
#ifdef COMPILE_ARM_INTRINSIC
	if (neon_supported)
	{
		uint8x16_t d0 = vqaddq_u16(vpaddlq_u8(vabdq_u8(v1->nn[0], v2->nn[0])), vpaddlq_u8(vabdq_u8(v1->nn[1], v2->nn[1])));
		uint8x16_t d1 = vqaddq_u16(vpaddlq_u8(vabdq_u8(v1->nn[2], v2->nn[2])), vpaddlq_u8(vabdq_u8(v1->nn[3], v2->nn[3])));
		uint32x4_t result = vpaddlq_u32(vpaddlq_u16(vqaddq_u16(d0, d1)));
		return vgetq_lane_s64(result, 0) + vgetq_lane_s64(result, 1);
	}
#endif

	return 0;
}

const vector64_t* make_hash(const char* src)
{
	auto result = (vector64_t*)_aligned_malloc(hash_size, 16);
	for (int i = 0; i < hash_size; i++) result->h[i] = src[i];
	return result;
}



///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

constexpr uint32_t CRCINIT = 0xFFFFFFFFu;


__forceinline static uint32_t calc_crc32c_neon(uint32_t crc, const void* data, const size_t len)
{
#if defined(COMPILE_ARM_INTRINSIC)
	const auto* p = static_cast<const uint8_t*>(data);
	const auto* const end = p + len;

	while (p < end && (std::bit_cast<uintptr_t>(p) & 0x0f))
	{
		crc = __crc32cb(crc, *p++);
	}

	while (p + (sizeof(uint32_t) - 1) < end)
	{
		crc = __crc32cw(crc, *std::bit_cast<const uint32_t*>(p));
		p += sizeof(uint32_t);
	}

	while (p < end)
	{
		crc = __crc32cb(crc, *p++);
	}
#endif

	return crc;
}


__forceinline static uint32_t calc_crc32c_sse(uint32_t crc, const void* data, const size_t len)
{
#ifdef COMPILE_SIMD_INTRINSIC
	const auto* p = static_cast<const uint8_t*>(data);
	const auto* const end = p + len;

	while (p < end && (std::bit_cast<uintptr_t>(p) & 0x0f))
	{
		crc = _mm_crc32_u8(crc, *p++);
	}

	while (p + (sizeof(uint32_t) - 1) < end)
	{
		crc = _mm_crc32_u32(crc, *std::bit_cast<const uint32_t*>(p));
		p += sizeof(uint32_t);
	}

	while (p < end)
	{
		crc = _mm_crc32_u8(crc, *p++);
	}

#endif

	return crc;
}


static std::array<std::array<uint32_t, 256>, 4> create_crc32_precalc()
{
	// CRC-32C (iSCSI) polynomial in reversed bit order.
	static constexpr uint32_t CRCPOLY = 0x82f63b78;

	std::array<std::array<uint32_t, 256>, 4> result;

	for (auto i = 0u; i <= 0xFF; i++)
	{
		uint32_t x = i;

		for (uint32_t j = 0; j < 8; j++)
			x = (x >> 1) ^ (CRCPOLY & (-static_cast<int32_t>(x & 1)));

		result[0][i] = x;
	}

	for (auto i = 0u; i <= 0xFF; i++)
	{
		uint32_t c = result[0][i];

		for (auto j = 1u; j < 4; j++)
		{
			c = result[0][c & 0xFF] ^ (c >> 8);
			result[j][i] = c;
		}
	}

	return result;
}

__forceinline static uint32_t calc_crc32c_c(uint32_t crc, const void* data, const size_t len)
{
	static const auto crc_precalc = create_crc32_precalc();

	const auto* p = static_cast<const uint8_t*>(data);
	const auto* const end = p + len;

	while (p < end && std::bit_cast<uintptr_t>(p) & 0x0f)
	{
		crc = crc_precalc[0][(crc ^ *p++) & 0xFF] ^ (crc >> 8);
	}

	while (p + (sizeof(uint32_t) - 1) < end)
	{
		crc ^= *std::bit_cast<const uint32_t*>(p);
		crc =
			crc_precalc[3][(crc) & 0xFF] ^
			crc_precalc[2][(crc >> 8) & 0xFF] ^
			crc_precalc[1][(crc >> 16) & 0xFF] ^
			crc_precalc[0][(crc >> 24) & 0xFF];

		p += sizeof(uint32_t);
	}

	while (p < end)
	{
		crc = crc_precalc[0][(crc ^ *p++) & 0xFF] ^ (crc >> 8);
	}

	return crc;
}

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

static const char* arch_text(WORD wIamgeFileMachine)
{
	switch (wIamgeFileMachine)
	{
	case IMAGE_FILE_MACHINE_I386:
		return "X86";
	case IMAGE_FILE_MACHINE_AMD64:
		return "X64";
	case IMAGE_FILE_MACHINE_ARMNT:
		return "ARM";
	case IMAGE_FILE_MACHINE_ARM64:
		return "ARM64";
	}
	return "Unknown";
}

static const char* get_machine()
{
	USHORT wProcessMachine = 0;
	USHORT wNativeMachine = 0;

	if (IsWow64Process2(GetCurrentProcess(), &wProcessMachine, &wNativeMachine) != 0)
	{
		return arch_text(wNativeMachine);
	}

	return "Unknown";
}

struct test
{
	std::string name;
	std::function<bool()> run;
	bool can_run = false;
	bool success = false;
};

int main()
{
	const auto v1 = make_hash("bq0zgkfbNEhAzGQ2V2W0stbpqQyQ04zrF0TgxmVoJf9O5Wk65EghJBca378cCggd");
	const auto v2 = make_hash("e0MiFoM5x53XfZrCCKuH1VovqgJatp2qTR6q9UZwHkhAszSnztPzTlhTHR2xiA41");
	const auto crc_data = "I believe in intuitions and inspirations. I sometimes feel that I am right. I do not know that I am. -- Albert Einstein";
	const auto crc_data_len = strlen(crc_data);
	const auto crc_expected = ~0XEC0CEEE5;

	test tests[] =
	{
		{ "distance C", [v1, v2] { return distance_c(v1, v2) == 1855; }, true, false },
		{ "distance SSE", [v1, v2] { return distance_sse(v1, v2) == 1855; }, sse2_supported, false },
		{ "distance AVX2", [v1, v2] { return distance_avx2(v1, v2) == 1855; }, avx2_supported, false },
		{ "distance AVX512", [v1, v2] { return distance_avx512(v1, v2) == 1855; }, avx512_supported, false },
		{ "distance NEON", [v1, v2] { return distance_neon(v1, v2) == 1855; }, neon_supported, false },

		{ "crc32 C", [crc_data, crc_data_len] { return ~calc_crc32c_c(CRCINIT, crc_data, crc_data_len) == crc_expected; }, true, false },
		{ "crc32 SSE", [crc_data, crc_data_len] { return ~calc_crc32c_sse(CRCINIT, crc_data, crc_data_len) == crc_expected; }, sse2_supported, false },
		{ "crc32 NEON", [crc_data, crc_data_len] { return ~calc_crc32c_neon(CRCINIT, crc_data, crc_data_len) == crc_expected; }, neon_supported, false },
	};

	std::cout << "| Host | Build | Test | Result | Time   |" << std::endl;
	std::cout << "| ---- | ----- | ---- | ------ | ------ |" << std::endl;

	const auto host = get_machine();
	const auto timing_iterations = 100000000ull;

	for (const auto& t : tests)
	{
		if (t.can_run)
		{
			auto success = true;
			auto start_time = now_ms();
			for (int i = 0ull; i < timing_iterations; i++)
			{
				success = success && t.run();
			}
			const auto time = now_ms() - start_time;
			const auto result = success ? "pass" : "fail";
			std::cout << "| " << host << " | " << build_arch << " | " << t.name << " | " << result << " | " << time << " | " << std::endl;
		}
	}

	return 0;
}
