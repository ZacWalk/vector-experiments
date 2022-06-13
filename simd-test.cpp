// simd-test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <windows.h>
#include <iostream>
#include <intrin.h>


#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#define COMPILE_SIMD_INTRINSIC
#endif

#if defined(_MSC_VER) && defined(_M_ARM64)
#define COMPILE_ARM_INTRINSIC
#endif

static uint64_t now_ms()
{
	LARGE_INTEGER tps = { 0 };
	QueryPerformanceFrequency(&tps);

	LARGE_INTEGER pc = { 0 };
	QueryPerformanceCounter(&pc);
	return (pc.QuadPart * 1000ll) / tps.QuadPart;
}


static BOOL sse2_supported = ::IsProcessorFeaturePresent(PF_XMMI64_INSTRUCTIONS_AVAILABLE);
static BOOL avx2_supported = ::IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE);
static BOOL avx512_supported = ::IsProcessorFeaturePresent(PF_AVX512F_INSTRUCTIONS_AVAILABLE);
static BOOL neon_supported = ::IsProcessorFeaturePresent(PF_ARM_NEON_INSTRUCTIONS_AVAILABLE);

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

__forceinline uint64_t distance_sse(const vector64_t* v1, const vector64_t* v2)
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

__forceinline uint64_t distance_avx2(const vector64_t* v1, const vector64_t* v2)
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

__forceinline uint64_t distance_avx512(const vector64_t* v1, const vector64_t* v2)
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

__forceinline uint64_t distance_neon(const vector64_t* v1, const vector64_t* v2)
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

int main()
{
	const auto v1 = make_hash("bq0zgkfbNEhAzGQ2V2W0stbpqQyQ04zrF0TgxmVoJf9O5Wk65EghJBca378cCggd");
	const auto v2 = make_hash("e0MiFoM5x53XfZrCCKuH1VovqgJatp2qTR6q9UZwHkhAszSnztPzTlhTHR2xiA41");

	// calculate distance various implimentations
	const auto dc = distance_c(v1, v2);
	const auto dsse = distance_sse(v1, v2);
	const auto davx2 = distance_avx2(v1, v2);
	const auto davx512 = distance_avx512(v1, v2);
	const auto dneon = distance_neon(v1, v2);

	const auto timing_iterations = 100000000ull;
	auto total_difference = 0ull; // needed to avoid opptimizing out code	

	// measure time of [timing_iterations] distance calulations
	auto start_time = now_ms();
	for (int i = 0ull; i < timing_iterations; i++)
	{
		total_difference += distance_c(v1, v2);
	}
	const auto time_c = now_ms() - start_time;

	start_time = now_ms();
	for (int i = 0ull; i < timing_iterations; i++)
	{
		total_difference += distance_sse(v1, v2);
	}
	const auto time_sse = now_ms() - start_time;

	start_time = now_ms();
	for (int i = 0ull; i < timing_iterations; i++)
	{
		total_difference += distance_avx2(v1, v2);
	}
	const auto time_avx2 = now_ms() - start_time;

	start_time = now_ms();
	for (int i = 0ull; i < timing_iterations; i++)
	{
		total_difference += distance_avx512(v1, v2);
	}
	const auto time_avx512 = now_ms() - start_time;

	start_time = now_ms();
	for (int i = 0ull; i < timing_iterations; i++)
	{
		total_difference += distance_neon(v1, v2);
	}
	const auto time_neon = now_ms() - start_time;

	// report results
	std::cout << "Results:" << std::endl << std::endl;
	std::cout << "Calc distance C:  " << dc << std::endl;
	std::cout << "Calc distance SSE:  " << dsse << std::endl;
	std::cout << "Calc distance AVX2:  " << davx2 << std::endl;
	std::cout << "Calc distance AVX512:  " << davx512 << std::endl;
	std::cout << "Calc distance NEON: " << dneon << std::endl;
	std::cout << "Total difference: " << total_difference << std::endl;
	std::cout << std::endl << std::endl << "Performance (Milliseconds for 100,000,000 iterations):" << std::endl << std::endl;

	std::cout << "|   C   | SSE  | AVX2 | AVX512 | NEON |" << std::endl;
	std::cout << "| ----- | ---- | ---- | ------ | ---- |" << std::endl;
	std::cout << "| " << time_c << " | ";
	
	if (sse2_supported)
		std::cout << time_sse;
	else
		std::cout << " N/A ";

	std::cout << " | ";

	if (avx2_supported)
		std::cout << time_avx2;
	else
		std::cout << " N/A ";
	
	std::cout << " | ";
	
	if (avx512_supported)
		std::cout << time_avx512;
	else
		std::cout << " N/A ";
	
	std::cout << " | ";
	
	if (neon_supported)
		std::cout << time_neon;
	else
		std::cout << " N/A ";
	
	std::cout << " | " << std::endl;

	return 0;
}
