// sorting.hpp
//
// A second experiment: contrast std::sort (libstdc++/MSVC introsort, i.e.
// "quicksort") against three integer-sort architectures.
//
// These are the portable, runnable embodiments of three AVX-512 designs. The
// designs' headline trick is a specific AVX-512 instruction (vcompress / a
// runtime JIT / gather) that this test hardware does not have, so what is
// measured here is each architecture's *algorithmic core*, in scalar form:
//
//   * sort_radix   (VMCRS)  - byte-wise LSD radix sort: no comparisons, purely
//                             sequential reads/writes. The vcompress partition is
//                             an accelerator for exactly this data movement.
//   * sort_bitonic (JIT-SN) - a bitonic sorting network: a fixed, branchless mesh
//                             of min/max compare-exchanges. The JIT emits the same
//                             min/max sequence as AVX-512 code at runtime.
//   * sort_kmerge  (RP-KWM) - a k-way merge that finds the minimum across K run
//                             heads by scanning them — the exact reduction the
//                             register-pinned SIMD version does in one instruction.
//
// All four share the signature `void(sort_key*, size_t, sort_key* scratch)` so the
// harness can treat them uniformly; `scratch` (length n) is ignored where unused.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "platform.hpp"

#if defined(VEXP_ARCH_X86)
#  include <immintrin.h>
#endif

namespace vexp {

using sort_key = uint32_t;

// Baseline: the standard library's introsort (quicksort + heapsort + insertion).
inline void sort_std(sort_key* a, size_t n, sort_key* /*scratch*/)
{
    std::sort(a, a + n);
}

// VMCRS essence: least-significant-digit, byte-wise (base-256) radix sort.
// The per-byte histograms are independent of element order, so all four are
// built in a single read pass; then four stable scatter passes finish the sort
// (5 passes over memory instead of the naive 8 — radix is memory-bound).
inline void sort_radix(sort_key* a, size_t n, sort_key* scratch)
{
    if (n < 2)
        return;

    size_t count[4][256] = {{0}};
    for (size_t i = 0; i < n; ++i)
    {
        const sort_key v = a[i];
        ++count[0][v & 0xFFu];
        ++count[1][(v >> 8) & 0xFFu];
        ++count[2][(v >> 16) & 0xFFu];
        ++count[3][(v >> 24) & 0xFFu];
    }

    for (int b = 0; b < 4; ++b)
    {
        size_t sum = 0;
        for (int d = 0; d < 256; ++d)
        {
            const size_t c = count[b][d];
            count[b][d] = sum;
            sum += c;
        }
    }

    sort_key* src = a;
    sort_key* dst = scratch;
    for (int b = 0; b < 4; ++b)
    {
        const int shift = b * 8;
        size_t* off = count[b];
        for (size_t i = 0; i < n; ++i)
        {
            const sort_key v = src[i];
            dst[off[(v >> shift) & 0xFFu]++] = v;
        }
        std::swap(src, dst);
    }
    // Four passes is an even number of swaps, so the sorted data is back in `a`.
}

// JIT-SN essence: a bitonic sorting network — branchless min/max compare-exchanges
// arranged in a fixed pattern. Requires a power-of-two length (the harness uses
// one); falls back to std::sort otherwise so it stays correct for any input.
inline void sort_bitonic(sort_key* a, size_t n, sort_key* /*scratch*/)
{
    if (n < 2 || (n & (n - 1)) != 0)
    {
        std::sort(a, a + n);
        return;
    }

    for (size_t k = 2; k <= n; k <<= 1)
        for (size_t j = k >> 1; j > 0; j >>= 1)
            for (size_t i = 0; i < n; ++i)
            {
                const size_t partner = i ^ j;
                if (partner > i)
                {
                    const bool ascending = (i & k) == 0;
                    const sort_key x = a[i];
                    const sort_key y = a[partner];
                    const sort_key lo = x < y ? x : y;
                    const sort_key hi = x < y ? y : x;
                    a[i] = ascending ? lo : hi;
                    a[partner] = ascending ? hi : lo;
                }
            }
}

// JIT-SN, accelerated for this machine (AVX2): the *same* bitonic network, but
// eight compare-exchanges run per instruction. Inter-register stages (stride >= 8)
// are straight `vpminud`/`vpmaxud`; intra-register stages (stride 4/2/1) use a
// lane permute + min/max + per-lane blend. Gated at runtime by `cpu().avx2`.
#if defined(VEXP_ARCH_X86)
VEXP_TARGET("avx2")
inline void sort_bitonic_avx2(sort_key* a, size_t n, sort_key* /*scratch*/)
{
    if (n < 16 || (n & (n - 1)) != 0)
    {
        std::sort(a, a + n);
        return;
    }

    const __m256i all_ones = _mm256_set1_epi32(-1);

    for (size_t k = 2; k <= n; k <<= 1)
    {
        for (size_t j = k >> 1; j > 0; j >>= 1)
        {
            if (j >= 8)
            {
                // Inter-register: lanes i..i+7 pair with l..l+7 element-wise.
                for (size_t i = 0; i < n; i += 8)
                {
                    const size_t l = i ^ j;
                    if (l > i)
                    {
                        const __m256i vi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
                        const __m256i vl = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + l));
                        const __m256i lo = _mm256_min_epu32(vi, vl);
                        const __m256i hi = _mm256_max_epu32(vi, vl);
                        const bool ascending = (i & k) == 0;
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + i), ascending ? lo : hi);
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + l), ascending ? hi : lo);
                    }
                }
            }
            else
            {
                // Intra-register: partner of lane r is lane r ^ j (j in {4,2,1}).
                alignas(32) uint32_t idx[8];
                alignas(32) uint32_t pat[8];  // ascending keep-min pattern: r < r^j
                for (uint32_t r = 0; r < 8; ++r)
                {
                    idx[r] = r ^ static_cast<uint32_t>(j);
                    pat[r] = (r < (r ^ static_cast<uint32_t>(j))) ? 0xFFFFFFFFu : 0u;
                }
                const __m256i vidx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(idx));
                const __m256i pmask = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pat));

                if (k >= 8)
                {
                    // Direction is uniform across each 8-lane block.
                    const __m256i pmask_desc = _mm256_xor_si256(pmask, all_ones);
                    for (size_t i = 0; i < n; i += 8)
                    {
                        const bool ascending = (i & k) == 0;
                        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
                        const __m256i p = _mm256_permutevar8x32_epi32(v, vidx);
                        const __m256i lo = _mm256_min_epu32(v, p);
                        const __m256i hi = _mm256_max_epu32(v, p);
                        const __m256i keepmin = ascending ? pmask : pmask_desc;
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + i),
                                            _mm256_blendv_epi8(hi, lo, keepmin));
                    }
                }
                else
                {
                    // k in {2,4}: direction alternates within the block by (r & k).
                    alignas(32) uint32_t km[8];
                    for (uint32_t r = 0; r < 8; ++r)
                    {
                        const bool low = r < (r ^ static_cast<uint32_t>(j));
                        const bool descending = (r & static_cast<uint32_t>(k)) != 0;
                        km[r] = (low != descending) ? 0xFFFFFFFFu : 0u;
                    }
                    const __m256i keepmin = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(km));
                    for (size_t i = 0; i < n; i += 8)
                    {
                        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
                        const __m256i p = _mm256_permutevar8x32_epi32(v, vidx);
                        const __m256i lo = _mm256_min_epu32(v, p);
                        const __m256i hi = _mm256_max_epu32(v, p);
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + i),
                                            _mm256_blendv_epi8(hi, lo, keepmin));
                    }
                }
            }
        }
    }
}
#endif  // VEXP_ARCH_X86

// RP-KWM, AVX2 take: a bottom-up SIMD merge sort. Blocks of 8 are sorted with an
// in-register bitonic network, then runs are merged with a vectorised bitonic
// 2-way merge that consumes/produces 8 elements per step. This is the
// AVX2-friendly way to make a *merge*-based sort beat quicksort; the K-way
// register reduction the spec describes is the AVX-512 variant of the same idea.
#if defined(VEXP_ARCH_X86)
// Compare-exchange 8 lanes with their partners (lane r <-> lane r^j given by idx);
// `keepmin` selects, per lane, whether to keep the min (set) or the max (clear).
VEXP_TARGET("avx2")
inline __m256i vexp_cex8(__m256i v, __m256i idx, __m256i keepmin)
{
    const __m256i p = _mm256_permutevar8x32_epi32(v, idx);
    const __m256i lo = _mm256_min_epu32(v, p);
    const __m256i hi = _mm256_max_epu32(v, p);
    return _mm256_blendv_epi8(hi, lo, keepmin);
}

// Sort 8 uint32 in a register, ascending (a fixed bitonic-8 network).
VEXP_TARGET("avx2")
inline __m256i vexp_sort8(__m256i v)
{
    const __m256i i1 = _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6);
    const __m256i i2 = _mm256_setr_epi32(2, 3, 0, 1, 6, 7, 4, 5);
    const __m256i i4 = _mm256_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3);
    v = vexp_cex8(v, i1, _mm256_setr_epi32(-1, 0, 0, -1, -1, 0, 0, -1));
    v = vexp_cex8(v, i2, _mm256_setr_epi32(-1, -1, 0, 0, 0, 0, -1, -1));
    v = vexp_cex8(v, i1, _mm256_setr_epi32(-1, 0, -1, 0, 0, -1, 0, -1));
    v = vexp_cex8(v, i4, _mm256_setr_epi32(-1, -1, -1, -1, 0, 0, 0, 0));
    v = vexp_cex8(v, i2, _mm256_setr_epi32(-1, -1, 0, 0, -1, -1, 0, 0));
    v = vexp_cex8(v, i1, _mm256_setr_epi32(-1, 0, -1, 0, -1, 0, -1, 0));
    return v;
}

// Merge two ascending 8-vectors into a sorted 16: out_lo = smallest 8, out_hi =
// largest 8 (both ascending). Reverse b to form a bitonic 16, then bitonic-merge.
VEXP_TARGET("avx2")
inline void vexp_merge16(__m256i a, __m256i b, __m256i& out_lo, __m256i& out_hi)
{
    const __m256i i1 = _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6);
    const __m256i i2 = _mm256_setr_epi32(2, 3, 0, 1, 6, 7, 4, 5);
    const __m256i i4 = _mm256_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3);
    const __m256i rev = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i a4 = _mm256_setr_epi32(-1, -1, -1, -1, 0, 0, 0, 0);
    const __m256i a2 = _mm256_setr_epi32(-1, -1, 0, 0, -1, -1, 0, 0);
    const __m256i a1 = _mm256_setr_epi32(-1, 0, -1, 0, -1, 0, -1, 0);

    const __m256i br = _mm256_permutevar8x32_epi32(b, rev);
    __m256i lo = _mm256_min_epu32(a, br);
    __m256i hi = _mm256_max_epu32(a, br);
    lo = vexp_cex8(lo, i4, a4); lo = vexp_cex8(lo, i2, a2); lo = vexp_cex8(lo, i1, a1);
    hi = vexp_cex8(hi, i4, a4); hi = vexp_cex8(hi, i2, a2); hi = vexp_cex8(hi, i1, a1);
    out_lo = lo;
    out_hi = hi;
}

VEXP_TARGET("avx2")
inline void sort_merge_avx2(sort_key* a, size_t n, sort_key* scratch)
{
    if (n < 16 || (n & (n - 1)) != 0)
    {
        std::sort(a, a + n);
        return;
    }

    // Phase 1: sort each block of 8 in-register.
    for (size_t i = 0; i < n; i += 8)
    {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        v = vexp_sort8(v);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + i), v);
    }

    // Phase 2: bottom-up vectorised 2-way merges, ping-ponging a <-> scratch.
    sort_key* src = a;
    sort_key* dst = scratch;
    for (size_t run = 8; run < n; run <<= 1)
    {
        for (size_t i = 0; i < n; i += 2 * run)
        {
            const sort_key* A = src + i;
            const sort_key* B = src + i + run;
            sort_key* out = dst + i;
            size_t ia = 8, ib = 8, io = 0;

            __m256i lo, hi;
            vexp_merge16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(A)),
                         _mm256_loadu_si256(reinterpret_cast<const __m256i*>(B)), lo, hi);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + io), lo);
            io += 8;

            // Pull the next 8 from whichever run has the smaller head, and merge
            // it with the carried top-8 (`hi`); emit the new bottom-8.
            while (ia < run && ib < run)
            {
                __m256i nxt;
                if (A[ia] <= B[ib]) { nxt = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(A + ia)); ia += 8; }
                else                { nxt = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(B + ib)); ib += 8; }
                vexp_merge16(hi, nxt, lo, hi);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + io), lo);
                io += 8;
            }
            while (ia < run)
            {
                __m256i nxt = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(A + ia)); ia += 8;
                vexp_merge16(hi, nxt, lo, hi);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + io), lo); io += 8;
            }
            while (ib < run)
            {
                __m256i nxt = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(B + ib)); ib += 8;
                vexp_merge16(hi, nxt, lo, hi);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + io), lo); io += 8;
            }
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + io), hi);
        }
        sort_key* tmp = src; src = dst; dst = tmp;
    }

    if (src != a)
        std::memcpy(a, src, n * sizeof(sort_key));
}
#endif  // VEXP_ARCH_X86

// RP-KWM essence: produce K sorted runs, then k-way merge them by repeatedly
// scanning the K run heads for the minimum — the horizontal reduction that the
// register-pinned SIMD version performs across vector lanes.
inline void sort_kmerge(sort_key* a, size_t n, sort_key* scratch)
{
    constexpr size_t K = 16;
    if (n < K * 2)
    {
        std::sort(a, a + n);
        return;
    }

    size_t start[K + 1];
    size_t head[K];
    const size_t base = n / K;
    const size_t rem = n % K;
    size_t s = 0;
    for (size_t r = 0; r < K; ++r)
    {
        start[r] = s;
        head[r] = s;
        const size_t len = base + (r < rem ? 1 : 0);
        std::sort(a + s, a + s + len);
        s += len;
    }
    start[K] = n;

    for (size_t out = 0; out < n; ++out)
    {
        sort_key best = 0;
        int best_run = -1;
        for (size_t r = 0; r < K; ++r)
        {
            if (head[r] < start[r + 1])
            {
                const sort_key v = a[head[r]];
                if (best_run < 0 || v < best)
                {
                    best = v;
                    best_run = static_cast<int>(r);
                }
            }
        }
        scratch[out] = best;
        ++head[static_cast<size_t>(best_run)];
    }

    std::memcpy(a, scratch, n * sizeof(sort_key));
}

}  // namespace vexp
