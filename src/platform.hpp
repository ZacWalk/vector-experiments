// platform.hpp
//
// Portable building blocks shared by every benchmark: architecture / compiler
// detection, a force-inline macro, a per-function ISA "target" attribute, a
// monotonic millisecond clock, optimisation barriers for micro-benchmarking and
// runtime CPU feature detection.
//
// Nothing in here is Windows specific at the API level: the few OS calls we do
// make (CPU features, host architecture) are selected behind #if blocks so the
// same source compiles on Windows, Linux and macOS with MSVC, Clang or GCC.
#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <atomic>
#include <chrono>
#include <string>

// ----------------------------------------------------------------------------
// Architecture detection
// ----------------------------------------------------------------------------
#if defined(__x86_64__) || defined(_M_X64)
#  define VEXP_ARCH_X86_64 1
#  define VEXP_ARCH_X86 1
#elif defined(__i386__) || defined(_M_IX86)
#  define VEXP_ARCH_X86_32 1
#  define VEXP_ARCH_X86 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#  define VEXP_ARCH_ARM64 1
#  define VEXP_ARCH_ARM 1
#elif defined(__arm__) || defined(_M_ARM)
#  define VEXP_ARCH_ARM32 1
#  define VEXP_ARCH_ARM 1
#endif

// ----------------------------------------------------------------------------
// Compiler detection / helpers
// ----------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
#  define VEXP_GCC_OR_CLANG 1
#endif

// GCC and Clang both provide `__attribute__((vector_size(N)))` "vector
// extensions": portable SIMD that the compiler lowers to SSE/AVX on x86 and to
// NEON on ARM from a single piece of source. MSVC does not implement them.
#if defined(VEXP_GCC_OR_CLANG)
#  define VEXP_HAVE_VECTOR_EXTENSIONS 1
#endif

#if defined(_MSC_VER)
#  define VEXP_FORCEINLINE __forceinline
#else
#  define VEXP_FORCEINLINE inline __attribute__((always_inline))
#endif

// Per-function target attribute. On GCC/Clang this lets a single binary contain
// AVX2 / AVX-512 / SSE4.2 code paths that are only entered after a runtime CPU
// check, without raising the baseline ISA for the whole program. MSVC makes all
// intrinsics available unconditionally, so the attribute is a no-op there.
#if defined(VEXP_GCC_OR_CLANG)
#  define VEXP_TARGET(spec) __attribute__((target(spec)))
#else
#  define VEXP_TARGET(spec)
#endif

// Best-effort "do not auto-vectorise this loop", used to keep the scalar
// baseline honestly scalar so the speed-up of the vector paths is visible.
#if defined(__clang__)
#  define VEXP_NO_VECTORIZE_LOOP _Pragma("clang loop vectorize(disable) interleave(disable)")
#elif defined(_MSC_VER)
#  define VEXP_NO_VECTORIZE_LOOP __pragma(loop(no_vector))
#else
#  define VEXP_NO_VECTORIZE_LOOP
#endif
// GCC has no per-loop pragma; disable vectorisation for the whole function.
#if defined(__GNUC__) && !defined(__clang__)
#  define VEXP_NO_VECTORIZE_FN __attribute__((optimize("no-tree-vectorize")))
#else
#  define VEXP_NO_VECTORIZE_FN
#endif

// ----------------------------------------------------------------------------
// OS headers (kept narrowly scoped)
// ----------------------------------------------------------------------------
#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  if defined(_MSC_VER)
#    include <intrin.h>
#  endif
#else
#  include <sys/utsname.h>
#  if defined(__linux__)
#    include <sys/auxv.h>
#    if defined(VEXP_ARCH_ARM)
#      include <asm/hwcap.h>
#    endif
#  elif defined(__APPLE__)
#    include <sys/sysctl.h>
#  endif
#endif

#if defined(VEXP_ARCH_X86) && defined(VEXP_GCC_OR_CLANG)
#  include <cpuid.h>
#endif

namespace vexp {

// ----------------------------------------------------------------------------
// Timing
// ----------------------------------------------------------------------------
inline uint64_t now_ms()
{
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count());
}

// ----------------------------------------------------------------------------
// Micro-benchmark optimisation barriers
//
// clobber_memory() tells the optimiser that memory may have changed, forcing it
// to re-read the inputs (so a loop-invariant computation cannot be hoisted out
// of the timing loop). do_not_optimize() forces a value to be materialised so
// the computation that produced it is not eliminated as dead code.
// ----------------------------------------------------------------------------
#if defined(VEXP_GCC_OR_CLANG)
VEXP_FORCEINLINE void clobber_memory()
{
    asm volatile("" : : : "memory");
}
template <class T>
VEXP_FORCEINLINE void do_not_optimize(const T& value)
{
    asm volatile("" : : "r,m"(value) : "memory");
}
#else
VEXP_FORCEINLINE void clobber_memory()
{
    std::atomic_signal_fence(std::memory_order_acq_rel);
}
template <class T>
VEXP_FORCEINLINE void do_not_optimize(const T& value)
{
    // Read the value through a volatile lvalue so it must be computed, then a
    // compiler fence prevents reordering across the consumption point.
    volatile char sink = *reinterpret_cast<const volatile char*>(&value);
    (void)sink;
    std::atomic_signal_fence(std::memory_order_acq_rel);
}
#endif

// ----------------------------------------------------------------------------
// CPU feature detection
// ----------------------------------------------------------------------------
struct cpu_features
{
    bool sse2 = false;
    bool sse42 = false;
    bool avx2 = false;
    bool avx512 = false;  // AVX-512 F + BW (enough for _mm512_sad_epu8)
    bool neon = false;
    bool arm_crc = false;
};

#if defined(VEXP_ARCH_X86)
inline void cpuid_ex(unsigned leaf, unsigned subleaf, unsigned regs[4])
{
#if defined(_MSC_VER)
    int r[4];
    __cpuidex(r, static_cast<int>(leaf), static_cast<int>(subleaf));
    regs[0] = static_cast<unsigned>(r[0]);
    regs[1] = static_cast<unsigned>(r[1]);
    regs[2] = static_cast<unsigned>(r[2]);
    regs[3] = static_cast<unsigned>(r[3]);
#else
    __cpuid_count(leaf, subleaf, regs[0], regs[1], regs[2], regs[3]);
#endif
}

inline uint64_t xgetbv0()
{
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    unsigned eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<uint64_t>(edx) << 32) | eax;
#endif
}
#endif  // VEXP_ARCH_X86

#if defined(VEXP_ARCH_ARM64)
inline bool detect_arm_crc()
{
#if defined(_WIN32)
    return ::IsProcessorFeaturePresent(PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE) != 0;
#elif defined(__APPLE__)
    int value = 0;
    size_t size = sizeof(value);
    if (::sysctlbyname("hw.optional.armv8_crc32", &value, &size, nullptr, 0) == 0)
        return value != 0;
    return false;
#elif defined(__linux__)
#  ifndef HWCAP_CRC32
#    define HWCAP_CRC32 (1 << 7)
#  endif
    return (::getauxval(AT_HWCAP) & HWCAP_CRC32) != 0;
#else
    return false;
#endif
}
#endif  // VEXP_ARCH_ARM64

inline cpu_features detect_cpu()
{
    cpu_features f;
#if defined(VEXP_ARCH_X86)
    unsigned r[4] = {0, 0, 0, 0};

    cpuid_ex(1, 0, r);
    f.sse2 = (r[3] >> 26) & 1u;
    f.sse42 = (r[2] >> 20) & 1u;
    const bool osxsave = (r[2] >> 27) & 1u;
    const bool avx = (r[2] >> 28) & 1u;

    bool ymm_enabled = false;
    bool zmm_enabled = false;
    if (osxsave)
    {
        const uint64_t xcr0 = xgetbv0();
        ymm_enabled = (xcr0 & 0x6u) == 0x6u;     // XMM + YMM state saved by OS
        zmm_enabled = (xcr0 & 0xE6u) == 0xE6u;   // + opmask + ZMM hi256 + hi16 ZMM
    }

    unsigned max_leaf[4] = {0, 0, 0, 0};
    cpuid_ex(0, 0, max_leaf);
    unsigned r7[4] = {0, 0, 0, 0};
    if (max_leaf[0] >= 7)
        cpuid_ex(7, 0, r7);

    const bool cpu_avx2 = (r7[1] >> 5) & 1u;
    const bool cpu_avx512f = (r7[1] >> 16) & 1u;
    const bool cpu_avx512bw = (r7[1] >> 30) & 1u;

    f.avx2 = avx && ymm_enabled && cpu_avx2;
    f.avx512 = avx && zmm_enabled && cpu_avx512f && cpu_avx512bw;
#elif defined(VEXP_ARCH_ARM64)
    f.neon = true;  // mandatory on AArch64
    f.arm_crc = detect_arm_crc();
#endif
    return f;
}

inline const cpu_features& cpu()
{
    static const cpu_features features = detect_cpu();
    return features;
}

// ----------------------------------------------------------------------------
// Architecture names
// ----------------------------------------------------------------------------
inline const char* build_arch_name()
{
#if defined(VEXP_ARCH_X86_64)
    return "x64";
#elif defined(VEXP_ARCH_X86_32)
    return "x86";
#elif defined(VEXP_ARCH_ARM64)
    return "ARM64";
#elif defined(VEXP_ARCH_ARM32)
    return "ARM";
#else
    return "unknown";
#endif
}

inline std::string host_arch_name()
{
#if defined(_WIN32)
    USHORT process_machine = 0;
    USHORT native_machine = 0;
    if (::IsWow64Process2(::GetCurrentProcess(), &process_machine, &native_machine))
    {
        switch (native_machine)
        {
        case IMAGE_FILE_MACHINE_I386:  return "x86";
        case IMAGE_FILE_MACHINE_AMD64: return "x64";
        case IMAGE_FILE_MACHINE_ARMNT: return "ARM";
        case IMAGE_FILE_MACHINE_ARM64: return "ARM64";
        default: break;
        }
    }
    return build_arch_name();
#else
    struct utsname u;
    if (::uname(&u) == 0)
        return u.machine;
    return build_arch_name();
#endif
}

}  // namespace vexp
