// main.cpp
//
// Benchmark harness: runs every available implementation of the three
// experiments (distance, CRC-32C, sorting), checks each produces the correct
// answer, times it and prints Markdown tables contrasting the scalar baseline
// against the vector / hardware variants.
//
// argv[1] is the measurement budget in milliseconds per row (default 300). Each
// row is timed by repeating its kernel in calibrated batches that fill that
// budget, so the numbers are independent of the clock resolution and of how fast
// the host is. See "Measurement" below for why that matters.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "platform.hpp"
#include "distance.hpp"
#include "crc32c.hpp"
#include "sorting.hpp"

using namespace vexp;

namespace {

constexpr uint64_t default_budget_ms = 300;
constexpr uint64_t max_budget_ms = 10000;
constexpr double ns_per_ms = 1000000.0;
constexpr int batches_per_measurement = 5;

// ----------------------------------------------------------------------------
// Measurement
//
// Calibrate a batch size that runs long enough to time accurately, then keep the
// fastest of several batches. Everything that perturbs a benchmark - a scheduler
// pre-emption, a frequency dip, a noisy neighbour - only ever *adds* time, so the
// minimum is the closest we can get to the cost of the kernel alone. A single
// batch is not enough: on a loaded machine successive runs of the same binary
// varied by 3x here.
// ----------------------------------------------------------------------------
template <class Loop>
double measure(uint64_t budget_ns, Loop loop)
{
    constexpr uint64_t max_ops = uint64_t(1) << 40;
    const uint64_t batch_ns = std::max<uint64_t>(budget_ns / batches_per_measurement, 1);

    uint64_t ops = 1;
    uint64_t best = 0;
    for (;;)
    {
        const uint64_t start = now_ns();
        loop(ops);
        best = now_ns() - start;
        if (best >= batch_ns || ops >= max_ops)
            break;

        // Aim straight at the batch length, but never grow more than 64x off a
        // single sample, which the clock may have quantised to nothing.
        uint64_t next = ops * 2;
        if (best > 0)
        {
            const double want = static_cast<double>(ops) * 1.1 *
                                static_cast<double>(batch_ns) / static_cast<double>(best);
            if (want > static_cast<double>(next))
                next = static_cast<uint64_t>(want);
        }
        ops = std::min(next, ops * 64);
    }

    for (int i = 1; i < batches_per_measurement; ++i)
    {
        const uint64_t start = now_ns();
        loop(ops);
        best = std::min(best, now_ns() - start);
    }
    return static_cast<double>(best) / static_cast<double>(ops);
}

using distance_fn = uint64_t (*)(const vector64_t*, const vector64_t*);
using crc_fn = uint32_t (*)(uint32_t, const void*, size_t);
using sort_fn = void (*)(sort_key*, size_t, sort_key*);

// ----------------------------------------------------------------------------
// Timing loops
//
// Two things have to be true for these numbers to mean anything.
//
// 1. The kernel must be inlined into the loop. A handful of instructions timed
//    through a call measures the call. GCC and Clang refuse to inline a function
//    carrying a higher `target(...)` attribute into a caller that lacks one, and
//    GCC appends a `vzeroupper` to every AVX function - measured that way the
//    AVX2 kernel came out *slower* than SSE2 (which does inline), i.e. we were
//    timing the runtime dispatch rather than the arithmetic. So each loop is
//    stamped out carrying its own kernel's ISA target.
//
// 2. The accumulator must stay in a register and become observable only once,
//    after the loop. Forcing it to memory on every iteration adds a
//    store-forwarding stall longer than any of these kernels, which flattens
//    every row to the same number.
// ----------------------------------------------------------------------------
#define VEXP_BASELINE  // this kernel already runs at the program's baseline ISA

#define VEXP_DISTANCE_BENCH(attr, name, kernel)                                   \
    attr void name##_loop(uint64_t ops, const vector64_t& a, const vector64_t& b) \
    {                                                                             \
        uint64_t sink = 0;                                                        \
        for (uint64_t i = 0; i < ops; ++i)                                        \
        {                                                                         \
            clobber_memory();                                                     \
            sink += kernel(&a, &b);                                               \
        }                                                                         \
        do_not_optimize(sink);                                                    \
    }                                                                             \
    double name(uint64_t budget_ns, const vector64_t& a, const vector64_t& b)     \
    {                                                                             \
        return measure(budget_ns, [&](uint64_t ops) { name##_loop(ops, a, b); }); \
    }

#define VEXP_CRC_BENCH(attr, name, kernel)                                           \
    attr void name##_loop(uint64_t ops, const void* data, size_t len)                \
    {                                                                                \
        uint32_t sink = 0;                                                           \
        for (uint64_t i = 0; i < ops; ++i)                                           \
        {                                                                            \
            clobber_memory();                                                        \
            sink += kernel(crc_init, data, len);                                     \
        }                                                                            \
        do_not_optimize(sink);                                                       \
    }                                                                                \
    double name(uint64_t budget_ns, const void* data, size_t len)                    \
    {                                                                                \
        return measure(budget_ns, [&](uint64_t ops) { name##_loop(ops, data, len); });\
    }

VEXP_DISTANCE_BENCH(VEXP_BASELINE, bench_distance_scalar, distance_scalar)
VEXP_DISTANCE_BENCH(VEXP_BASELINE, bench_distance_autovec, distance_autovec)
#if defined(VEXP_HAVE_VECTOR_EXTENSIONS)
VEXP_DISTANCE_BENCH(VEXP_BASELINE, bench_distance_vext, distance_vext)
#endif
#if defined(VEXP_ARCH_X86)
VEXP_DISTANCE_BENCH(VEXP_TARGET("sse2"), bench_distance_sse2, distance_sse2)
VEXP_DISTANCE_BENCH(VEXP_TARGET("avx2"), bench_distance_avx2, distance_avx2)
VEXP_DISTANCE_BENCH(VEXP_TARGET("avx512f,avx512bw"), bench_distance_avx512, distance_avx512)
#endif
#if defined(VEXP_ARCH_ARM64)
VEXP_DISTANCE_BENCH(VEXP_BASELINE, bench_distance_neon, distance_neon)
#endif

VEXP_CRC_BENCH(VEXP_BASELINE, bench_crc_table, calc_crc32c_table)
#if defined(VEXP_ARCH_X86)
VEXP_CRC_BENCH(VEXP_TARGET("sse4.2"), bench_crc_sse, calc_crc32c_sse)
#endif
#if defined(VEXP_HAVE_ARM_CRC)
VEXP_CRC_BENCH(VEXP_BASELINE, bench_crc_arm, calc_crc32c_arm)
#endif

// A whole sort takes milliseconds, so the dispatch call is irrelevant here and
// the kernel can stay an ordinary template argument.
//
// One "op" includes the copy that restores the unsorted input. That copy is a
// fraction of a percent of a sort this size and is charged to every contestant
// equally, so it does not distort the comparison.
template <sort_fn Fn>
double bench_sort(uint64_t budget_ns, const sort_key* master, sort_key* work,
                  sort_key* scratch, size_t n)
{
    return measure(budget_ns, [&](uint64_t ops) {
        for (uint64_t i = 0; i < ops; ++i)
        {
            std::memcpy(work, master, n * sizeof(sort_key));
            clobber_memory();
            Fn(work, n, scratch);
            do_not_optimize(work[0]);
        }
    });
}

// ----------------------------------------------------------------------------
// Reporting
// ----------------------------------------------------------------------------
struct row
{
    const char* name;
    bool ok;
    double ns_per_op;
};

// One Markdown table per experiment. `unit_ns` scales raw nanoseconds into the
// unit that suits the kernel (ns for the tiny ones, ms for a whole sort); the
// speed-up column is relative to the first row, that experiment's baseline.
void print_table(const char* what, const char* unit, double unit_ns, const std::vector<row>& rows)
{
    if (rows.empty())
        return;

    const double base = rows.front().ns_per_op;
    std::printf("\n| %s | Result | %s | vs %s |\n", what, unit, rows.front().name);
    std::printf("| --- | --- | ---: | ---: |\n");
    for (const auto& r : rows)
        std::printf("| %s | %s | %.2f | %.2fx |\n",
                    r.name, r.ok ? "pass" : "fail",
                    r.ns_per_op / unit_ns,
                    r.ns_per_op > 0.0 ? base / r.ns_per_op : 0.0);
}

// ----------------------------------------------------------------------------
// Contestants: `verify` is called once for the correctness check, `bench` is the
// timing loop that has this same kernel baked into it at compile time.
// ----------------------------------------------------------------------------
struct distance_contestant
{
    const char* name;
    bool available;
    distance_fn verify;
    double (*bench)(uint64_t, const vector64_t&, const vector64_t&);
};

struct crc_contestant
{
    const char* name;
    bool available;
    crc_fn verify;
    double (*bench)(uint64_t, const void*, size_t);
};

struct sort_contestant
{
    const char* name;
    bool available;
    sort_fn verify;
    double (*bench)(uint64_t, const sort_key*, sort_key*, sort_key*, size_t);
};

}  // namespace

int main(int argc, char** argv)
{
    uint64_t budget_ms = default_budget_ms;
    if (argc > 1)
    {
        // Anything unparseable, trailing, negative or zero falls back to the
        // default; note that strtoull silently *wraps* "-5" into a huge value,
        // which would otherwise clamp to the longest run we allow.
        char* end = nullptr;
        const unsigned long long requested = std::strtoull(argv[1], &end, 10);
        if (end != argv[1] && *end == '\0' && argv[1][0] != '-' && requested > 0)
            budget_ms = std::min<uint64_t>(requested, max_budget_ms);
    }
    const uint64_t budget_ns = budget_ms * 1000000ull;

    bool all_passed = true;

    std::fprintf(stderr, "host=%s build=%s budget=%llums per measurement\n",
                 host_arch_name().c_str(), build_arch_name(),
                 static_cast<unsigned long long>(budget_ms));

    // -- experiment 1: sum of absolute differences over two 64-byte vectors ---
    const vector64_t v1 = make_hash("bq0zgkfbNEhAzGQ2V2W0stbpqQyQ04zrF0TgxmVoJf9O5Wk65EghJBca378cCggd");
    const vector64_t v2 = make_hash("e0MiFoM5x53XfZrCCKuH1VovqgJatp2qTR6q9UZwHkhAszSnztPzTlhTHR2xiA41");
    constexpr uint64_t distance_expected = 1855;

    const distance_contestant distance_tests[] = {
        {"scalar", true, distance_scalar, bench_distance_scalar},
        {"autovec", true, distance_autovec, bench_distance_autovec},
#if defined(VEXP_HAVE_VECTOR_EXTENSIONS)
        {"vector-ext", true, distance_vext, bench_distance_vext},
#endif
#if defined(VEXP_ARCH_X86)
        {"SSE2", cpu().sse2, distance_sse2, bench_distance_sse2},
        {"AVX2", cpu().avx2, distance_avx2, bench_distance_avx2},
        {"AVX512", cpu().avx512, distance_avx512, bench_distance_avx512},
#endif
#if defined(VEXP_ARCH_ARM64)
        {"NEON", cpu().neon, distance_neon, bench_distance_neon},
#endif
    };

    std::vector<row> distance_rows;
    for (const auto& t : distance_tests)
    {
        if (!t.available)
            continue;
        const bool ok = (t.verify(&v1, &v2) == distance_expected);
        all_passed = all_passed && ok;
        distance_rows.push_back({t.name, ok, t.bench(budget_ns, v1, v2)});
    }

    // -- experiment 2: CRC-32C over a short buffer ----------------------------
    const char* const crc_data =
        "I believe in intuitions and inspirations. I sometimes feel that I am "
        "right. I do not know that I am. -- Albert Einstein";
    const size_t crc_len = std::strlen(crc_data);
    constexpr uint32_t crc_expected = 0x13F3111Au;  // after the customary final complement

    const crc_contestant crc_tests[] = {
        {"table", true, calc_crc32c_table, bench_crc_table},
#if defined(VEXP_ARCH_X86)
        {"SSE4.2", cpu().sse42, calc_crc32c_sse, bench_crc_sse},
#endif
#if defined(VEXP_HAVE_ARM_CRC)
        {"ARM", cpu().arm_crc, calc_crc32c_arm, bench_crc_arm},
#endif
    };

    std::vector<row> crc_rows;
    for (const auto& t : crc_tests)
    {
        if (!t.available)
            continue;
        const bool ok = (~t.verify(crc_init, crc_data, crc_len)) == crc_expected;
        all_passed = all_passed && ok;
        crc_rows.push_back({t.name, ok, t.bench(budget_ns, crc_data, crc_len)});
    }

    // -- experiment 3: sorting 65,536 random 32-bit integers ------------------
    constexpr size_t sort_n = size_t(1) << 16;

    std::vector<sort_key> master(sort_n), work(sort_n), scratch(sort_n), reference(sort_n);
    std::mt19937 rng(0xC0FFEEu);
    for (auto& value : master)
        value = rng();
    reference = master;
    std::sort(reference.begin(), reference.end());

    const sort_contestant sort_tests[] = {
        {"std (quicksort)", true, sort_std, bench_sort<sort_std>},
        {"radix (VMCRS)", true, sort_radix, bench_sort<sort_radix>},
        {"bitonic (JIT-SN)", true, sort_bitonic, bench_sort<sort_bitonic>},
#if defined(VEXP_ARCH_X86)
        {"bitonic AVX2 (JIT-SN)", cpu().avx2, sort_bitonic_avx2, bench_sort<sort_bitonic_avx2>},
#endif
        {"k-way merge (RP-KWM)", true, sort_kmerge, bench_sort<sort_kmerge>},
#if defined(VEXP_ARCH_X86)
        {"merge AVX2 (RP-KWM)", cpu().avx2, sort_merge_avx2, bench_sort<sort_merge_avx2>},
#endif
    };

    std::vector<row> sort_rows;
    for (const auto& t : sort_tests)
    {
        if (!t.available)
            continue;
        std::memcpy(work.data(), master.data(), sort_n * sizeof(sort_key));
        t.verify(work.data(), sort_n, scratch.data());
        const bool ok = std::equal(work.begin(), work.end(), reference.begin());
        all_passed = all_passed && ok;
        sort_rows.push_back(
            {t.name, ok,
             t.bench(budget_ns, master.data(), work.data(), scratch.data(), sort_n)});
    }

    print_table("Distance (64 bytes)", "ns/call", 1.0, distance_rows);
    print_table("CRC-32C (129 bytes)", "ns/call", 1.0, crc_rows);
    print_table("Sort (65,536 x uint32)", "ms/sort", ns_per_ms, sort_rows);

    return all_passed ? 0 : 1;
}
