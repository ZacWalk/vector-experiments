// main.cpp
//
// Benchmark harness: runs every available "distance" and "crc32" implementation,
// checks each produces the correct answer, times it and prints a Markdown table
// contrasting the scalar baseline against the vector / hardware variants.
//
// Pass an iteration count as argv[1] to override the default (handy for quick
// runs), e.g. `vector-experiments 5000000`.
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

// Time `iters` calls of fn(args...) and return the elapsed milliseconds. The
// barriers stop the optimiser hoisting the (loop-invariant) call out of the
// loop or discarding its result.
template <class Fn, class... Args>
uint64_t time_calls(uint64_t iters, Fn fn, Args... args)
{
    uint64_t sink = 0;
    const uint64_t start = now_ms();
    for (uint64_t i = 0; i < iters; ++i)
    {
        clobber_memory();
        sink += static_cast<uint64_t>(fn(args...));
        do_not_optimize(sink);
    }
    do_not_optimize(sink);
    return now_ms() - start;
}

struct row
{
    std::string host;
    std::string build;
    std::string test;
    std::string result;
    uint64_t ms;
};

void print_table(const std::vector<row>& rows)
{
    std::printf("| Host | Build | Test | Result | Time |\n");
    std::printf("| ---- | ----- | ---- | ------ | ---- |\n");
    for (const auto& r : rows)
        std::printf("| %s | %s | %s | %s | %llu |\n",
                    r.host.c_str(), r.build.c_str(), r.test.c_str(),
                    r.result.c_str(), static_cast<unsigned long long>(r.ms));
}

// Sort table: like print_table, but with a speed-up column relative to the first
// row (std::sort), which is what the sort experiment is measured against.
void print_sort_table(const std::vector<row>& rows)
{
    const double base = rows.empty() ? 0.0 : static_cast<double>(rows[0].ms);
    std::printf("| Host | Build | Sort | Result | Time | vs std::sort |\n");
    std::printf("| ---- | ----- | ---- | ------ | ---- | ------------ |\n");
    for (const auto& r : rows)
    {
        char speed[32];
        if (base > 0.0 && r.ms > 0)
            std::snprintf(speed, sizeof(speed), "%.2fx", base / static_cast<double>(r.ms));
        else
            std::snprintf(speed, sizeof(speed), "-");
        std::printf("| %s | %s | %s | %s | %llu | %s |\n",
                    r.host.c_str(), r.build.c_str(), r.test.c_str(),
                    r.result.c_str(), static_cast<unsigned long long>(r.ms), speed);
    }
}

struct distance_contestant
{
    const char* name;
    bool available;
    uint64_t (*fn)(const vector64_t*, const vector64_t*);
};

struct crc_contestant
{
    const char* name;
    bool available;
    uint32_t (*fn)(uint32_t, const void*, size_t);
};

}  // namespace

int main(int argc, char** argv)
{
    uint64_t iterations = 100000000ull;
    if (argc > 1)
    {
        const uint64_t requested = std::strtoull(argv[1], nullptr, 10);
        if (requested != 0)
            iterations = requested;
    }

    vector64_t* v1 = make_hash("bq0zgkfbNEhAzGQ2V2W0stbpqQyQ04zrF0TgxmVoJf9O5Wk65EghJBca378cCggd");
    vector64_t* v2 = make_hash("e0MiFoM5x53XfZrCCKuH1VovqgJatp2qTR6q9UZwHkhAszSnztPzTlhTHR2xiA41");

    const char* crc_data =
        "I believe in intuitions and inspirations. I sometimes feel that I am "
        "right. I do not know that I am. -- Albert Einstein";
    const size_t crc_len = std::strlen(crc_data);

    const uint64_t distance_expected = 1855;
    const uint32_t crc_expected = ~0xEC0CEEE5u;

    const std::string host = host_arch_name();
    const std::string build = build_arch_name();

    const distance_contestant distance_tests[] = {
        {"distance scalar", true, distance_scalar},
        {"distance autovec", true, distance_autovec},
#if defined(VEXP_HAVE_VECTOR_EXTENSIONS)
        {"distance vector-ext", true, distance_vext},
#endif
#if defined(VEXP_ARCH_X86)
        {"distance SSE2", cpu().sse2, distance_sse2},
        {"distance AVX2", cpu().avx2, distance_avx2},
        {"distance AVX512", cpu().avx512, distance_avx512},
#endif
#if defined(VEXP_ARCH_ARM64)
        {"distance NEON", cpu().neon, distance_neon},
#endif
    };

    const crc_contestant crc_tests[] = {
        {"crc32 table", true, calc_crc32c_table},
#if defined(VEXP_ARCH_X86)
        {"crc32 SSE4.2", cpu().sse42, calc_crc32c_sse},
#endif
#if defined(VEXP_HAVE_ARM_CRC)
        {"crc32 ARM", cpu().arm_crc, calc_crc32c_arm},
#endif
    };

    std::fprintf(stderr, "host=%s build=%s iterations=%llu\n",
                 host.c_str(), build.c_str(),
                 static_cast<unsigned long long>(iterations));

    std::vector<row> rows;
    bool all_passed = true;

    for (const auto& t : distance_tests)
    {
        if (!t.available)
            continue;
        const bool ok = (t.fn(v1, v2) == distance_expected);
        all_passed = all_passed && ok;
        const uint64_t ms = time_calls(iterations, t.fn, v1, v2);
        rows.push_back({host, build, t.name, ok ? "pass" : "fail", ms});
    }

    for (const auto& t : crc_tests)
    {
        if (!t.available)
            continue;
        const bool ok = (~t.fn(crc_init, crc_data, crc_len)) == crc_expected;
        all_passed = all_passed && ok;
        const uint64_t ms = time_calls(iterations, t.fn, crc_init, crc_data, crc_len);
        rows.push_back({host, build, t.name, ok ? "pass" : "fail", ms});
    }

    print_table(rows);

    // -- second experiment: sorting (std::sort vs the three novel architectures) --
    const size_t sort_n = size_t(1) << 16;
    const double sort_scale = static_cast<double>(iterations) / 100000000.0;
    const uint64_t sort_repeats =
        std::max<uint64_t>(1, static_cast<uint64_t>(300.0 * sort_scale));

    std::vector<sort_key> master(sort_n), work(sort_n), scratch(sort_n), reference(sort_n);
    std::mt19937 rng(0xC0FFEEu);
    for (auto& value : master)
        value = rng();
    reference = master;
    std::sort(reference.begin(), reference.end());

    struct sort_contestant
    {
        const char* name;
        bool available;
        void (*fn)(sort_key*, size_t, sort_key*);
    };
    const sort_contestant sort_tests[] = {
        {"sort std (quicksort)", true, sort_std},
        {"sort radix (VMCRS)", true, sort_radix},
        {"sort bitonic (JIT-SN)", true, sort_bitonic},
#if defined(VEXP_ARCH_X86)
        {"sort bitonic AVX2 (JIT-SN)", cpu().avx2, sort_bitonic_avx2},
#endif
        {"sort k-way merge (RP-KWM)", true, sort_kmerge},
#if defined(VEXP_ARCH_X86)
        {"sort merge AVX2 (RP-KWM)", cpu().avx2, sort_merge_avx2},
#endif
    };

    std::vector<row> sort_rows;
    for (const auto& s : sort_tests)
    {
        if (!s.available)
            continue;
        std::memcpy(work.data(), master.data(), sort_n * sizeof(sort_key));
        s.fn(work.data(), sort_n, scratch.data());
        const bool ok = std::equal(work.begin(), work.end(), reference.begin());
        all_passed = all_passed && ok;

        const uint64_t start = now_ms();
        for (uint64_t r = 0; r < sort_repeats; ++r)
        {
            std::memcpy(work.data(), master.data(), sort_n * sizeof(sort_key));
            clobber_memory();
            s.fn(work.data(), sort_n, scratch.data());
            do_not_optimize(work[0]);
        }
        const uint64_t ms = now_ms() - start;
        sort_rows.push_back({host, build, s.name, ok ? "pass" : "fail", ms});
    }

    std::fprintf(stderr, "sort: n=%llu repeats=%llu\n",
                 static_cast<unsigned long long>(sort_n),
                 static_cast<unsigned long long>(sort_repeats));
    std::printf("\n");
    print_sort_table(sort_rows);

    free_hash(v1);
    free_hash(v2);
    return all_passed ? 0 : 1;
}
