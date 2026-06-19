cmake_minimum_required(VERSION 3.20)

project(vector-experiments LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Language standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Keep GNU extensions enabled: the portable "vector_size" SIMD path relies on a
# GCC/Clang language extension.

# Default to an optimised build for single-config generators (Ninja, Makefiles).
get_property(_is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _is_multi_config AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

add_executable(vector-experiments src/main.cpp)
target_include_directories(vector-experiments PRIVATE src)

# Treat the headers as part of the target for IDEs.
target_sources(vector-experiments PRIVATE
  src/platform.hpp
  src/distance.hpp
  src/crc32c.hpp
  src/sorting.hpp
)

# ---------------------------------------------------------------------------
# Optimisation flags
# ---------------------------------------------------------------------------
if(MSVC)
  target_compile_options(vector-experiments PRIVATE
    $<$<CONFIG:Release>:/O2>
    $<$<CONFIG:RelWithDebInfo>:/O2>)
else()
  target_compile_options(vector-experiments PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:RelWithDebInfo>:-O2>)
endif()

# ---------------------------------------------------------------------------
# AArch64: enable the optional CRC32 instructions so the hardware CRC path can
# be compiled. Execution is still gated behind a runtime CPU check. Apple
# silicon has CRC32 in its baseline, so no flag is needed there.
# ---------------------------------------------------------------------------
if(NOT MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
  if(NOT APPLE)
    target_compile_options(vector-experiments PRIVATE -march=armv8-a+crc)
  endif()
endif()

# ---------------------------------------------------------------------------
# Convenience: `cmake --build . --target run`
# ---------------------------------------------------------------------------
add_custom_target(run
  COMMAND vector-experiments
  DEPENDS vector-experiments
  USES_TERMINAL)
