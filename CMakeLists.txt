# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.16)
project(rsfec VERSION 0.1.0 LANGUAGES CXX
        DESCRIPTION "Systematic Reed-Solomon erasure coding over GF(256)")

option(RSFEC_BUILD_TESTS "Build the test binary" ON)
option(RSFEC_BUILD_BENCH "Build the Google Benchmark microbenchmarks" OFF)

# Header-only: consumers just need the include path. -mssse3 is an interface
# requirement rather than a hard one -- gf256.h compiles without it and falls
# back to the scalar kernel, but every x86-64 CPU made since 2006 has SSSE3,
# so the default is to let the dispatcher have its baseline SIMD kernel.
add_library(rsfec INTERFACE)
add_library(rsfec::rsfec ALIAS rsfec)
target_include_directories(rsfec INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(rsfec INTERFACE cxx_std_17)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_compile_options(rsfec INTERFACE -mssse3)
endif()

include(GNUInstallDirs)
install(DIRECTORY include/rsfec DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS rsfec EXPORT rsfecTargets)
install(EXPORT rsfecTargets NAMESPACE rsfec::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rsfec)

if(RSFEC_BUILD_TESTS)
  enable_testing()
  add_executable(test_fec tests/test_fec.cpp)
  target_link_libraries(test_fec PRIVATE rsfec)
  add_test(NAME test_fec COMMAND test_fec)

  # The dispatcher picks the best kernel the CPU offers, so one run only
  # exercises one path. FEC_NO_SIMD forces the scalar reference so CI proves
  # both ends of the dispatch table on any runner.
  add_executable(test_fec_scalar tests/test_fec.cpp)
  target_link_libraries(test_fec_scalar PRIVATE rsfec)
  target_compile_definitions(test_fec_scalar PRIVATE FEC_NO_SIMD)
  add_test(NAME test_fec_scalar COMMAND test_fec_scalar)
endif()

if(RSFEC_BUILD_TESTS)
  add_executable(roundtrip examples/roundtrip.cpp)
  target_link_libraries(roundtrip PRIVATE rsfec)
  add_test(NAME example_roundtrip COMMAND roundtrip)
endif()

if(RSFEC_BUILD_BENCH)
  find_package(benchmark REQUIRED)
  add_executable(bench_fec bench/bench_fec.cpp)
  target_link_libraries(bench_fec PRIVATE rsfec benchmark::benchmark)

  add_executable(bench_fec_cache bench/bench_fec_cache.cpp)
  target_include_directories(bench_fec_cache PRIVATE bench)
  target_link_libraries(bench_fec_cache PRIVATE rsfec)
endif()
