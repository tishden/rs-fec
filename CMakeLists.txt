# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.16)
project(rsfec VERSION 0.1.0 LANGUAGES CXX
        DESCRIPTION "Systematic Reed-Solomon erasure coding over GF(256)")

# Tests default on only when rsfec is the top-level project, not when it is
# pulled in with add_subdirectory/FetchContent.
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  set(_rsfec_top ON)
else()
  set(_rsfec_top OFF)
endif()
option(RSFEC_BUILD_TESTS "Build the tests and the example" ${_rsfec_top})
option(RSFEC_BUILD_BENCH "Build the Google Benchmark microbenchmarks" OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Header-only: consumers just need the include path. No architecture flags
# are imposed on them -- the SIMD kernels carry per-function target
# attributes and are picked at runtime by CPUID, so a baseline build still
# runs GFNI on hardware that has it.
add_library(rsfec INTERFACE)
add_library(rsfec::rsfec ALIAS rsfec)
target_include_directories(rsfec INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(rsfec INTERFACE cxx_std_17)

install(DIRECTORY include/rsfec DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS rsfec EXPORT rsfecTargets)
install(EXPORT rsfecTargets NAMESPACE rsfec::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rsfec)
# find_package(rsfec) needs a Config file next to the exported targets.
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/rsfecConfig.cmake
     "include(\"\${CMAKE_CURRENT_LIST_DIR}/rsfecTargets.cmake\")\n")
# Pre-1.0: the wire format may still change between minor versions.
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/rsfecConfigVersion.cmake
  COMPATIBILITY SameMinorVersion ARCH_INDEPENDENT)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/rsfecConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/rsfecConfigVersion.cmake
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
