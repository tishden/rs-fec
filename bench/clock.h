// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Denis Tishkov

// Wall clock for the microbenchmarks. CLOCK_REALTIME rather than a monotonic
// clock so that numbers are comparable with latency samples stamped by
// applications, which is what those usually use; the runs are far shorter
// than any clock adjustment.
#pragma once

#include <chrono>
#include <cstdint>

namespace util {

inline uint64_t now_ns() {
  using namespace std::chrono;
  return static_cast<uint64_t>(
      duration_cast<nanoseconds>(system_clock::now().time_since_epoch())
          .count());
}

}  // namespace util
