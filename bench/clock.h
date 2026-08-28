// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Denis Tishkov

// Monotonic-ish wall clock for the microbenchmarks. Deliberately the same
// CLOCK_REALTIME reading the transport this codec came from uses to stamp
// messages, so a benchmark number and a production latency sample are drawn
// from one clock.
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
