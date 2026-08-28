// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Denis Tishkov

// Deterministic packet-loss injection with configurable burstiness.
//
// Model: Gilbert-Elliott, the standard two-state Markov loss model. The
// channel is either Good (delivers) or Bad (drops). Parameterized by the two
// numbers an operator actually thinks in:
//
//   pct        stationary loss rate, percent (fraction of packets dropped)
//   burst_len  mean length of a loss burst, packets (1 = independent losses,
//              i.e. the classic Bernoulli coin flip)
//
// Internally: P(Bad->Good) = beta = 1/burst_len (bursts are geometric with
// mean burst_len), and P(Good->Bad) = alpha = p*beta/(1-p) so the stationary
// Bad probability is alpha/(alpha+beta) = p.
//
// Fixed-run variant (fixed_len >= 1): every loss event drops EXACTLY
// fixed_len consecutive packets -- the "0.01%/0.1% loss with runs of 1/2/3"
// axis of the final matrix, where geometric burst lengths would blur the
// point. A run is triggered on a delivering packet with probability
//   q = p / (L*(1-p) + p)
// so the stationary drop fraction is L / (1/q + L - 1) = p exactly. Back-to-
// back triggers can concatenate runs into multiples of L; at matrix rates
// (p <= 1%) that is ~q*L^2 rare and the run-length histogram stays pinned
// at L.
//
// Driven by xorshift64*, seeded explicitly: a given (pct, burst, seed) always
// drops exactly the same packets, so experiments are reproducible and any
// case can be replayed -- the reason we do not use netem for this.
#pragma once

#include <cstdint>

namespace loss {

class Model {
 public:
  Model(double pct, double burst_len, uint64_t seed, unsigned fixed_len = 0)
      : rng_(seed | 1), fixed_len_(fixed_len) {
    const double p = pct / 100.0;
    if (p <= 0.0) return;
    if (fixed_len >= 1) {
      const double q = p / (fixed_len * (1.0 - p) + p);
      to_bad_ = static_cast<uint64_t>(q * 4294967296.0);
      enabled_ = true;
    } else if (burst_len >= 1.0) {
      const double beta = 1.0 / burst_len;
      const double alpha = p * beta / (1.0 - p);
      // Thresholds in 32-bit space for integer-only comparisons on the path.
      to_bad_ = static_cast<uint64_t>(alpha * 4294967296.0);
      to_good_ = static_cast<uint64_t>(beta * 4294967296.0);
      enabled_ = true;
    }
  }

  // One decision per packet: true = drop it. GE mode: the state transition
  // happens first, then the packet shares the fate of the resulting state --
  // this ordering makes burst lengths exactly geometric with mean burst_len.
  bool drop() {
    if (!enabled_) return false;
    if (fixed_len_ >= 1) {
      if (run_left_ > 0) {
        --run_left_;
        ++dropped_;
        return true;
      }
      if ((next() >> 32) < to_bad_) {
        run_left_ = fixed_len_ - 1;
        ++dropped_;
        return true;
      }
      return false;
    }
    const uint64_t roll = next() >> 32;
    if (bad_) {
      if (roll < to_good_) bad_ = false;
    } else {
      if (roll < to_bad_) bad_ = true;
    }
    if (bad_) {
      ++dropped_;
      return true;
    }
    return false;
  }

  uint64_t dropped() const { return dropped_; }

 private:
  uint64_t next() {
    rng_ ^= rng_ >> 12;
    rng_ ^= rng_ << 25;
    rng_ ^= rng_ >> 27;
    return rng_ * 0x2545f4914f6cdd1dull;
  }

  uint64_t rng_;
  uint64_t to_bad_ = 0;
  uint64_t to_good_ = 0;
  unsigned fixed_len_ = 0;
  unsigned run_left_ = 0;
  bool bad_ = false;
  bool enabled_ = false;
  uint64_t dropped_ = 0;
};

}  // namespace loss
