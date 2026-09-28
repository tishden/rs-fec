// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Denis Tishkov

// Forward error correction for the sender -> receiver datagram stream.
//
// ============================================================================
// What this is
// ============================================================================
//
// A systematic Reed-Solomon erasure code over GF(256) with optional
// interleaving, packaged as two standalone classes:
//
//   fec::Encoder -- sender side. Feed every outgoing datagram; it groups them
//                   k at a time and emits r parity datagrams per group.
//   fec::Decoder -- receiver side. Feed every received data and parity
//                   datagram; when a group is solvable it returns the
//                   reconstructed original datagrams.
//
// "Systematic" means data packets go on the wire unmodified -- FEC is purely
// additive, a receiver that ignores parity packets keeps working. Recovery
// needs no round-trip: any k of a group's k+r packets reconstruct the rest,
// so a loss is repaired as soon as enough of the group has arrived --
// typically microseconds after the loss, versus an RTT-scale NACK retransmit.
//
// Sequence numbers: group membership is positional (member j of a group is
// first_seq + j*depth), so the encoder needs to know which seqs are adjacent.
// Feed seqs that number the data stream consecutively. A gap, repeat or
// reset is tolerated -- the affected group is closed early, exactly like a
// flush, and a new one opens at the new seq -- it just costs a short group.
//
// Trust: the decoder bounds its memory (FEC_MAX_PENDING) and never reads or
// writes out of bounds on malformed parity, but it has no authentication.
// Anyone who can inject parity packets can make it reconstruct garbage;
// authenticate the stream below this layer if that matters.
//
// ============================================================================
// Parameters and what they buy (runtime Params, bounded at compile time)
// ============================================================================
//
//   k      data packets per group.       overhead = r/k, larger k amortizes
//   r      parity packets per group.     parity but widens the loss window.
//   depth  interleaving depth D: consecutive packets round-robin across D
//          concurrently open groups (packet with seq s joins group s mod D).
//   flush_us  close a partially filled group this long after its first
//          packet, so parity is never postponed indefinitely at low rate.
//
//   overhead            r/k parity packets per data packet
//   burst tolerance     any r losses per group; a burst of B consecutive
//                       packets touches ceil(B/D) packets in each group =>
//                       fully recoverable while B <= r*D
//   recovery latency    the rest of the group must arrive: <= k*D packet
//                       intervals (+ flush_us bound at low rate)
//
// Picking parameters is a three-way trade between overhead, burst tolerance
// and recovery latency; there is no free lunch:
//
//   k=8 r=1 D=1   12.5% overhead, singles only        -- iid loss, tight budget
//   k=8 r=2 D=1   25% overhead, bursts <= 2           -- default
//   k=8 r=2 D=4   25% overhead, bursts <= 8, 4x latency -- bursty channels
//
// Residual failure (> r losses in a group) at iid loss p is
// ~ C(k+r, r+1) p^(r+1): for k=8 r=2 at p=1% that is ~1e-4 per group, ~1e-5
// per packet -- pair with a retransmit fallback if delivery must be complete.
//
// ============================================================================
// Wire format of one coded unit
// ============================================================================
//
// Each datagram is coded as the unit [u16 length | datagram bytes | zeros],
// padded to the longest unit in its group; parity packets carry the group
// descriptor (first_seq, k_actual, depth, index) plus the coded bytes, so a
// reconstructed unit is self-describing: read the length, and the bytes are
// the original datagram, headers included. Partial (flushed) groups simply
// carry k_actual < k. Parity of group [f, f+D, ..., f+(k-1)D], index i, is
//
//     parity_i = sum_j  C(i,j) * unit_j        (sums in GF(256))
//
// with C a Cauchy matrix: C(i,j) = 1/(x_i ^ y_j), x_i = 255-i, y_j = j.
// Every square submatrix of a Cauchy matrix is invertible, which is exactly
// the erasure-code property: any e <= r missing units are found by
// subtracting the present units from e parities and solving the e x e system.
//
// ============================================================================
// Cost
// ============================================================================
//
// Encoding is r addmul passes (gf256.h) per fed packet; decoding e losses is
// ~(k_present * e + e^2) addmul passes over one group. With the SSSE3 kernel
// an addmul over a 1400-byte unit is ~100ns; k=8 r=2 encode adds ~200ns per
// data packet. Buffers are preallocated; no heap traffic on the hot path
// (Decoder::pending_ is a vector, reserved up front to FEC_MAX_PENDING).
//
// Unit lengths ride in a u16, so a zero-length datagram is a valid unit and
// is recovered like any other.
//
// Compile-time bounds (override with -D):
//   FEC_MAX_UNIT   max datagram size codable            (default 1500)
//   FEC_MAX_K      max data packets per group           (default 32)
//   FEC_MAX_R      max parity packets per group         (default 8)
//   FEC_MAX_DEPTH  max interleaving depth               (default 16)
//   FEC_WINDOW     decoder's data-retention ring, slots (default 1024, pow2)
//                  Reconstruction only needs the packets of groups still in
//                  flight, which span k*depth; the assert below fixes the
//                  requirement at twice the widest configurable geometry.
//                  Each slot costs FEC_MAX_UNIT bytes, so this is the single
//                  number that decides whether the receive path fits in cache:
//                  1024 slots is 1.5 MB, the former 8192 was 12 MB -- more
//                  than the last-level cache of the machines it was measured
//                  on, so the ring evicted itself on every lap.
//   FEC_MAX_PENDING  groups the decoder keeps waiting for more packets
//                  (default 64). When full, the oldest is abandoned. This is
//                  what bounds decoder memory against a stream of parity for
//                  groups whose data never arrives.
//   FEC_NO_SIMD    force the scalar GF kernel
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "rsfec/gf256.h"

#ifndef FEC_MAX_UNIT
#define FEC_MAX_UNIT 1500
#endif
#ifndef FEC_MAX_K
#define FEC_MAX_K 32
#endif
#ifndef FEC_MAX_R
#define FEC_MAX_R 8
#endif
#ifndef FEC_MAX_DEPTH
#define FEC_MAX_DEPTH 16
#endif
#ifndef FEC_WINDOW
#define FEC_WINDOW 1024
#endif
#ifndef FEC_MAX_PENDING
#define FEC_MAX_PENDING 64
#endif

namespace fec {

inline constexpr uint32_t kMaxUnit = FEC_MAX_UNIT;
inline constexpr uint32_t kMaxK = FEC_MAX_K;
inline constexpr uint32_t kMaxR = FEC_MAX_R;
inline constexpr uint32_t kMaxDepth = FEC_MAX_DEPTH;
inline constexpr uint32_t kWindow = FEC_WINDOW;
inline constexpr uint32_t kMaxPending = FEC_MAX_PENDING;
inline constexpr uint32_t kUnitCap = 2 + kMaxUnit;  // [u16 len | bytes]
// Parity rows are padded to whole cache lines (see Encoder::Group).
inline constexpr uint32_t kAccStride = (kUnitCap + 63) / 64 * 64;

static_assert(kMaxK + kMaxR <= 200, "Cauchy points must stay distinct");
static_assert((kWindow & (kWindow - 1)) == 0, "FEC_WINDOW must be pow2");
static_assert(kWindow >= 2 * kMaxK * kMaxDepth, "window smaller than a block");
static_assert(kMaxUnit <= 0xffff - 2, "unit length and coded_len are u16");
static_assert(kMaxK <= 0xff && kMaxDepth <= 0xff, "k and depth ride in u8");
static_assert(kMaxPending >= kMaxDepth, "one pending group per open group");

struct Params {
  uint16_t k = 8;
  uint16_t r = 2;
  uint16_t depth = 1;
  uint32_t flush_us = 100;

  bool valid() const {
    return k >= 1 && k <= kMaxK && r >= 1 && r <= kMaxR && depth >= 1 &&
           depth <= kMaxDepth;
  }
};

// Cauchy coefficient for parity row i, data position j.
inline uint8_t coeff(uint32_t i, uint32_t j) {
  return gf256::inv(static_cast<uint8_t>((255 - i) ^ j));
}

// One parity datagram produced by the Encoder. `data` points into the
// encoder's group buffer and is valid until the next feed()/poll() call.
struct Parity {
  uint64_t first_seq;   // seq of the group's first data packet
  uint8_t k_actual;     // data packets actually in the group (< k if flushed)
  uint8_t index;        // parity row, 0..r-1
  uint8_t depth;        // group stride: members are first_seq + j*depth
  uint16_t coded_len;   // bytes of coded payload (2 + longest datagram)
  const uint8_t* data;
};

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------
class Encoder {
 public:
  struct Stats {
    uint64_t groups_full = 0;   // closed by reaching k
    uint64_t groups_flushed = 0;  // closed by flush timeout
    uint64_t groups_cut = 0;      // closed early by a seq gap/repeat/reset
    uint64_t rejected = 0;        // datagrams longer than FEC_MAX_UNIT
    uint64_t parity_out = 0;
    uint64_t parity_bytes = 0;
  };

  // Throws std::invalid_argument unless p.valid().
  explicit Encoder(const Params& p) : p_(checked(p)), r_now_(p.r) {
    for (auto& g : groups_) g.open = false;
    for (uint32_t d = 0; d < kMaxDepth; ++d) slot_[d] = static_cast<uint8_t>(d);
    spare_ = kMaxDepth;
  }

  // Adjust the parity count for groups opened from now on (capped at the
  // configured r; 0 suspends parity emission entirely). Safe mid-stream: a
  // group carries its own parity set and the decoder works per group, so the
  // sender may cheapen redundancy under load and restore it when pressure
  // subsides.
  void set_r(uint16_t r) { r_now_ = r < p_.r ? r : p_.r; }
  uint16_t current_r() const { return r_now_; }

  // Feed one outgoing data datagram. seq should number the data stream
  // consecutively (see "Sequence numbers" above). Emitted parity packets, if
  // the packet closed a group, are appended to `out`.
  //
  // Returns false, and codes nothing, for a datagram longer than
  // FEC_MAX_UNIT: it still goes on the wire but is not protected. The seq it
  // occupied then reads as a gap, which closes its group early, so the
  // decoder never mistakes a neighbour for it.
  bool feed(uint64_t seq, const uint8_t* pkt, uint32_t len, uint64_t now_ns,
            std::vector<Parity>& out) {
    if (len > kMaxUnit) {
      ++stats_.rejected;
      return false;
    }
    const uint32_t d = static_cast<uint32_t>(seq % p_.depth);
    // Membership is positional: the decoder will look for member j at
    // first_seq + j*depth. A seq that is not that slot would be attributed
    // to the wrong packet on recovery, so close what we have instead. The
    // closed group's rows are the parity just handed out, so the new group
    // takes the spare buffer rather than overwriting them in this same call.
    {
      Group& old = groups_[slot_[d]];
      if (old.open &&
          seq != old.first_seq + static_cast<uint64_t>(old.count) * p_.depth) {
        close(old, out);
        ++stats_.groups_cut;
        const uint8_t t = slot_[d];
        slot_[d] = spare_;
        spare_ = t;
      }
    }
    Group& g = groups_[slot_[d]];
    if (!g.open) {
      g.open = true;
      g.first_seq = seq;
      g.count = 0;
      g.max_len = 0;
      g.open_ts = now_ns;
      g.r = r_now_;
      // Zeroing is deferred: clearing the full 1502-byte row here costs the
      // same memory traffic whether the packets are 100 bytes or 1400, and at
      // small payloads that is an order of magnitude more than the group will
      // ever touch. Rows are cleared lazily below, up to the longest packet
      // seen -- which is exactly the region that becomes the parity payload.
      g.zeroed = 0;
    }
    {
      const uint32_t need = 2 + len;
      if (need > g.zeroed) {
        for (uint32_t i = 0; i < g.r; ++i)
          std::memset(g.acc[i] + g.zeroed, 0, need - g.zeroed);
        g.zeroed = need;
      }
    }
    const uint8_t j = static_cast<uint8_t>(g.count);
    uint8_t len_le[2] = {static_cast<uint8_t>(len & 0xff),
                         static_cast<uint8_t>(len >> 8)};
    // Parity rows are fed in fused pairs: one pass over the packet updates
    // two accumulators (gf256::addmul2) -- the hot shape for the r=2 default.
    uint32_t i = 0;
    for (; i + 1 < g.r; i += 2) {
      const uint8_t ca = coeff(i, j);
      const uint8_t cb = coeff(i + 1, j);
      gf256::addmul2(g.acc[i], g.acc[i + 1], len_le, 2, ca, cb);
      gf256::addmul2(g.acc[i] + 2, g.acc[i + 1] + 2, pkt, len, ca, cb);
    }
    for (; i < g.r; ++i) {
      const uint8_t c = coeff(i, j);
      gf256::addmul(g.acc[i], len_le, 2, c);
      gf256::addmul(g.acc[i] + 2, pkt, len, c);
    }
    if (len > g.max_len) g.max_len = len;
    ++g.count;
    if (g.count == p_.k) {
      close(g, out);
      ++stats_.groups_full;
    }
    return true;
  }

  // Close overdue partial groups (call regularly from the idle loop).
  void poll(uint64_t now_ns, std::vector<Parity>& out) {
    const uint64_t flush_ns = static_cast<uint64_t>(p_.flush_us) * 1000;
    for (uint32_t d = 0; d < p_.depth; ++d) {
      Group& g = groups_[slot_[d]];
      if (g.open && g.count > 0 && now_ns - g.open_ts >= flush_ns) {
        close(g, out);
        ++stats_.groups_flushed;
      }
    }
  }

  const Stats& stats() const { return stats_; }

 private:
  struct Group {
    bool open;
    uint64_t first_seq;
    uint32_t count;
    uint32_t max_len;
    uint64_t open_ts;
    uint16_t r;  // parity rows for THIS group (r_now_ at open time)
    uint32_t zeroed;  // prefix of every row known to be clear
    // Rows are padded to a cache line so each starts on its own, and the
    // accumulate loop's loads and stores stay aligned.
    alignas(64) uint8_t acc[kMaxR][kAccStride];
  };

  static const Params& checked(const Params& p) {
    if (!p.valid()) throw std::invalid_argument("fec::Params out of range");
    return p;
  }

  void close(Group& g, std::vector<Parity>& out) {
    const uint16_t coded_len = static_cast<uint16_t>(2 + g.max_len);
    for (uint32_t i = 0; i < g.r; ++i) {
      out.push_back(Parity{g.first_seq, static_cast<uint8_t>(g.count),
                           static_cast<uint8_t>(i),
                           static_cast<uint8_t>(p_.depth), coded_len,
                           g.acc[i]});
      ++stats_.parity_out;
      stats_.parity_bytes += coded_len;
    }
    g.open = false;
  }

  Params p_;
  uint16_t r_now_;
  // One buffer per interleave lane plus a spare that a cut group retires
  // into; slot_ maps lane -> buffer. Indices, not pointers, so the Encoder
  // stays copyable.
  Group groups_[kMaxDepth + 1];
  uint8_t slot_[kMaxDepth];
  uint8_t spare_;
  Stats stats_;
};

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------
// Keeps a ring of the last FEC_WINDOW received datagrams (reconstruction
// needs the group's surviving packets), and a small list of groups that are
// waiting for enough packets to solve. Groups that fall behind the stream by
// more than half a window, or are pushed out of a full pending list, are
// abandoned (`unrecoverable`) -- a retransmit layer, if there is one, remains
// responsible for them.
class Decoder {
 public:
  struct Stats {
    uint64_t recovered = 0;          // data packets reconstructed
    uint64_t groups_solved = 0;      // groups that needed reconstruction
    uint64_t groups_intact = 0;      // parity arrived but nothing was missing
    uint64_t unrecoverable = 0;      // groups abandoned (losses > parity)
    uint64_t evicted = 0;            // of those, pushed out by FEC_MAX_PENDING
    uint64_t parity_rx = 0;
    uint64_t parity_dup = 0;
  };

  // A datagram reconstructed by feed_parity()/feed_data(). `data` points into
  // the decoder's scratch buffer, valid until the next feed_* call.
  struct Recovered {
    uint64_t seq;
    uint32_t len;
    const uint8_t* data;
  };

  Decoder() {
    for (auto& s : window_) s.seq = kNoSeq;
    pending_.reserve(kMaxPending);
  }

  // Feed every received data datagram (including retransmitted ones).
  // May complete a pending group: reconstructed packets land in `out`.
  void feed_data(uint64_t seq, const uint8_t* pkt, uint32_t len,
                 std::vector<Recovered>& out) {
    if (len > kMaxUnit) return;
    Slot& s = window_[seq & (kWindow - 1)];
    s.seq = seq;
    s.len = len;
    std::memcpy(s.data, pkt, len);
    if (seq > max_seq_) max_seq_ = seq;

    for (size_t i = 0; i < pending_.size();) {
      PGroup& g = pending_[i];
      if (member_pos(g, seq) >= 0 && try_solve(g, out)) {
        remove_at(i);
        continue;
      }
      ++i;
    }
    gc();
  }

  // Feed a received parity datagram.
  void feed_parity(uint64_t first_seq, uint8_t k_actual, uint8_t depth,
                   uint8_t index, const uint8_t* coded, uint16_t coded_len,
                   std::vector<Recovered>& out) {
    ++stats_.parity_rx;
    if (k_actual == 0 || k_actual > kMaxK || depth == 0 ||
        depth > kMaxDepth || index >= kMaxR || coded_len > kUnitCap) {
      return;
    }
    PGroup* g = nullptr;
    for (auto& p : pending_) {
      if (p.first_seq == first_seq) { g = &p; break; }
    }
    if (!g) {
      if (pending_.size() == kMaxPending) evict_oldest();
      pending_.emplace_back();
      g = &pending_.back();
      g->first_seq = first_seq;
      g->k = k_actual;
      g->depth = depth;
      g->n_par = 0;
    }
    for (uint8_t i = 0; i < g->n_par; ++i) {
      if (g->par_idx[i] == index) { ++stats_.parity_dup; return; }
    }
    if (g->n_par == kMaxR) return;
    g->par_idx[g->n_par] = index;
    g->par_len[g->n_par] = coded_len;
    std::memcpy(g->par[g->n_par], coded, coded_len);
    ++g->n_par;

    if (try_solve(*g, out)) remove_at(static_cast<size_t>(g - pending_.data()));
    gc();
  }

  const Stats& stats() const { return stats_; }
  size_t pending_groups() const { return pending_.size(); }

 private:
  static constexpr uint64_t kNoSeq = ~0ull;

  struct Slot {
    uint64_t seq;
    uint32_t len;
    uint8_t data[kMaxUnit];
  };

  struct PGroup {
    uint64_t first_seq;
    uint8_t k, depth, n_par;
    uint8_t par_idx[kMaxR];
    uint16_t par_len[kMaxR];
    alignas(16) uint8_t par[kMaxR][kUnitCap];
  };

  // Position of seq inside group g, or -1.
  int member_pos(const PGroup& g, uint64_t seq) const {
    if (seq < g.first_seq) return -1;
    const uint64_t d = seq - g.first_seq;
    if (d % g.depth != 0) return -1;
    const uint64_t j = d / g.depth;
    return j < g.k ? static_cast<int>(j) : -1;
  }

  const Slot* present(uint64_t seq) const {
    const Slot& s = window_[seq & (kWindow - 1)];
    return s.seq == seq ? &s : nullptr;
  }

  // Order of pending_ carries no meaning, so removal is a swap with the last
  // entry rather than an erase that shifts every later ~12 KB group down.
  void remove_at(size_t i) {
    if (i + 1 != pending_.size()) pending_[i] = pending_.back();
    pending_.pop_back();
  }

  void evict_oldest() {
    size_t oldest = 0;
    for (size_t i = 1; i < pending_.size(); ++i)
      if (pending_[i].first_seq < pending_[oldest].first_seq) oldest = i;
    remove_at(oldest);
    ++stats_.unrecoverable;
    ++stats_.evicted;
  }

  // Attempt reconstruction. True if the group is finished (solved or intact).
  bool try_solve(PGroup& g, std::vector<Recovered>& out) {
    uint8_t miss[kMaxR];
    uint32_t n_miss = 0;
    for (uint32_t j = 0; j < g.k; ++j) {
      if (!present(g.first_seq + static_cast<uint64_t>(j) * g.depth)) {
        if (n_miss == kMaxR) return false;  // more losses than any parity
        miss[n_miss++] = static_cast<uint8_t>(j);
      }
    }
    if (n_miss == 0) { ++stats_.groups_intact; return true; }
    if (n_miss > g.n_par) return false;  // keep waiting

    const uint32_t e = n_miss;
    uint16_t coded_len = 0;
    for (uint8_t i = 0; i < g.n_par; ++i)
      if (g.par_len[i] > coded_len) coded_len = g.par_len[i];

    // residual_i = parity_i - sum(present units); done in place on e parities,
    // with parity rows updated in fused pairs per present unit (addmul2).
    for (uint32_t j = 0; j < g.k; ++j) {
      const Slot* s = present(g.first_seq + static_cast<uint64_t>(j) * g.depth);
      if (!s) continue;
      uint8_t len_le[2] = {static_cast<uint8_t>(s->len & 0xff),
                           static_cast<uint8_t>(s->len >> 8)};
      uint32_t a = 0;
      for (; a + 1 < e; a += 2) {
        const uint8_t ca = coeff(g.par_idx[a], j);
        const uint8_t cb = coeff(g.par_idx[a + 1], j);
        gf256::addmul2(g.par[a], g.par[a + 1], len_le, 2, ca, cb);
        gf256::addmul2(g.par[a] + 2, g.par[a + 1] + 2, s->data, s->len, ca,
                       cb);
      }
      for (; a < e; ++a) {
        const uint8_t c = coeff(g.par_idx[a], j);
        gf256::addmul(g.par[a], len_le, 2, c);
        gf256::addmul(g.par[a] + 2, s->data, s->len, c);
      }
    }

    // Solve the e x e system M * units = residuals, M[a][b] = C(idx_a, miss_b).
    uint8_t m[kMaxR][kMaxR], id[kMaxR][kMaxR];
    for (uint32_t a = 0; a < e; ++a) {
      for (uint32_t b = 0; b < e; ++b) {
        m[a][b] = coeff(g.par_idx[a], miss[b]);
        id[a][b] = (a == b) ? 1 : 0;
      }
    }
    for (uint32_t col = 0; col < e; ++col) {  // Gauss-Jordan over GF(256)
      uint32_t piv = col;
      while (piv < e && m[piv][col] == 0) ++piv;
      if (piv == e) return false;  // cannot happen for Cauchy; be safe
      if (piv != col) {
        for (uint32_t b = 0; b < e; ++b) {
          std::swap(m[piv][b], m[col][b]);
          std::swap(id[piv][b], id[col][b]);
        }
      }
      const uint8_t s = gf256::inv(m[col][col]);
      for (uint32_t b = 0; b < e; ++b) {
        m[col][b] = gf256::mul(m[col][b], s);
        id[col][b] = gf256::mul(id[col][b], s);
      }
      for (uint32_t a = 0; a < e; ++a) {
        if (a == col || m[a][col] == 0) continue;
        const uint8_t f = m[a][col];
        for (uint32_t b = 0; b < e; ++b) {
          m[a][b] ^= gf256::mul(f, m[col][b]);
          id[a][b] ^= gf256::mul(f, id[col][b]);
        }
      }
    }

    // units_b = sum_a inv[b][a] * residual_a; emit and reinsert into window.
    for (uint32_t b = 0; b < e; ++b) {
      uint8_t* unit = scratch_[b];
      std::memset(unit, 0, coded_len);
      for (uint32_t a = 0; a < e; ++a)
        gf256::addmul(unit, g.par[a], coded_len, id[b][a]);
      const uint32_t len = unit[0] | (unit[1] << 8);
      if (len > kMaxUnit || len + 2u > coded_len) continue;
      const uint64_t seq =
          g.first_seq + static_cast<uint64_t>(miss[b]) * g.depth;
      Slot& s = window_[seq & (kWindow - 1)];
      s.seq = seq;
      s.len = len;
      std::memcpy(s.data, unit + 2, len);
      out.push_back(Recovered{seq, len, unit + 2});
      ++stats_.recovered;
    }
    ++stats_.groups_solved;
    return true;
  }

  // Abandon groups the stream has left more than a window behind.
  void gc() {
    for (size_t i = 0; i < pending_.size();) {
      const PGroup& g = pending_[i];
      const uint64_t horizon =
          g.first_seq + static_cast<uint64_t>(g.k) * g.depth + kWindow / 2;
      if (max_seq_ > horizon) {
        ++stats_.unrecoverable;
        remove_at(i);
        continue;
      }
      ++i;
    }
  }

  Slot window_[kWindow];

  std::vector<PGroup> pending_;
  alignas(16) uint8_t scratch_[kMaxR][kUnitCap];
  uint64_t max_seq_ = 0;
  Stats stats_;
};

}  // namespace fec
