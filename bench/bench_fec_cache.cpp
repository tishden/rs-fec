// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Denis Tishkov

// Cache behaviour of the FEC codec, isolated from the network.
//
// The end-to-end runs say the decoder costs about 0.34 extra cache misses per
// message (0.107 on a clean receive against 0.448 with FEC, measured on bare
// metal). That number cannot be optimised against directly -- it mixes the
// codec with everything else the receiver does. This benchmark runs the codec
// alone over a realistic packet stream so `perf stat` attributes misses to it,
// and prints ns and bytes per packet so a change can be judged on both.
//
// No google-benchmark dependency on purpose: it has to build anywhere the
// transport builds.
//
//   make bin/bench_fec_cache
//   perf stat -e cache-references,cache-misses ./bin/bench_fec_cache decode ...
//
// Args: <encode|decode> <k> <r> <depth> <unit_bytes> <packets>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "rsfec/fec.h"
#include "clock.h"

namespace {

// A stream of packets whose payloads differ, so the codec cannot benefit from
// the same bytes staying hot by accident.
std::vector<std::vector<uint8_t>> make_packets(uint32_t n, uint32_t unit) {
  std::vector<std::vector<uint8_t>> pkts(n);
  uint64_t x = 0x9e3779b97f4a7c15ull;
  for (uint32_t i = 0; i < n; ++i) {
    pkts[i].resize(unit);
    for (uint32_t j = 0; j < unit; ++j) {
      x ^= x << 13; x ^= x >> 7; x ^= x << 17;
      pkts[i][j] = static_cast<uint8_t>(x);
    }
  }
  return pkts;
}

int run_encode(const fec::Params& p, uint32_t unit, uint32_t n) {
  auto pkts = make_packets(n, unit);
  fec::Encoder enc(p);
  std::vector<fec::Parity> out;
  out.reserve(64);
  const uint64_t t0 = util::now_ns();
  for (uint32_t i = 0; i < n; ++i) {
    out.clear();
    enc.feed(i, pkts[i].data(), unit, t0, out);
  }
  const uint64_t dt = util::now_ns() - t0;
  const auto& s = enc.stats();
  printf("encode k=%u r=%u d=%u unit=%u packets=%u\n", p.k, p.r, p.depth, unit, n);
  printf("  %.1f ns/packet, parity out %llu (%.2f per packet), %.2f parity bytes/packet\n",
         double(dt) / n, (unsigned long long)s.parity_out,
         double(s.parity_out) / n, double(s.parity_bytes) / n);
  return 0;
}

int run_decode(const fec::Params& p, uint32_t unit, uint32_t n) {
  // Encode first and keep the parity, then replay the stream with one data
  // packet dropped per group -- the shape a real loss takes, and the only one
  // that makes the decoder do its work.
  auto pkts = make_packets(n, unit);
  fec::Encoder enc(p);
  // 12 MB of retention ring -- heap, not stack (the receiver does the same).
  auto decp = std::make_unique<fec::Decoder>();
  fec::Decoder& dec = *decp;
  std::vector<fec::Parity> parity;
  std::vector<std::vector<uint8_t>> par_bytes;
  struct Par { uint64_t seq; uint8_t k, idx, depth; uint16_t len; size_t at; };
  std::vector<Par> pars;
  const uint64_t t0 = util::now_ns();
  for (uint32_t i = 0; i < n; ++i) {
    parity.clear();
    enc.feed(i, pkts[i].data(), unit, t0, parity);
    for (const auto& q : parity) {
      par_bytes.emplace_back(q.data, q.data + q.coded_len);
      pars.push_back({q.first_seq, q.k_actual, q.index, q.depth, q.coded_len,
                      par_bytes.size() - 1});
    }
  }
  const uint64_t t_enc = util::now_ns() - t0;

  std::vector<fec::Decoder::Recovered> rec;
  const uint64_t t1 = util::now_ns();
  size_t pi = 0;
  for (uint32_t i = 0; i < n; ++i) {
    rec.clear();
    if (i % p.k != 0) dec.feed_data(i, pkts[i].data(), unit, rec);
    while (pi < pars.size() && pars[pi].seq <= i) {
      const Par& q = pars[pi];
      dec.feed_parity(q.seq, q.k, q.depth, q.idx, par_bytes[q.at].data(), q.len,
                      rec);
      ++pi;
    }
  }
  const uint64_t dt = util::now_ns() - t1;
  const auto& s2 = dec.stats();
  printf("decode k=%u r=%u d=%u unit=%u packets=%u\n", p.k, p.r, p.depth, unit, n);
  printf("  %.1f ns/packet (encode side %.1f), recovered %llu, groups solved %llu\n",
         double(dt) / n, double(t_enc) / n,
         (unsigned long long)s2.recovered, (unsigned long long)s2.groups_solved);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string what = argc > 1 ? argv[1] : "encode";
  fec::Params p;
  p.k = argc > 2 ? std::atoi(argv[2]) : 8;
  p.r = argc > 3 ? std::atoi(argv[3]) : 3;
  p.depth = argc > 4 ? std::atoi(argv[4]) : 4;
  p.flush_us = 1000000;  // never flush on time: the benchmark drives groups full
  const uint32_t unit = argc > 5 ? std::atoi(argv[5]) : 100;
  const uint32_t n = argc > 6 ? std::atoi(argv[6]) : 200000;
  if (!p.valid()) { fprintf(stderr, "bad params\n"); return 2; }
  return what == "decode" ? run_decode(p, unit, n) : run_encode(p, unit, n);
}
