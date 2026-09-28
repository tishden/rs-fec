// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Denis Tishkov

// Assertion tests for the FEC stack: GF(256) kernels, encoder/decoder
// round-trips under loss patterns (random, consecutive bursts, partial
// groups, lost parity), and the Gilbert-Elliott loss model statistics.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include "rsfec/fec.h"
#include "rsfec/gf256.h"
#include "rsfec/loss.h"

// assert() vanishes under NDEBUG, which every Release build defines -- and a
// test binary that checks nothing still exits 0. CHECK is unconditional.
#define CHECK(cond)                                                      \
  do {                                                                   \
    if (!(cond)) {                                                       \
      std::fflush(stdout);                                               \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__,        \
                   __LINE__, #cond);                                     \
      std::abort();                                                      \
    }                                                                    \
  } while (0)

namespace {

std::mt19937 rng(12345);

uint32_t rnd(uint32_t lo, uint32_t hi) {  // inclusive
  return lo + rng() % (hi - lo + 1);
}

void test_gf256_field() {
  for (int a = 1; a < 256; ++a) {
    CHECK(gf256::mul(a, gf256::inv(a)) == 1);
    CHECK(gf256::mul(a, 1) == a);
    CHECK(gf256::mul(a, 0) == 0);
  }
  for (int i = 0; i < 10000; ++i) {
    uint8_t a = rng(), b = rng(), c = rng();
    CHECK(gf256::mul(a, b) == gf256::mul(b, a));
    CHECK(gf256::mul(a, static_cast<uint8_t>(b ^ c)) ==
           (gf256::mul(a, b) ^ gf256::mul(a, c)));
    CHECK(gf256::mul(gf256::mul(a, b), c) ==
           gf256::mul(a, gf256::mul(b, c)));
  }
  printf("test_gf256_field OK\n");
}

// Every kernel the CPU can run must agree byte-for-byte with scalar, both the
// single addmul and the fused addmul2 (vs two independent scalar passes).
void test_gf256_kernels_agree() {
  size_t nk = 0;
  const gf256::KernelInfo* ks = gf256::kernel_table(&nk);
  for (size_t ki = 0; ki < nk; ++ki) {
    const gf256::KernelInfo& k = ks[ki];
    if ((k.need & ~gf256::caps()) != 0) {
      printf("  kernel %-12s SKIPPED (cpu)\n", k.name);
      continue;
    }
    for (int iter = 0; iter < 100; ++iter) {
      const size_t n = rnd(1, 1500);
      std::vector<uint8_t> src(n), a(n), b(n), a2(n), b2(n);
      for (auto& x : src) x = rng();
      for (size_t i = 0; i < n; ++i) {
        a[i] = a2[i] = rng();
        b[i] = b2[i] = rng();
      }
      const uint8_t c0 = rnd(2, 255), c1 = rnd(2, 255);
      gf256::addmul_scalar(a.data(), src.data(), n, c0);
      k.addmul(a2.data(), src.data(), n, c0);
      CHECK(a == a2);
      gf256::addmul_scalar(a.data(), src.data(), n, c0);
      gf256::addmul_scalar(b.data(), src.data(), n, c1);
      k.addmul2(a2.data(), b2.data(), src.data(), n, c0, c1);
      CHECK(a == a2 && b == b2);
    }
    printf("  kernel %-12s OK\n", k.name);
  }
  printf("test_gf256_kernels_agree OK (dispatch=%s)\n", gf256::kernel_name());
}

struct Pkt {
  uint64_t seq;
  std::vector<uint8_t> data;
};

// End-to-end: encode `pkts`, drop `lost_data` seqs and `lost_parity` (by
// emission order), feed the rest to a Decoder (order: parity first if
// parity_first). Returns recovered seq -> bytes.
struct RunResult {
  std::vector<Pkt> recovered;
  fec::Decoder::Stats stats;
};

RunResult run_codec(const fec::Params& prm, const std::vector<Pkt>& pkts,
                    const std::set<uint64_t>& lost_data,
                    const std::set<size_t>& lost_parity, bool parity_first,
                    uint64_t flush_at_ns = 0) {
  fec::Encoder enc(prm);
  std::vector<fec::Parity> par;
  struct StoredParity {
    uint64_t first_seq;
    uint8_t k_actual, index, depth;
    uint16_t coded_len;
    std::vector<uint8_t> data;
  };
  std::vector<StoredParity> stored;
  auto drain = [&]() {
    for (const auto& p : par) {
      stored.push_back({p.first_seq, p.k_actual, p.index, p.depth,
                        p.coded_len,
                        std::vector<uint8_t>(p.data, p.data + p.coded_len)});
    }
    par.clear();
  };
  uint64_t now = 1000;
  for (const auto& p : pkts) {
    enc.feed(p.seq, p.data.data(), static_cast<uint32_t>(p.data.size()), now,
             par);
    drain();
    now += 1000;
  }
  if (flush_at_ns) {
    enc.poll(now + flush_at_ns, par);
    drain();
  }

  auto dec = std::make_unique<fec::Decoder>();
  std::vector<fec::Decoder::Recovered> rec;
  RunResult res;
  auto take = [&]() {
    for (const auto& r : rec) {
      res.recovered.push_back(
          {r.seq, std::vector<uint8_t>(r.data, r.data + r.len)});
    }
    rec.clear();
  };
  auto feed_parities = [&]() {
    for (size_t i = 0; i < stored.size(); ++i) {
      if (lost_parity.count(i)) continue;
      const auto& s = stored[i];
      dec->feed_parity(s.first_seq, s.k_actual, s.depth, s.index,
                       s.data.data(), s.coded_len, rec);
      take();
    }
  };
  auto feed_data = [&]() {
    for (const auto& p : pkts) {
      if (lost_data.count(p.seq)) continue;
      dec->feed_data(p.seq, p.data.data(),
                     static_cast<uint32_t>(p.data.size()), rec);
      take();
    }
  };
  if (parity_first) { feed_parities(); feed_data(); }
  else { feed_data(); feed_parities(); }
  res.stats = dec->stats();
  return res;
}

std::vector<Pkt> make_pkts(uint64_t first_seq, uint32_t n, uint32_t min_len,
                           uint32_t max_len) {
  std::vector<Pkt> v;
  for (uint32_t i = 0; i < n; ++i) {
    Pkt p;
    p.seq = first_seq + i;
    p.data.resize(rnd(min_len, max_len));
    for (auto& b : p.data) b = rng();
    v.push_back(std::move(p));
  }
  return v;
}

// With in-order delivery (data before its group's parity) the decoder must
// recover exactly the lost set. With parity delivered first the decoder is
// deliberately greedy: it may reconstruct packets that were merely still in
// flight (harmless duplicates downstream) -- then we only require that every
// lost packet is among the recoveries and every recovery is byte-exact.
void check_recovered_exact(const RunResult& res, const std::vector<Pkt>& pkts,
                           const std::set<uint64_t>& lost_data,
                           bool allow_extra = false) {
  if (!allow_extra) CHECK(res.recovered.size() == lost_data.size());
  std::set<uint64_t> got;
  for (const auto& r : res.recovered) {
    got.insert(r.seq);
    const Pkt* orig = nullptr;
    for (const auto& p : pkts)
      if (p.seq == r.seq) { orig = &p; break; }
    CHECK(orig && orig->data == r.data);
  }
  for (uint64_t s : lost_data) CHECK(got.count(s));
}

void test_roundtrip_random_losses() {
  for (uint16_t k : {1, 2, 3, 8, 16}) {
    for (uint16_t r : {1, 2, 4}) {
      for (uint16_t d : {1, 2, 4}) {
        fec::Params prm{k, r, d, 100};
        const uint32_t n = k * d * 3;  // three full blocks
        auto pkts = make_pkts(1, n, 1, 1400);
        // Per group, drop up to r random members.
        std::set<uint64_t> lost;
        for (uint32_t g = 0; g < 3 * d; ++g) {
          const uint32_t nl = rnd(0, r);
          for (uint32_t t = 0; t < nl; ++t) {
            // group g covers seqs base + j*d where base = 1 + (g/d)*k*d + g%d
            const uint64_t base = 1 + (g / d) * (uint64_t)k * d + g % d;
            lost.insert(base + static_cast<uint64_t>(rnd(0, k - 1)) * d);
          }
        }
        for (bool pf : {false, true}) {
          auto res = run_codec(prm, pkts, lost, {}, pf);
          check_recovered_exact(res, pkts, lost, /*allow_extra=*/pf);
          CHECK(res.stats.unrecoverable == 0);
        }
      }
    }
  }
  printf("test_roundtrip_random_losses OK\n");
}

void test_burst_recovery() {
  // A burst of exactly r*d consecutive packets must always be recoverable.
  for (uint16_t r : {1, 2}) {
    for (uint16_t d : {1, 2, 4}) {
      const uint16_t k = 8;
      fec::Params prm{k, r, d, 100};
      const uint32_t n = k * d * 2;
      auto pkts = make_pkts(1, n, 100, 1400);
      const uint32_t burst = static_cast<uint32_t>(r) * d;
      const uint64_t start = 1 + d;  // somewhere inside the first block
      std::set<uint64_t> lost;
      for (uint32_t i = 0; i < burst; ++i) lost.insert(start + i);
      auto res = run_codec(prm, pkts, lost, {}, false);
      check_recovered_exact(res, pkts, lost);

      // One more than r*d consecutive: some group now has r+1 losses.
      std::set<uint64_t> lost2 = lost;
      lost2.insert(start + burst);
      auto res2 = run_codec(prm, pkts, lost2, {}, false);
      CHECK(res2.recovered.size() < lost2.size());
    }
  }
  printf("test_burst_recovery OK\n");
}

void test_partial_group_flush() {
  const fec::Params prm{8, 2, 1, 100};
  // Only 3 packets arrive, then the stream pauses: poll() must flush a
  // k_actual=3 group whose parity still recovers losses.
  auto pkts = make_pkts(1, 3, 1, 200);
  std::set<uint64_t> lost{2};
  auto res = run_codec(prm, pkts, lost, {}, false, /*flush_at_ns=*/200000);
  check_recovered_exact(res, pkts, lost);
  printf("test_partial_group_flush OK\n");
}

void test_lost_parity() {
  const fec::Params prm{8, 2, 1, 100};
  auto pkts = make_pkts(1, 16, 1, 1400);
  // Group 1 loses one data packet and one of its two parities -> still fine.
  std::set<uint64_t> lost{3};
  std::set<size_t> lost_par{0};  // first emitted parity (group 1, index 0)
  auto res = run_codec(prm, pkts, lost, lost_par, false);
  check_recovered_exact(res, pkts, lost);
  // Two data losses + one parity lost -> 2 > 1 available: unrecoverable.
  std::set<uint64_t> lost2{3, 5};
  auto res2 = run_codec(prm, pkts, lost2, lost_par, false);
  CHECK(res2.recovered.empty());
  printf("test_lost_parity OK\n");
}

void test_intact_group_and_dup_parity() {
  const fec::Params prm{4, 1, 1, 100};
  auto pkts = make_pkts(1, 4, 10, 100);
  auto res = run_codec(prm, pkts, {}, {}, false);
  CHECK(res.recovered.empty());
  CHECK(res.stats.groups_intact == 1);
  CHECK(res.stats.recovered == 0);

  // The same parity twice while its group is still pending: the second copy
  // is counted and ignored, and the group still solves with one parity.
  const fec::Params p2{4, 2, 1, 100};
  fec::Encoder enc(p2);
  std::vector<fec::Parity> par;
  for (const auto& p : pkts)
    enc.feed(p.seq, p.data.data(), static_cast<uint32_t>(p.data.size()), 0, par);
  CHECK(par.size() == 2);
  auto dec = std::make_unique<fec::Decoder>();
  std::vector<fec::Decoder::Recovered> rec;
  for (int t = 0; t < 2; ++t)
    dec->feed_parity(par[0].first_seq, par[0].k_actual, par[0].depth,
                     par[0].index, par[0].data, par[0].coded_len, rec);
  CHECK(dec->stats().parity_dup == 1);
  CHECK(dec->pending_groups() == 1);
  for (size_t j = 0; j < pkts.size(); ++j) {
    if (j == 2) continue;
    dec->feed_data(pkts[j].seq, pkts[j].data.data(),
                   static_cast<uint32_t>(pkts[j].data.size()), rec);
  }
  CHECK(rec.size() == 1 && rec[0].seq == pkts[2].seq);
  CHECK(dec->pending_groups() == 0);
  printf("test_intact_group_and_dup_parity OK\n");
}

// set_r applies at group-open time; each group carries its own parity count
// and the decoder handles any per-group r (a reduced-r group is just the
// r=1/r=0 params case, whose recovery the roundtrip tests already cover).
void test_encoder_set_r() {
  fec::Params prm{4, 2, 1, 100};
  fec::Encoder enc(prm);
  std::vector<fec::Parity> out;
  uint8_t data[64];
  for (auto& b : data) b = rng();
  uint64_t seq = 1;
  auto feed_group = [&]() {
    out.clear();
    for (int j = 0; j < 4; ++j) enc.feed(seq++, data, sizeof(data), 1, out);
    return out.size();
  };
  CHECK(feed_group() == 2);      // configured r
  enc.set_r(1);
  CHECK(feed_group() == 1);
  enc.set_r(0);
  CHECK(feed_group() == 0);      // parity suspended
  enc.set_r(2);
  CHECK(feed_group() == 2);      // restored
  enc.set_r(7);                   // capped at configured r
  CHECK(enc.current_r() == 2);
  CHECK(feed_group() == 2);
  // Mid-group change affects only the NEXT group.
  out.clear();
  enc.feed(seq++, data, sizeof(data), 1, out);
  enc.set_r(0);
  for (int j = 0; j < 3; ++j) enc.feed(seq++, data, sizeof(data), 1, out);
  CHECK(out.size() == 2);        // group opened at r=2 closes with 2
  CHECK(feed_group() == 0);      // next group uses r=0
  printf("test_encoder_set_r OK\n");
}

// Params outside the compile-time bounds are refused at construction rather
// than faulting later (depth=0 used to be a division by zero in feed()).
void test_params_rejected() {
  const fec::Params bad[] = {{0, 2, 1, 100},
                             {8, 0, 1, 100},
                             {8, 2, 0, 100},
                             {fec::kMaxK + 1, 2, 1, 100},
                             {8, fec::kMaxR + 1, 1, 100},
                             {8, 2, fec::kMaxDepth + 1, 100}};
  for (const auto& p : bad) {
    bool threw = false;
    try {
      fec::Encoder enc(p);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CHECK(threw);
  }
  printf("test_params_rejected OK\n");
}

// A datagram longer than FEC_MAX_UNIT is refused, not coded -- it used to be
// accumulated past the end of the parity row. Its seq then reads as a gap,
// so the group around it closes early and still protects its neighbours.
void test_oversize_rejected() {
  const fec::Params prm{8, 8, fec::kMaxDepth, 100};  // widest rows, last group
  fec::Encoder enc(prm);
  std::vector<fec::Parity> par;
  std::vector<uint8_t> big(fec::kMaxUnit + 1, 0xab);
  CHECK(!enc.feed(fec::kMaxDepth - 1, big.data(),
                  static_cast<uint32_t>(big.size()), 0, par));
  CHECK(enc.stats().rejected == 1 && par.empty());
  big.resize(fec::kMaxUnit);  // exactly at the bound is fine
  CHECK(enc.feed(fec::kMaxDepth - 1, big.data(),
                 static_cast<uint32_t>(big.size()), 0, par));

  // In a stream: seqs 1..8, seq 4 oversize. Group [1,2,3] is cut, [5..8]
  // opens fresh; losing 2 and 6 is still recovered, byte-exact.
  const fec::Params p1{8, 2, 1, 100};
  auto pkts = make_pkts(1, 8, 1, 200);
  pkts[3].data.assign(fec::kMaxUnit + 1, 0xcd);
  std::vector<Pkt> sent;
  for (const auto& p : pkts)
    if (p.data.size() <= fec::kMaxUnit) sent.push_back(p);
  auto res = run_codec(p1, pkts, {2, 6}, {}, false, /*flush_at_ns=*/200000);
  check_recovered_exact(res, sent, {2, 6});
  printf("test_oversize_rejected OK\n");
}

// A gap in seq inside a group used to shift every later member by one slot:
// the decoder "recovered" the seq that was never sent, carrying the bytes of
// the packet after it. Now the encoder cuts the group at the gap.
void test_seq_gap() {
  for (uint16_t d : {1, 2, 4}) {
    const fec::Params prm{8, 2, d, 100};
    auto all = make_pkts(1, 8u * d * 3, 1, 300);
    std::vector<Pkt> pkts;  // drop a few seqs from the stream entirely
    for (const auto& p : all)
      if (p.seq != 7 && p.seq != 8 && p.seq != 20 && p.seq != 33) pkts.push_back(p);
    const std::set<uint64_t> lost{3, 10, 22};
    for (bool pf : {false, true}) {
      auto res = run_codec(prm, pkts, lost, {}, pf, /*flush_at_ns=*/200000);
      check_recovered_exact(res, pkts, lost, /*allow_extra=*/pf);
      for (const auto& r : res.recovered)
        CHECK(r.seq != 7 && r.seq != 8 && r.seq != 20 && r.seq != 33);
    }
  }
  // Repeats and resets cut the group too, and nothing is misattributed.
  fec::Encoder enc(fec::Params{8, 2, 1, 100});
  std::vector<fec::Parity> par;
  uint8_t b[16] = {};
  for (uint64_t s : {10, 11, 11, 3, 4}) enc.feed(s, b, sizeof(b), 0, par);
  CHECK(enc.stats().groups_cut == 2);
  printf("test_seq_gap OK\n");
}

// Zero-length datagrams are valid units and must come back when lost (they
// used to be dropped while the group was still counted as solved).
void test_zero_length() {
  const fec::Params prm{4, 1, 1, 100};
  auto pkts = make_pkts(1, 4, 1, 50);
  pkts[1].data.clear();
  auto res = run_codec(prm, pkts, {2}, {}, false);
  check_recovered_exact(res, pkts, {2});
  CHECK(res.recovered[0].data.empty());
  printf("test_zero_length OK\n");
}

// Parity for groups whose data never shows up must not grow the decoder
// without bound; the oldest pending group gives way.
void test_pending_bounded() {
  auto dec = std::make_unique<fec::Decoder>();
  std::vector<fec::Decoder::Recovered> rec;
  uint8_t z[16] = {};
  const uint64_t n = 10 * fec::kMaxPending;
  for (uint64_t i = 0; i < n; ++i)
    dec->feed_parity(1000000 + i * 64, 8, 1, 0, z, sizeof(z), rec);
  CHECK(dec->pending_groups() == fec::kMaxPending);
  CHECK(dec->stats().evicted == n - fec::kMaxPending);
  CHECK(rec.empty());

  // Legit traffic afterwards still decodes.
  auto pkts = make_pkts(1, 8, 1, 200);
  fec::Encoder enc(fec::Params{8, 1, 1, 100});
  std::vector<fec::Parity> par;
  for (const auto& p : pkts)
    enc.feed(p.seq, p.data.data(), static_cast<uint32_t>(p.data.size()), 0, par);
  for (const auto& p : pkts)
    if (p.seq != 5)
      dec->feed_data(p.seq, p.data.data(),
                     static_cast<uint32_t>(p.data.size()), rec);
  dec->feed_parity(par[0].first_seq, par[0].k_actual, par[0].depth,
                   par[0].index, par[0].data, par[0].coded_len, rec);
  CHECK(rec.size() == 1 && rec[0].seq == 5);
  CHECK(std::memcmp(rec[0].data, pkts[4].data.data(), rec[0].len) == 0);
  printf("test_pending_bounded OK\n");
}

void test_loss_model_stats() {
  for (double pct : {0.1, 1.0, 5.0}) {
    for (double burst : {1.0, 4.0, 8.0}) {
      loss::Model m(pct, burst, 42);
      const uint64_t n = 4000000;
      uint64_t drops = 0, bursts = 0, cur = 0, burst_total = 0;
      for (uint64_t i = 0; i < n; ++i) {
        if (m.drop()) {
          ++drops;
          ++cur;
        } else if (cur) {
          ++bursts;
          burst_total += cur;
          cur = 0;
        }
      }
      const double rate = 100.0 * static_cast<double>(drops) / n;
      const double mean_burst =
          bursts ? static_cast<double>(burst_total) / bursts : 0.0;
      CHECK(rate > pct * 0.85 && rate < pct * 1.15);
      CHECK(mean_burst > burst * 0.85 && mean_burst < burst * 1.15);
    }
  }
  printf("test_loss_model_stats OK\n");
}

// Fixed-run mode: the drop rate must hit the target AND every loss run must
// be exactly L packets (multiples of L only via back-to-back triggers, which
// must stay rare at low loss rates).
void test_loss_model_fixed() {
  for (double pct : {0.01, 0.1, 1.0}) {
    for (unsigned L : {1u, 2u, 3u}) {
      loss::Model m(pct, 1.0, 42, L);
      const uint64_t n = 20000000;
      uint64_t drops = 0, runs = 0, exact = 0, cur = 0;
      for (uint64_t i = 0; i < n; ++i) {
        if (m.drop()) {
          ++drops;
          ++cur;
        } else if (cur) {
          ++runs;
          CHECK(cur % L == 0);  // runs are always whole multiples of L
          if (cur == L) ++exact;
          cur = 0;
        }
      }
      const double rate = 100.0 * static_cast<double>(drops) / n;
      CHECK(rate > pct * 0.9 && rate < pct * 1.1);
      // Concatenated runs (2L+) appear with probability ~q per run boundary;
      // worst case here is p=1%, L=1 (q=1% -> ~99% exact). 0.97 keeps margin.
      CHECK(runs > 0 && exact > runs * 0.97);
      CHECK(m.dropped() == drops);
    }
  }
  printf("test_loss_model_fixed OK\n");
}

// Rates the two-state model cannot reach with the requested burst length
// keep the rate (bursts lengthen instead); 100% drops everything. Both used
// to go wrong silently: a capped rate, and a division by zero.
void test_loss_model_edges() {
  for (double pct : {60.0, 90.0}) {
    loss::Model m(pct, 1.0, 7);
    const uint64_t n = 2000000;
    uint64_t drops = 0;
    for (uint64_t i = 0; i < n; ++i) drops += m.drop();
    const double rate = 100.0 * static_cast<double>(drops) / n;
    CHECK(rate > pct * 0.97 && rate < pct * 1.03);
  }
  loss::Model all(100.0, 4.0, 7);
  for (int i = 0; i < 1000; ++i) CHECK(all.drop());
  loss::Model none(0.0, 4.0, 7);
  for (int i = 0; i < 1000; ++i) CHECK(!none.drop());
  printf("test_loss_model_edges OK\n");
}

}  // namespace

int main() {
  test_gf256_field();
  test_gf256_kernels_agree();
  test_roundtrip_random_losses();
  test_burst_recovery();
  test_partial_group_flush();
  test_lost_parity();
  test_intact_group_and_dup_parity();
  test_encoder_set_r();
  test_params_rejected();
  test_oversize_rejected();
  test_seq_gap();
  test_zero_length();
  test_pending_bounded();
  test_loss_model_stats();
  test_loss_model_fixed();
  test_loss_model_edges();
  printf("ALL FEC TESTS PASSED\n");
  return 0;
}
