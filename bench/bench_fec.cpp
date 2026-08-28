// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Denis Tishkov

// Microbenchmarks (Google Benchmark) for the FEC stack: GF(256) kernels,
// encoder feed cost per packet, and full group decode with 1..r losses.
//
// Build: make bench   Run: ./bin/bench_fec [--benchmark_filter=...]
#include <benchmark/benchmark.h>

#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "rsfec/fec.h"
#include "rsfec/gf256.h"

namespace {

constexpr size_t kPkt = 1400;

std::vector<uint8_t> random_buf(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<uint8_t> v(n);
  for (auto& b : v) b = static_cast<uint8_t>(rng());
  return v;
}

void BM_memxor(benchmark::State& state) {
  auto src = random_buf(kPkt, 1);
  auto dst = random_buf(kPkt, 2);
  for (auto _ : state) {
    gf256::memxor(dst.data(), src.data(), kPkt);
    benchmark::DoNotOptimize(dst.data());
  }
  state.SetBytesProcessed(state.iterations() * kPkt);
}
BENCHMARK(BM_memxor);

// Per-kernel addmul / fused addmul2 benchmarks are registered from main() for
// exactly the kernels this CPU can run (see gf256::kernel_table).
void register_kernel_benchmarks() {
  size_t nk = 0;
  const gf256::KernelInfo* ks = gf256::kernel_table(&nk);
  for (size_t ki = 0; ki < nk; ++ki) {
    const gf256::KernelInfo& k = ks[ki];
    if ((k.need & ~gf256::caps()) != 0) continue;
    benchmark::RegisterBenchmark(
        (std::string("BM_addmul_") + k.name).c_str(),
        [fn = k.addmul](benchmark::State& state) {
          auto src = random_buf(kPkt, 1);
          auto dst = random_buf(kPkt, 2);
          for (auto _ : state) {
            fn(dst.data(), src.data(), kPkt, 0x8e);
            benchmark::DoNotOptimize(dst.data());
          }
          state.SetBytesProcessed(state.iterations() * kPkt);
        });
    benchmark::RegisterBenchmark(
        (std::string("BM_addmul2_") + k.name).c_str(),
        [fn = k.addmul2](benchmark::State& state) {
          auto src = random_buf(kPkt, 1);
          auto d0 = random_buf(kPkt, 2);
          auto d1 = random_buf(kPkt, 3);
          for (auto _ : state) {
            fn(d0.data(), d1.data(), src.data(), kPkt, 0x8e, 0x37);
            benchmark::DoNotOptimize(d0.data());
          }
          // Two logical addmuls per pass -- report the effective rate.
          state.SetBytesProcessed(state.iterations() * kPkt * 2);
        });
  }
}

// Per-packet encoder cost, steady state: k/r from the benchmark arguments.
void BM_encoder_feed(benchmark::State& state) {
  const uint16_t k = static_cast<uint16_t>(state.range(0));
  const uint16_t r = static_cast<uint16_t>(state.range(1));
  fec::Encoder enc(fec::Params{k, r, 1, 100});
  auto pkt = random_buf(kPkt, 3);
  std::vector<fec::Parity> out;
  out.reserve(fec::kMaxR);
  uint64_t seq = 1, now = 1;
  for (auto _ : state) {
    enc.feed(seq++, pkt.data(), kPkt, now++, out);
    out.clear();
    benchmark::DoNotOptimize(out.data());
  }
  state.SetBytesProcessed(state.iterations() * kPkt);
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_encoder_feed)
    ->Args({8, 1})->Args({8, 2})->Args({8, 4})->Args({16, 2});

// Whole-group cycle: feed k-e data packets and parities, decoder solves e
// losses. Cost includes the decoder window copies -- i.e. the real receive-
// path price of one group with e losses.
void BM_decode_group(benchmark::State& state) {
  const uint16_t k = static_cast<uint16_t>(state.range(0));
  const uint16_t r = static_cast<uint16_t>(state.range(1));
  const uint32_t e = static_cast<uint32_t>(state.range(2));
  fec::Encoder enc(fec::Params{k, r, 1, 100});
  auto dec = std::make_unique<fec::Decoder>();
  std::vector<fec::Parity> par;
  std::vector<fec::Decoder::Recovered> rec;
  std::vector<std::vector<uint8_t>> pkts;
  for (uint16_t j = 0; j < k; ++j) pkts.push_back(random_buf(kPkt, 10 + j));

  uint64_t seq = 1, now = 1;
  for (auto _ : state) {
    par.clear();
    const uint64_t first = seq;
    for (uint16_t j = 0; j < k; ++j)
      enc.feed(seq++, pkts[j].data(), kPkt, now++, par);
    // Deliver all but the first e data packets, then parities.
    for (uint16_t j = e; j < k; ++j) {
      rec.clear();
      dec->feed_data(first + j, pkts[j].data(), kPkt, rec);
    }
    uint64_t recovered = 0;
    for (const auto& p : par) {
      rec.clear();
      dec->feed_parity(p.first_seq, p.k_actual, p.depth, p.index, p.data,
                       p.coded_len, rec);
      recovered += rec.size();
    }
    if (recovered != e) state.SkipWithError("recovery failed");
    benchmark::DoNotOptimize(recovered);
  }
  state.SetItemsProcessed(state.iterations());  // groups per second
}
BENCHMARK(BM_decode_group)
    ->Args({8, 2, 1})->Args({8, 2, 2})->Args({8, 4, 4})->Args({16, 2, 2});

}  // namespace

int main(int argc, char** argv) {
  printf("gf256 dispatch: %s\n", gf256::kernel_name());
  register_kernel_benchmarks();
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
