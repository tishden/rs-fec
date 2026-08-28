// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Denis Tishkov

// FEC csim, two claims:
//   (1) the FPGA encoder's parity is byte-identical to the golden fec::Encoder;
//   (2) the FPGA decoder reconstructs the missing datagrams whenever
//       losses <= r -- checked over every one of the C(8,2) loss pairs.
#include <cstdio>
#include <cstring>
#include <vector>
#include "fec_hls.h"
#include "rsfec/fec.h"  // golden: the scalar C++ reference

static std::vector<uint8_t> gen(int len, uint32_t seed) {
  std::vector<uint8_t> v(len);
  uint32_t x = seed * 2654435761u + 1;
  for (int i = 0; i < len; ++i) { x = x * 1664525u + 1013904223u; v[i] = x >> 24; }
  return v;
}

int main() {
  Gf256Tab t; gf256_init(t);
  const int K = 8, R = 2;

  // A group of K datagrams of differing length, like a mixed stream.
  std::vector<std::vector<uint8_t>> data;
  for (int j = 0; j < K; ++j) data.push_back(gen(j % 3 == 2 ? 300 : 100 + j, 7 + j));
  int maxlen = 0; for (auto &d : data) maxlen = std::max(maxlen, (int)d.size());

  // --- (1) FPGA encoder vs golden ---
  FecEnc g; fec_enc_open(g, R);
  for (int j = 0; j < K; ++j) {
    ap_uint<8> buf[FEC_HLS_MAXUNIT];
    for (size_t b = 0; b < data[j].size(); ++b) buf[b] = data[j][b];
    fec_enc_add(g, t, buf, data[j].size());
  }
  const int coded = 2 + maxlen;

  fec::Params pp; pp.k = K; pp.r = R; pp.depth = 1; pp.flush_us = 100000;
  fec::Encoder enc(pp);
  std::vector<fec::Parity> parity;
  for (int j = 0; j < K; ++j)
    enc.feed(j, data[j].data(), data[j].size(), 0, parity);
  if ((int)parity.size() != R) { fprintf(stderr, "tb_fec: golden emitted %zu parity packets\n", parity.size()); return 1; }

  for (int i = 0; i < R; ++i) {
    if (parity[i].coded_len != coded) { fprintf(stderr, "coded_len %d!=%d\n", parity[i].coded_len, coded); return 1; }
    for (int b = 0; b < coded; ++b)
      if ((uint8_t)g.acc[i][b] != parity[i].data[b]) {
        fprintf(stderr, "parity[%d] byte %d: fpga %d golden %d\n", i, b, (int)g.acc[i][b], parity[i].data[b]);
        return 1;
      }
  }
  printf("tb_fec: encoder byte-identical to golden (K=%d R=%d, coded=%d)\n", K, R, coded);

  // --- (2) decoder: drop 2 datagrams, rebuild them from the 2 parities ---
  for (int drop_a = 0; drop_a < K; ++drop_a)
    for (int drop_b = drop_a + 1; drop_b < K; ++drop_b) {
      static ap_uint<8> dbuf[FEC_HLS_MAXK][FEC_HLS_UNITCAP];
      int dl[FEC_HLS_MAXK]; bool dpres[FEC_HLS_MAXK];
      for (int j = 0; j < K; ++j) {
        if (j == drop_a || j == drop_b) { dpres[j] = false; dl[j] = 0; continue; }
        for (size_t b = 0; b < data[j].size(); ++b) dbuf[j][b] = data[j][b];
        dpres[j] = true; dl[j] = data[j].size();
      }
      static ap_uint<8> pbuf[FEC_HLS_MAXR][FEC_HLS_UNITCAP]; bool ppres[FEC_HLS_MAXR];
      for (int i = 0; i < FEC_HLS_MAXR; ++i) ppres[i] = false;
      for (int i = 0; i < R; ++i) {
        for (int b = 0; b < coded; ++b) pbuf[i][b] = parity[i].data[b];
        ppres[i] = true;
      }
      FecDecCtx ctx{K, coded};
      static ap_uint<8> outd[FEC_HLS_MAXK][FEC_HLS_UNITCAP]; int outl[FEC_HLS_MAXK];
      bool ok = fec_decode(t, ctx, dbuf, dl, dpres, pbuf, ppres, outd, outl);
      if (!ok) { fprintf(stderr, "decode fail drop %d,%d\n", drop_a, drop_b); return 1; }
      for (int dj : {drop_a, drop_b}) {
        if (outl[dj] != (int)data[dj].size()) { fprintf(stderr, "len %d: %d!=%zu\n", dj, outl[dj], data[dj].size()); return 1; }
        for (size_t b = 0; b < data[dj].size(); ++b)
          if ((uint8_t)outd[dj][b] != data[dj][b]) { fprintf(stderr, "recover %d byte %zu\n", dj, b); return 1; }
      }
    }
  printf("tb_fec: decoder recovered all C(8,2)=28 loss pairs\n");
  return 0;
}
