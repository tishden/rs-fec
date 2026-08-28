// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Denis Tishkov

// Cross-check the FPGA GF(256) against the golden rsfec/gf256.h: every
// product, every inverse and the Cauchy coefficients must agree byte for byte.
// This is what licenses the claim that host parity and FPGA parity interoperate.
#include <cstdio>
#include "gf256_hls.h"
#include "rsfec/gf256.h"  // golden: the scalar C++ reference

int main() {
  Gf256Tab t;
  gf256_init(t);
  const auto &g = gf256::tables();

  int fails = 0;
  for (int a = 0; a < 256 && fails < 10; ++a)
    for (int b = 0; b < 256; ++b) {
      ap_uint<8> got = gf_mul(t, a, b);
      uint8_t want = g.mul[a][b];
      if ((uint8_t)got != want) {
        fprintf(stderr, "mul(%d,%d): got %d want %d\n", a, b, (int)got, want);
        if (++fails >= 10) break;
      }
    }
  for (int a = 1; a < 256; ++a) {
    ap_uint<8> got = gf_inv(t, a);
    if ((uint8_t)got != g.inv[a]) {
      fprintf(stderr, "inv(%d): got %d want %d\n", a, (int)got, g.inv[a]);
      ++fails;
    }
  }
  // Cauchy: C(i,j) = inv((255-i)^j), the construction fec.h uses
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 32; ++j) {
      ap_uint<8> got = gf_cauchy(t, i, j);
      uint8_t want = g.inv[(uint8_t)(255 - i) ^ (uint8_t)j];
      if ((uint8_t)got != want) {
        fprintf(stderr, "cauchy(%d,%d): got %d want %d\n", i, j, (int)got, want);
        ++fails;
      }
    }
  if (fails) { fprintf(stderr, "tb_gf: FAIL (%d)\n", fails); return 1; }
  printf("tb_gf: OK (65536 mul + 255 inv + cauchy)\n");
  return 0;
}
