// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Denis Tishkov

// GF(2^8) for the FPGA side of the codec. Same field as include/rsfec/gf256.h:
// GF(256) under the AES polynomial 0x11b. It has to be the same field, or host
// parity and FPGA parity do not agree and a packet repaired on one side is
// garbage on the other.
//
// The host builds its tables generator-free (peasant multiplication); here
// they are log/exp tables over the primitive element 3 (0x03 is primitive for
// 0x11b). The product does not depend on how it was derived -- it is one
// field -- and tb/tb_gf.cpp proves it by checking all 256x256 products and all
// inverses against the host implementation.
//
// The log/exp tables are 2x256 bytes (BRAM or LUT), built once.
#pragma once
#include <ap_int.h>

struct Gf256Tab {
  // exp[i] = 3^i, duplicated out to 512 so (a+b) needs no modulo on lookup.
  ap_uint<8> exp_[512];
  ap_uint<8> log_[256];
};

// Called once in the kernel before the datapath starts (and in csim likewise).
inline void gf256_init(Gf256Tab &t) {
  ap_uint<16> x = 1;
  for (int i = 0; i < 255; ++i) {
    t.exp_[i] = x.range(7, 0);
    t.log_[x.range(7, 0)] = i;
    // x *= 3, i.e. x ^ (x<<1) reduced by 0x11b
    ap_uint<16> x2 = x << 1;
    if (x2[8]) x2 ^= 0x11b;
    ap_uint<16> x3 = x2 ^ x;  // 3x = 2x + x
    if (x3[8]) x3 ^= 0x11b;
    x = x3 & 0xff;
  }
  t.exp_[255] = 1;  // the cycle has length 255
  for (int i = 255; i < 512; ++i) t.exp_[i] = t.exp_[i - 255];
  t.log_[0] = 0;  // never read: multiplication by 0 is handled explicitly
}

// COMBINATIONAL GF(256) multiply, poly 0x11b: peasant multiplication whose
// eight iterations unroll into plain logic. No tables and therefore no BRAM
// ports, which is the point -- it replicates freely under the unrolled byte
// loops (SIMD on silicon), where a table-based multiply would serialise on
// memory ports. Byte-identical to the table version; tb_gf checks all 256x256
// products. The tables stay for gf_inv, which is off the hot path.
inline ap_uint<8> gf_mul(const Gf256Tab &t, ap_uint<8> a, ap_uint<8> b) {
#pragma HLS inline
  (void)t;
  ap_uint<8> p = 0;
  ap_uint<8> aa = a;
  ap_uint<8> bb = b;
  for (int i = 0; i < 8; ++i) {
#pragma HLS unroll
    if (bb[0]) p ^= aa;
    ap_uint<1> hi = aa[7];
    aa = aa << 1;
    if (hi) aa ^= 0x1b;
    bb = bb >> 1;
  }
  return p;
}

inline ap_uint<8> gf_inv(const Gf256Tab &t, ap_uint<8> a) {
#pragma HLS inline
  // 1/a = 3^(255 - log a)
  return t.exp_[255 - t.log_[a]];
}

// Cauchy matrix coefficient, same construction as rsfec/fec.h:
// C(i,j) = 1/((255-i) ^ j).
inline ap_uint<8> gf_cauchy(const Gf256Tab &t, int i, int j) {
#pragma HLS inline
  ap_uint<8> x = (ap_uint<8>)(255 - i) ^ (ap_uint<8>)j;
  return gf_inv(t, x);
}
