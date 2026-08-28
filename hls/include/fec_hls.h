// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Denis Tishkov

// RS-FEC over GF(256) for FPGA -- byte-compatible with include/rsfec/fec.h.
// Systematic code: data goes on the wire unmodified, parity is purely
// additive. A unit is [u16 len | datagram | zeros], padded to the longest
// datagram in the group. parity_i = sum_j C(i,j) * unit_j, with C the Cauchy
// matrix from gf256_hls.h.
//
// depth = 1 here (the default geometry is r=2, D=1). Interleaving D > 1 is the
// same code applied to D independent groups -- whoever feeds this routes
// seq % D -- so it is not unrolled at this level.
#pragma once
#include "gf256_hls.h"

#ifndef FEC_HLS_MAXUNIT
#define FEC_HLS_MAXUNIT 1500
#endif
#define FEC_HLS_UNITCAP (2 + FEC_HLS_MAXUNIT)  // == fec::kUnitCap
#define FEC_HLS_MAXR 8
#define FEC_HLS_MAXK 32

// --- Group encoder: r accumulators, units added one at a time ---
struct FecEnc {
  ap_uint<8> acc[FEC_HLS_MAXR][FEC_HLS_UNITCAP];
  int count;    // units added so far (this is the position j)
  int max_len;  // longest datagram in the group
  int r;
};

inline void fec_enc_open(FecEnc &g, int r) {
  g.count = 0;
  g.max_len = 0;
  g.r = r;
  for (int i = 0; i < r; ++i)
    for (int b = 0; b < FEC_HLS_UNITCAP; ++b) g.acc[i][b] = 0;
}

// Add a datagram (len + bytes) as the unit at position j = count.
inline void fec_enc_add(FecEnc &g, const Gf256Tab &t, const ap_uint<8> *dgram,
                        int len) {
  const int j = g.count;
  ap_uint<8> unit_len0 = len & 0xff, unit_len1 = (len >> 8) & 0xff;
  for (int i = 0; i < g.r; ++i) {
    const ap_uint<8> c = gf_cauchy(t, i, j);
    // acc[i] ^= c * unit, where unit = [len_le | dgram]
    g.acc[i][0] ^= gf_mul(t, c, unit_len0);
    g.acc[i][1] ^= gf_mul(t, c, unit_len1);
  add_body:
    for (int b = 0; b < len; ++b) {
#pragma HLS pipeline II = 1
      g.acc[i][2 + b] ^= gf_mul(t, c, dgram[b]);
    }
  }
  if (len > g.max_len) g.max_len = len;
  g.count++;
}

// The finished parity's coded_len is 2 + max_len (same as fec.h); acc[i] holds
// the parity bytes.

// --- Decoder: reconstruct the group's missing units ---
// HLS does not support pointer-to-pointer in synthesisable functions, so data
// and parity arrive as 2D arrays plus presence masks rather than as arrays of
// pointers. data[j] is a datagram (length in data_len[j], present when
// data_present[j]); par[i] holds coded bytes (present when par_present[i]).
// Missing units are reconstructed into out_data[j], length into out_len[j].
struct FecDecCtx {
  int k;          // data units in the group (k_actual)
  int coded_len;  // 2 + max_len
};

inline bool fec_decode(const Gf256Tab &t, const FecDecCtx &ctx,
                       const ap_uint<8> data[FEC_HLS_MAXK][FEC_HLS_UNITCAP],
                       const int data_len[FEC_HLS_MAXK],
                       const bool data_present[FEC_HLS_MAXK],
                       const ap_uint<8> par[FEC_HLS_MAXR][FEC_HLS_UNITCAP],
                       const bool par_present[FEC_HLS_MAXR],
                       ap_uint<8> out_data[FEC_HLS_MAXK][FEC_HLS_UNITCAP],
                       int out_len[FEC_HLS_MAXK]) {
  // Partition along bytes (dim 2) so the unrolled byte loops below get
  // independent banks instead of queueing on one memory port.
#pragma HLS array_partition variable = data cyclic factor = 8 dim = 2
#pragma HLS array_partition variable = out_data cyclic factor = 8 dim = 2
  int miss[FEC_HLS_MAXR];
  int e = 0;
  for (int j = 0; j < ctx.k; ++j)
    if (!data_present[j]) {
      if (e >= FEC_HLS_MAXR) return false;
      miss[e++] = j;
    }
  if (e == 0) return true;

  // Pick e available parities.
  int par_idx[FEC_HLS_MAXR];
  int p = 0;
  for (int i = 0; i < FEC_HLS_MAXR && p < e; ++i)
    if (par_present[i]) par_idx[p++] = i;
  if (p < e) return false;  // not enough parity to solve

  // Residual[row][byte] = par_i XOR sum_{j present} C(i,j) * unit_j
  // (unit_j = [len_le | dgram | zeros], coded_len bytes long)
  static ap_uint<8> res[FEC_HLS_MAXR][FEC_HLS_UNITCAP];
  // Complete partition along rows: Gaussian elimination touches res[row] and
  // res[col] in the same cycle, and only separate banks keep that free of
  // port contention (this is what removes the II violations in norm_b/elim_b).
#pragma HLS array_partition variable = res complete dim = 1
  // Cyclic along bytes: the unrolled byte loops then issue N gf_mul per cycle.
#pragma HLS array_partition variable = res cyclic factor = 8 dim = 2
  for (int row = 0; row < e; ++row) {
    const int i = par_idx[row];
    for (int b = 0; b < ctx.coded_len; ++b) res[row][b] = par[i][b];
    for (int j = 0; j < ctx.k; ++j) {
      if (!data_present[j]) continue;
      const ap_uint<8> c = gf_cauchy(t, i, j);
      const int len = data_len[j];
      ap_uint<8> l0 = len & 0xff, l1 = (len >> 8) & 0xff;
      res[row][0] ^= gf_mul(t, c, l0);
      res[row][1] ^= gf_mul(t, c, l1);
    resid_b:  // pipelining breaks the gf_mul combinational path (closes 300 MHz)
      for (int b = 0; b < len; ++b) {
#pragma HLS pipeline II = 1
#pragma HLS unroll factor = 8
        res[row][2 + b] ^= gf_mul(t, c, data[j][b]);
      }
    }
  }

  // Matrix A[row][col] = C(par_idx[row], miss[col]); solve A * X = res.
  static ap_uint<8> A[FEC_HLS_MAXR][FEC_HLS_MAXR];
#pragma HLS array_partition variable = A complete dim = 0
  for (int row = 0; row < e; ++row)
    for (int col = 0; col < e; ++col)
      A[row][col] = gf_cauchy(t, par_idx[row], miss[col]);

  // Gaussian elimination in GF(256): reduce [A | res] to identity on the left.
  for (int col = 0; col < e; ++col) {
    // pivot
    int piv = -1;
    for (int row = col; row < e; ++row)
      if (A[row][col] != 0) { piv = row; break; }
    // A Cauchy submatrix is always invertible, so this cannot fire on
    // well-formed input; it is here so corrupt metadata fails cleanly.
    if (piv < 0) return false;
    if (piv != col) {
      for (int c2 = 0; c2 < e; ++c2) { auto tmp = A[col][c2]; A[col][c2] = A[piv][c2]; A[piv][c2] = tmp; }
      for (int b = 0; b < ctx.coded_len; ++b) { auto tmp = res[col][b]; res[col][b] = res[piv][b]; res[piv][b] = tmp; }
    }
    const ap_uint<8> invp = gf_inv(t, A[col][col]);
    for (int c2 = 0; c2 < e; ++c2) A[col][c2] = gf_mul(t, A[col][c2], invp);
  norm_b:
    for (int b = 0; b < ctx.coded_len; ++b) {
#pragma HLS pipeline II = 1
#pragma HLS unroll factor = 8
      res[col][b] = gf_mul(t, res[col][b], invp);
    }
    for (int row = 0; row < e; ++row) {
      if (row == col) continue;
      const ap_uint<8> f = A[row][col];
      if (f == 0) continue;
      for (int c2 = 0; c2 < e; ++c2) A[row][c2] ^= gf_mul(t, f, A[col][c2]);
    elim_b:
      for (int b = 0; b < ctx.coded_len; ++b) {
#pragma HLS pipeline II = 1
#pragma HLS unroll factor = 8
        res[row][b] ^= gf_mul(t, f, res[col][b]);
      }
    }
  }

  // res[col] is now the reconstructed unit_j for miss[col].
  for (int col = 0; col < e; ++col) {
    const int j = miss[col];
    int len = (int)res[col][0] | ((int)res[col][1] << 8);
    out_len[j] = len;
  out_b:  // pipelined output: breaks the len_reg -> out_data BRAM address path
    for (int b = 0; b < len; ++b) {
#pragma HLS pipeline II = 1
      out_data[j][b] = res[col][2 + b];
    }
  }
  return true;
}
