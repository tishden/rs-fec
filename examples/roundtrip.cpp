// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Denis Tishkov

// Smallest useful example: encode a group of 8 datagrams with 2 parity, drop
// two of the data packets on the way, and watch the decoder rebuild them
// without asking the sender for anything.
#include <cstdio>
#include <cstring>
#include <vector>

#include "rsfec/fec.h"

int main() {
  fec::Params p;  // k=8, r=2, depth=1
  fec::Encoder enc(p);
  fec::Decoder dec;  // the decoder learns the geometry from the parity packets

  // Eight datagrams of differing length, as a real stream would be.
  std::vector<std::vector<uint8_t>> sent;
  for (int j = 0; j < p.k; ++j) sent.emplace_back(100 + 17 * j, uint8_t('a' + j));

  std::vector<fec::Parity> parity;
  for (int j = 0; j < p.k; ++j)
    enc.feed(/*seq=*/j, sent[j].data(), sent[j].size(), /*now_ns=*/0, parity);
  printf("encoder emitted %zu parity packets\n", parity.size());

  // The channel eats packets 2 and 5.
  std::vector<fec::Decoder::Recovered> out;
  for (int j = 0; j < p.k; ++j) {
    if (j == 2 || j == 5) continue;
    dec.feed_data(j, sent[j].data(), sent[j].size(), out);
  }
  printf("after %d of %d data packets: %zu recovered\n", p.k - 2, p.k, out.size());

  // Parity arrives; the group becomes solvable on the last one.
  for (const auto& par : parity)
    dec.feed_parity(par.first_seq, par.k_actual, par.depth, par.index,
                    par.data, par.coded_len, out);

  printf("after parity: %zu recovered\n", out.size());
  for (const auto& r : out) {
    const bool ok = r.len == sent[r.seq].size() &&
                    std::memcmp(r.data, sent[r.seq].data(), r.len) == 0;
    printf("  seq %llu: %u bytes, %s\n", (unsigned long long)r.seq, r.len,
           ok ? "byte-identical to what was sent" : "MISMATCH");
    if (!ok) return 1;
  }
  return out.size() == 2 ? 0 : 1;
}
