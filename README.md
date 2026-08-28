# rsfec

Systematic Reed-Solomon erasure coding over GF(256), for packet streams that
cannot afford a round trip.

Header-only C++17, no dependencies. Ships with an HLS mirror of the same codec
that is proven byte-identical to the C++ one, so a host and an FPGA can encode
and repair each other's packets.

```cpp
#include <rsfec/fec.h>

fec::Params p;                       // k=8 data, r=2 parity, no interleaving
fec::Encoder enc(p);
std::vector<fec::Parity> parity;

// Sender: feed every outgoing datagram. Completing a group appends r parity
// packets to `parity`, which you put on the wire alongside the data.
enc.feed(seq, datagram, len, now_ns, parity);

// Receiver: feed everything that arrives. Reconstructed datagrams land in
// `out` as soon as a group becomes solvable -- no round trip involved.
fec::Decoder dec;                    // geometry is read off the parity packets
std::vector<fec::Decoder::Recovered> out;
dec.feed_data(seq, datagram, len, out);
dec.feed_parity(par.first_seq, par.k_actual, par.depth, par.index,
                par.data, par.coded_len, out);
```

A complete runnable version is [`examples/roundtrip.cpp`](examples/roundtrip.cpp)
(`make example`), which drops two of eight packets and prints the rebuilt ones.

## What it is for

A dropped packet normally costs a round trip: the receiver notices the gap,
asks, the sender resends. On a 100 us path that is 100 us of added tail latency
on top of however long detection took, and it happens precisely when the
network is already unhappy.

This is the other trade. The sender emits `r` parity packets per `k` data
packets; any `k` of the `k+r` reconstruct the rest. A loss is repaired as soon
as enough of its group has arrived -- microseconds, no round trip, and the
repair path does not touch the network again. You pay `r/k` bandwidth for it,
always, whether or not anything was lost.

"Systematic" means data packets go on the wire unmodified. FEC is purely
additive: a receiver that ignores parity packets keeps working, which makes it
deployable one end at a time.

## Choosing k, r and depth

Three knobs, a three-way trade, and no free lunch:

| | overhead | burst tolerance | recovery latency |
|---|---|---|---|
| `k=8 r=1 D=1` | 12.5% | single losses | short |
| `k=8 r=2 D=1` | 25% | bursts <= 2 | short |
| `k=8 r=2 D=4` | 25% | bursts <= 8 | 4x |

`depth` (D) is interleaving: consecutive packets round-robin across D
concurrently open groups, so a burst of B consecutive losses hits `ceil(B/D)`
packets in each group. It buys burst tolerance with latency, not bandwidth.

Residual failure -- more than `r` losses in one group -- is about
`C(k+r, r+1) * p^(r+1)` at iid loss `p`. For `k=8 r=2` at `p=1%` that is ~1e-4
per group. If delivery has to be complete rather than merely much better, pair
this with a retransmit fallback; the two compose well, because FEC removes the
common case and leaves the retransmit path handling only the rare one.

## Performance

The hot operation is `addmul(dst, src, n, c): dst[i] ^= c * src[i]`. Seven
implementations of it are compiled into the binary and selected at runtime by
CPUID, so a build made on an old machine still uses GFNI on a new one.

Measured with `bench/bench_fec.cpp` (1400-byte packets):

| kernel | AWS m7i (Sapphire Rapids) | i7-3820 (Sandy Bridge) |
|---|---|---|
| `scalar` | 2.61 GiB/s | 1.5 GB/s |
| `ssse3` PSHUFB | -- | 10.3 GB/s |
| `avx128` VEX PSHUFB | -- | 13.2 GB/s |
| `avx2` | 33.8 GiB/s | dispatch-ready |
| `gfni128` | 25.0 GiB/s | not available |
| `gfni_avx512` | 43.0 GiB/s | not available |
| `gfni_avx2` | **47.6 GiB/s** | not available |

End to end that is 147 ns per packet to encode at `k=8 r=4` with SIMD, against
~2.4 us scalar. On Sandy Bridge, `addmul2` -- the fused form that feeds two
parity accumulators from one pass over the source -- is worth a further 19%,
and moved `k=8 r=2` encode from 297 to 208 ns/packet.

Two honest caveats from those runs:

* **The dispatcher is not always right.** On m7i it picks `gfni_avx512`, which
  is ~10% *slower* than `gfni_avx2` there -- the usual 512-bit frequency
  penalty. Pinning the kernel is a one-line change; auto-detecting this is not,
  and has not been done.
* **SIMD buys headroom, not median latency.** In the transport this came from,
  `--fec-simd 1` vs `0` at 200-300k messages/s was p50 14.0 vs 13.9 us -- no
  difference. The scalar kernel merely used ~70% of a core instead of ~4%. The
  threshold where it starts to matter was around 400k messages/s.

## Field choice

GF(256) under the AES reduction polynomial 0x11b, chosen so that Intel's GFNI
`GF2P8MULB` multiplies natively in exactly this field -- one instruction per
16/32/64 bytes. Erasure coding needs only field axioms and inverses (the Cauchy
construction does not need a primitive element), so any irreducible polynomial
would do; this one is free on modern hardware.

Parity row `i` of a group is `sum_j C(i,j) * unit_j` with `C` a Cauchy matrix,
`C(i,j) = 1/(x_i ^ y_j)`, `x_i = 255-i`, `y_j = j`. Every square submatrix of a
Cauchy matrix is invertible, which is exactly the erasure property: any `e <= r`
missing units are recovered by subtracting the present ones from `e` parities
and solving the resulting `e x e` system.

## The HLS mirror

`hls/` is the same codec for AMD/Xilinx Vitis HLS -- the encoder and the
Gaussian-elimination decoder, written for synthesis (combinational GF multiply,
banked accumulators, `II=1` byte loops).

It is not a port that is *believed* to match. `hls/tb/tb_gf.cpp` checks all
65536 products, all 255 inverses and the Cauchy coefficients against the C++
implementation; `hls/tb/tb_fec.cpp` checks that the encoder's parity is
byte-identical and that the decoder recovers every one of the C(8,2) loss pairs.
Both run under plain `g++` -- the only thing Vitis HLS is needed for is its
`ap_int.h`:

```
make -C hls tb HLS_INC=/path/to/Vitis_HLS/2024.1/include
```

(csim prints `Bitsize mismatch` warnings from `ap_int`'s runtime checks on the
`^=` compound assignments. They are cosmetic; the values are verified against
the golden implementation on every path.)

This codec has run on silicon -- an Alveo U50 exchanging FEC-protected
datagrams with a host running the C++ side of the same code.

## Build

```
make test              # both dispatch ends: best-available kernel, and scalar
make bench             # needs Google Benchmark
```

or

```
cmake -B build -DRSFEC_BUILD_TESTS=ON && cmake --build build && ctest --test-dir build
```

Header-only, so integrating is just an include path; a CMake `rsfec::rsfec`
interface target is exported for convenience.

## Compile-time bounds

Override with `-D`:

| | default | |
|---|---|---|
| `FEC_MAX_UNIT` | 1500 | largest codable datagram |
| `FEC_MAX_K` | 32 | data packets per group |
| `FEC_MAX_R` | 8 | parity packets per group |
| `FEC_MAX_DEPTH` | 16 | interleaving depth |
| `FEC_WINDOW` | 1024 | decoder retention ring, slots (power of two) |
| `FEC_NO_SIMD` | -- | force the scalar kernel |

`FEC_WINDOW` is the one that decides whether the receive path fits in cache:
1024 slots is 1.5 MB. An earlier 8192 was 12 MB -- larger than the last-level
cache of the machines this ran on, so the ring evicted itself every lap.

## Also here

`include/rsfec/loss.h` is a deterministic Gilbert-Elliott loss model
(two-state Markov, parameterised as loss rate + mean burst length, plus a
fixed-run variant). Seeded explicitly, so a given `(pct, burst, seed)` always
drops exactly the same packets and any experiment replays. That property is
why it exists instead of `netem`.

## License and contributing

Apache-2.0, Copyright 2026 Denis Tishkov. Contributions require a signed CLA —
see [CONTRIBUTING.md](CONTRIBUTING.md).

Written with Claude (Anthropic). The design decisions, the measurements and
the silicon runs are mine; the code was written in collaboration with it.
