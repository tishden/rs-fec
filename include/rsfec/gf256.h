// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Denis Tishkov

// GF(2^8) arithmetic for the FEC codec (see fec.h).
//
// ============================================================================
// Field choice
// ============================================================================
// GF(256) with the AES reduction polynomial x^8+x^4+x^3+x+1 (0x11b). Chosen
// deliberately: Intel GFNI's GF2P8MULB instruction multiplies in exactly this
// field, so on GFNI hardware a coefficient multiply is one instruction per
// 16/32/64 bytes. Erasure coding only needs field axioms and inverses (the
// Cauchy matrix construction does not need a primitive element), so any
// irreducible polynomial works; tables are built generator-free by
// carry-less "peasant" multiplication.
//
// ============================================================================
// Kernels
// ============================================================================
// The one hot operation is
//
//     addmul (dst, src, n, c):          dst[i] ^= c * src[i]
//     addmul2(d0, d1, src, n, c0, c1):  d0[i] ^= c0*src[i], d1[i] ^= c1*src[i]
//
// addmul2 is the fused form the encoder wants for r=2 parity: one pass over
// src feeds two accumulators, halving loads and loop overhead.
//
// Implementations, best-first (selected at runtime by CPUID, no recompilation
// needed to benefit from newer hardware; the binary carries all of them via
// per-function target attributes):
//
//   gfni_avx512  GF2P8MULB zmm, 64 B/step
//   gfni_avx2    GF2P8MULB ymm, 32 B/step
//   gfni128      GF2P8MULB xmm, 16 B/step
//   avx2         PSHUFB split-nibble, ymm, 32 B/step
//   avx128       same as ssse3 but VEX-encoded: 3-operand PSHUFB avoids the
//                per-shuffle table-register copy (helps Sandy/Ivy Bridge)
//   ssse3        classic PSHUFB split-nibble (as in Intel ISA-L), unrolled x2
//   scalar       256-byte table row per coefficient; portable baseline
//
// The full kernel list with availability flags is exposed (kernel_table) so
// tests can cross-check every runnable kernel against scalar and benchmarks
// can measure each one. FEC_NO_SIMD forces scalar dispatch.
//
// All lookup tables (~73 KB) are built once, lazily, at first use.
#pragma once

#include <cpuid.h>
#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace gf256 {

inline constexpr uint16_t kPoly = 0x11b;  // AES field: GF2P8MULB-compatible

struct Tables {
  uint8_t mul[256][256];
  uint8_t inv[256];
  // PSHUFB nibble tables: shuf_lo[c][x] = c*x, shuf_hi[c][x] = c*(x<<4).
  alignas(16) uint8_t shuf_lo[256][16];
  alignas(16) uint8_t shuf_hi[256][16];
};

inline const Tables& tables() {
  static const Tables t = [] {
    Tables tb{};
    auto mul_slow = [](uint8_t a, uint8_t b) {
      uint8_t p = 0;
      while (b) {
        if (b & 1) p ^= a;
        b >>= 1;
        a = static_cast<uint8_t>((a << 1) ^ ((a & 0x80) ? (kPoly & 0xff) : 0));
      }
      return p;
    };
    for (int a = 0; a < 256; ++a)
      for (int b = 0; b < 256; ++b)
        tb.mul[a][b] = mul_slow(static_cast<uint8_t>(a),
                                static_cast<uint8_t>(b));
    for (int a = 1; a < 256; ++a)
      for (int b = 1; b < 256; ++b)
        if (tb.mul[a][b] == 1) { tb.inv[a] = static_cast<uint8_t>(b); break; }
    for (int c = 0; c < 256; ++c) {
      for (int x = 0; x < 16; ++x) {
        tb.shuf_lo[c][x] = tb.mul[c][x];
        tb.shuf_hi[c][x] = tb.mul[c][x << 4];
      }
    }
    return tb;
  }();
  return t;
}

inline uint8_t mul(uint8_t a, uint8_t b) { return tables().mul[a][b]; }
inline uint8_t inv(uint8_t a) { return tables().inv[a]; }

// ---------------------------------------------------------------------------
// CPU feature detection
// ---------------------------------------------------------------------------
enum Caps : uint32_t {
  kSSSE3 = 1u << 0,
  kAVX = 1u << 1,       // + OS ymm state
  kAVX2 = 1u << 2,
  kGFNI = 1u << 3,
  kAVX512BW = 1u << 4,  // + OS zmm state
};

inline uint32_t caps() {
  static const uint32_t c = [] {
    uint32_t r = 0;
    unsigned a, b, cx, d;
    if (!__get_cpuid(1, &a, &b, &cx, &d)) return r;
    if (cx & (1u << 9)) r |= kSSSE3;
    const bool osxsave = cx & (1u << 27);
    uint64_t xcr0 = 0;
    if (osxsave) {  // xgetbv via asm: the intrinsic needs -mxsave to compile
      uint32_t lo, hi;
      __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
      xcr0 = (static_cast<uint64_t>(hi) << 32) | lo;
    }
    const bool ymm_os = (xcr0 & 0x6) == 0x6;
    const bool zmm_os = (xcr0 & 0xe6) == 0xe6;
    if ((cx & (1u << 28)) && ymm_os) r |= kAVX;
    unsigned a7, b7, c7, d7;
    if (__get_cpuid_count(7, 0, &a7, &b7, &c7, &d7)) {
      if ((b7 & (1u << 5)) && ymm_os) r |= kAVX2;
      if (c7 & (1u << 8)) r |= kGFNI;
      if ((b7 & (1u << 30)) && zmm_os) r |= kAVX512BW;
    }
#ifdef FEC_NO_SIMD
    r = 0;
#endif
    return r;
  }();
  return c;
}

// ---------------------------------------------------------------------------
// Scalar kernels
// ---------------------------------------------------------------------------
inline void memxor(uint8_t* dst, const uint8_t* src, size_t n) {
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    uint64_t a, b;
    std::memcpy(&a, dst + i, 8);
    std::memcpy(&b, src + i, 8);
    a ^= b;
    std::memcpy(dst + i, &a, 8);
  }
  for (; i < n; ++i) dst[i] ^= src[i];
}

inline void addmul_scalar(uint8_t* dst, const uint8_t* src, size_t n,
                          uint8_t c) {
  const uint8_t* row = tables().mul[c];
  for (size_t i = 0; i < n; ++i) dst[i] ^= row[src[i]];
}

inline void addmul2_scalar(uint8_t* d0, uint8_t* d1, const uint8_t* src,
                           size_t n, uint8_t c0, uint8_t c1) {
  const uint8_t* r0 = tables().mul[c0];
  const uint8_t* r1 = tables().mul[c1];
  for (size_t i = 0; i < n; ++i) {
    const uint8_t s = src[i];
    d0[i] ^= r0[s];
    d1[i] ^= r1[s];
  }
}

// ---------------------------------------------------------------------------
// PSHUFB split-nibble kernels. The 16/32-byte bodies are macro-shared between
// the SSSE3 and VEX (avx128) builds -- identical intrinsics, different
// encodings; VEX's 3-operand PSHUFB drops the table-register copy that the
// legacy encoding forces on every shuffle.
// ---------------------------------------------------------------------------
#define GF256_SHUF16(SRC, DST_IN)                                             \
  _mm_xor_si128(                                                              \
      DST_IN,                                                                 \
      _mm_xor_si128(                                                          \
          _mm_shuffle_epi8(lo, _mm_and_si128(SRC, nib)),                      \
          _mm_shuffle_epi8(                                                   \
              hi, _mm_and_si128(_mm_srli_epi64(SRC, 4), nib))))

#define GF256_SHUF_KERNEL_BODY()                                              \
  const Tables& t = tables();                                                 \
  const __m128i lo =                                                          \
      _mm_load_si128(reinterpret_cast<const __m128i*>(t.shuf_lo[c]));         \
  const __m128i hi =                                                          \
      _mm_load_si128(reinterpret_cast<const __m128i*>(t.shuf_hi[c]));         \
  const __m128i nib = _mm_set1_epi8(0x0f);                                    \
  size_t i = 0;                                                               \
  for (; i + 32 <= n; i += 32) {                                              \
    const __m128i s0 =                                                        \
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));           \
    const __m128i s1 =                                                        \
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i + 16));      \
    __m128i d0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(dst + i));        \
    __m128i d1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(dst + i + 16));   \
    d0 = GF256_SHUF16(s0, d0);                                                \
    d1 = GF256_SHUF16(s1, d1);                                                \
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), d0);                \
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i + 16), d1);           \
  }                                                                           \
  for (; i + 16 <= n; i += 16) {                                              \
    const __m128i s =                                                         \
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));           \
    __m128i d = _mm_loadu_si128(reinterpret_cast<__m128i*>(dst + i));         \
    d = GF256_SHUF16(s, d);                                                   \
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), d);                 \
  }                                                                           \
  if (i < n) addmul_scalar(dst + i, src + i, n - i, c);

#define GF256_SHUF16_2(SRC, DST_IN, LO, HI)                                   \
  _mm_xor_si128(                                                              \
      DST_IN,                                                                 \
      _mm_xor_si128(                                                          \
          _mm_shuffle_epi8(LO, _mm_and_si128(SRC, nib)),                      \
          _mm_shuffle_epi8(                                                   \
              HI, _mm_and_si128(_mm_srli_epi64(SRC, 4), nib))))

#define GF256_SHUF_KERNEL2_BODY()                                             \
  const Tables& t = tables();                                                 \
  const __m128i lo0 =                                                         \
      _mm_load_si128(reinterpret_cast<const __m128i*>(t.shuf_lo[c0]));        \
  const __m128i hi0 =                                                         \
      _mm_load_si128(reinterpret_cast<const __m128i*>(t.shuf_hi[c0]));        \
  const __m128i lo1 =                                                         \
      _mm_load_si128(reinterpret_cast<const __m128i*>(t.shuf_lo[c1]));        \
  const __m128i hi1 =                                                         \
      _mm_load_si128(reinterpret_cast<const __m128i*>(t.shuf_hi[c1]));        \
  const __m128i nib = _mm_set1_epi8(0x0f);                                    \
  size_t i = 0;                                                               \
  for (; i + 16 <= n; i += 16) {                                              \
    const __m128i s =                                                         \
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));           \
    __m128i a = _mm_loadu_si128(reinterpret_cast<__m128i*>(d0 + i));          \
    __m128i b = _mm_loadu_si128(reinterpret_cast<__m128i*>(d1 + i));          \
    a = GF256_SHUF16_2(s, a, lo0, hi0);                                       \
    b = GF256_SHUF16_2(s, b, lo1, hi1);                                       \
    _mm_storeu_si128(reinterpret_cast<__m128i*>(d0 + i), a);                  \
    _mm_storeu_si128(reinterpret_cast<__m128i*>(d1 + i), b);                  \
  }                                                                           \
  if (i < n) {                                                                \
    addmul_scalar(d0 + i, src + i, n - i, c0);                                \
    addmul_scalar(d1 + i, src + i, n - i, c1);                                \
  }

__attribute__((target("ssse3"))) inline void addmul_ssse3(
    uint8_t* dst, const uint8_t* src, size_t n, uint8_t c) {
  GF256_SHUF_KERNEL_BODY()
}

__attribute__((target("ssse3"))) inline void addmul2_ssse3(
    uint8_t* d0, uint8_t* d1, const uint8_t* src, size_t n, uint8_t c0,
    uint8_t c1) {
  GF256_SHUF_KERNEL2_BODY()
}

__attribute__((target("avx"))) inline void addmul_avx128(
    uint8_t* dst, const uint8_t* src, size_t n, uint8_t c) {
  GF256_SHUF_KERNEL_BODY()
}

__attribute__((target("avx"))) inline void addmul2_avx128(
    uint8_t* d0, uint8_t* d1, const uint8_t* src, size_t n, uint8_t c0,
    uint8_t c1) {
  GF256_SHUF_KERNEL2_BODY()
}

__attribute__((target("avx2"))) inline void addmul_avx2(
    uint8_t* dst, const uint8_t* src, size_t n, uint8_t c) {
  const Tables& t = tables();
  const __m256i lo = _mm256_broadcastsi128_si256(
      _mm_load_si128(reinterpret_cast<const __m128i*>(t.shuf_lo[c])));
  const __m256i hi = _mm256_broadcastsi128_si256(
      _mm_load_si128(reinterpret_cast<const __m128i*>(t.shuf_hi[c])));
  const __m256i nib = _mm256_set1_epi8(0x0f);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    const __m256i s =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
    const __m256i p = _mm256_xor_si256(
        _mm256_shuffle_epi8(lo, _mm256_and_si256(s, nib)),
        _mm256_shuffle_epi8(hi,
                            _mm256_and_si256(_mm256_srli_epi64(s, 4), nib)));
    __m256i d = _mm256_loadu_si256(reinterpret_cast<__m256i*>(dst + i));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i),
                        _mm256_xor_si256(d, p));
  }
  if (i < n) addmul_scalar(dst + i, src + i, n - i, c);
}

__attribute__((target("avx2"))) inline void addmul2_avx2(
    uint8_t* d0, uint8_t* d1, const uint8_t* src, size_t n, uint8_t c0,
    uint8_t c1) {
  const Tables& t = tables();
  const __m256i lo0 = _mm256_broadcastsi128_si256(
      _mm_load_si128(reinterpret_cast<const __m128i*>(t.shuf_lo[c0])));
  const __m256i hi0 = _mm256_broadcastsi128_si256(
      _mm_load_si128(reinterpret_cast<const __m128i*>(t.shuf_hi[c0])));
  const __m256i lo1 = _mm256_broadcastsi128_si256(
      _mm_load_si128(reinterpret_cast<const __m128i*>(t.shuf_lo[c1])));
  const __m256i hi1 = _mm256_broadcastsi128_si256(
      _mm_load_si128(reinterpret_cast<const __m128i*>(t.shuf_hi[c1])));
  const __m256i nib = _mm256_set1_epi8(0x0f);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    const __m256i s =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
    const __m256i sl = _mm256_and_si256(s, nib);
    const __m256i sh = _mm256_and_si256(_mm256_srli_epi64(s, 4), nib);
    const __m256i p0 = _mm256_xor_si256(_mm256_shuffle_epi8(lo0, sl),
                                        _mm256_shuffle_epi8(hi0, sh));
    const __m256i p1 = _mm256_xor_si256(_mm256_shuffle_epi8(lo1, sl),
                                        _mm256_shuffle_epi8(hi1, sh));
    __m256i a = _mm256_loadu_si256(reinterpret_cast<__m256i*>(d0 + i));
    __m256i b = _mm256_loadu_si256(reinterpret_cast<__m256i*>(d1 + i));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(d0 + i),
                        _mm256_xor_si256(a, p0));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(d1 + i),
                        _mm256_xor_si256(b, p1));
  }
  if (i < n) {
    addmul_scalar(d0 + i, src + i, n - i, c0);
    addmul_scalar(d1 + i, src + i, n - i, c1);
  }
}

// ---------------------------------------------------------------------------
// GFNI kernels: c*x is a single GF2P8MULB against a broadcast coefficient
// (native in this field -- see the polynomial note above). Untested on hosts
// without GFNI; guarded by CPUID dispatch and cross-checked by tests wherever
// the hardware allows.
// ---------------------------------------------------------------------------
__attribute__((target("gfni,avx"))) inline void addmul_gfni128(
    uint8_t* dst, const uint8_t* src, size_t n, uint8_t c) {
  const __m128i vc = _mm_set1_epi8(static_cast<char>(c));
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m128i s =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
    const __m128i d = _mm_loadu_si128(reinterpret_cast<__m128i*>(dst + i));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i),
                     _mm_xor_si128(d, _mm_gf2p8mul_epi8(s, vc)));
  }
  if (i < n) addmul_scalar(dst + i, src + i, n - i, c);
}

__attribute__((target("gfni,avx"))) inline void addmul2_gfni128(
    uint8_t* d0, uint8_t* d1, const uint8_t* src, size_t n, uint8_t c0,
    uint8_t c1) {
  const __m128i v0 = _mm_set1_epi8(static_cast<char>(c0));
  const __m128i v1 = _mm_set1_epi8(static_cast<char>(c1));
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m128i s =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
    const __m128i a = _mm_loadu_si128(reinterpret_cast<__m128i*>(d0 + i));
    const __m128i b = _mm_loadu_si128(reinterpret_cast<__m128i*>(d1 + i));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(d0 + i),
                     _mm_xor_si128(a, _mm_gf2p8mul_epi8(s, v0)));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(d1 + i),
                     _mm_xor_si128(b, _mm_gf2p8mul_epi8(s, v1)));
  }
  if (i < n) {
    addmul_scalar(d0 + i, src + i, n - i, c0);
    addmul_scalar(d1 + i, src + i, n - i, c1);
  }
}

__attribute__((target("gfni,avx2"))) inline void addmul_gfni_avx2(
    uint8_t* dst, const uint8_t* src, size_t n, uint8_t c) {
  const __m256i vc = _mm256_set1_epi8(static_cast<char>(c));
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    const __m256i s =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
    const __m256i d = _mm256_loadu_si256(reinterpret_cast<__m256i*>(dst + i));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i),
                        _mm256_xor_si256(d, _mm256_gf2p8mul_epi8(s, vc)));
  }
  if (i < n) addmul_scalar(dst + i, src + i, n - i, c);
}

__attribute__((target("gfni,avx2"))) inline void addmul2_gfni_avx2(
    uint8_t* d0, uint8_t* d1, const uint8_t* src, size_t n, uint8_t c0,
    uint8_t c1) {
  const __m256i v0 = _mm256_set1_epi8(static_cast<char>(c0));
  const __m256i v1 = _mm256_set1_epi8(static_cast<char>(c1));
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    const __m256i s =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
    const __m256i a = _mm256_loadu_si256(reinterpret_cast<__m256i*>(d0 + i));
    const __m256i b = _mm256_loadu_si256(reinterpret_cast<__m256i*>(d1 + i));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(d0 + i),
                        _mm256_xor_si256(a, _mm256_gf2p8mul_epi8(s, v0)));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(d1 + i),
                        _mm256_xor_si256(b, _mm256_gf2p8mul_epi8(s, v1)));
  }
  if (i < n) {
    addmul_scalar(d0 + i, src + i, n - i, c0);
    addmul_scalar(d1 + i, src + i, n - i, c1);
  }
}

__attribute__((target("gfni,avx512f,avx512bw"))) inline void
addmul_gfni_avx512(uint8_t* dst, const uint8_t* src, size_t n, uint8_t c) {
  const __m512i vc = _mm512_set1_epi8(static_cast<char>(c));
  size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    const __m512i s = _mm512_loadu_si512(src + i);
    const __m512i d = _mm512_loadu_si512(dst + i);
    _mm512_storeu_si512(dst + i,
                        _mm512_xor_si512(d, _mm512_gf2p8mul_epi8(s, vc)));
  }
  if (i < n) addmul_scalar(dst + i, src + i, n - i, c);
}

__attribute__((target("gfni,avx512f,avx512bw"))) inline void
addmul2_gfni_avx512(uint8_t* d0, uint8_t* d1, const uint8_t* src, size_t n,
                    uint8_t c0, uint8_t c1) {
  const __m512i v0 = _mm512_set1_epi8(static_cast<char>(c0));
  const __m512i v1 = _mm512_set1_epi8(static_cast<char>(c1));
  size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    const __m512i s = _mm512_loadu_si512(src + i);
    const __m512i a = _mm512_loadu_si512(d0 + i);
    const __m512i b = _mm512_loadu_si512(d1 + i);
    _mm512_storeu_si512(d0 + i,
                        _mm512_xor_si512(a, _mm512_gf2p8mul_epi8(s, v0)));
    _mm512_storeu_si512(d1 + i,
                        _mm512_xor_si512(b, _mm512_gf2p8mul_epi8(s, v1)));
  }
  if (i < n) {
    addmul_scalar(d0 + i, src + i, n - i, c0);
    addmul_scalar(d1 + i, src + i, n - i, c1);
  }
}

// ---------------------------------------------------------------------------
// Registry and dispatch
// ---------------------------------------------------------------------------
using AddmulFn = void (*)(uint8_t*, const uint8_t*, size_t, uint8_t);
using Addmul2Fn = void (*)(uint8_t*, uint8_t*, const uint8_t*, size_t,
                           uint8_t, uint8_t);

struct KernelInfo {
  const char* name;
  AddmulFn addmul;
  Addmul2Fn addmul2;
  uint32_t need;  // caps() bits required
};

// Best-first. kernel_table()[i].need & ~caps() == 0 => runnable.
inline const KernelInfo* kernel_table(size_t* n) {
  static const KernelInfo k[] = {
      {"gfni_avx512", addmul_gfni_avx512, addmul2_gfni_avx512,
       kGFNI | kAVX512BW},
      {"gfni_avx2", addmul_gfni_avx2, addmul2_gfni_avx2, kGFNI | kAVX2},
      {"gfni128", addmul_gfni128, addmul2_gfni128, kGFNI | kAVX},
      {"avx2", addmul_avx2, addmul2_avx2, kAVX2},
      {"avx128", addmul_avx128, addmul2_avx128, kAVX},
      {"ssse3", addmul_ssse3, addmul2_ssse3, kSSSE3},
      {"scalar", addmul_scalar, addmul2_scalar, 0},
  };
  *n = sizeof(k) / sizeof(k[0]);
  return k;
}

// Runtime override (--fec-simd): lets one binary produce the "+SIMD" and
// "-SIMD" rows of the final matrix. Must be set before the first FEC call;
// the extra pointer check is one predicted-not-taken branch per addmul --
// noise against the O(n) kernel body.
inline const KernelInfo*& kernel_override() {
  static const KernelInfo* k = nullptr;
  return k;
}

// name = "scalar"/"ssse3"/.../"gfni_avx512"; false if unknown or the CPU
// lacks the required features. nullptr/"auto" clears the override.
inline bool force_kernel(const char* name) {
  if (!name || !*name || strcmp(name, "auto") == 0) {
    kernel_override() = nullptr;
    return true;
  }
  size_t n;
  const KernelInfo* k = kernel_table(&n);
  for (size_t i = 0; i < n; ++i) {
    if (strcmp(k[i].name, name) == 0 && (k[i].need & ~caps()) == 0) {
      kernel_override() = &k[i];
      return true;
    }
  }
  return false;
}

inline const KernelInfo& best_kernel() {
  if (kernel_override()) return *kernel_override();
  static const KernelInfo& best = []() -> const KernelInfo& {
    size_t n;
    const KernelInfo* k = kernel_table(&n);
    for (size_t i = 0; i < n; ++i)
      if ((k[i].need & ~caps()) == 0) return k[i];
    return k[n - 1];
  }();
  return best;
}

inline const char* kernel_name() { return best_kernel().name; }

// dst[i] ^= c * src[i]
inline void addmul(uint8_t* dst, const uint8_t* src, size_t n, uint8_t c) {
  if (c == 0) return;
  if (c == 1) { memxor(dst, src, n); return; }
  best_kernel().addmul(dst, src, n, c);
}

// d0[i] ^= c0 * src[i];  d1[i] ^= c1 * src[i]  -- one pass over src.
inline void addmul2(uint8_t* d0, uint8_t* d1, const uint8_t* src, size_t n,
                    uint8_t c0, uint8_t c1) {
  if (c0 < 2 || c1 < 2) {  // rare degenerate coefficients: take simple paths
    addmul(d0, src, n, c0);
    addmul(d1, src, n, c1);
    return;
  }
  best_kernel().addmul2(d0, d1, src, n, c0, c1);
}

}  // namespace gf256
