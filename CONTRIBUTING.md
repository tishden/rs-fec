# Contributing to rsfec

Bug reports, failing test cases and measurements on hardware I do not have are
all welcome — open an issue.

## The CLA

Code contributions require a signed Contributor License Agreement
([CLA.md](CLA.md)) before they can be merged. This is a copyright *licence*,
not an assignment: you keep the copyright in what you write, and you grant me
the right to relicense the project as a whole later.

To sign, comment on your pull request with:

```
I have read the CLA document and I hereby sign the CLA.
```

You only sign once; it covers all of your future contributions here.

If your employer owns work you do, get their sign-off before contributing —
the CLA asks you to confirm you have the right to grant these terms.

## What makes a good pull request here

This is a codec whose whole value is that it produces exactly the right bytes,
fast. Two things follow.

**Anything touching arithmetic must be proven against the reference.** The
scalar kernel in `gf256.h` is the definition of correct; `tests/test_fec.cpp`
cross-checks every runnable SIMD kernel against it, and `hls/tb/` does the same
for the FPGA mirror. A new kernel that is not in that cross-check is not
finished. Run both dispatch ends:

```
make test          # best available kernel, and FEC_NO_SIMD scalar
```

**Wire-format changes are breaking changes.** The coded unit layout and the
Cauchy construction are what let a host and an FPGA repair each other's
packets. If a change alters the bytes, say so explicitly in the PR — it needs a
version bump, not a patch release.

Performance claims want a number and the machine it came from. "Faster" without
either is not reviewable; `bench/bench_fec.cpp` exists for this.

## Style

Match the file you are editing. Google-ish C++ with two-space indent, comments
in English that explain *why* rather than restate the code. The existing header
comments are the standard to aim for: someone should be able to choose `k`, `r`
and `depth` for their channel without reading the implementation.
