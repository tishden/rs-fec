# SPDX-License-Identifier: Apache-2.0
# Plain-make path for people who would rather not run cmake. Same two tests.
CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -mssse3 -Iinclude
BIN      := bin

.PHONY: all test bench example clean

all: test

$(BIN):
	mkdir -p $(BIN)

# Both dispatch ends: whatever the CPU offers, and the forced scalar reference.
test: $(BIN)/test_fec $(BIN)/test_fec_scalar
	./$(BIN)/test_fec
	./$(BIN)/test_fec_scalar

$(BIN)/test_fec: tests/test_fec.cpp include/rsfec/*.h | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN)/test_fec_scalar: tests/test_fec.cpp include/rsfec/*.h | $(BIN)
	$(CXX) $(CXXFLAGS) -DFEC_NO_SIMD $< -o $@

example: $(BIN)/roundtrip
	./$(BIN)/roundtrip

$(BIN)/roundtrip: examples/roundtrip.cpp include/rsfec/*.h | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@

# Needs Google Benchmark (dnf install google-benchmark-devel / apt install
# libbenchmark-dev).
bench: $(BIN)/bench_fec
	./$(BIN)/bench_fec

$(BIN)/bench_fec: bench/bench_fec.cpp include/rsfec/*.h | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@ -lbenchmark -lpthread

# Cache-behaviour probe: no benchmark dependency, run it under perf stat.
$(BIN)/bench_fec_cache: bench/bench_fec_cache.cpp include/rsfec/*.h | $(BIN)
	$(CXX) $(CXXFLAGS) -Ibench $< -o $@

clean:
	rm -rf $(BIN)
