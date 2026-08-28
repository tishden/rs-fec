# SPDX-License-Identifier: Apache-2.0
# csim cross-check: the HLS codec against the scalar C++ reference. Runs on a
# plain g++ -- the only thing Vitis HLS is needed for is its ap_int.h.
HLS_INC ?= /opt/Xilinx/Vitis_HLS/2024.1/include
CXXFLAGS := -std=c++17 -O1 -w -I$(HLS_INC) -Iinclude -I../include
BIN := ../bin/hls

.PHONY: tb clean

tb: $(BIN)/tb_gf $(BIN)/tb_fec
	./$(BIN)/tb_gf
	./$(BIN)/tb_fec

$(BIN):
	mkdir -p $(BIN)

$(BIN)/tb_gf: tb/tb_gf.cpp include/gf256_hls.h ../include/rsfec/gf256.h | $(BIN)
	g++ $(CXXFLAGS) -o $@ $<

$(BIN)/tb_fec: tb/tb_fec.cpp include/fec_hls.h include/gf256_hls.h ../include/rsfec/fec.h | $(BIN)
	g++ $(CXXFLAGS) -o $@ $<

clean:
	rm -rf $(BIN)
