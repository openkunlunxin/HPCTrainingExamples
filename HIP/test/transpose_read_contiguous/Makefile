# MIT License
#
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

CXX   = hipcc
ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
CXXFLAGS = --offload-arch=${ROCM_GPU}

EXAMPLE := main
COMMON_INCLUDE_DIR := ../../Common
EXTERNAL_DIR := ../../External
GPU_RUNTIME ?= HIP

# HIP variables
ROCM_INSTALL_DIR ?= /opt/rocm
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include

HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

# Common variables and flags
CXX_STD   := c++17
ICXXFLAGS := -std=$(CXX_STD)
ICPPFLAGS := -I $(COMMON_INCLUDE_DIR) -I $(EXTERNAL_DIR)
ILDFLAGS  :=
ILDLIBS   :=
BP_LV     ?= l1

ifeq ($(GPU_RUNTIME), CUDA)
  ICXXFLAGS += -x cu
  ICPPFLAGS += -isystem $(HIP_INCLUDE_DIR)
else ifeq ($(GPU_RUNTIME), HIP)
  CXXFLAGS ?= -Wall -Wextra
else
  $(error GPU_RUNTIME is set to "$(GPU_RUNTIME)". GPU_RUNTIME must be either CUDA or HIP)
endif

ICXXFLAGS += $(CXXFLAGS)
ICPPFLAGS += $(CPPFLAGS)
#ICPPFLAGS += -save-temps
ILDFLAGS  += $(LDFLAGS)
ILDLIBS   += $(LDLIBS)
ILDLIBS   += -lrocblas

transpose_timed.o: transpose_kernels.h
transpose_kernel_read_contiguous.o: transpose_kernels.h
transpose_kernel_write_contiguous.o: transpose_kernels.h
transpose_kernel_tiled.o: transpose_kernels.h

DEV_ASM: $(EXAMPLE)-dev.s

ASM_BP: $(EXAMPLE)-dev_bp.s

HOST_O: $(EXAMPLE)-host.o

DEV_BP: $(EXAMPLE)-dev_bp_$(BP_LV).o

GEN_MCIN: $(EXAMPLE)-host.mcin

DEV_EM_BP: $(EXAMPLE)-dev_em_bp_$(BP_LV).o

HIPFB_BP: $(EXAMPLE)-dev_bp_$(BP_LV).hipfb

BIN_BP: $(EXAMPLE)_bp_$(BP_LV)

BIN: $(EXAMPLE)

SRCS :=  transpose_timed.o transpose_kernel_read_contiguous.o transpose_kernel_write_contiguous.o  transpose_kernel_tiled.o

$(EXAMPLE)-dev_bp_$(BP_LV).s: bp.py $(EXAMPLE)-dev.s
	python3 $^ $(BP_LV) $@

$(EXAMPLE)-dev.s: transpose_kernels.h
	$(HIPCXX) -S --cuda-device-only --offload-arch=gfx942 $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) -o $@ $(SRCS) $(ILDLIBS)

$(EXAMPLE)-host.o: $(SRCS)
	$(HIPCXX) -c --cuda-host-only $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) -o $@ $(SRCS) $(ILDLIBS)

$(EXAMPLE)-host.mcin: $(EXAMPLE)-host.o $(EXAMPLE)-dev_bp_$(BP_LV).hipfb
	./gen_mcin $^ > $@

$(EXAMPLE)-dev_bp_$(BP_LV).o: $(EXAMPLE)-dev_bp_$(BP_LV).s
	clang -target amdgcn-amd-amdhsa -mcpu=gfx942 $< -o $@

$(EXAMPLE)-dev_bp_$(BP_LV).hipfb: $(EXAMPLE)-dev_bp_$(BP_LV).o
	clang-offload-bundler -type=o -bundle-align=4096 -targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx942 -input=/dev/null -input=$<  -output=$@

$(EXAMPLE)-dev_em_bp_$(BP_LV).o: $(EXAMPLE)-host.mcin $(EXAMPLE)-dev_bp_$(BP_LV).hipfb
	llvm-mc -triple x86_64-unknown-linux-gnu -o $@ $< --filetype=obj

$(EXAMPLE)_bp_$(BP_LV): $(EXAMPLE)-host.o $(EXAMPLE)-dev_em_bp_$(BP_LV).o
	hipcc $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) -o $@ $^ $(ILDLIBS)


$(EXAMPLE):$(SRCS)
	$(HIPCXX) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) -o $@ $(SRCS) $(ILDLIBS)

clean:
	$(RM) $(EXAMPLE) $(EXAMPLE)-host* $(EXAMPLE)-dev* $(EXAMPLE)_bp*

profile_bp: $(EXAMPLE)_bp_$(BP_LV)
	rocprof-compute profile -n $< -- ./$<
	rocprof-compute analyze -p workloads/$</MI300X_A1/ -b 3

profile: $(EXAMPLE)
	rocprof-compute profile -n $< -- ./$<
	rocprof-compute analyze -p workloads/$</MI300X_A1/ -b 3

ana: $(EXAMPLE)
	rocprof-compute analyze -p workloads/$</MI300X_A1/ -b 3

ana_bp: $(EXAMPLE)_bp_$(BP_LV)
	rocprof-compute analyze -p workloads/$</MI300X_A1/ -b 3

.PHONY: clean
