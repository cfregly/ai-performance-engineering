# Shared CUDA architecture configuration for Blackwell and Grace-Blackwell builds.
#
# Usage from chapter Makefiles (located under ch*/):
#   include ../core/common/cuda_arch.mk
#   NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS) ...
#   # optional: USE_ARCH_SUFFIX := 0  # to disable suffixing targets
#
# Exposes:
#   ARCH                - Selected GPU architecture (default: sm_100)
#   ARCH_NAME           - Human-readable architecture label
#   ARCH_SUFFIX         - Suffix (_sm100, _sm103, _sm120, _sm121) for architecture-specific binaries
#   TARGET_SUFFIX       - Suffix applied when USE_ARCH_SUFFIX is 1
#   CUDA_NVCC_ARCH_FLAGS- Baseline nvcc flags for the selected architecture
#   ARCH_LIST           - Ordered list of repository targets supported by CUDA 13.0

CUDA_VERSION ?= 13.0
NVCC ?= nvcc
PYTHON ?= python3

# Some environments install nvcc under /usr/local/cuda but don't add it to PATH.
# Make builds more robust by falling back to common install locations when NVCC
# is left at its default value.
ifeq ($(NVCC),nvcc)
  _NVCC_IN_PATH := $(strip $(shell command -v nvcc 2>/dev/null))
  ifeq ($(_NVCC_IN_PATH),)
    ifneq ($(wildcard /usr/local/cuda/bin/nvcc),)
      NVCC := /usr/local/cuda/bin/nvcc
    else ifneq ($(wildcard /usr/local/cuda-$(CUDA_VERSION)/bin/nvcc),)
      NVCC := /usr/local/cuda-$(CUDA_VERSION)/bin/nvcc
    endif
  endif
endif

CUDA_ARCH_MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CUDA_COMMON_DIR := $(dir $(CUDA_ARCH_MK_PATH))

DEFAULT_ARCH := sm_100
AUTO_ARCH_DETECTION ?= 1

ifeq ($(origin ARCH), undefined)
  ifeq ($(AUTO_ARCH_DETECTION),1)
    DETECTED_ARCH := $(strip $(shell $(PYTHON) $(CUDA_COMMON_DIR)/../benchmark/detect_sm.py 2>/dev/null))
  endif
endif

ifeq ($(strip $(DETECTED_ARCH)),)
  ifeq ($(AUTO_ARCH_DETECTION),1)
    ifeq ($(origin ARCH), undefined)
      $(error [cuda_arch] Unable to auto-detect GPU architecture. Set ARCH=<sm_100|sm_103|sm_120|sm_121> explicitly.)
    endif
  endif
  ARCH ?= $(DEFAULT_ARCH)
  ARCH_SOURCE := default
else
  ARCH := $(DETECTED_ARCH)
  ARCH_SOURCE := auto
endif

ifeq ($(origin ARCH), command line)
  ARCH_SOURCE := user
endif
ifeq ($(origin ARCH), environment)
  ARCH_SOURCE := user
endif

ifeq ($(ARCH_SOURCE),auto)
$(info [cuda_arch] Auto-detected GPU architecture $(ARCH) (override with ARCH=<sm_100|sm_103|sm_120|sm_121>))
endif
NVTX_STUB_DIR := $(abspath $(CUDA_COMMON_DIR)/nvtx_stub)
NVTX_STUB_LIB := $(NVTX_STUB_DIR)/libnvToolsExt.a
NVTX_STUB_SCRIPT := $(abspath $(CUDA_COMMON_DIR)/../profiling/nvtx_stub.py)

# CUDA 13.0 documents these Blackwell targets. Keep this overrideable so a CI
# job can state its compiler contract without requiring a visible GPU.
CUDA_13_ARCH_LIST := sm_100 sm_103 sm_120 sm_121
ARCH_LIST ?= $(CUDA_13_ARCH_LIST)

ifeq ($(ARCH),sm_121)
ARCH_NAME := Grace-Blackwell GB10 (CC 12.1)
ARCH_SUFFIX := _sm121
CUDA_ARCH_GENCODE := -gencode arch=compute_121,code=[sm_121,compute_121]
HOST_ARCH_FLAGS := -Xcompiler -mcpu=native
else ifeq ($(ARCH),sm_100)
ARCH_NAME := Blackwell B200/B300 (CC 10.0)
ARCH_SUFFIX := _sm100
# Use sm_100a for cluster/DSMEM support on Blackwell (CUDA 13.0+)
CUDA_ARCH_GENCODE := -gencode arch=compute_100a,code=[sm_100a,compute_100a]
HOST_ARCH_FLAGS :=
else ifeq ($(ARCH),sm_120)
ARCH_NAME := Grace-Blackwell GB200 (CC 12.0)
ARCH_SUFFIX := _sm120
CUDA_ARCH_GENCODE := -gencode arch=compute_120,code=[sm_120,compute_120]
HOST_ARCH_FLAGS := -Xcompiler -mcpu=native
else ifeq ($(ARCH),sm_103)
ARCH_NAME := Blackwell Ultra B300 (CC 10.3)
ARCH_SUFFIX := _sm103
# Use sm_103a for cluster/DSMEM support on Blackwell Ultra (CUDA 13.0+)
CUDA_ARCH_GENCODE := -gencode arch=compute_103a,code=[sm_103a,compute_103a]
HOST_ARCH_FLAGS :=
else
$(error Unsupported ARCH=$(ARCH). Supported values: $(CUDA_13_ARCH_LIST))
endif

# Base nvcc flags shared across the project. Chapters may append additional flags as needed.
CUDA_CXX_STANDARD ?= 17
CUDA_NVCC_BASE_FLAGS ?= -O3 -std=c++$(CUDA_CXX_STANDARD) $(CUDA_ARCH_GENCODE) --expt-relaxed-constexpr -Xcompiler -fPIC
CUDA_NVCC_ARCH_FLAGS := $(CUDA_NVCC_BASE_FLAGS) $(HOST_ARCH_FLAGS)

# Control whether binaries get suffixed with architecture-specific suffixes.
USE_ARCH_SUFFIX ?= 1
ifeq ($(USE_ARCH_SUFFIX),1)
TARGET_SUFFIX := $(ARCH_SUFFIX)
else
TARGET_SUFFIX :=
endif

$(NVTX_STUB_LIB):
	$(PYTHON) $(NVTX_STUB_SCRIPT) --output $@

# NVTX profiling helpers are opt-in. Set NVTX_ENABLED=1 to enable.
NVTX_ENABLED ?= 0
ifeq ($(strip $(NVTX_ENABLED)),1)
CUDA_NVTX_CFLAGS := -DENABLE_NVTX_PROFILING
CUDA_NVTX_LDFLAGS := -L$(NVTX_STUB_DIR) -lnvToolsExt
CUDA_NVTX_DEPS := $(NVTX_STUB_LIB)
else
CUDA_NVTX_CFLAGS :=
CUDA_NVTX_LDFLAGS :=
CUDA_NVTX_DEPS :=
endif
