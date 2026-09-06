// profiling_helpers.cuh - NVTX markers and profiling utilities
// Integrates with Nsight Systems for visual timeline profiling

#ifndef PROFILING_HELPERS_CUH
#define PROFILING_HELPERS_CUH

#include <cstdint>
#include <string>

#include "../nvtx_utils.cuh"

// Color palette for NVTX markers
namespace nvtx {
  constexpr uint32_t COLOR_RED     = 0xFFFF0000;
  constexpr uint32_t COLOR_GREEN   = 0xFF00FF00;
  constexpr uint32_t COLOR_BLUE    = 0xFF0000FF;
  constexpr uint32_t COLOR_YELLOW  = 0xFFFFFF00;
  constexpr uint32_t COLOR_CYAN    = 0xFF00FFFF;
  constexpr uint32_t COLOR_MAGENTA = 0xFFFF00FF;
  constexpr uint32_t COLOR_WHITE   = 0xFFFFFFFF;
  constexpr uint32_t COLOR_ORANGE  = 0xFFFFA500;
  constexpr uint32_t COLOR_PURPLE  = 0xFF800080;
}

// nvtx_utils.cuh owns NVTX_RANGE and NVTX_RANGE_COLOR for both enabled and
// disabled builds. Keep this alias for code that names the original helper type.
using NvtxRange = aisp_nvtx::NvtxRange;

// Mark specific operations for profiling
inline void NvtxMark(const char* name) {
#if AISP_NVTX_ENABLED
  std::string label = aisp_nvtx::standardize_nvtx_label(name);
  nvtxMarkA(label.c_str());
#else
  (void)name;
#endif
}

#define NVTX_MARK_COMPUTE(name) NvtxMark(name)
#define NVTX_MARK_MEMORY(name) NvtxMark(name)

// Common profiling ranges
#define PROFILE_KERNEL_LAUNCH(name) NVTX_RANGE_COLOR(name, nvtx::COLOR_BLUE)
#define PROFILE_MEMORY_COPY(name) NVTX_RANGE_COLOR(name, nvtx::COLOR_YELLOW)
#define PROFILE_HOST_COMPUTE(name) NVTX_RANGE_COLOR(name, nvtx::COLOR_CYAN)
#define PROFILE_DATA_PREP(name) NVTX_RANGE_COLOR(name, nvtx::COLOR_MAGENTA)

#endif // PROFILING_HELPERS_CUH
