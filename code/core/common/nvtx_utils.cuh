// Shared NVTX helpers for chapter CUDA examples.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>

#include "nvtx_label_utils.cuh"

#if defined(ENABLE_NVTX_PROFILING)
#define AISP_NVTX_ENABLED 1
#include <nvtx3/nvToolsExt.h>

namespace aisp_nvtx {
inline uint32_t nvtx_color_for_label(const char* name) {
  if (!name) {
    return 0xFF00BFFF;  // Default: DeepSkyBlue
  }
  const char* colon = std::strchr(name, ':');
  const size_t len = colon ? static_cast<size_t>(colon - name) : std::strlen(name);
  auto matches = [&](const char* key) {
    return std::strlen(key) == len && std::strncmp(name, key, len) == 0;
  };
  if (matches("setup")) return 0xFF1E90FF;           // DodgerBlue
  if (matches("warmup")) return 0xFF00BFFF;          // DeepSkyBlue
  if (matches("compute_kernel")) return 0xFF32CD32;  // LimeGreen
  if (matches("compute_math")) return 0xFF228B22;    // ForestGreen
  if (matches("compute_graph")) return 0xFF2E8B57;   // SeaGreen
  if (matches("transfer_async")) return 0xFF8A2BE2;  // BlueViolet
  if (matches("transfer_sync")) return 0xFFFF8C00;   // DarkOrange
  if (matches("prefetch")) return 0xFF20B2AA;        // LightSeaGreen
  if (matches("barrier")) return 0xFF708090;         // SlateGray
  if (matches("reduce")) return 0xFFFFC107;          // Amber
  if (matches("verify")) return 0xFFFFD700;          // Gold
  if (matches("cleanup")) return 0xFFB22222;         // FireBrick
  if (matches("batch")) return 0xFF00CED1;           // DarkTurquoise
  if (matches("tile")) return 0xFFDA70D6;            // Orchid
  if (matches("iteration")) return 0xFFB0C4DE;       // LightSteelBlue
  if (matches("step")) return 0xFFB0E0E6;            // PowderBlue
  return 0xFF00BFFF;  // Default: DeepSkyBlue
}

struct NvtxRange {
  explicit NvtxRange(const char* name, uint32_t color = 0) {
    label_ = standardize_nvtx_label(name);
    const uint32_t resolved_color =
        color != 0 ? color : nvtx_color_for_label(label_.c_str());
    nvtxEventAttributes_t attr{};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = resolved_color;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = label_.c_str();
    nvtxRangePushEx(&attr);
  }
  ~NvtxRange() { nvtxRangePop(); }
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;

 private:
  std::string label_;
};
}  // namespace aisp_nvtx

#define AISP_NVTX_CONCAT_IMPL(lhs, rhs) lhs##rhs
#define AISP_NVTX_CONCAT(lhs, rhs) AISP_NVTX_CONCAT_IMPL(lhs, rhs)
#define NVTX_RANGE(name) \
  aisp_nvtx::NvtxRange AISP_NVTX_CONCAT(_aisp_nvtx_range_, __LINE__){name}
#define NVTX_RANGE_COLOR(name, color) \
  aisp_nvtx::NvtxRange AISP_NVTX_CONCAT(_aisp_nvtx_range_, __LINE__){name, color}

#else
#define AISP_NVTX_ENABLED 0

namespace aisp_nvtx {
inline uint32_t nvtx_color_for_label(const char*) { return 0; }
struct NvtxRange {
  explicit NvtxRange(const char*, uint32_t = 0) {}
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
};
}  // namespace aisp_nvtx

#define NVTX_RANGE(name) ((void)0)
#define NVTX_RANGE_COLOR(name, color) ((void)0)

#endif
