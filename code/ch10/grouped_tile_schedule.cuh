#pragma once

#ifdef __CUDACC__
#define CH10_HOST_DEVICE __host__ __device__
#else
#define CH10_HOST_DEVICE
#endif

namespace ch10 {
struct TileCoord { int m; int n; };

// Rasterize complete N groups, then the possibly narrower final group.
// Preconditions: 0 <= linear < grid_m * grid_n; all dimensions are positive.
CH10_HOST_DEVICE constexpr TileCoord grouped_tile_coord(
    int linear, int grid_m, int grid_n, int group_size = 8) {
  const int group = linear / (group_size * grid_m);
  const int first_n = group * group_size;
  const int remaining_n = grid_n - first_n;
  const int width = remaining_n < group_size ? remaining_n : group_size;
  const int local = linear - first_n * grid_m;
  return {local / width, first_n + local % width};
}
}  // namespace ch10

#undef CH10_HOST_DEVICE
