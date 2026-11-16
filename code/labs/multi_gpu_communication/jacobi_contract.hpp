#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace multi_gpu_communication {

struct Domain {
    int rank;
    int ranks;
    int rows;
    int global_start;
};

inline void validate_shape(int nx, int ny, int iterations) {
    if (nx < 3 || ny < 3 || iterations < 0) {
        throw std::invalid_argument("Require nx >= 3, ny >= 3, and iterations >= 0");
    }
    const auto count = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::invalid_argument("The mesh allocation exceeds the addressable size");
    }
}

inline Domain decompose(int ny, int ranks, int rank) {
    if (ny < 3 || ranks < 1 || rank < 0 || rank >= ranks || ranks > ny - 2) {
        throw std::invalid_argument("Each rank must own at least one of the ny - 2 interior rows");
    }
    const int base = (ny - 2) / ranks;
    const int remainder = (ny - 2) % ranks;
    return {rank, ranks, base + (rank < remainder ? 1 : 0),
            1 + rank * base + std::min(rank, remainder)};
}

inline float initial_value(int nx, int ny, int x, int y) {
    validate_shape(nx, ny, 0);
    if (x < 0 || x >= nx || y < 0 || y >= ny) {
        throw std::invalid_argument("Initial mesh coordinates are out of range");
    }
    const int periodic_y = y == 0 ? ny - 2 : (y == ny - 1 ? 1 : y);
    constexpr double pi = 3.1415926535897932384626433832795;
    const double phase = 2.0 * pi * (periodic_y - 1) / (ny - 2);
    if (x == 0 || x == nx - 1) {
        return static_cast<float>(std::sin(phase));
    }
    return static_cast<float>(0.05 * std::sin(pi * x / (nx - 1)) * std::cos(phase));
}

inline std::vector<float> initial_mesh(int nx, int ny) {
    validate_shape(nx, ny, 0);
    std::vector<float> values(static_cast<std::size_t>(nx) * ny);
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            values[static_cast<std::size_t>(y) * nx + x] = initial_value(nx, ny, x, y);
        }
    }
    return values;
}

inline std::vector<float> reference_jacobi(int nx, int ny, int iterations) {
    validate_shape(nx, ny, iterations);
    auto current = initial_mesh(nx, ny);
    auto next = current;
    for (int step = 0; step < iterations; ++step) {
        for (int y = 1; y < ny - 1; ++y) {
            for (int x = 1; x < nx - 1; ++x) {
                const auto i = static_cast<std::size_t>(y) * nx + x;
                next[i] = 0.25f * (current[i - 1] + current[i + 1] +
                                    current[i - nx] + current[i + nx]);
            }
        }
        std::copy_n(next.data() + static_cast<std::size_t>(ny - 2) * nx, nx, next.data());
        std::copy_n(next.data() + nx, nx, next.data() + static_cast<std::size_t>(ny - 1) * nx);
        current.swap(next);
    }
    return current;
}

}  // namespace multi_gpu_communication
