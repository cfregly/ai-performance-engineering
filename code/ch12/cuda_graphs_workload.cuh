#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>

struct StageSpec {
    float scale;
    float bias;
    float frequency;
};

inline constexpr StageSpec kStageSpecs[] = {
    {1.01f, 0.05f, 0.10f},
    {0.97f, -0.02f, 0.25f},
    {1.08f, 0.12f, 0.40f},
    {0.92f, -0.04f, 0.60f},
    {1.04f, 0.03f, 0.85f},
    {0.99f, -0.08f, 1.05f},
};

inline constexpr int kStageCount = sizeof(kStageSpecs) / sizeof(StageSpec);
inline constexpr int kInnerPasses = 3;

// Independent scalar reference, used by the bounded --verify acceptance run.
inline bool verify_graph_output(const float* output, int count, int iterations) {
    for (int i = 0; i < count; ++i) {
        double value = std::sin(0.001f * static_cast<float>(i));
        for (int iteration = 0; iteration < iterations; ++iteration) {
            for (const auto& stage : kStageSpecs) {
                for (int pass = 0; pass < kInnerPasses; ++pass) {
                    value = std::tanh(value * stage.scale + stage.bias);
                    value = 0.65 * std::sin(value * stage.frequency + 0.05 * pass)
                          + 0.35 * std::cos(value * 0.35 + 0.02 * pass);
                }
            }
        }
        if (!std::isfinite(output[i]) || std::abs(output[i] - value) > 2e-5 * std::max(1.0, std::abs(value))) {
            std::fprintf(stderr, "graph mismatch at %d: %.9g versus %.9g\n", i, output[i], value);
            return false;
        }
    }
    return true;
}
