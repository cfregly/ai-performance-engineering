#pragma once

// Host-only full-array check, shared by the CUDA executable and CPU regression tests.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ozaki_scheme {
struct AccuracyMetrics {
    double max_abs_error = 0;
    double mean_abs_error = 0;
    double relative_l2 = 0;
    double normalized_max_abs = 0;
};

inline AccuracyMetrics measure_accuracy(const double* actual, const double* reference, std::size_t count) {
    if (!count || !actual || !reference || count > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
        throw std::runtime_error("Accuracy requires nonempty independent candidate/reference arrays");
    }
    const auto a = reinterpret_cast<std::uintptr_t>(actual);
    const auto r = reinterpret_cast<std::uintptr_t>(reference);
    if ((a <= r ? r - a : a - r) < count * sizeof(double)) {
        throw std::runtime_error("Accuracy candidate/reference storage overlaps");
    }
    long double error_squared = 0, reference_squared = 0, error_sum = 0;
    double max_error = 0, max_reference = 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(reference[i])) {
            throw std::runtime_error("Non-finite candidate/reference in full-array accuracy check");
        }
        const long double difference = static_cast<long double>(actual[i]) - reference[i];
        error_squared += difference * difference;
        reference_squared += static_cast<long double>(reference[i]) * reference[i];
        error_sum += std::abs(difference);
        max_error = std::max(max_error, static_cast<double>(std::abs(difference)));
        max_reference = std::max(max_reference, std::abs(reference[i]));
    }
    AccuracyMetrics result;
    result.max_abs_error = max_error;
    result.mean_abs_error = static_cast<double>(error_sum / count);
    result.relative_l2 = reference_squared > 0
        ? static_cast<double>(std::sqrt(error_squared / reference_squared))
        : (error_squared == 0 ? 0 : std::numeric_limits<double>::infinity());
    result.normalized_max_abs = max_reference > 0 ? max_error / max_reference
        : (max_error == 0 ? 0 : std::numeric_limits<double>::infinity());
    return result;
}

inline void validate_accuracy_limits(double relative_l2_limit, double normalized_max_abs_limit) {
    if (!std::isfinite(relative_l2_limit) || relative_l2_limit < 0 || relative_l2_limit >= 1 ||
        !std::isfinite(normalized_max_abs_limit) || normalized_max_abs_limit < 0 || normalized_max_abs_limit >= 1) {
        throw std::runtime_error("Uncalibrated accuracy: explicitly configure finite relative-L2 and "
                                 "normalized-max-abs limits in [0,1); no permissive default exists");
    }
}

inline void assert_accuracy(const AccuracyMetrics& metrics, double relative_l2_limit, double normalized_max_abs_limit) {
    validate_accuracy_limits(relative_l2_limit, normalized_max_abs_limit);
    if (!std::isfinite(metrics.relative_l2) || !std::isfinite(metrics.normalized_max_abs) ||
        metrics.relative_l2 > relative_l2_limit || metrics.normalized_max_abs > normalized_max_abs_limit) {
        throw std::runtime_error("Full-array accuracy failed against independent native FP64 reference");
    }
}
} // namespace ozaki_scheme
