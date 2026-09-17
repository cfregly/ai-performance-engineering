"""Memory-goal comparison of byte-per-code and packed KIVI representations."""

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.kv_cache_compression.kivi_cache import encode_cache


class KiviBenchmark(VerificationPayloadMixin, BaseBenchmark):
    def __init__(self, packed, shape=(2, 8, 2048, 128)):
        super().__init__()
        self.packed = packed
        self.shape = shape
        self.keys = self.values = self.cache = None
        self.register_workload_metadata(
            custom_units_per_iteration=shape[0] * shape[1] * shape[2], custom_unit_name="kv_tokens"
        )

    def setup(self):
        self.cache = None
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: KIVI benchmark pairs require CUDA")
        self.keys = torch.randn(self.shape, device=self.device, dtype=torch.bfloat16)
        self.values = torch.randn_like(self.keys)

    def benchmark_fn(self):
        if self.keys is None:
            raise RuntimeError("setup() must run first")
        self.cache = None
        self.cache = encode_cache(self.keys, self.values, packed=self.packed)

    def capture_verification_payload(self):
        if self.cache is None:
            raise RuntimeError("benchmark_fn() must run before verification")
        keys, values = self.cache.decode()
        reference = encode_cache(self.keys, self.values, packed=False)
        expected_keys, expected_values = reference.decode()
        torch.testing.assert_close(keys, expected_keys, rtol=0, atol=0)
        torch.testing.assert_close(values, expected_values, rtol=0, atol=0)
        self._set_verification_payload(
            inputs={"keys": self.keys, "values": self.values},
            output=torch.stack((keys, values)),
            batch_size=self.shape[0],
            precision_flags={"bf16": True},
            output_tolerance=(0.0, 0.0),
        )

    def get_optimization_goal(self):
        return "memory"

    def get_custom_metrics(self):
        if self.cache is None:
            return {}
        return {
            "retained_cache_bytes": self.cache.nbytes,
            "quantization_bits": 2,
            "storage_bits_per_code": 2 if self.packed else 8,
            "residual_tokens": self.shape[2] - self.cache.prefix_tokens,
        }

    def get_config(self):
        return BenchmarkConfig(iterations=10, warmup=5)

    def teardown(self):
        self.keys = self.values = self.cache = None
