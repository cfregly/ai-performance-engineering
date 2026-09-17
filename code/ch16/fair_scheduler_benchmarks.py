"""CPU token-fair admission policy comparison alongside runtime scheduling."""

import torch

from ch16.fair_scheduler import Request, dispatch_trace
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


class FairSchedulerBenchmark(VerificationPayloadMixin, BaseBenchmark):
    allow_cpu = True

    def __init__(self, optimized, size=1024):
        super().__init__()
        if type(size) is not int or size <= 0:
            raise ValueError("size must be a positive integer")
        self.device = torch.device("cpu")
        self.optimized, self.size, self.output = optimized, size, None
        self.register_workload_metadata(
            custom_units_per_iteration=size, custom_unit_name="policy_queries"
        )

    def setup(self):
        self.output = None
        self.inputs = torch.stack(
            (
                torch.randint(0, 128, (self.size,)),
                torch.randint(1, 100, (self.size,)),
                torch.randint(1, 30, (self.size,)),
            ),
            dim=1,
        )
        self.requests = [
            Request(i, client, tokens) for i, (client, tokens, _) in enumerate(self.inputs.tolist())
        ]
        self.observed_outputs = {i: row[2] for i, row in enumerate(self.inputs.tolist())}

    def benchmark_fn(self):
        self.output = None
        self.output = dispatch_trace(
            self.requests, self.observed_outputs, backend="heap" if self.optimized else "linear"
        )

    def capture_verification_payload(self):
        if self.output is None:
            raise RuntimeError("benchmark_fn() must run before verification")
        expected = dispatch_trace(self.requests, self.observed_outputs, backend="linear")
        if self.output != expected:
            raise AssertionError("admission decisions differ from the linear reference")
        self._set_verification_payload(
            inputs={"workload": self.inputs},
            output=torch.tensor(self.output),
            batch_size=self.size,
            parameter_count=0,
            output_tolerance=(0.0, 0.0),
        )

    def get_config(self):
        return BenchmarkConfig(iterations=10, warmup=5)

    def teardown(self):
        self.output = None
