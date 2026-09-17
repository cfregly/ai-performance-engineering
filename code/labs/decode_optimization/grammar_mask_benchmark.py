"""Compile legal-token masks for the existing guided-decode concept."""

import time

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.decode_optimization.token_grammar import TokenGrammar


class GrammarMaskBenchmark(VerificationPayloadMixin, BaseBenchmark):
    allow_cpu = True

    def __init__(self, optimized, size=1024):
        super().__init__()
        if type(size) is not int or size <= 0:
            raise ValueError("size must be a positive integer")
        self.device = torch.device("cpu")
        self.optimized, self.size, self.output = optimized, size, None
        self.compile_ms = 0.0
        self.register_workload_metadata(
            custom_units_per_iteration=size, custom_unit_name="policy_queries"
        )

    def setup(self):
        self.output, self.compile_ms = None, 0.0
        self.vocabulary = (
            (b"",)
            + tuple(bytes([i]) for i in range(256))
            + tuple(str(i).encode() for i in range(1000))
        )
        self.alternatives = [f'{{"value":{i}}}'.encode() for i in range(32)]
        self.grammar = TokenGrammar.literals(
            self.alternatives, self.vocabulary, eos_token_id=0, cache=self.optimized
        )
        self.inputs = torch.randint(0, len(self.grammar.transitions), (self.size,))
        self.states = self.inputs.tolist()
        if self.optimized:
            start = time.perf_counter()
            self.grammar.precompile()
            self.compile_ms = (time.perf_counter() - start) * 1000

    def benchmark_fn(self):
        self.output = None
        self.output = tuple(self.grammar.allowed_mask(state) for state in self.states)

    def capture_verification_payload(self):
        if self.output is None:
            raise RuntimeError("benchmark_fn() must run before verification")
        reference = TokenGrammar.literals(
            self.alternatives, self.vocabulary, eos_token_id=0, cache=False
        )
        expected = tuple(reference.allowed_mask(state) for state in self.states)
        if self.output != expected:
            raise AssertionError("token masks differ from the uncached reference")
        self._set_verification_payload(
            inputs={"workload": self.inputs},
            output=torch.tensor(self.output),
            batch_size=self.size,
            parameter_count=0,
            output_tolerance=(0.0, 0.0),
        )

    def get_custom_metrics(self):
        return {"grammar_precompile_ms": self.compile_ms}

    def get_config(self):
        return BenchmarkConfig(iterations=10, warmup=5)

    def teardown(self):
        self.output = None
