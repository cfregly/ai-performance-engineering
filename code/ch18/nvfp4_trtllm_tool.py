"""Run generation with an explicitly supplied NVFP4 TensorRT-LLM engine.

Set ``TRT_LLM_ENGINE`` to a single-rank TensorRT-LLM engine directory. This
tool deliberately has no framework fallback: a missing dependency, unusable
engine asset, or non-NVFP4 engine is reported as ``SKIPPED`` and exits nonzero.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402

_NVFP4_QUANT_ALGOS = frozenset(
    {
        "NVFP4",
        "W4A8_NVFP4_FP8",
    }
)


@dataclass(frozen=True)
class _NVFP4EngineAssets:
    engine_dir: Path
    config_path: Path
    engine_path: Path
    quant_algo: str
    vocab_size: int


def _skip(message: str) -> RuntimeError:
    return RuntimeError(f"SKIPPED: {message}")


def _require_mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _skip(f"TensorRT-LLM engine config field {field!r} must be an object")
    return value


def _inspect_nvfp4_engine_assets(engine_dir: str | os.PathLike[str]) -> _NVFP4EngineAssets:
    """Validate the on-disk evidence needed for the supported engine path."""
    try:
        resolved_dir = Path(engine_dir).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _skip(f"TensorRT-LLM engine directory is unavailable: {engine_dir!s}") from exc
    if not resolved_dir.is_dir():
        raise _skip(f"TensorRT-LLM engine path is not a directory: {resolved_dir}")

    config_path = resolved_dir / "config.json"
    if not config_path.is_file():
        raise _skip(f"TensorRT-LLM engine config is missing: {config_path}")
    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _skip(f"TensorRT-LLM engine config is unreadable or invalid JSON: {config_path}") from exc

    config = _require_mapping(raw_config, field="<root>")
    pretrained = _require_mapping(config.get("pretrained_config"), field="pretrained_config")
    quantization = _require_mapping(
        pretrained.get("quantization"),
        field="pretrained_config.quantization",
    )
    quant_algo = quantization.get("quant_algo")
    if not isinstance(quant_algo, str) or quant_algo not in _NVFP4_QUANT_ALGOS:
        raise _skip(
            "TensorRT-LLM engine must declare an NVFP4 quantization algorithm at "
            "pretrained_config.quantization.quant_algo; "
            f"observed {quant_algo!r}"
        )

    mapping = _require_mapping(pretrained.get("mapping"), field="pretrained_config.mapping")
    world_size = mapping.get("world_size")
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size != 1:
        raise _skip(
            "the standalone NVFP4/TRT-LLM tool requires a single-rank engine "
            f"(pretrained_config.mapping.world_size=1); observed {world_size!r}"
        )

    vocab_size = pretrained.get("vocab_size")
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size < 2:
        raise _skip(
            "TensorRT-LLM engine config must declare pretrained_config.vocab_size >= 2; "
            f"observed {vocab_size!r}"
        )

    engine_path = resolved_dir / "rank0.engine"
    try:
        engine_size = engine_path.stat().st_size
    except OSError as exc:
        raise _skip(f"TensorRT-LLM rank-0 engine is missing or unreadable: {engine_path}") from exc
    if not engine_path.is_file() or engine_size <= 0:
        raise _skip(f"TensorRT-LLM rank-0 engine must be a non-empty file: {engine_path}")

    return _NVFP4EngineAssets(
        engine_dir=resolved_dir,
        config_path=config_path,
        engine_path=engine_path,
        quant_algo=quant_algo,
        vocab_size=vocab_size,
    )


def _load_model_runner() -> Any:
    try:
        from tensorrt_llm.runtime import ModelRunner  # type: ignore[import-not-found]
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        raise _skip("TensorRT-LLM runtime dependency is unavailable") from exc
    if not callable(getattr(ModelRunner, "from_dir", None)):
        raise _skip("installed TensorRT-LLM runtime does not provide ModelRunner.from_dir")
    return ModelRunner


class NVFP4TRTLLMBenchmark(VerificationPayloadMixin, BaseBenchmark):
    def __init__(
        self,
        engine_dir: str | os.PathLike[str] | None = None,
        *,
        prompt_len: int = 32,
        max_new_tokens: int = 1,
    ) -> None:
        super().__init__()
        if prompt_len <= 0:
            raise ValueError("prompt_len must be positive")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.engine_dir = Path(engine_dir).expanduser() if engine_dir is not None else None
        self.prompt_len = int(prompt_len)
        self.max_new_tokens = int(max_new_tokens)
        self.input_ids: torch.Tensor | None = None
        self._batch_input_ids: list[torch.Tensor] | None = None
        self.output: torch.Tensor | None = None
        self._trt_runner: Any = None
        self._quant_algo: str | None = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.prompt_len + self.max_new_tokens),
        )
        self._verification_payload = None
        self._enable_nvtx = False
        self._empty_iteration_result: dict[str, float] = {}

    def _configured_engine_dir(self) -> Path:
        if self.engine_dir is not None:
            return self.engine_dir
        configured = os.getenv("TRT_LLM_ENGINE", "").strip()
        if not configured:
            raise _skip(
                "an explicit NVFP4 TensorRT-LLM engine directory is required; "
                "set TRT_LLM_ENGINE"
            )
        return Path(configured).expanduser()

    def setup(self) -> None:
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        assets = _inspect_nvfp4_engine_assets(self._configured_engine_dir())

        if not torch.cuda.is_available():
            raise _skip("CUDA is required for the NVFP4 TensorRT-LLM engine")
        capability = torch.cuda.get_device_capability(self.device)
        if capability[0] < 10:
            raise _skip(
                "NVFP4 TensorRT-LLM execution requires a Blackwell-class GPU "
                f"(compute capability >= 10.0); observed {capability[0]}.{capability[1]}"
            )

        ModelRunner = _load_model_runner()

        try:
            self._trt_runner = ModelRunner.from_dir(
                str(assets.engine_dir),
                rank=0,
                debug_mode=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "FAIL FAST: TensorRT-LLM failed to load the validated NVFP4 engine "
                f"at {assets.engine_dir} ({type(exc).__name__}: {exc})"
            ) from exc

        self._quant_algo = assets.quant_algo
        self.input_ids = torch.arange(self.prompt_len, dtype=torch.int32).remainder_(
            assets.vocab_size
        )
        self._batch_input_ids = [self.input_ids]
        torch.cuda.synchronize(self.device)

    @staticmethod
    def _require_output_ids(outputs: object) -> torch.Tensor:
        if not isinstance(outputs, dict) or "output_ids" not in outputs:
            raise RuntimeError(
                "FAIL FAST: TRT-LLM generate must return a dict containing output_ids"
            )
        output_ids = outputs["output_ids"]
        if not isinstance(output_ids, torch.Tensor):
            raise RuntimeError(
                "FAIL FAST: TRT-LLM generate returned non-Tensor output_ids "
                f"({type(output_ids).__name__})"
            )
        if output_ids.ndim not in (2, 3) or output_ids.shape[0] != 1 or output_ids.numel() == 0:
            raise RuntimeError(
                "FAIL FAST: TRT-LLM output_ids must be non-empty with shape "
                f"[1, seq] or [1, beam, seq]; observed {tuple(output_ids.shape)}"
            )
        if output_ids.dtype not in (torch.int32, torch.int64):
            raise RuntimeError(
                "FAIL FAST: TRT-LLM output_ids must use an integer token dtype; "
                f"observed {output_ids.dtype}"
            )
        return output_ids

    def benchmark_fn(self) -> dict[str, float]:
        if self._trt_runner is None or self._batch_input_ids is None:
            raise RuntimeError("NVFP4 TensorRT-LLM engine is not initialized")

        with nvtx_range("nvfp4_trtllm_engine", enable=self._enable_nvtx):
            try:
                outputs = self._trt_runner.generate(
                    self._batch_input_ids,
                    max_new_tokens=self.max_new_tokens,
                    end_id=0,
                    pad_id=0,
                    top_k=1,
                    top_p=0.0,
                    temperature=1.0,
                    random_seed=0,
                    return_dict=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    "FAIL FAST: TensorRT-LLM NVFP4 generate failed "
                    f"({type(exc).__name__}: {exc})"
                ) from exc
            self.output = self._require_output_ids(outputs)
        return self._empty_iteration_result

    def capture_verification_payload(self) -> None:
        if self.input_ids is None or self.output is None or self._quant_algo is None:
            raise RuntimeError("setup() and benchmark_fn() must run before verification capture")
        self._set_verification_payload(
            inputs={"input_ids": self.input_ids},
            output=self.output,
            batch_size=1,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": False, "fp8": False, "tf32": False},
            output_tolerance=(0.0, 0.0),
            signature_overrides={"quantization_mode": self._quant_algo},
        )

    def teardown(self) -> None:
        self._trt_runner = None
        self.input_ids = None
        self._batch_input_ids = None
        self.output = None
        self._quant_algo = None

    def get_workload_metadata(self) -> WorkloadMetadata | None:
        return self._workload


def get_benchmark() -> BaseBenchmark:
    return NVFP4TRTLLMBenchmark()


def main() -> None:
    """Run the chapter tool through the shared standalone benchmark helper."""
    if not torch.cuda.is_available():
        raise RuntimeError("SKIPPED: CUDA required for ch18 NVFP4/TRT-LLM tool")

    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(
        get_benchmark,
        iterations=10,
        warmup=5,
        name="ch18_nvfp4_trtllm_tool",
    )


if __name__ == "__main__":
    main()
