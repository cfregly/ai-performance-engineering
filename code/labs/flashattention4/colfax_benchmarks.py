"""Pinned Colfax FA4 kernel ablations, measured as steady-state CUDA graph replay.

The optional upstream packages are experiment-specific; see
``colfax_optimization_diaries.md``. Neither pair uses FlexAttention as a proxy.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


@dataclass(frozen=True)
class ColfaxConfig:
    kind: str
    batch_size: int
    seqlen_q: int
    seqlen_k: int
    query_heads: int
    kv_heads: int
    head_dim: int
    causal: bool = False

    def validate(self) -> None:
        if self.kind not in ("decode", "backward"):
            raise ValueError("kind must be decode or backward")
        for name in ("batch_size", "seqlen_q", "seqlen_k", "query_heads", "kv_heads", "head_dim"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.query_heads % self.kv_heads:
            raise ValueError("query_heads must be divisible by kv_heads")
        if self.seqlen_q > self.seqlen_k:
            raise ValueError("seqlen_q must not exceed seqlen_k")
        if self.kind == "decode":
            if self.head_dim not in (64, 128):
                raise ValueError("decode ping-pong requires head_dim 64 or 128")
            if self.seqlen_q * (self.query_heads // self.kv_heads) > 128:
                raise ValueError("packed decode queries must fit one 128-row Q stage")
            if self.seqlen_k < 256:
                raise ValueError("decode needs at least two 128-token KV tiles to overlap")
        else:
            if self.head_dim != 64:
                raise ValueError("backward de-aliasing requires head_dim 64")
            if self.seqlen_q != self.seqlen_k:
                raise ValueError("this backward lab uses equal Q and KV sequence lengths")


def default_config(kind: str) -> ColfaxConfig:
    if kind == "decode":
        return ColfaxConfig(kind, 32, 1, 131072, 16, 1, 64)
    if kind == "backward":
        return ColfaxConfig(kind, 4, 16384, 16384, 32, 32, 64)
    raise ValueError("kind must be decode or backward")


def build_inputs(config: ColfaxConfig, device: torch.device) -> dict[str, torch.Tensor]:
    """Use the caller's RNG; both arms consume identical random draws."""
    config.validate()
    b, q, k = config.batch_size, config.seqlen_q, config.seqlen_k
    hq, hkv, d = config.query_heads, config.kv_heads, config.head_dim
    tensors = {
        "q": torch.randn(b, q, hq, d, device=device, dtype=torch.bfloat16),
        "k": torch.randn(b, k, hkv, d, device=device, dtype=torch.bfloat16),
        "v": torch.randn(b, k, hkv, d, device=device, dtype=torch.bfloat16),
    }
    if config.kind == "backward":
        tensors["dout"] = torch.randn(b, q, hq, d, device=device, dtype=torch.bfloat16)
    return tensors


def reference_attention(q, k, v, causal: bool = False):
    """Small-shape independent oracle with FA's bottom-right causal alignment.

    This materializes scores and is only for correctness tests, never timing.
    Preserve double inputs for gradient checks; otherwise accumulate in FP32.
    """
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    group = q.shape[2] // k.shape[2]
    qh = q.to(dtype).transpose(1, 2)
    kh = k.to(dtype).repeat_interleave(group, dim=2).transpose(1, 2)
    vh = v.to(dtype).repeat_interleave(group, dim=2).transpose(1, 2)
    scores = (qh @ kh.transpose(-1, -2)) * q.shape[-1] ** -0.5
    if causal:
        rows = torch.arange(q.shape[1], device=q.device)[:, None]
        cols = torch.arange(k.shape[1], device=q.device)[None, :]
        scores = scores.masked_fill(cols > rows + k.shape[1] - q.shape[1], float("-inf"))
    return (scores.softmax(dim=-1) @ vh).transpose(1, 2)


def source_manifest(kind: str) -> dict:
    if kind not in ("decode", "backward"):
        raise ValueError("kind must be decode or backward")
    return json.loads(
        Path(__file__).with_name("colfax_upstream.json").read_text(encoding="utf-8")
    )[kind]


_SOURCE_TREE_FORMAT = "sha256-path-content-sha256-v1"


def python_source_tree_fingerprint(root: Path) -> dict[str, int | str]:
    """Fingerprint every importable Python source path and its exact bytes."""
    sources = sorted(
        (path.relative_to(root).as_posix(), path)
        for path in root.rglob("*.py")
        if path.is_file()
    )
    digest = hashlib.sha256()
    for relative_path, path in sources:
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\0")
    return {
        "format": _SOURCE_TREE_FORMAT,
        "file_count": len(sources),
        "sha256": digest.hexdigest(),
    }


def verify_python_source_tree(
    root: Path,
    expected: dict,
    *,
    kind: str,
    commit: str,
) -> None:
    if expected.get("format") != _SOURCE_TREE_FORMAT:
        raise RuntimeError(f"Invalid Colfax {kind} source-tree manifest format")
    actual = python_source_tree_fingerprint(root)
    expected_fingerprint = {
        "format": expected.get("format"),
        "file_count": expected.get("file_count"),
        "sha256": expected.get("sha256"),
    }
    if actual != expected_fingerprint:
        raise RuntimeError(
            f"SKIPPED: Colfax {kind} requires unmodified FA4 {commit}; "
            "Python source tree mismatch. "
            f"See requirements_colfax_{kind}.txt."
        )


def verify_vcs_direct_url(direct_url: dict, kind: str) -> dict:
    """Validate PEP 610 metadata without importing the CUDA package."""
    manifest = source_manifest(kind)
    expected = manifest["installed_vcs"]
    vcs_info = direct_url.get("vcs_info")
    actual_url = direct_url.get("url")
    expected_url = manifest["repository"]
    if isinstance(actual_url, str):
        actual_url = actual_url.removeprefix("git+").rstrip("/").removesuffix(".git")
    if isinstance(expected_url, str):
        expected_url = expected_url.removeprefix("git+").rstrip("/").removesuffix(".git")
    valid = (
        isinstance(vcs_info, dict)
        and vcs_info.get("vcs") == "git"
        and vcs_info.get("commit_id") == manifest["commit"]
        and vcs_info.get("requested_revision") == manifest["commit"]
        and actual_url == expected_url
        and direct_url.get("subdirectory") == expected["subdirectory"]
    )
    if not valid:
        raise RuntimeError(
            f"SKIPPED: Colfax {kind} requires an exact VCS install of FA4 "
            f"{manifest['commit']} from {manifest['repository']}#{expected['subdirectory']}"
        )
    return manifest


def verify_installed_vcs_commit(kind: str) -> dict:
    manifest = source_manifest(kind)
    installed_vcs = manifest["installed_vcs"]
    distribution_name = installed_vcs["distribution"]
    metadata_name = installed_vcs["metadata"]
    try:
        distribution = importlib_metadata.distribution(distribution_name)
        direct_url_text = distribution.read_text(metadata_name)
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"SKIPPED: install requirements_colfax_{kind}.txt in a dedicated environment"
        ) from exc
    if direct_url_text is None:
        raise RuntimeError(
            f"SKIPPED: Colfax {kind} requires PEP 610 {metadata_name} VCS provenance"
        )
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"SKIPPED: Colfax {kind} has invalid PEP 610 {metadata_name} provenance"
        ) from exc
    if not isinstance(direct_url, dict):
        raise RuntimeError(
            f"SKIPPED: Colfax {kind} has invalid PEP 610 {metadata_name} provenance"
        )
    return verify_vcs_direct_url(direct_url, kind)


def verify_source_files(root: Path, kind: str) -> dict:
    """Reject drift in every runtime Python source before importing CUDA code."""
    manifest = source_manifest(kind)
    verify_python_source_tree(
        root,
        manifest["python_source_tree"],
        kind=kind,
        commit=manifest["commit"],
    )
    return manifest


def load_upstream(kind: str):
    verify_installed_vcs_commit(kind)
    try:
        spec = importlib.util.find_spec("flash_attn.cute")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"SKIPPED: install requirements_colfax_{kind}.txt in a dedicated environment"
        ) from exc
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            f"SKIPPED: install requirements_colfax_{kind}.txt in a dedicated environment"
        )
    root = Path(next(iter(spec.submodule_search_locations)))
    verify_source_files(root, kind)
    # Missing dependencies and compiler failures must retain their actual errors.
    interface = importlib.import_module("flash_attn.cute.interface")
    if Path(interface.__file__).resolve() != (root / "interface.py").resolve():
        raise RuntimeError("SKIPPED: loaded FA4 interface does not match verified source")
    return interface


@contextmanager
def decode_control(interface, optimized: bool):
    """Bridge the pinned import-time env switch to a scoped capture-time control.

    FA_DISABLE_S_PING_PONG is read only when upstream utils is imported. Setting
    os.environ per arm would silently reuse the first arm's setting. The pinned
    boolean is read by _flash_attn_fwd and enters its JIT key as use_s_ping_pong.
    Restore it even if compilation fails. Replays no longer read this global.
    """
    previous = interface.utils._fa_disable_s_ping_pong_enabled
    interface.utils._fa_disable_s_ping_pong_enabled = not optimized
    try:
        yield
    finally:
        interface.utils._fa_disable_s_ping_pong_enabled = previous


class ColfaxBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """One real FA4 invocation per replay; full timed outputs are verified."""

    def __init__(self, kind: str, optimized: bool, config: ColfaxConfig | None = None):
        super().__init__()
        self.spec = config or default_config(kind)
        self.spec.validate()
        if self.spec.kind != kind:
            raise ValueError("config kind must match benchmark kind")
        self.kind = kind
        self.optimized = optimized
        self.inputs = None
        self.output = None
        self.graph = None
        self._ran = False
        self._forward_state = None

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: Colfax FA4 labs require a CUDA Blackwell SM100 GPU")
        if torch.cuda.get_device_capability(self.device) != (10, 0):
            raise RuntimeError(
                "SKIPPED: Colfax FA4 labs are scoped to Blackwell SM100 (B200/GB200)"
            )
        if torch.version.cuda is None or int(torch.version.cuda.split(".")[0]) < 13:
            raise RuntimeError(
                "SKIPPED: the Colfax experiment recipes require a CUDA 13+ PyTorch build"
            )
        interface = load_upstream(self.kind)
        if interface.is_fake_mode():
            raise RuntimeError("SKIPPED: FA4 fake-tensor mode cannot produce benchmark evidence")
        if interface._get_device_arch() != 100:
            raise RuntimeError("SKIPPED: FA4 architecture override must resolve to SM100")
        self.inputs = build_inputs(self.spec, self.device)
        self._ran = False
        cfg, x = self.spec, self.inputs
        fwd_kwargs = {
            "causal": cfg.causal,
            "tile_mn": (128, 128),
            "num_splits": 1,
            "pack_gqa": True,
        }

        if self.kind == "decode":
            dispatch = interface._get_fwd_config(
                arch=100,
                head_dim=cfg.head_dim,
                head_dim_v=cfg.head_dim,
                max_seqlen_q=cfg.seqlen_q,
                max_seqlen_k=cfg.seqlen_k,
                num_head_kv=cfg.kv_heads,
                qhead_per_kvhead=cfg.query_heads // cfg.kv_heads,
                pack_gqa=True,
                batch_size=cfg.batch_size,
                causal=cfg.causal,
                local=False,
                window_size_left=None,
                window_size_right=None,
                num_splits=1,
                device=self.device,
                tile_mn=(128, 128),
            )
            if (
                dispatch.m_block_size,
                dispatch.n_block_size,
                dispatch.q_stage,
                dispatch.num_splits,
            ) != (128, 128, 1, 1):
                raise RuntimeError(
                    "SKIPPED: upstream dispatch is not eligible for the decode ping-pong ablation"
                )

            def run():
                return interface._flash_attn_fwd(x["q"], x["k"], x["v"], **fwd_kwargs)[0]

            # All dispatch predicates are fixed: SM100, Q stage 1, 128x128,
            # D=64/128, no 2CTA, no modifiers/sparsity, unsplit contiguous KV.
            with decode_control(interface, self.optimized):
                self._capture(run)
        else:
            # Backward-only contract: forward output/LSE are common setup state.
            out, lse, _, _ = interface._flash_attn_fwd(
                x["q"], x["k"], x["v"], return_lse=True, **fwd_kwargs
            )
            self._forward_state = (out, lse)
            self.inputs.update(out=out, lse=lse)

            def run():
                return interface._flash_attn_bwd(
                    x["q"],
                    x["k"],
                    x["v"],
                    out,
                    x["dout"],
                    lse,
                    causal=cfg.causal,
                    deterministic=True,
                    split_P_dS=self.optimized,
                    warp_sync=self.optimized,
                )[:3]

            self._capture(run)

    def _capture(self, run) -> None:
        """JIT/warmup/capture are setup costs, identical lifecycle for both arms."""
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = run()
        torch.cuda.current_stream(self.device).wait_stream(stream)

    def benchmark_fn(self) -> None:
        if self.graph is None:
            raise RuntimeError("setup() must complete before benchmark_fn()")
        with self._nvtx_range(
            f"colfax_{self.kind}_{'optimized' if self.optimized else 'baseline'}"
        ):
            self.graph.replay()
        self._ran = True

    def capture_verification_payload(self) -> None:
        if not self._ran or self.inputs is None or self.output is None:
            raise RuntimeError(
                "benchmark_fn() must replay before verification; setup outputs are not evidence"
            )
        # Concatenation is post-timing. Verify every gradient, not a checksum or slice.
        output = (
            self.output
            if self.kind == "decode"
            else torch.cat([t.reshape(-1) for t in self.output])
        )
        self._set_verification_payload(
            inputs=self.inputs,
            output=output,
            batch_size=self.spec.batch_size,
            parameter_count=0,
            precision_flags={"bf16": True, "fp16": False, "tf32": False},
            # PR2804 reports bit-identical gradients with deterministic=True.
            output_tolerance=(0.0, 0.0) if self.kind == "backward" else (2e-2, 2e-2),
            signature_overrides={"graph_capture_enabled": True},
        )

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=20, warmup=5, enable_cuda_graph=False, measurement_timeout_seconds=180
        )

    def get_workload_metadata(self) -> WorkloadMetadata:
        return WorkloadMetadata(
            requests_per_iteration=float(self.spec.batch_size),
            tokens_per_iteration=float(self.spec.batch_size * self.spec.seqlen_q),
        )

    def get_custom_metrics(self) -> dict[str, float]:
        return {
            "colfax.pr": 2817.0 if self.kind == "decode" else 2804.0,
            "colfax.optimization_enabled": float(self.optimized),
            "colfax.head_dim": float(self.spec.head_dim),
            "colfax.seqlen_q": float(self.spec.seqlen_q),
            "colfax.seqlen_k": float(self.spec.seqlen_k),
            "colfax.query_heads": float(self.spec.query_heads),
            "colfax.kv_heads": float(self.spec.kv_heads),
            "colfax.causal": float(self.spec.causal),
            "colfax.deterministic_backward": float(self.kind == "backward"),
            "colfax.cuda_graph_replay": 1.0,
        }

    def validate_result(self) -> str | None:
        return None if self._ran and self.output is not None else "No timed graph replay output"

    def teardown(self) -> None:
        self.graph = None
        self.output = None
        self.inputs = None
        self._forward_state = None
        self._verification_payload = None
        self._ran = False
