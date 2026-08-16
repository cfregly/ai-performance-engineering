from __future__ import annotations

from pathlib import Path

import torch

from core.scripts.audit_verification_compliance import (
    audit_directory,
    load_benchmark_class,
)


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_audit_loader_captures_python_output_without_process_fd_hijacking(
    tmp_path: Path,
) -> None:
    benchmark_path = tmp_path / "baseline_output.py"
    _write(
        benchmark_path,
        (
            "import sys\n"
            "print('import stdout')\n"
            "print('import stderr', file=sys.stderr)\n"
            "class _Bench:\n"
            "    pass\n"
            "def get_benchmark():\n"
            "    print('factory stdout')\n"
            "    return _Bench()\n"
        ),
    )

    loaded = load_benchmark_class(benchmark_path)

    assert loaded is not None
    _, class_name, captured = loaded
    assert class_name == "_Bench"
    assert captured == ["import stdout", "factory stdout", "import stderr"]


def test_audit_loader_leaves_native_and_child_output_on_process_descriptors(
    tmp_path: Path,
    capfd,
) -> None:
    benchmark_path = tmp_path / "baseline_process_output.py"
    _write(
        benchmark_path,
        (
            "import os\n"
            "import subprocess\n"
            "import sys\n"
            "print('python stdout')\n"
            "os.write(1, b'native stdout\\n')\n"
            "subprocess.run([sys.executable, '-c', \"print('child stdout')\"], check=True)\n"
            "class _Bench:\n"
            "    pass\n"
            "def get_benchmark():\n"
            "    return _Bench()\n"
        ),
    )

    loaded = load_benchmark_class(benchmark_path)
    process_output = capfd.readouterr()

    assert loaded is not None
    _, _, captured = loaded
    assert captured == ["python stdout"]
    assert "native stdout" in process_output.out
    assert "child stdout" in process_output.out


def test_audit_loader_resolves_sibling_imports(tmp_path: Path) -> None:
    _write(tmp_path / "helper_mod.py", "VALUE = 7\n")
    _write(
        tmp_path / "baseline_sibling_import.py",
        (
            "from __future__ import annotations\n"
            "from helper_mod import VALUE\n"
            "import torch\n"
            "from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig\n"
            "from core.benchmark.verification_mixin import VerificationPayloadMixin\n"
            "\n"
            "class _Bench(VerificationPayloadMixin, BaseBenchmark):\n"
            "    allow_cpu = True\n"
            "    def __init__(self) -> None:\n"
            "        super().__init__()\n"
            "        self.value = VALUE\n"
            "        self.device = torch.device('cpu')\n"
            "        self.output = None\n"
            "        self.register_workload_metadata(requests_per_iteration=1.0)\n"
            "    def setup(self) -> None:\n"
            "        self.output = torch.tensor([float(self.value)], device=self.device)\n"
            "    def benchmark_fn(self) -> None:\n"
            "        if self.output is None:\n"
            "            raise RuntimeError('missing output')\n"
            "    def capture_verification_payload(self) -> None:\n"
            "        if self.output is None:\n"
            "            raise RuntimeError('missing output')\n"
            "        self._set_verification_payload(\n"
            "            inputs={'probe': torch.tensor([1.0])},\n"
            "            output=self.output,\n"
            "            batch_size=1,\n"
            "            parameter_count=0,\n"
            "            output_tolerance=(0.0, 0.0),\n"
            "        )\n"
            "    def validate_result(self):\n"
            "        return None\n"
            "    def get_config(self) -> BenchmarkConfig:\n"
            "        return BenchmarkConfig(iterations=1, warmup=0, device=self.device)\n"
            "\n"
            "def get_benchmark():\n"
            "    return _Bench()\n"
        ),
    )

    results = audit_directory(tmp_path)
    result = next(iter(results.values()))
    assert result["status"] == "compliant"


def test_audit_loader_retries_multigpu_benchmark_on_single_gpu(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    _write(
        tmp_path / "baseline_multigpu_retry.py",
        (
            "from __future__ import annotations\n"
            "import torch\n"
            "from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig\n"
            "from core.benchmark.verification_mixin import VerificationPayloadMixin\n"
            "\n"
            "class _Bench(VerificationPayloadMixin, BaseBenchmark):\n"
            "    def __init__(self) -> None:\n"
            "        super().__init__()\n"
            "        self.world_size = torch.cuda.device_count()\n"
            "        if self.world_size < 2:\n"
            "            raise RuntimeError('example requires >=2 GPUs')\n"
            "        self.output = None\n"
            "        self.register_workload_metadata(requests_per_iteration=1.0)\n"
            "    def setup(self) -> None:\n"
            "        self.output = torch.zeros(1)\n"
            "    def benchmark_fn(self) -> None:\n"
            "        if self.output is None:\n"
            "            raise RuntimeError('missing output')\n"
            "    def capture_verification_payload(self) -> None:\n"
            "        if self.output is None:\n"
            "            raise RuntimeError('missing output')\n"
            "        self._set_verification_payload(\n"
            "            inputs={'probe': torch.zeros(1)},\n"
            "            output=self.output,\n"
            "            batch_size=1,\n"
            "            parameter_count=0,\n"
            "            output_tolerance=(0.0, 0.0),\n"
            "            signature_overrides={'world_size': self.world_size},\n"
            "        )\n"
            "    def validate_result(self):\n"
            "        return None\n"
            "    def get_config(self) -> BenchmarkConfig:\n"
            "        return BenchmarkConfig(iterations=1, warmup=0)\n"
            "\n"
            "def get_benchmark():\n"
            "    return _Bench()\n"
        ),
    )

    results = audit_directory(tmp_path)
    result = next(iter(results.values()))
    assert result["status"] == "compliant"


def test_audit_loader_skips_non_benchmark_submission_files(tmp_path: Path) -> None:
    _write(
        tmp_path / "optimized_submission.py",
        (
            "from __future__ import annotations\n"
            "VALUE = 1\n"
        ),
    )

    assert audit_directory(tmp_path) == {}
