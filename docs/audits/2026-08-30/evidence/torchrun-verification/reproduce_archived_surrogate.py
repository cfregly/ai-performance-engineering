"""Replay the preserved old wrapper's actual CPU mechanism, never a GPU benchmark."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile

import torch


def main():
    evidence = Path(__file__).resolve().parent
    root = evidence.parents[4]
    sys.path.insert(0, str(root / "code"))
    archived = evidence / "torchrun_harness.before.py.txt"
    expected = json.loads((evidence / "before-source.json").read_text())["source_files"][
        "code/labs/train_distributed/training_utils/torchrun_harness.py"
    ]
    assert hashlib.sha256(archived.read_bytes()).hexdigest() == expected
    namespace = {"__name__": "local019_archived_wrapper"}
    exec(compile(archived.read_text(), str(archived), "exec"), namespace)
    old_wrapper = namespace["TorchrunScriptBenchmark"]
    with tempfile.TemporaryDirectory(prefix="local019-surrogate-replay-") as temporary:
        directory = Path(temporary)
        marker = directory / "child-executed"
        child = directory / "actual_training.py"
        child.write_text(
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('executed')\n"
            "raise SystemExit(37)\n"
        )
        outputs, launch_args = [], []
        for mode in ("baseline", "corrupt"):
            benchmark = old_wrapper(
                script_path=child, base_args=["--mode", mode],
                target_label="local019:actual_training", multi_gpu_required=False,
                default_nproc_per_node=1,
            )
            benchmark.device = torch.device("cpu")
            spec = benchmark.get_torchrun_spec()
            outputs.append(benchmark._subprocess_verify_output.clone())
            launch_args.append(spec.script_args)
        assert launch_args[0] != launch_args[1]
        assert torch.equal(outputs[0], outputs[1]) and not marker.exists()
        print(json.dumps({
            "scope": "Archived original CPU toy computation; no GPU or child-training acceptance",
            "archived_source_sha256": expected,
            "child_launched": marker.exists(),
            "child_would_exit": 37,
            "different_child_args": launch_args,
            "identical_verification_outputs": True,
            "output_shape": list(outputs[0].shape),
            "output_abs_max": outputs[0].abs().max().item(),
        }, indent=2))


if __name__ == "__main__":
    main()
