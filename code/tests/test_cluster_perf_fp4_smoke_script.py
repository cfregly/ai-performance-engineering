from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys


def _write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_container_smoke_mounts_external_structured_dir_and_preserves_argv(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_cluster = repo_root / "cluster"
    test_cluster = tmp_path / "repo with spaces" / "cluster"
    scripts_dir = test_cluster / "scripts"
    scripts_dir.mkdir(parents=True)

    for relative_path in (
        "scripts/run_cluster_perf_fp4_smoke.sh",
        "scripts/lib_artifact_dirs.sh",
        "scripts/cluster_perf_stack_profiles.sh",
        "configs/cluster_perf_stack_profiles.json",
    ):
        source = source_cluster / relative_path
        destination = test_cluster / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    fake_python = test_cluster / "env" / "venv" / "bin" / "python"
    _write_executable(
        fake_python,
        """#!/usr/bin/env bash
set -euo pipefail
out_json=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--out-json" ]]; then
    out_json="$2"
    break
  fi
  shift
done
if [[ -n "$out_json" ]]; then
  mkdir -p "$(dirname "$out_json")"
  printf '{"status":"ok"}\\n' > "$out_json"
fi
""",
    )
    (scripts_dir / "preflight_cluster_perf_runtime.py").write_text("", encoding="utf-8")
    _write_executable(
        scripts_dir / "run_with_gpu_clocks.sh",
        """#!/usr/bin/env bash
set -euo pipefail
lock_meta=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lock-meta-out) lock_meta="$2"; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
if [[ -n "$lock_meta" ]]; then
  mkdir -p "$(dirname "$lock_meta")"
  printf '{"status":"ok"}\\n' > "$lock_meta"
fi
exec "$@"
""",
    )

    fake_bin = tmp_path / "fake bin"
    argv_log = tmp_path / "docker-argv.json"
    fake_docker = fake_bin / "docker"
    _write_executable(
        fake_docker,
        f"""#!{sys.executable}
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
Path(os.environ["DOCKER_ARGV_LOG"]).write_text(json.dumps(args), encoding="utf-8")
volumes = [args[index + 1] for index, arg in enumerate(args[:-1]) if arg == "--volume"]
output = args[args.index("--out-json") + 1]
for volume in volumes:
    host_dir, container_dir = volume.rsplit(":", 1)
    if output == container_dir or output.startswith(container_dir + "/"):
        relative = output.removeprefix(container_dir).lstrip("/")
        destination = Path(host_dir) / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text('{{"status":"ok","source":"fake-docker"}}\\n', encoding="utf-8")
        print("fake container smoke passed")
        raise SystemExit(0)
print("missing output bind mount", file=sys.stderr)
raise SystemExit(3)
""",
    )
    _write_executable(fake_bin / "nvidia-smi", "#!/usr/bin/env bash\nexit 0\n")

    structured_dir = tmp_path / "external structured results"
    raw_dir = tmp_path / "external raw results"
    injection_marker = tmp_path / "unsafe-shell-expansion"
    injected_m = f"$(touch {injection_marker})"
    run_id = "external-results"
    label = "node one"
    image = "sha256:" + "1" * 64
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "DOCKER_ARGV_LOG": str(argv_log),
            "CLUSTER_RESULTS_STRUCTURED_DIR": str(structured_dir),
            "CLUSTER_RESULTS_RAW_DIR": str(raw_dir),
        }
    )
    proc = subprocess.run(
        [
            "bash",
            str(scripts_dir / "run_cluster_perf_fp4_smoke.sh"),
            "--run-id",
            run_id,
            "--label",
            label,
            "--runtime",
            "container",
            "--stack-profile",
            "orig_parity_container",
            "--image",
            image,
            "--m",
            injected_m,
            "--n",
            "2",
            "--k",
            "3",
            "--warmup",
            "0",
            "--iters",
            "1",
        ],
        cwd=test_cluster,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output
    assert not injection_marker.exists()

    output_name = f"{run_id}_{label}_cluster_perf_fp4_smoke.json"
    output_path = structured_dir / output_name
    assert json.loads(output_path.read_text(encoding="utf-8"))["source"] == "fake-docker"
    assert (raw_dir / f"{run_id}_{label}_cluster_perf_fp4_smoke.log").read_text(
        encoding="utf-8"
    ).strip() == "fake container smoke passed"
    assert str(output_path) in output

    argv = json.loads(argv_log.read_text(encoding="utf-8"))
    volumes = [argv[index + 1] for index, arg in enumerate(argv[:-1]) if arg == "--volume"]
    assert f"{structured_dir}:/cluster-results/structured" in volumes
    assert argv[argv.index("--out-json") + 1] == f"/cluster-results/structured/{output_name}"
    assert argv[argv.index("--m") + 1] == injected_m
    assert not any(str(structured_dir) in arg and arg.startswith("/workspace/") for arg in argv)
