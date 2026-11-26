import subprocess
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "cluster" / "scripts"


def test_activity_logging_stays_outside_publication_and_manifest(tmp_path, monkeypatch):
    run = tmp_path / "run"
    monkeypatch.setenv("CLUSTER_RUN_DIR", str(run))
    root = tmp_path / "cluster"
    root.mkdir()
    scripts = root / "scripts"
    scripts.mkdir()
    for name in ("log_activity.sh", "enable_activity_log.sh", "lib_artifact_dirs.sh"):
        (scripts / name).write_bytes((SCRIPTS / name).read_bytes())
    report = root / "field-report.md"
    report.write_text("existing publication\n")

    subprocess.run(
        ["bash", str(scripts / "log_activity.sh"), "decision", "--run-id", "example", "--note", "test decision"],
        check=True,
    )
    shell = subprocess.run(
        ["bash", "-c", 'source "$1" --run-id example --label local && printf "%s" "$ACTIVITY_LOG_FILE"',
         "activity-test", str(scripts / "enable_activity_log.sh")],
        check=True,
        capture_output=True,
        text=True,
    )

    activity = run / "progress" / "activity"
    assert str(activity / "example_shell.log") in shell.stdout
    assert "test decision" in (activity / "example_activity.md").read_text()
    assert "test decision" in (activity / "example_activity.log").read_text()
    assert report.read_text() == "existing publication\n"
    assert not (run / "raw").exists()
