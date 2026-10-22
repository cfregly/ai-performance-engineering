from __future__ import annotations

from pathlib import Path

from core.scripts.audit_silent_fallbacks import collect_findings


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_audit_scans_core_and_flags_global_warning_filter(tmp_path: Path) -> None:
    _write(
        tmp_path / "core" / "bad_filter.py",
        "import warnings\nwarnings.filterwarnings('ignore')\n",
    )

    findings = collect_findings(tmp_path, ["core"])

    assert any(f.category == "global_warning_filter" for f in findings)


def test_audit_ignores_scoped_warning_filter_inside_catch_warnings(tmp_path: Path) -> None:
    _write(
        tmp_path / "core" / "good_filter.py",
        "import warnings\n"
        "with warnings.catch_warnings():\n"
        "    warnings.filterwarnings('ignore')\n",
    )

    findings = collect_findings(tmp_path, ["core"])

    assert findings == []


def test_audit_flags_stdio_hijack_and_silent_except_pass(tmp_path: Path) -> None:
    _write(
        tmp_path / "ch01" / "bad_entrypoint.py",
        "import os\n"
        "import sys\n"
        "sys.stderr = open(os.devnull, 'w')\n"
        "os.dup2(1, 2)\n"
        "try:\n"
        "    do_thing()\n"
        "except Exception:\n"
        "    pass\n",
    )

    findings = collect_findings(tmp_path, ["ch*"])
    categories = {finding.category for finding in findings}

    assert "stderr_reassignment" in categories
    assert "stdio_dup2_hijack" in categories
    assert "silent_except_pass" in categories


def test_audit_allows_stdio_recovery_when_stream_is_none(tmp_path: Path) -> None:
    _write(
        tmp_path / "core" / "recover_stdio.py",
        "import os\n"
        "import sys\n"
        "if sys.stderr is None:\n"
        "    sys.stderr = open(os.devnull, 'w')\n",
    )

    findings = collect_findings(tmp_path, ["core"])

    assert findings == []


def test_audit_exclusions_are_relative_to_repository_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "artifacts" / "repository"
    _write(
        repo_root / "core" / "included.py",
        "import warnings\nwarnings.filterwarnings('ignore')\n",
    )
    _write(
        repo_root / "core" / "artifacts" / "excluded.py",
        "import warnings\nwarnings.filterwarnings('ignore')\n",
    )

    findings = collect_findings(repo_root, ["core"])

    assert [finding.path.relative_to(repo_root).as_posix() for finding in findings] == [
        "core/included.py"
    ]
    assert findings[0].category == "global_warning_filter"
