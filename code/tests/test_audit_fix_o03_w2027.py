"""Regression tests for W2-027 and the W2-084 residual in the dynamic-router eval stack.

W2-027: a drop count must be divided by the tokens it covers, never by the number of rows.
W2-084 residual: per-window router rows must summarize the whole window, not its last step.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

from labs.dynamic_router import scorecard
from labs.dynamic_router.eval_stack import CheapEvalStack, EvalConfig


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_metrics_dir(
    metrics_dir: Path,
    router_rows: list[dict],
    *,
    write_router: bool = True,
) -> None:
    metrics_dir.mkdir(parents=True)
    _write_jsonl(
        metrics_dir / "quality.jsonl",
        [
            {"task": "mmlu-mini", "correct": True},
            {"task": "gsm8k-lite", "correct": False},
        ],
    )
    _write_jsonl(
        metrics_dir / "latency.jsonl",
        [
            {"ttft_ms": 300.0, "decode_ms": 900.0, "output_tokens": 128, "case": "warm"},
            {"ttft_ms": 450.0, "decode_ms": 1500.0, "output_tokens": 512, "case": "cold"},
        ],
    )
    if write_router:
        _write_jsonl(metrics_dir / "moe_router.jsonl", router_rows)
    _write_jsonl(
        metrics_dir / "moe_traffic.jsonl",
        [{"window": 0, "expert_hist": [10, 12, 9, 11], "imbalance_cv": 0.1}],
    )


def _replay(tmp_path: Path, metrics_dir: Path, **overrides) -> dict[str, object]:
    overrides.setdefault("experts", 4)
    cfg = EvalConfig(run_root=tmp_path / "replay_runs", metrics_dir=metrics_dir, **overrides)
    return CheapEvalStack(cfg).run()


def test_metrics_replay_rejects_window_rows_without_token_counts(tmp_path: Path) -> None:
    # Before the fix these two windows reported a 150% drop rate: 3 drops over 2 rows.
    metrics_dir = tmp_path / "metrics"
    _write_metrics_dir(
        metrics_dir,
        [{"drops": 1, "entropy": 1.0}, {"drops": 2, "entropy": 1.0}],
    )

    with pytest.raises(ValueError, match="row 0 lacks 'total_tokens'"):
        _replay(tmp_path, metrics_dir)


def test_metrics_replay_divides_drops_by_tokens_and_weights_window_means(
    tmp_path: Path,
) -> None:
    metrics_dir = tmp_path / "metrics"
    _write_metrics_dir(
        metrics_dir,
        [
            {"drops": 1, "total_tokens": 64, "entropy": 1.0, "margin": 0.2},
            {"drops": 2, "total_tokens": 36, "entropy": 2.0, "margin": 0.4},
        ],
    )

    summary = _replay(tmp_path, metrics_dir)
    written = json.loads(Path(summary["scorecard_path"]).read_text(encoding="utf-8"))

    assert summary["component_sources"]["moe_router"] == "metrics_replay"
    assert summary["token_drop_rate"] == pytest.approx(3 / 100)
    assert written["moe"]["token_drop_rate"] == pytest.approx(3 / 100)
    assert summary["router_entropy"] == pytest.approx((64 * 1.0 + 36 * 2.0) / 100)
    assert summary["gate_margin"] == pytest.approx((64 * 0.2 + 36 * 0.4) / 100)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"total_tokens": 64}, "row 0 lacks 'drops'"),
        ({"drops": -1, "total_tokens": 64}, "drops=-1"),
        ({"drops": True, "total_tokens": 64}, "drops=True"),
        ({"drops": 1, "total_tokens": "64"}, "total_tokens='64'"),
        ({"drops": 5, "total_tokens": 4}, "exceeds total_tokens=4"),
        ({"drops": 0, "total_tokens": 0}, "zero total_tokens"),
    ],
)
def test_metrics_replay_rejects_invalid_router_counts(
    tmp_path: Path,
    row: dict,
    message: str,
) -> None:
    metrics_dir = tmp_path / "metrics"
    _write_metrics_dir(metrics_dir, [row])

    with pytest.raises(ValueError, match=message):
        _replay(tmp_path, metrics_dir)


def test_metrics_replay_without_router_rows_does_not_report_zero_drops(
    tmp_path: Path,
) -> None:
    # Before the fix a missing moe_router.jsonl next to real traffic rows reported 0.0 drops.
    metrics_dir = tmp_path / "metrics"
    _write_metrics_dir(metrics_dir, [], write_router=False)

    with pytest.raises(ValueError, match="needs moe_router.jsonl rows"):
        _replay(tmp_path, metrics_dir, allow_missing_metrics=True)


def test_synthetic_artifacts_replay_to_the_same_moe_summary(tmp_path: Path) -> None:
    source = CheapEvalStack(
        EvalConfig(run_root=tmp_path / "source_runs", request_count=200, use_vllm=False)
    ).run()
    run_dir = Path(source["run_dir"])
    router_rows = [
        json.loads(line)
        for line in (run_dir / "moe_router.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    # 4000 routing steps give 62 full 64-token windows and a 32-token tail window.
    assert [row["total_tokens"] for row in router_rows] == [64] * 62 + [32]
    assert sum(row["drops"] for row in router_rows) > 0

    replayed = _replay(tmp_path, run_dir, experts=8)

    assert replayed["component_sources"]["moe_router"] == "metrics_replay"
    for key in ("token_drop_rate", "expert_imbalance_cv", "router_entropy", "gate_margin"):
        assert replayed[key] == pytest.approx(source[key], rel=1e-12), key
    assert scorecard._load_run(run_dir)["drop_rate"] == pytest.approx(
        source["token_drop_rate"], rel=1e-12
    )


def test_scorecard_rejects_rows_without_drop_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Before the fix the second row counted as zero drops and halved the reported rate.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(
        run_dir / "moe_router.jsonl",
        [{"drops": 1, "total_tokens": 64}, {"total_tokens": 64}],
    )

    assert math.isnan(scorecard._load_run(run_dir)["drop_rate"])

    monkeypatch.setattr(sys, "argv", ["scorecard.py", str(run_dir), "--min-acc", "0"])
    scorecard.main()
    output = capsys.readouterr().out

    assert "- run: BLOCK" in output
    assert "drop rate unknown: every moe_router.jsonl row needs drops and total_tokens" in output
