#!/usr/bin/env python3
"""Render the retained startup decomposition.

This script reads only the bounded public data extract beside it. It does
not import torch, run the benchmark, access a GPU, or change qualification.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "startup-decomposition-data.json"
SVG_PATH = ROOT / "startup-decomposition.svg"
PNG_PATH = ROOT / "startup-decomposition.png"

FALSE_COLOR = "#285D8F"
TRUE_COLOR = "#E36A2E"
PAIR_COLOR = "#AAB4BF"
TEXT_COLOR = "#17212B"
MUTED_COLOR = "#586574"
GRID_COLOR = "#DDE3E9"
FAIL_COLOR = "#B42318"
FIGURE_COLOR = "#F5F7FA"
MARKERS = ["o", "s", "^", "D"]
JITTER = [-0.06, -0.02, 0.02, 0.06]


def normalize_svg(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    normalized = re.sub(r"[ \t]+(?=\n|\Z)", "", text)
    if normalized != text:
        path.write_text(normalized, encoding="utf-8")


def load_data() -> dict:
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    expected_seeds = [0, 1, 2, 3]
    if [row["seed"] for row in data["primary"]] != expected_seeds:
        raise RuntimeError("primary rows do not contain all retained seeds")
    if [row["seed"] for row in data["no_break"]] != expected_seeds:
        raise RuntimeError("control rows do not contain all retained seeds")
    return data


def style_axis(axis: plt.Axes, ylabel: str) -> None:
    axis.set_xlim(-0.22, 1.22)
    axis.set_xticks([0, 1], ["False", "True"])
    axis.set_xlabel("nested_graph_breaks", labelpad=8)
    axis.set_ylabel(ylabel, labelpad=8)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.9)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#94A0AD")
    axis.spines["bottom"].set_color("#94A0AD")
    axis.tick_params(colors=MUTED_COLOR)


def draw_pairs(
    axis: plt.Axes,
    rows: list[dict],
    false_key: str,
    true_key: str,
) -> None:
    for row, marker, offset in zip(rows, MARKERS, JITTER):
        false_value = row[false_key]
        true_value = row[true_key]
        xs = [offset, 1 + offset]
        axis.plot(
            xs,
            [false_value, true_value],
            color=PAIR_COLOR,
            linewidth=1.8,
            alpha=0.95,
            zorder=1,
        )
        axis.scatter(
            xs[0],
            false_value,
            marker=marker,
            s=78,
            facecolor=FALSE_COLOR,
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        axis.scatter(
            xs[1],
            true_value,
            marker=marker,
            s=78,
            facecolor=TRUE_COLOR,
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.15,
        1.13,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": TEXT_COLOR,
            "edgecolor": TEXT_COLOR,
        },
    )


def main() -> None:
    data = load_data()
    primary = data["primary"]
    control = data["no_break"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "figure.facecolor": FIGURE_COLOR,
            "axes.facecolor": "white",
            "savefig.facecolor": FIGURE_COLOR,
            "svg.fonttype": "none",
            "svg.hashsalt": "nested-graph-breaks-study",
        }
    )

    figure, axes = plt.subplots(1, 3, figsize=(15.5, 7.2))
    figure.subplots_adjust(
        left=0.065,
        right=0.985,
        top=0.72,
        bottom=0.19,
        wspace=0.28,
    )

    figure.text(
        0.065,
        0.925,
        "Retained cold timing: the startup delta is concentrated in call 1",
        fontsize=20,
        fontweight="bold",
        color=TEXT_COLOR,
        ha="left",
    )
    figure.text(
        0.065,
        0.875,
        "Paired fresh-process workers. Every retained seed is shown.",
        fontsize=12,
        color=MUTED_COLOR,
        ha="left",
    )

    arm_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=FALSE_COLOR,
            markeredgecolor="white",
            markersize=9,
            label="False",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=TRUE_COLOR,
            markeredgecolor="white",
            markersize=9,
            label="True",
        ),
    ]
    seed_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color=PAIR_COLOR,
            markerfacecolor="white",
            markeredgecolor=MUTED_COLOR,
            markersize=7,
            label=f"Seed {seed}",
        )
        for seed, marker in enumerate(MARKERS)
    ]
    figure.legend(
        handles=arm_handles + seed_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.895),
        frameon=False,
        ncol=6,
        columnspacing=1.2,
        handletextpad=0.45,
    )

    first_axis, later_axis, control_axis = axes

    draw_pairs(
        first_axis,
        primary,
        "call_1_false_s",
        "call_1_true_s",
    )
    style_axis(first_axis, "Call 1 wall time (seconds)")
    first_axis.set_ylim(2.25, 3.34)
    first_axis.set_title(
        "Primary call 1\nTrue is slower in all four seeds",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    first_axis.text(
        0.04,
        0.92,
        "True − False\n+0.786 to +0.876 s",
        transform=first_axis.transAxes,
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=TRUE_COLOR,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "#FFF2EA",
            "edgecolor": "#F4C5AC",
        },
    )
    add_panel_label(first_axis, "A")

    draw_pairs(
        later_axis,
        primary,
        "calls_2_10_false_ms",
        "calls_2_10_true_ms",
    )
    style_axis(later_axis, "Sum of calls 2–10 (milliseconds)")
    later_axis.set_ylim(15.0, 41.5)
    later_axis.set_title(
        "Primary calls 2–10\nTrue is faster in all four seeds",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    later_axis.text(
        0.53,
        0.92,
        "True − False\n−18.6 to −19.3 ms",
        transform=later_axis.transAxes,
        va="top",
        ha="left",
        fontsize=10.5,
        fontweight="bold",
        color=FALSE_COLOR,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "#EDF5FC",
            "edgecolor": "#B8D3EA",
        },
    )
    add_panel_label(later_axis, "B")

    draw_pairs(
        control_axis,
        control,
        "call_1_false_s",
        "call_1_true_s",
    )
    style_axis(control_axis, "Call 1 wall time (seconds)")
    control_axis.set_ylim(1.84, 2.075)
    control_axis.set_title(
        "No-break control call 1\nSeed 1 drives the startup failure",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    failed = next(row for row in control if row.get("startup_control_failure"))
    failed_x = 1 + JITTER[failed["seed"]]
    failed_y = failed["call_1_true_s"]
    control_axis.scatter(
        failed_x,
        failed_y,
        marker=MARKERS[failed["seed"]],
        s=180,
        facecolor="none",
        edgecolor=FAIL_COLOR,
        linewidth=2.2,
        zorder=4,
    )
    control_axis.annotate(
        "Seed 1 control failure\n"
        "call 1 ratio 0.941900\n"
        "first-10 ratio 0.942412\n"
        "outside [0.95, 1.05]",
        xy=(failed_x, failed_y),
        xytext=(0.02, 2.057),
        textcoords="data",
        fontsize=9.7,
        fontweight="bold",
        color=FAIL_COLOR,
        ha="left",
        va="top",
        arrowprops={
            "arrowstyle": "-|>",
            "color": FAIL_COLOR,
            "linewidth": 1.3,
            "connectionstyle": "arc3,rad=-0.15",
        },
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "#FFF1F0",
            "edgecolor": "#F3B6B2",
        },
    )
    add_panel_label(control_axis, "C")

    figure.text(
        0.065,
        0.075,
        "False / True = torch._dynamo.config.nested_graph_breaks   •   "
        "Fixed gates unchanged   •   Exact compiler phase unresolved   •   "
        "Public evidence projection",
        fontsize=10.2,
        color=MUTED_COLOR,
        ha="left",
    )

    metadata = {
        "Title": "Retained startup decomposition",
        "Description": (
            "Paired retained seed timings for the primary first call, "
            "primary calls 2 through 10, and no-break first call."
        ),
        "Creator": "plot_startup_decomposition.py",
        "Date": "2026-10-21T00:00:00Z",
    }
    figure.savefig(SVG_PATH, format="svg", metadata=metadata)
    figure.savefig(
        PNG_PATH,
        format="png",
        dpi=180,
        metadata={"Title": metadata["Title"], "Author": metadata["Creator"]},
    )
    plt.close(figure)
    normalize_svg(SVG_PATH)
    print(SVG_PATH)
    print(PNG_PATH)


if __name__ == "__main__":
    main()
