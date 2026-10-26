#!/usr/bin/env python3
"""Render the nested-graph-breaks Whiteboard visual package.

The renderer reads retained public evidence. It does not execute the workload.
It deliberately keeps the original historical cohort separate from fresh run
022. The code-native drawings make every count deterministic and keep the
claim-bearing labels out of image-generation models.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import random
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent

BG = "#FBFBF8"
INK = "#202326"
BLUE = "#3478BA"
ORANGE = "#E87823"
BLUE_LIGHT = "#DCEAF6"
ORANGE_LIGHT = "#FBE6D7"
MUTED = "#6E7478"
GRID = "#D9DAD5"
PANEL = "#F3F1EA"
WHITE = "#FFFFFF"

HEADING_FONT = Path("/System/Library/Fonts/MarkerFelt.ttc")
BODY_FONT = Path("/System/Library/Fonts/Avenir Next.ttc")

ANALYZER_PATH = PACKAGE / "results" / "analyzer-result.json"
BASELINE_PATH = PACKAGE / "results" / "nested_break_depth8-seed0-flag0" / "result.json"
CANDIDATE_PATH = PACKAGE / "results" / "nested_break_depth8-seed0-flag1" / "result.json"
FRESH_PATH = PACKAGE / "profiles" / "fresh022" / "qualification" / "FINAL-SCIENTIFIC-QUALIFICATION.json"
README_PATH = PACKAGE / "README.md"
PROTOCOL_PATH = PACKAGE / "protocol.json"
IMAGEGEN_CANDIDATE = HERE / "sources" / "unused-imagegen-cover-candidate.png"
GAMMA_EXPORT_ZIP = HERE / "sources" / "gamma-453-faster-at-steady-state-original.zip"
GAMMA_EXPORT_DIR = HERE / "sources" / "gamma-original-export"
GAMMA_CARD_1 = GAMMA_EXPORT_DIR / "1_453-faster-at-steady-state.png"
GAMMA_CARD_2 = GAMMA_EXPORT_DIR / "2_Actual-emitted-Inductor-regions-depth-8-seed-0-Each-dot-represents-one-ATen-operation-FP32-1048576-e.png"
GAMMA_CARD_3 = GAMMA_EXPORT_DIR / "3_Faster-reuse-Longer-first-call.png"
GAMMA_URL = "https://gamma.app/docs/453-faster-at-steady-state-u5q8j0ibrd2qhu1?mode=doc"
GAMMA_EXPORT_SHA256 = "89aeb6360afbb8ce289896d8530d5927bb0caf78b78613db947cdb57794d5f63"
GAMMA_CARD_1_SHA256 = "d43959205955cc07b463d05f5ad2805efeb16e9d275202b6a5b575f4eb375be6"
GAMMA_CARD_2_SHA256 = "87dbcb6e74dbb6eea2d8d54b8a11f533f00f2c5d752854674dc7d163cdc97698"
GAMMA_CARD_3_SHA256 = "2d3a4534ac2280884c203273634d5cb1f34eebfc86b158e49d2824e5b46a61c4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def gamma_source_metadata() -> dict:
    """Verify the immutable Gamma export used as the public cover source.

    The public cover is a deterministic landscape/square adaptation of card 1.
    It does not carry the source card's ambiguous False/True labels or its
    rounded display values. The exact measured values are read independently
    from retained JSON evidence below.
    """
    assert sha256(GAMMA_EXPORT_ZIP) == GAMMA_EXPORT_SHA256
    expected_cards = {
        GAMMA_CARD_1: GAMMA_CARD_1_SHA256,
        GAMMA_CARD_2: GAMMA_CARD_2_SHA256,
        GAMMA_CARD_3: GAMMA_CARD_3_SHA256,
    }
    for path, expected in expected_cards.items():
        assert sha256(path) == expected

    with Image.open(GAMMA_CARD_1) as source:
        image = source.convert("RGB")
        assert image.size == (2400, 3000)
        sampled = {
            "background": image.getpixel((100, 100)),
            "blue": image.getpixel((500, 1000)),
            "orange": image.getpixel((500, 1950)),
        }
    expected_palette = {
        "background": (251, 251, 248),
        "blue": (52, 120, 186),
        "orange": (232, 120, 35),
    }
    assert sampled == expected_palette
    assert BG.upper() == "#%02X%02X%02X" % sampled["background"]
    assert BLUE.upper() == "#%02X%02X%02X" % sampled["blue"]
    assert ORANGE.upper() == "#%02X%02X%02X" % sampled["orange"]
    return {
        "dimensions": [2400, 3000],
        "sampled_palette": {
            name: "#%02X%02X%02X" % value for name, value in sampled.items()
        },
    }


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def load_evidence() -> dict:
    analyzer = json.loads(ANALYZER_PATH.read_text())
    baseline = json.loads(BASELINE_PATH.read_text())
    candidate = json.loads(CANDIDATE_PATH.read_text())
    fresh = json.loads(FRESH_PATH.read_text())

    case = analyzer["cases"]["nested_break_depth8"]
    seeds = case["per_seed"]
    baseline_steady_us = geometric_mean(
        [row["baseline_enabled_0"]["steady_median_invocation_host_ms"] for row in seeds]
    ) * 1000
    candidate_steady_us = geometric_mean(
        [row["candidate_enabled_1"]["steady_median_invocation_host_ms"] for row in seeds]
    ) * 1000
    baseline_first10_ms = geometric_mean(
        [row["baseline_enabled_0"]["cold_first_10_total_host_ms"] for row in seeds]
    )
    candidate_first10_ms = geometric_mean(
        [row["candidate_enabled_1"]["cold_first_10_total_host_ms"] for row in seeds]
    )
    historical_speedup = case["steady_median_invocation"]["geometric_mean_ratio"]
    startup_penalty_pct = (candidate_first10_ms / baseline_first10_ms - 1) * 100

    baseline_invocations = baseline["profile"]["invocation_count_completed"]
    candidate_invocations = candidate["profile"]["invocation_count_completed"]
    baseline_launches = baseline["profile"]["cuda_graph_launch_event_count"] // baseline_invocations
    candidate_launches = candidate["profile"]["cuda_graph_launch_event_count"] // candidate_invocations
    baseline_counters = baseline["profile"]["compiler_counter_audit"]["before_profile"]
    candidate_counters = candidate["profile"]["compiler_counter_audit"]["before_profile"]
    baseline_regions = baseline_counters["stats.unique_graphs"]
    candidate_regions = candidate_counters["stats.unique_graphs"]
    operations = baseline_counters["stats.calls_captured"]
    assert candidate_counters["stats.calls_captured"] == operations

    fresh_speedup = fresh["saved_primary_summary"]["steady_median_invocation"]["geometric_mean_ratio"]

    evidence = {
        "baseline_steady_us": baseline_steady_us,
        "candidate_steady_us": candidate_steady_us,
        "baseline_first10_ms": baseline_first10_ms,
        "candidate_first10_ms": candidate_first10_ms,
        "historical_speedup": historical_speedup,
        "startup_penalty_pct": startup_penalty_pct,
        "seed_ratios": case["steady_median_invocation"]["observed_seed_ratios"],
        "steady_min_ratio": case["steady_median_invocation"]["observed_min_seed_ratio"],
        "steady_max_ratio": case["steady_median_invocation"]["observed_max_seed_ratio"],
        "steady_pass": case["steady_win_criteria_pass"],
        "startup_pass": case["startup_win_criteria_pass"],
        "baseline_launches": baseline_launches,
        "candidate_launches": candidate_launches,
        "baseline_regions": baseline_regions,
        "candidate_regions": candidate_regions,
        "operations": operations,
        "baseline_ops_per_region": operations // baseline_regions,
        "candidate_ops_per_region": operations // candidate_regions,
        "profiled_invocations": baseline_invocations,
        "fresh022_speedup": fresh_speedup,
    }

    assert f"{baseline_steady_us:.3f}" == "407.567"
    assert f"{candidate_steady_us:.3f}" == "89.935"
    assert f"{historical_speedup:.6f}" == "4.531795"
    assert f"{baseline_first10_ms:.3f}" == "2411.177"
    assert f"{candidate_first10_ms:.3f}" == "3211.677"
    assert f"{startup_penalty_pct:.1f}" == "33.2"
    assert baseline_launches == 16 and candidate_launches == 2
    assert baseline_regions == 16 and candidate_regions == 2 and operations == 64
    assert case["steady_win_criteria_pass"] is True
    assert case["startup_win_criteria_pass"] is False
    assert f"{fresh_speedup:.6f}" == "4.541758"
    return evidence


E = load_evidence()


@lru_cache(maxsize=None)
def font(size: int, heading: bool = False) -> ImageFont.FreeTypeFont:
    path = HEADING_FONT if heading else BODY_FONT
    return ImageFont.truetype(str(path), max(8, int(size)))


def fit_font(draw: ImageDraw.ImageDraw, text: str, max_width: int, start: int, minimum: int, heading: bool = False):
    for size in range(start, minimum - 1, -1):
        value = font(size, heading)
        box = draw.textbbox((0, 0), text, font=value)
        if box[2] - box[0] <= max_width:
            return value
    return font(minimum, heading)


def wrap_lines(draw: ImageDraw.ImageDraw, text: str, face, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        box = draw.textbbox((0, 0), trial, font=face)
        if box[2] - box[0] <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


class Board:
    def __init__(self, width: int, height: int, key: str, title: str, eyebrow: str, square: bool = False):
        self.w = width
        self.h = height
        self.square = square
        self.key = key
        self.image = Image.new("RGB", (width, height), BG)
        self.draw = ImageDraw.Draw(self.image)
        self.rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) + width)
        self.margin = 54 if width >= 1000 else 28
        self.top = 178 if not square else 190
        self.bottom = height - (112 if not square else 138)
        self.header(title, eyebrow)
        setting_y = 153 if not square else 164
        self.draw.text(
            (self.margin, setting_y),
            "Nested graph-break support: off versus on",
            fill=MUTED,
            font=font(18),
        )

    def header(self, title: str, eyebrow: str) -> None:
        eyebrow_face = font(24 if self.w >= 1000 else 18)
        self.draw.text((self.margin, 28), eyebrow.upper(), fill=MUTED, font=eyebrow_face)
        max_width = self.w - self.margin * 2
        title_face = fit_font(self.draw, title, max_width, 66 if self.w >= 1200 else 54, 36, heading=True)
        self.draw.text((self.margin, 62), title, fill=INK, font=title_face)
        y = 139 if not self.square else 150
        self.marker_line((self.margin, y), (self.w - self.margin, y), INK, width=3)

    def footer(self, *parts: str) -> None:
        text = "  ·  ".join(part for part in parts if part)
        y = self.h - (82 if not self.square else 104)
        self.marker_line((self.margin, y - 14), (self.w - self.margin, y - 14), GRID, width=2)
        face = fit_font(self.draw, text, self.w - self.margin * 2, 23 if self.w >= 1000 else 16, 15)
        self.draw.text((self.margin, y), text, fill=MUTED, font=face)

    def marker_line(self, start: tuple[float, float], end: tuple[float, float], color: str, width: int = 4) -> None:
        for offset in (-1, 0, 1):
            jitter = self.rng.uniform(-1.5, 1.5)
            points = [(start[0], start[1] + offset + jitter), (end[0], end[1] + offset - jitter)]
            self.draw.line(points, fill=color, width=max(1, width - abs(offset)))

    def arrow(self, start: tuple[float, float], end: tuple[float, float], color: str, width: int = 7) -> None:
        self.marker_line(start, end, color, width)
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        size = max(14, width * 3)
        left = (end[0] - size * math.cos(angle - 0.55), end[1] - size * math.sin(angle - 0.55))
        right = (end[0] - size * math.cos(angle + 0.55), end[1] - size * math.sin(angle + 0.55))
        self.draw.polygon([end, left, right], fill=color)

    def rough_rect(self, box: tuple[float, float, float, float], outline: str = INK, fill: str | None = None,
                   width: int = 4, radius: int = 14, hatch: bool = False) -> None:
        x0, y0, x1, y1 = box
        self.draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
        self.draw.rounded_rectangle((x0 + 2, y0 - 1, x1 - 1, y1 + 1), radius=radius, outline=outline, width=1)
        if hatch:
            step = 18
            for x in range(int(x0 - (y1 - y0)), int(x1), step):
                sx = max(x0 + 4, x)
                sy = y1 - 4 - max(0, x0 - x)
                ex = min(x1 - 4, x + (y1 - y0))
                ey = y0 + 4 + max(0, x + (y1 - y0) - x1)
                if sx < ex and sy > ey:
                    self.draw.line((sx, sy, ex, ey), fill=outline, width=2)

    def panel(self, box: tuple[float, float, float, float], label: str | None = None, accent: str = INK) -> None:
        self.rough_rect(box, outline=GRID, fill=WHITE, width=3, radius=20)
        if label:
            self.draw.text((box[0] + 22, box[1] + 16), label.upper(), fill=accent, font=font(22))

    def pill(self, box: tuple[float, float, float, float], text: str, fill: str, ink: str = INK, size: int = 24) -> None:
        self.draw.rounded_rectangle(box, radius=(box[3] - box[1]) // 2, fill=fill)
        face = fit_font(self.draw, text, int(box[2] - box[0] - 20), size, max(11, size - 8))
        bbox = self.draw.textbbox((0, 0), text, font=face)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        self.draw.text(((box[0] + box[2] - tw) / 2, (box[1] + box[3] - th) / 2 - bbox[1]), text, fill=ink, font=face)

    def text(self, xy: tuple[float, float], value: str, size: int = 30, fill: str = INK,
             heading: bool = False, max_width: int | None = None, anchor: str | None = None) -> None:
        face = font(size, heading)
        if max_width:
            face = fit_font(self.draw, value, max_width, size, max(12, int(size * 0.66)), heading)
        self.draw.text(xy, value, fill=fill, font=face, anchor=anchor)

    def paragraph(self, xy: tuple[int, int], value: str, max_width: int, size: int = 28, fill: str = INK,
                  spacing: int = 8) -> int:
        face = font(size)
        lines = wrap_lines(self.draw, value, face, max_width)
        line_height = self.draw.textbbox((0, 0), "Ag", font=face)[3] + spacing
        for index, line in enumerate(lines):
            self.draw.text((xy[0], xy[1] + index * line_height), line, fill=fill, font=face)
        return xy[1] + len(lines) * line_height

    def save(self, path: Path) -> None:
        self.image.save(path, format="PNG", optimize=True)


def draw_region_chain(board: Board, x: int, y: int, count: int, total_width: int, height: int, color: str,
                      columns: int | None = None) -> None:
    columns = columns or count
    rows = math.ceil(count / columns)
    gap = 12
    cell_w = (total_width - gap * (columns - 1)) / columns
    cell_h = (height - gap * (rows - 1)) / rows
    for index in range(count):
        row, col = divmod(index, columns)
        x0 = x + col * (cell_w + gap)
        y0 = y + row * (cell_h + gap)
        board.rough_rect((x0, y0, x0 + cell_w, y0 + cell_h), outline=color, fill=WHITE, width=3,
                         radius=max(5, int(cell_h / 6)), hatch=True)


def draw_bar(board: Board, x: int, y: int, max_width: int, value: float, maximum: float, label: str,
             value_text: str, color: str, height: int = 70) -> None:
    board.text((x, y - 36), label, 25, fill=INK, max_width=max_width)
    board.draw.rounded_rectangle((x, y, x + max_width, y + height), radius=height // 2, fill=PANEL)
    used = max(height, max_width * value / maximum)
    board.draw.rounded_rectangle((x, y, x + used, y + height), radius=height // 2, fill=color)
    board.text((x + used + 18, y + height / 2), value_text, 28, fill=color, anchor="lm")


def render_fun(width: int, height: int, square: bool) -> Image.Image:
    # The immutable Gamma card is the design and palette source. Public values
    # are redrawn from retained JSON so the adapted layouts can use exact copy
    # and the required natural support labels.
    gamma_source_metadata()
    board = Board(width, height, "fun", f"{E['historical_speedup']:.6f}x Faster at Steady State", "B200 compiler microbenchmark", square)
    if square:
        x, bar_width = 110, 700
        board.text((540, 225), "Synchronized steady median · lower is better", 27, fill=INK, anchor="mm")
        draw_bar(board, x, 345, bar_width, E["baseline_steady_us"], E["baseline_steady_us"],
                 "Support off", f"{E['baseline_steady_us']:.3f} µs", BLUE, 90)
        draw_bar(board, x, 575, bar_width, E["candidate_steady_us"], E["baseline_steady_us"],
                 "Support on", f"{E['candidate_steady_us']:.3f} µs", ORANGE, 90)
        board.pill((175, 760, 905, 855), "same synthetic work · steady route PASS", ORANGE_LIGHT, ORANGE, 27)
        board.footer("same 64 captured operations", "four paired seeds", "startup route FAIL")
    else:
        x, bar_width = 190, 1060
        board.text((800, 215), "Synchronized steady median · lower is better", 30, fill=INK, anchor="mm")
        draw_bar(board, x, 330, bar_width, E["baseline_steady_us"], E["baseline_steady_us"],
                 "Support off", f"{E['baseline_steady_us']:.3f} µs", BLUE, 100)
        draw_bar(board, x, 555, bar_width, E["candidate_steady_us"], E["baseline_steady_us"],
                 "Support on", f"{E['candidate_steady_us']:.3f} µs", ORANGE, 100)
        board.pill((455, 705, 1145, 770), "same synthetic work · steady route PASS", ORANGE_LIGHT, ORANGE, 24)
        board.footer("same 64 captured operations", "four paired seeds", "startup route FAIL")
    return board.image


def render_a(width: int, height: int, square: bool) -> Image.Image:
    board = Board(width, height, "a", "Same Work, Fewer Replay Regions", "Explainer A · mechanism", square)
    if square:
        x, w = 80, width - 160
        board.text((x, 210), "Support off · 16 emitted regions", 29, fill=BLUE)
        draw_region_chain(board, x, 260, 16, w, 210, BLUE, columns=8)
        board.text((width / 2, 505), "16 × 4 ops = 64", 34, fill=BLUE, anchor="mm")
        board.arrow((width / 2, 550), (width / 2, 610), INK, 7)
        board.text((x, 635), "Support on · 2 emitted regions", 29, fill=ORANGE)
        draw_region_chain(board, x, 685, 2, w, 120, ORANGE, columns=2)
        board.text((width / 2, 855), "2 × 32 ops = 64", 34, fill=ORANGE, anchor="mm")
    else:
        board.text((86, 195), "Support off", 31, fill=BLUE, heading=True)
        draw_region_chain(board, 86, 250, 16, 1428, 150, BLUE, columns=16)
        board.text((800, 445), "16 regions × 4 ATen ops", 38, fill=BLUE, anchor="mm")
        board.arrow((800, 480), (800, 545), INK, 7)
        board.text((86, 560), "Support on", 31, fill=ORANGE, heading=True)
        draw_region_chain(board, 330, 615, 2, 940, 110, ORANGE, columns=2)
        board.text((800, 762), "2 regions × 32 ATen ops", 38, fill=ORANGE, anchor="mm")
    board.footer("64 captured operations in both arms", "depth-8 seed-0 emitted regions", "measured")
    return board.image


def render_b(width: int, height: int, square: bool) -> Image.Image:
    board = Board(width, height, "b", "Steady Wins, Startup Does Not", "Explainer B · numerical result", square)
    if square:
        board.panel((60, 200, 1020, 520), "Synchronized steady median", ORANGE)
        draw_bar(board, 105, 300, 600, E["baseline_steady_us"], E["baseline_steady_us"], "Support off", f"{E['baseline_steady_us']:.3f} µs", BLUE, 58)
        draw_bar(board, 105, 420, 600, E["candidate_steady_us"], E["baseline_steady_us"], "Support on", f"{E['candidate_steady_us']:.3f} µs", ORANGE, 58)
        board.text((815, 385), f"{E['historical_speedup']:.6f}×", 42, fill=ORANGE, anchor="mm")
        board.panel((60, 555, 1020, 895), "First ten compiled-call intervals", BLUE)
        draw_bar(board, 105, 655, 600, E["baseline_first10_ms"], E["candidate_first10_ms"], "Support off", f"{E['baseline_first10_ms']:.3f} ms", BLUE, 58)
        draw_bar(board, 105, 775, 600, E["candidate_first10_ms"], E["candidate_first10_ms"], "Support on", f"{E['candidate_first10_ms']:.3f} ms", ORANGE, 58)
        board.pill((735, 690, 970, 780), f"Support on +{E['startup_penalty_pct']:.1f}% · FAIL", ORANGE_LIGHT, ORANGE, 22)
    else:
        left = (65, 200, 785, 745)
        right = (815, 200, 1535, 745)
        board.panel(left, "Steady · lower is better", ORANGE)
        draw_bar(board, 110, 350, 460, E["baseline_steady_us"], E["baseline_steady_us"], "Support off", f"{E['baseline_steady_us']:.3f} µs", BLUE)
        draw_bar(board, 110, 510, 460, E["candidate_steady_us"], E["baseline_steady_us"], "Support on", f"{E['candidate_steady_us']:.3f} µs", ORANGE)
        board.text((425, 665), f"{E['historical_speedup']:.6f}×", 50, fill=ORANGE, anchor="mm")
        board.panel(right, "First ten · lower is better", BLUE)
        draw_bar(board, 860, 350, 460, E["baseline_first10_ms"], E["candidate_first10_ms"], "Support off", f"{E['baseline_first10_ms']:.3f} ms", BLUE)
        draw_bar(board, 860, 510, 460, E["candidate_first10_ms"], E["candidate_first10_ms"], "Support on", f"{E['candidate_first10_ms']:.3f} ms", ORANGE)
        board.pill((1030, 630, 1430, 706), f"Support on +{E['startup_penalty_pct']:.1f}% · route FAIL", ORANGE_LIGHT, ORANGE, 21)
    board.footer("geometric means across 4 paired seeds", "PyTorch 2.14.1+cu130", "NVIDIA B200")
    return board.image


def draw_trace(board: Board, x: int, y: int, width: int, count: int, color: str, label: str) -> None:
    board.text((x, y - 44), label, 27, fill=color)
    board.marker_line((x, y + 40), (x + width, y + 40), GRID, 3)
    gap = width / count
    for index in range(count):
        x0 = x + index * gap + 3
        x1 = x + (index + 1) * gap - 3
        board.draw.rounded_rectangle((x0, y, x1, y + 80), radius=8, fill=BLUE_LIGHT if color == BLUE else ORANGE_LIGHT,
                                     outline=color, width=3)


def render_c(width: int, height: int, square: bool) -> Image.Image:
    board = Board(width, height, "c", "How to Spot Replay Fragmentation", "Explainer C · diagnostic pattern", square)
    if square:
        x, w = 80, width - 160
        draw_trace(board, x, 275, w, 16, BLUE, "One profiled invocation · support off")
        board.text((width / 2, 420), "16 cudaGraphLaunch events", 34, fill=BLUE, heading=True, anchor="mm")
        draw_trace(board, x, 575, w, 2, ORANGE, "One profiled invocation · support on")
        board.text((width / 2, 720), "2 cudaGraphLaunch events", 34, fill=ORANGE, heading=True, anchor="mm")
        board.pill((205, 790, 875, 865), "Count launches inside one invocation", PANEL, INK, 25)
    else:
        x, w = 100, width - 200
        draw_trace(board, x, 280, w, 16, BLUE, "Support off · repeated small replay launches")
        draw_trace(board, x, 540, w, 2, ORANGE, "Support on · two larger replay launches")
        board.text((800, 430), "16 launches", 43, fill=BLUE, heading=True, anchor="mm")
        board.text((800, 690), "2 launches", 43, fill=ORANGE, heading=True, anchor="mm")
    board.footer("seed 0 · 80/5 versus 10/5 cudaGraphLaunch events", "diagram, not a profiler screenshot")
    return board.image


def render_curious(width: int, height: int, square: bool) -> Image.Image:
    board = Board(width, height, "aud-curious", "Same Work, Fewer Trips", "Curious non-specialist", square)
    if square:
        board.text((90, 225), "Support off", 30, fill=BLUE, heading=True)
        draw_region_chain(board, 90, 275, 16, 900, 200, BLUE, columns=8)
        board.text((540, 515), "16 small trips", 38, fill=BLUE, heading=True, anchor="mm")
        board.arrow((540, 555), (540, 620), INK, 7)
        board.text((90, 650), "Support on", 30, fill=ORANGE, heading=True)
        draw_region_chain(board, 180, 700, 2, 720, 120, ORANGE, columns=2)
        board.text((540, 865), "2 larger trips", 38, fill=ORANGE, heading=True, anchor="mm")
    else:
        board.panel((70, 220, 760, 660), "Support off · many small trips", BLUE)
        draw_region_chain(board, 120, 320, 16, 590, 180, BLUE, columns=8)
        board.text((415, 555), "407.567 µs", 47, fill=BLUE, heading=True, anchor="mm")
        board.panel((840, 220, 1530, 660), "Support on · two larger trips", ORANGE)
        draw_region_chain(board, 980, 320, 2, 410, 180, ORANGE, columns=2)
        board.text((1185, 555), "89.935 µs", 47, fill=ORANGE, heading=True, anchor="mm")
        board.text((800, 715), f"{E['historical_speedup']:.6f}× faster after warmup", 42, fill=ORANGE, anchor="mm")
    board.footer("same synthetic depth-8 work", "same 64 operations", "startup was 33.2% longer")
    return board.image


def render_ml(width: int, height: int, square: bool) -> Image.Image:
    board = Board(width, height, "aud-ml-engineer", "Nested Support Changes the Compiled Shape", "ML and AI engineer", square)
    items = [
        ("Support off", "16 regions", "16 replay launches", BLUE),
        ("Support on", "2 regions", "2 replay launches", ORANGE),
    ]
    if square:
        y = 225
        for flag, regions, launches, color in items:
            board.panel((70, y, 1010, y + 260), flag, color)
            board.text((130, y + 110), regions, 43, fill=color, heading=True)
            board.arrow((410, y + 140), (600, y + 140), color, 6)
            board.text((650, y + 110), launches, 36, fill=color, heading=True)
            y += 310
        board.pill((190, 865, 890, 930), "fullgraph=False · reduce-overhead", PANEL, INK, 24)
    else:
        x_positions = [80, 830]
        for x, (flag, regions, launches, color) in zip(x_positions, items):
            board.panel((x, 220, x + 690, 670), flag, color)
            board.text((x + 345, 345), regions, 54, fill=color, heading=True, anchor="mm")
            board.arrow((x + 180, 430), (x + 510, 430), color, 7)
            board.text((x + 345, 525), launches, 42, fill=color, heading=True, anchor="mm")
            board.text((x + 345, 600), "64 captured ATen ops", 27, fill=INK, anchor="mm")
        board.text((800, 730), "Compile shape changed. Workload code did not.", 38, fill=INK, heading=True, anchor="mm")
    board.footer("same explicit graph-break workload", "depth 8", "PyTorch 2.14.1+cu130")
    return board.image


def render_performance(width: int, height: int, square: bool) -> Image.Image:
    board = Board(width, height, "aud-performance-engineer", "Four Seeds Agree on the Steady Win", "Performance and capacity engineer", square)
    ratios = E["seed_ratios"]
    if square:
        x0, x1 = 140, 940
        y0 = 600
        board.marker_line((x0, y0), (x1, y0), INK, 4)
        minimum, maximum = 4.45, 4.62
        for tick in (4.45, 4.50, 4.55, 4.60):
            x = x0 + (tick - minimum) / (maximum - minimum) * (x1 - x0)
            board.marker_line((x, y0 - 12), (x, y0 + 12), INK, 2)
            board.text((x, y0 + 28), f"{tick:.2f}×", 21, fill=MUTED, anchor="ma")
        for index, value in enumerate(ratios):
            x = x0 + (value - minimum) / (maximum - minimum) * (x1 - x0)
            y = 300 + index * 70
            board.marker_line((x, y + 18), (x, y0), GRID, 2)
            board.draw.ellipse((x - 18, y, x + 18, y + 36), fill=ORANGE, outline=INK, width=2)
            board.text((x, y - 18), f"seed {index}\n{value:.4f}×", 20, fill=INK, anchor="ms")
        board.text((540, 760), f"geometric mean {E['historical_speedup']:.6f}×", 37, fill=ORANGE, anchor="mm")
        board.pill((155, 825, 925, 895), "steady PASS · startup FAIL", PANEL, INK, 27)
    else:
        x0, x1, y = 170, 1430, 450
        board.marker_line((x0, y), (x1, y), INK, 5)
        minimum, maximum = 4.45, 4.62
        for tick in (4.45, 4.50, 4.55, 4.60):
            x = x0 + (tick - minimum) / (maximum - minimum) * (x1 - x0)
            board.marker_line((x, y - 18), (x, y + 18), INK, 3)
            board.text((x, y + 40), f"{tick:.2f}×", 24, fill=MUTED, anchor="ma")
        offsets = [-120, -55, 55, 120]
        for index, (value, dy) in enumerate(zip(ratios, offsets)):
            x = x0 + (value - minimum) / (maximum - minimum) * (x1 - x0)
            board.marker_line((x, y + dy), (x, y), GRID, 2)
            board.draw.ellipse((x - 22, y + dy - 22, x + 22, y + dy + 22), fill=ORANGE, outline=INK, width=2)
            board.text((x, y + dy - 35 if dy < 0 else y + dy + 35), f"seed {index} · {value:.6f}×", 23, fill=INK,
                       anchor="ms" if dy < 0 else "ma")
        board.text((800, 690), f"geometric mean {E['historical_speedup']:.6f}×", 43, fill=ORANGE, anchor="mm")
        board.pill((550, 730, 1050, 790), "steady PASS · startup FAIL", PANEL, INK, 25)
    board.footer("ratio = support-off latency / support-on latency", "100 synchronized calls after 25 warmups", "4 paired seeds")
    return board.image


def render_kernel(width: int, height: int, square: bool) -> Image.Image:
    board = Board(width, height, "aud-kernel-engineer", "Replay Count Fell While Work Stayed Fixed", "CUDA and hardware engineer", square)
    if square:
        draw_trace(board, 80, 285, 920, 16, BLUE, "Support off · 16 graph launches")
        board.text((540, 440), "4 ATen ops per emitted region", 29, fill=BLUE, anchor="mm")
        draw_trace(board, 80, 590, 920, 2, ORANGE, "Support on · 2 graph launches")
        board.text((540, 745), "32 ATen ops per emitted region", 29, fill=ORANGE, anchor="mm")
        board.text((540, 850), "64 captured operations in both arms", 38, fill=INK, heading=True, anchor="mm")
    else:
        board.panel((60, 220, 780, 700), "Support off", BLUE)
        draw_trace(board, 110, 340, 620, 16, BLUE, "One profiled invocation")
        board.text((420, 535), "16 launches · 4 ops/region", 36, fill=BLUE, heading=True, anchor="mm")
        board.panel((820, 220, 1540, 700), "Support on", ORANGE)
        draw_trace(board, 870, 340, 620, 2, ORANGE, "One profiled invocation")
        board.text((1180, 535), "2 launches · 32 ops/region", 36, fill=ORANGE, heading=True, anchor="mm")
        board.text((800, 755), "Same 64 captured ATen operations", 42, fill=INK, heading=True, anchor="mm")
    board.footer("historical PyTorch profile · seed 0", "five untimed invocations", "launch counts are not latency")
    return board.image


def render_operations(width: int, height: int, square: bool) -> Image.Image:
    board = Board(width, height, "aud-platform-operations", "Monitor Cold and Steady Separately", "Platform operations", square)
    if square:
        x = 90
        steps = [
            ("25 warmups", BLUE, "prepare steady path"),
            ("100 synchronized calls", ORANGE, "measure steady path"),
            (f"{E['historical_speedup']:.6f}×", ORANGE, "steady PASS"),
        ]
        y = 245
        for index, (hero, color, note) in enumerate(steps):
            board.panel((x, y, 990, y + 150), None, color)
            board.text((x + 50, y + 52), hero, 38, fill=color, heading=index < 2)
            board.text((x + 600, y + 62), note, 26, fill=INK)
            if index < len(steps) - 1:
                board.arrow((540, y + 155), (540, y + 205), INK, 6)
            y += 205
        board.pill((190, 875, 890, 940), "Support on first ten +33.2% · startup FAIL", ORANGE_LIGHT, ORANGE, 21)
    else:
        x_positions = [100, 550, 1000]
        steps = [
            ("25", "warmups", BLUE),
            ("100", "synchronized calls", INK),
            (f"{E['historical_speedup']:.6f}×", "steady PASS", ORANGE),
        ]
        for index, (x, (hero, note, color)) in enumerate(zip(x_positions, steps)):
            board.rough_rect((x, 300, x + 350, 575), outline=color, fill=WHITE, width=5, radius=26)
            board.text((x + 175, 395), hero, 58, fill=color, heading=index < 2, anchor="mm")
            board.text((x + 175, 495), note, 28, fill=INK, anchor="mm")
            if index < 2:
                board.arrow((x + 360, 438), (x + 435, 438), INK, 7)
        board.pill((410, 650, 1190, 740), "Support on first ten +33.2% · startup route FAIL", ORANGE_LIGHT, ORANGE, 24)
    board.footer("one B200", "historical standalone protocol", "cold behavior did not qualify")
    return board.image


def render_business(width: int, height: int, square: bool) -> Image.Image:
    board = Board(width, height, "aud-business-capacity", "A Fast Steady Loop With a Startup Cost", "Business and capacity reader", square)
    measured = [
        f"Support on steady: {E['historical_speedup']:.6f}× faster",
        f"Support on first ten: {E['startup_penalty_pct']:.1f}% longer",
        "same synthetic depth-8 work",
    ]
    unmeasured = [
        "model throughput",
        "serving latency",
        "cost or capacity",
    ]
    if square:
        board.panel((60, 210, 1020, 560), "Measured", ORANGE)
        y = 310
        for item in measured:
            board.draw.ellipse((100, y, 124, y + 24), fill=ORANGE)
            board.text((150, y - 6), item, 32, fill=INK)
            y += 78
        board.panel((60, 610, 1020, 925), "Not measured", BLUE)
        y = 705
        for item in unmeasured:
            board.marker_line((100, y + 13), (124, y + 13), BLUE, 4)
            board.text((150, y - 6), item, 31, fill=INK)
            y += 70
    else:
        board.panel((70, 220, 770, 710), "Measured in this microbenchmark", ORANGE)
        y = 340
        for index, item in enumerate(measured):
            board.draw.ellipse((120, y, 150, y + 30), fill=ORANGE)
            board.text((185, y - 8), item, 34 if index < 2 else 29, fill=INK)
            y += 120
        board.panel((830, 220, 1530, 710), "Not established", BLUE)
        y = 340
        for item in unmeasured:
            board.marker_line((880, y + 15), (915, y + 15), BLUE, 5)
            board.text((950, y - 8), item, 34, fill=INK)
            y += 120
        board.text((800, 770), "Use the result for compiler diagnosis, not a capacity forecast.", 34, fill=INK, heading=True, anchor="mm")
    board.footer("PyTorch flag comparison", "B200", "no serving or cost claim")
    return board.image


RENDERERS: dict[str, Callable[[int, int, bool], Image.Image]] = {
    "a": render_a,
    "b": render_b,
    "c": render_c,
    "fun": render_fun,
    "aud-curious": render_curious,
    "aud-ml-engineer": render_ml,
    "aud-performance-engineer": render_performance,
    "aud-kernel-engineer": render_kernel,
    "aud-platform-operations": render_operations,
    "aud-business-capacity": render_business,
}


@dataclass(frozen=True)
class AssetSpec:
    key: str
    role: str
    audience: str
    title: str
    finding: str
    composition: str
    caveat: str
    sources: tuple[str, ...]
    visible_text: tuple[str, ...]


SPECS = [
    AssetSpec("a", "explainer a", "technical reader", "Same Work, Fewer Replay Regions",
              "The flag changes emitted regions from 16 x 4 operations to 2 x 32 operations while preserving 64 captured operations.",
              "A deterministic region-chain comparison with equal operation totals.",
              "Emitted wrapper regions are not complete Dynamo or CUDA Graph topologies.",
              ("results/nested_break_depth8-seed0-flag0/result.json", "results/nested_break_depth8-seed0-flag1/result.json"),
              ("Support off · 16 emitted regions", "16 × 4 ops = 64", "Support on · 2 emitted regions", "2 × 32 ops = 64")),
    AssetSpec("b", "explainer b", "technical reader", "Steady Wins, Startup Does Not",
              "The historical cohort measured a 4.531795x synchronized steady speedup and a 33.2 percent longer first-ten interval total.",
              "Two separately scaled bar panels for steady latency and first-ten startup intervals.",
              "The startup route failed. Milliseconds and microseconds are kept in separate panels.",
              ("results/analyzer-result.json", "README.md"),
              ("Support off · 407.567 µs", "Support on · 89.935 µs", "4.531795×", "Support off · 2411.177 ms", "Support on · 3211.677 ms", "Support on +33.2% · route FAIL")),
    AssetSpec("c", "explainer c", "technical reader", "How to Spot Replay Fragmentation",
              "The retained historical profile recorded 16 versus 2 cudaGraphLaunch events per profiled invocation.",
              "A trace-like count diagram that shows what to count inside one invocation.",
              "This is a diagram of retained counts, not a profiler screenshot or latency measurement.",
              ("results/nested_break_depth8-seed0-flag0/result.json", "results/nested_break_depth8-seed0-flag1/result.json"),
              ("One profiled invocation", "16 cudaGraphLaunch events", "2 cudaGraphLaunch events", "not a profiler screenshot")),
    AssetSpec("fun", "cover", "general technical reader", "4.531795x Faster at Steady State",
              "The historical depth-8 B200 cohort measured 407.567 microseconds with support off and 89.935 microseconds with support on, a 4.531795x steady speedup over the same synthetic work.",
              "A deterministic landscape and square adaptation of the verified existing Gamma first card, retaining its blue/orange bar-chart concept and sampled palette while replacing rounded values and False/True labels with exact evidence and natural support labels.",
              "Steady execution passed, but the first-ten startup route failed. This is a compiler microbenchmark, not a serving claim.",
              ("images/sources/gamma-original-export/1_453-faster-at-steady-state.png", "README.md", "results/analyzer-result.json"),
              ("Support off · 407.567 µs", "Support on · 89.935 µs", "4.531795x faster at steady state", "same 64 captured operations", "startup route FAIL")),
    AssetSpec("aud-curious", "audience image", "Curious non-specialists", "Same Work, Fewer Trips",
              "After warmup, grouping the same synthetic work into fewer replays reduced synchronized latency from 407.567 to 89.935 microseconds.",
              "A before-and-after trip metaphor backed by exact measured latency labels.",
              "The trip metaphor maps to replay launches and does not establish serving performance.",
              ("README.md", "results/analyzer-result.json"),
              ("16 small trips", "2 larger trips", "407.567 µs", "89.935 µs", "4.531795× faster after warmup")),
    AssetSpec("aud-ml-engineer", "audience image", "ML/AI engineers", "Nested Support Changes the Compiled Shape",
              "The same workload under fullgraph false emitted 16 versus 2 regions and replay launches when nested graph-break support changed.",
              "Two compiler-shape cards connected from emitted regions to replay launches.",
              "This does not compare models, serving systems, or PyTorch versions.",
              ("protocol.json", "benchmark.py", "results/nested_break_depth8-seed0-flag0/result.json", "results/nested_break_depth8-seed0-flag1/result.json"),
              ("Support off", "16 regions", "16 replay launches", "Support on", "2 regions", "2 replay launches")),
    AssetSpec("aud-performance-engineer", "audience image", "Performance and capacity engineers", "Four Seeds Agree on the Steady Win",
              "All four paired seeds measured a steady speedup between 4.495352x and 4.579454x with a 4.531795x geometric mean.",
              "A four-point seed plot with the descriptive geometric mean and route verdicts.",
              "No confidence interval or population-coverage claim is made.",
              ("results/analyzer-result.json", "README.md"),
              tuple([f"seed {index} · {value:.6f}×" for index, value in enumerate(E["seed_ratios"])] + ["geometric mean 4.531795×", "steady PASS · startup FAIL"])),
    AssetSpec("aud-kernel-engineer", "audience image", "CUDA and hardware engineers", "Replay Count Fell While Work Stayed Fixed",
              "The historical seed-0 profile recorded 16 versus 2 graph launches per invocation while compiler counters retained 64 captured ATen operations.",
              "Launch strips paired with operations per emitted region.",
              "Launch counts are not latency and the diagram is not a profiler screenshot.",
              ("results/nested_break_depth8-seed0-flag0/result.json", "results/nested_break_depth8-seed0-flag1/result.json"),
              ("Support off · 16 graph launches", "4 ATen ops per emitted region", "Support on · 2 graph launches", "32 ATen ops per emitted region", "64 captured operations in both arms")),
    AssetSpec("aud-platform-operations", "audience image", "Performance and capacity engineers, including platform operations", "Monitor Cold and Steady Separately",
              "The steady route passed after 25 warmups and 100 synchronized calls while the first-ten startup route failed.",
              "A three-stage operational measurement flow plus a separate startup warning.",
              "The result is a standalone compiler microbenchmark, not a serving SLO.",
              ("README.md", "protocol.json", "results/analyzer-result.json"),
              ("25 warmups", "100 synchronized calls", "Support on · 4.531795× steady PASS", "Support on first ten +33.2% · startup FAIL")),
    AssetSpec("aud-business-capacity", "audience image", "VCs and business readers", "A Fast Steady Loop With a Startup Cost",
              "The microbenchmark measured a 4.531795x steady speedup and a 33.2 percent first-ten penalty but did not measure serving, cost, or capacity.",
              "A measured-versus-not-established decision card.",
              "No cost, capacity, revenue, model-throughput, or serving-latency claim.",
              ("README.md", "results/analyzer-result.json"),
              ("Support on steady · 4.531795× faster", "Support on first ten · 33.2% longer", "model throughput · not measured", "serving latency · not measured", "cost or capacity · not measured")),
]


def render_preview(key: str, square: bool) -> Image.Image:
    width, height = (400, 400) if square else (400, 225)
    source = RENDERERS[key](1080, 1080, True) if square else RENDERERS[key](1600, 900, False)
    image = source.resize((width, height), Image.Resampling.LANCZOS)
    return image


def write_sidecars() -> None:
    for spec in SPECS:
        gamma = {
            "state": "not_a_gamma_origin_asset",
            "note": "This deterministic explainer or audience asset was rendered locally from retained evidence.",
        }
        if spec.key == "fun":
            gamma = {
                "state": "derived_from_verified_existing_gamma_export",
                "canonical_url": GAMMA_URL,
                "source_export": "images/sources/gamma-453-faster-at-steady-state-original.zip",
                "source_export_sha256": GAMMA_EXPORT_SHA256,
                "source_card": "images/sources/gamma-original-export/1_453-faster-at-steady-state.png",
                "source_card_sha256": GAMMA_CARD_1_SHA256,
                "derivation": "Landscape and square scientific chart recreation from Gamma card 1. Palette and chart concept are source-derived; exact values and natural support labels are composited from retained JSON evidence.",
            }
        lines = [
            spec.title,
            f"Audience: {spec.audience}",
            f"Evidence state: measured historical cohort",
            f"Core finding: {spec.finding}",
            "Visible copy:",
            *[f"- {line}" for line in spec.visible_text],
            f"Caveat: {spec.caveat}",
            "Cohort boundary: Fresh run 022 measured 4.541758x independently. It is not the source of this visual's 4.531795x claim.",
        ]
        (HERE / f"{spec.key}.txt").write_text("\n".join(lines) + "\n")
        prompt = {
            "schema_version": 1,
            "asset": spec.role,
            "audience": spec.audience,
            "core_finding": spec.finding,
            "exact_heading": spec.title,
            "exact_labels_and_values": list(spec.visible_text),
            "evidence_state": "measured",
            "composition": spec.composition,
            "required_caveat_or_uncertainty": spec.caveat,
            "source_references": list(spec.sources),
            "provenance_flag_identifier": "torch._dynamo.config.nested_graph_breaks",
            "public_setting_labels": ["Support off", "Support on"],
            "output": ["1600x900", "1080x1080", "400x225", "400x400"],
            "style": {
                "authority": "/Users/admin/dev/posts/style/whiteboard.md",
                "background": BG,
                "ink": INK,
                "blue": BLUE,
                "orange": ORANGE,
                "heading_font": "Marker Felt local substitute for Caveat Brush",
                "label_font": "Avenir Next local substitute for Inter",
            },
            "renderer": "images/render_whiteboard_visuals.py",
            "gamma": gamma,
        }
        (HERE / f"{spec.key}.json").write_text(json.dumps(prompt, indent=2) + "\n")


def make_contact_sheets() -> None:
    keys = [spec.key for spec in SPECS]
    sheet = Image.new("RGB", (1600, 900), BG)
    draw = ImageDraw.Draw(sheet)
    draw.text((45, 24), "Nested graph breaks · visual index", fill=INK, font=font(45, True))
    thumb_w, thumb_h = 282, 159
    for index, key in enumerate(keys):
        row, col = divmod(index, 5)
        x = 40 + col * 312
        y = 100 + row * 355
        image = Image.open(HERE / f"{key}.png").convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(image, (x, y))
        draw.rounded_rectangle((x, y, x + thumb_w, y + thumb_h), radius=8, outline=GRID, width=2)
        key_face = fit_font(draw, key, thumb_w, 24, 14)
        draw.text((x, y + thumb_h + 12), key, fill=INK, font=key_face)
        spec = next(item for item in SPECS if item.key == key)
        lines = textwrap.wrap(spec.title, width=28)[:2]
        for line_index, line in enumerate(lines):
            draw.text((x, y + thumb_h + 50 + line_index * 24), line, fill=MUTED, font=font(16))
    sheet.save(HERE / "contact-sheet.png", optimize=True)
    sheet.resize((400, 225), Image.Resampling.LANCZOS).save(HERE / "contact-sheet-400.png", optimize=True)

    square_sheet = Image.new("RGB", (1080, 1080), BG)
    draw = ImageDraw.Draw(square_sheet)
    draw.text((34, 22), "Visual index", fill=INK, font=font(48, True))
    thumb = 150
    for index, key in enumerate(keys):
        row, col = divmod(index, 2)
        x = 44 + col * 510
        y = 92 + row * 192
        image = Image.open(HERE / f"{key}-square.png").convert("RGB").resize((thumb, thumb), Image.Resampling.LANCZOS)
        square_sheet.paste(image, (x, y))
        draw.rounded_rectangle((x, y, x + thumb, y + thumb), radius=8, outline=GRID, width=2)
        spec = next(item for item in SPECS if item.key == key)
        draw.text((x + thumb + 18, y + 14), key, fill=INK, font=font(18))
        lines = textwrap.wrap(spec.title, width=27)[:3]
        for line_index, line in enumerate(lines):
            draw.text((x + thumb + 18, y + 48 + line_index * 24), line, fill=MUTED, font=font(16))
    square_sheet.save(HERE / "contact-sheet-square.png", optimize=True)
    square_sheet.resize((400, 400), Image.Resampling.LANCZOS).save(HERE / "contact-sheet-square-400.png", optimize=True)


def write_gallery() -> None:
    cards = []
    for spec in SPECS:
        cards.append(
            f"<article><h2>{html.escape(spec.key)} · {html.escape(spec.title)}</h2>"
            f"<img src='{html.escape(spec.key)}.png' alt='{html.escape(spec.title)}'>"
            f"<p>{html.escape(spec.finding)}</p>"
            f"<p><a href='{html.escape(spec.key)}-square.png'>square</a> · "
            f"<a href='{html.escape(spec.key)}-400.png'>400 px</a> · "
            f"<a href='{html.escape(spec.key)}.txt'>text sidecar</a> · "
            f"<a href='{html.escape(spec.key)}.json'>prompt and provenance</a></p></article>"
        )
    body = "\n".join(cards)
    css = (
        "body{font-family:system-ui;background:%s;color:%s;max-width:1200px;margin:auto;padding:32px}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:24px}"
        "article{background:white;border:2px solid %s;border-radius:18px;padding:18px}"
        "img{width:100%%;height:auto;border:1px solid %s}a{color:%s}p{line-height:1.45}"
    ) % (BG, INK, GRID, GRID, BLUE)
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nested graph breaks visual index</title>
<style>{css}</style></head>
<body><h1>Nested graph breaks visual index</h1><p>All claim-bearing labels come from retained public evidence. Fresh run 022 stays separate.</p><main>{body}</main></body></html>
"""
    (HERE / "gallery.html").write_text(document)


def write_provenance_and_receipt() -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    gamma_meta = gamma_source_metadata()
    cover_outputs = []
    for name in ("fun.png", "fun-square.png", "fun-400.png", "fun-square-400.png"):
        path = HERE / name
        with Image.open(path) as image:
            dimensions = list(image.size)
        cover_outputs.append({
            "path": str(path.relative_to(PACKAGE)),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "dimensions": dimensions,
        })

    gamma_derivation = {
        "schema_version": 1,
        "created_at": created_at,
        "state": "complete",
        "canonical_gamma_url": GAMMA_URL,
        "source": {
            "export_zip": "images/sources/gamma-453-faster-at-steady-state-original.zip",
            "export_zip_sha256": GAMMA_EXPORT_SHA256,
            "export_kind": "existing three-card Gamma PNG export",
            "export_was_newly_generated_for_this_recovery": False,
            "source_card": "images/sources/gamma-original-export/1_453-faster-at-steady-state.png",
            "source_card_sha256": GAMMA_CARD_1_SHA256,
            "source_card_dimensions": gamma_meta["dimensions"],
            "sampled_palette": gamma_meta["sampled_palette"],
        },
        "derivation": {
            "method": "deterministic scientific chart recreation from the verified Gamma first card",
            "preserved": [
                "single steady-state latency comparison",
                "horizontal-bar composition",
                "warm-white, blue, orange, and ink palette sampled from source pixels",
                "historical B200 compiler-microbenchmark cohort",
            ],
            "replaced": [
                "ambiguous False/True setting labels",
                "rounded 4.53x, 407.57 microseconds, and 89.94 microseconds display copy",
                "portrait layout that does not meet package export dimensions",
            ],
            "public_labels": ["Support off", "Support on"],
            "exact_values_from_retained_json": {
                "support_off_steady_us": round(E["baseline_steady_us"], 3),
                "support_on_steady_us": round(E["candidate_steady_us"], 3),
                "steady_speedup": round(E["historical_speedup"], 6),
            },
            "renderer": "images/render_whiteboard_visuals.py",
            "direct_bitmap_crop": False,
            "claim_bearing_text_from_image_generation": False,
        },
        "outputs": cover_outputs,
    }
    (HERE / "gamma-cover-derivation.json").write_text(json.dumps(gamma_derivation, indent=2) + "\n")

    gamma_status = {
        "schema_version": 1,
        "state": "integrated_existing_gamma_export",
        "retained_existing_job_found": True,
        "canonical_gamma_url": GAMMA_URL,
        "export_path": "images/sources/gamma-453-faster-at-steady-state-original.zip",
        "export_sha256": GAMMA_EXPORT_SHA256,
        "source_card": "images/sources/gamma-original-export/1_453-faster-at-steady-state.png",
        "source_card_sha256": GAMMA_CARD_1_SHA256,
        "current_cover": "images/fun.png",
        "current_cover_origin": "deterministic public-label adaptation of verified Gamma card 1",
        "current_cover_is_direct_gamma_export": False,
        "current_cover_is_gamma_origin": True,
        "derivation_receipt": "images/gamma-cover-derivation.json",
        "next_step": "Independent visual review of final landscape, square, and 400-pixel outputs.",
    }
    (HERE / "gamma-cover-status.json").write_text(json.dumps(gamma_status, indent=2) + "\n")

    output_paths = []
    for spec in SPECS:
        for suffix in (".png", "-square.png", "-400.png", "-square-400.png", ".txt", ".json"):
            output_paths.append(HERE / f"{spec.key}{suffix}")
    for name in ("contact-sheet.png", "contact-sheet-square.png", "contact-sheet-400.png", "contact-sheet-square-400.png", "gallery.html", "gamma-cover-status.json", "gamma-cover-derivation.json"):
        output_paths.append(HERE / name)

    files = []
    for path in sorted(output_paths):
        entry = {
            "path": str(path.relative_to(PACKAGE)),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        if path.suffix == ".png":
            with Image.open(path) as image:
                entry["dimensions"] = list(image.size)
        files.append(entry)

    source_files = []
    for path in (README_PATH, PROTOCOL_PATH, ANALYZER_PATH, BASELINE_PATH, CANDIDATE_PATH, FRESH_PATH,
                 GAMMA_EXPORT_ZIP, GAMMA_CARD_1, GAMMA_CARD_2, GAMMA_CARD_3):
        source_files.append({"path": str(path.relative_to(PACKAGE)), "sha256": sha256(path), "bytes": path.stat().st_size})

    provenance = {
        "schema_version": 1,
        "created_at": created_at,
        "renderer": {
            "path": "images/render_whiteboard_visuals.py",
            "sha256": sha256(Path(__file__)),
            "runtime": "Codex bundled Python with Pillow",
            "workload_executed": False,
        },
        "style_authorities": [
            {"path": "/Users/admin/dev/posts/style/whiteboard.md", "sha256": sha256(Path("/Users/admin/dev/posts/style/whiteboard.md"))},
            {"path": "/Users/admin/dev/posts/style/diagram.md", "sha256": sha256(Path("/Users/admin/dev/posts/style/diagram.md"))},
            {"path": "/Users/admin/dev/posts/templates/gamma-whiteboard-prompt.md", "sha256": sha256(Path("/Users/admin/dev/posts/templates/gamma-whiteboard-prompt.md"))},
        ],
        "source_files": source_files,
        "primary_visual_cohort": {
            "name": "historical retained standalone cohort",
            "hardware": "NVIDIA B200",
            "runtime": "PyTorch 2.14.1+cu130",
            "case": "nested_break_depth8",
            "baseline_steady_us": round(E["baseline_steady_us"], 3),
            "candidate_steady_us": round(E["candidate_steady_us"], 3),
            "steady_geometric_mean_speedup": round(E["historical_speedup"], 6),
            "baseline_first10_ms": round(E["baseline_first10_ms"], 3),
            "candidate_first10_ms": round(E["candidate_first10_ms"], 3),
            "candidate_first10_penalty_pct": round(E["startup_penalty_pct"], 1),
            "steady_route": "PASS",
            "startup_route": "FAIL",
            "baseline_replays_per_profiled_invocation": E["baseline_launches"],
            "candidate_replays_per_profiled_invocation": E["candidate_launches"],
            "captured_operations_each_arm": E["operations"],
        },
        "excluded_separate_cohort": {
            "name": "fresh run 022",
            "steady_geometric_mean_speedup": round(E["fresh022_speedup"], 6),
            "reason": "Independent capture cohort. It is not the source of the original 4.531795x claim and is not mixed into these visuals.",
            "native_profiler_images_used_as_audience_assets": False,
        },
        "gamma_cover": {
            "state": "integrated_existing_gamma_export",
            "canonical_url": GAMMA_URL,
            "retained_export": "images/sources/gamma-453-faster-at-steady-state-original.zip",
            "retained_export_sha256": GAMMA_EXPORT_SHA256,
            "source_card": "images/sources/gamma-original-export/1_453-faster-at-steady-state.png",
            "source_card_sha256": GAMMA_CARD_1_SHA256,
            "public_cover": "images/fun.png",
            "public_cover_is_direct_export": False,
            "public_cover_is_gamma_origin": True,
            "derivation_receipt": "images/gamma-cover-derivation.json",
            "boundary": "Only the fun cover family is Gamma-origin. Explainers and audience assets are deterministic local scientific renders.",
        },
        "unused_imagegen_candidate": {
            "path": "images/sources/unused-imagegen-cover-candidate.png",
            "sha256": sha256(IMAGEGEN_CANDIDATE),
            "generator": "built-in imagegen",
            "used_in_final_assets": False,
            "rejection_reason": "The bitmap rendered 15 small tiles rather than the requested exact 16. Claim-bearing counts require deterministic output.",
        },
        "files": files,
    }
    (HERE / "visual-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    receipt = {
        "schema_version": 1,
        "state": "rendered_pending_independent_review",
        "created_at": provenance["created_at"],
        "families": len(SPECS),
        "landscape_dimensions": [1600, 900],
        "square_dimensions": [1080, 1080],
        "preview_dimensions": [400, 225],
        "square_preview_dimensions": [400, 400],
        "families_are_distinct": True,
        "gamma_stage": "integrated_existing_export_as_public_label_adaptation",
        "files": files,
    }
    (HERE / "render-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    image_hashes = [f"{sha256(path)}  {path.name}" for path in sorted(HERE.glob("*.png"))]
    (HERE / "final-image-hashes.txt").write_text("\n".join(image_hashes) + "\n")

    review_inventory = {
        "review_state": "pending_independent_review",
        "review_requirement": "Inspect every landscape, square, and native 400 export. Confirm exact copy, readability, no clipping, no overlap, and distinct audience composition.",
        "reviewer_must_be_independent": True,
        "families": len(SPECS),
        "images": [item for item in files if item["path"].endswith(".png") and "contact-sheet" not in item["path"]],
        "cohort_boundary": "All claim-bearing images use the historical 4.531795x cohort. Fresh run 022 remains separate.",
        "gamma_boundary": "fun.png is a deterministic public-label adaptation of verified Gamma card 1. The other nine families are not Gamma-origin.",
    }
    (HERE / "reviewable-inventory.json").write_text(json.dumps(review_inventory, indent=2) + "\n")


def render_all() -> None:
    write_sidecars()
    for spec in SPECS:
        landscape = RENDERERS[spec.key](1600, 900, False)
        landscape.save(HERE / f"{spec.key}.png", optimize=True)
        square = RENDERERS[spec.key](1080, 1080, True)
        square.save(HERE / f"{spec.key}-square.png", optimize=True)
        render_preview(spec.key, False).save(HERE / f"{spec.key}-400.png", optimize=True)
        render_preview(spec.key, True).save(HERE / f"{spec.key}-square-400.png", optimize=True)
    make_contact_sheets()
    write_gallery()
    write_provenance_and_receipt()


if __name__ == "__main__":
    render_all()
    print(f"rendered {len(SPECS)} Whiteboard families into {HERE}")

