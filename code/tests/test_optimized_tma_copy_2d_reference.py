"""CPU controls for optimized_tma_copy's tiled 2D neighbor oracle."""

from __future__ import annotations

from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "ch07" / "optimized_tma_copy.cu"
TILE_ROWS = 64
TILE_COLUMNS = 64
LOOKAHEAD = 64


def _combine(center: float, near: float, far: float) -> float:
    return center * 0.75 + near * 0.25 + far * 0.125


def _source_coordinates(
    row: int,
    column: int,
    *,
    rows: int,
    columns: int,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    row_origin = (row // TILE_ROWS) * TILE_ROWS
    column_origin = (column // TILE_COLUMNS) * TILE_COLUMNS
    tile_rows = min(TILE_ROWS, rows - row_origin)
    tile_columns = min(TILE_COLUMNS, columns - column_origin)
    tile_elements = tile_rows * tile_columns
    local = (row - row_origin) * tile_columns + column - column_origin

    def coordinate(tile_index: int) -> tuple[int, int]:
        return (
            row_origin + tile_index // tile_columns,
            column_origin + tile_index % tile_columns,
        )

    return (
        (row, column),
        coordinate(min(local + 1, tile_elements - 1)),
        coordinate(min(local + LOOKAHEAD, tile_elements - 1)),
    )


def _reference_by_coordinate(
    values: list[float], *, rows: int, columns: int
) -> list[float]:
    output = []
    for row in range(rows):
        for column in range(columns):
            coordinates = _source_coordinates(
                row,
                column,
                rows=rows,
                columns=columns,
            )
            source = [values[source_row * columns + source_column] for source_row, source_column in coordinates]
            output.append(_combine(*source))
    return output


def _reference_by_tile_loop(
    values: list[float], *, rows: int, columns: int
) -> list[float]:
    output = [0.0] * (rows * columns)
    for row_origin in range(0, rows, TILE_ROWS):
        for column_origin in range(0, columns, TILE_COLUMNS):
            tile_rows = min(TILE_ROWS, rows - row_origin)
            tile_columns = min(TILE_COLUMNS, columns - column_origin)
            tile = [
                values[(row_origin + row) * columns + column_origin + column]
                for row in range(tile_rows)
                for column in range(tile_columns)
            ]
            for local, center in enumerate(tile):
                near = tile[min(local + 1, len(tile) - 1)]
                far = tile[min(local + LOOKAHEAD, len(tile) - 1)]
                row = row_origin + local // tile_columns
                column = column_origin + local % tile_columns
                output[row * columns + column] = _combine(center, near, far)
    return output


def test_non_square_reference_covers_full_and_partial_tile_edges() -> None:
    rows = 65
    columns = 70
    values = [
        float(row * 1009 + column * 3) + 0.25
        for row in range(rows)
        for column in range(columns)
    ]

    coordinate_reference = _reference_by_coordinate(
        values,
        rows=rows,
        columns=columns,
    )
    tile_reference = _reference_by_tile_loop(
        values,
        rows=rows,
        columns=columns,
    )

    assert coordinate_reference == tile_reference
    assert _source_coordinates(0, 0, rows=rows, columns=columns) == (
        (0, 0),
        (0, 1),
        (1, 0),
    )
    assert _source_coordinates(63, 63, rows=rows, columns=columns) == (
        (63, 63),
        (63, 63),
        (63, 63),
    )
    assert _source_coordinates(0, 64, rows=rows, columns=columns) == (
        (0, 64),
        (0, 65),
        (10, 68),
    )
    assert _source_coordinates(64, 0, rows=rows, columns=columns) == (
        (64, 0),
        (64, 1),
        (64, 63),
    )
    assert _source_coordinates(64, 64, rows=rows, columns=columns) == (
        (64, 64),
        (64, 65),
        (64, 69),
    )
    assert _source_coordinates(64, 69, rows=rows, columns=columns) == (
        (64, 69),
        (64, 69),
        (64, 69),
    )


def test_default_matrix_index_zero_matches_observed_neighbor_transform() -> None:
    columns = 4096

    def value(row: int, column: int) -> float:
        return float((row * columns + column) % 1000) / 1000.0

    center, near, far = _source_coordinates(
        0,
        0,
        rows=4096,
        columns=columns,
    )
    expected = _combine(value(*center), value(*near), value(*far))

    assert (center, near, far) == ((0, 0), (0, 1), (1, 0))
    assert expected == pytest.approx(0.01225)
    assert expected != value(0, 0)


def test_benchmark_validation_uses_tiled_neighbor_oracle_and_description() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    validation = source.split(
        "std::vector<float> h_matrix_out",
        maxsplit=1,
    )[1].split("float tma2d_ms", maxsplit=1)[0]

    assert "tiled_neighbor_reference_at(h_matrix, M, N, idx)" in validation
    assert "h_matrix_out[idx] != h_matrix[idx]" not in validation
    assert "Descriptor-backed 2D TMA tiled-neighbor transform" in source
