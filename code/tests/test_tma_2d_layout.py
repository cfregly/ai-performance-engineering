"""Host descriptor-contract checks, not CUDA execution or performance evidence."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def layout_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for the real host metadata helper")
    directory = tmp_path_factory.mktemp("tma-layout-host")
    source = directory / "layout.cpp"
    binary = directory / "layout"
    source.write_text(
        '#include "core/common/headers/tma_2d_layout.hpp"\n'
        "#include <cstdlib>\n#include <iostream>\n"
        "int main(int argc, char** argv) {\n"
        "  if (argc != 6) return 2;\n"
        "  const auto layout = cuda_tma::make_2d_tensor_map_layout(\n"
        "    std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3]),\n"
        "    std::atoi(argv[4]), std::atoi(argv[5]));\n"
        "  for (auto v : layout.dimensions) std::cout << v << ' ';\n"
        "  for (auto v : layout.strides_bytes) std::cout << v << ' ';\n"
        "  for (auto v : layout.box) std::cout << v << ' ';\n"
        "  for (auto v : layout.element_strides) std::cout << v << ' ';\n"
        "}\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(CODE_ROOT),
            str(source), "-o", str(binary),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


@pytest.mark.parametrize(
    ("width", "height", "ld", "box_width", "box_height"),
    [(196, 259, 208, 64, 128), (385, 129, 400, 128, 32), (64, 257, 64, 64, 128)],
)
def test_driver_metadata_preserves_row_major_axes_and_padded_pitch(
    layout_probe: Path, width: int, height: int, ld: int, box_width: int, box_height: int
) -> None:
    result = subprocess.run(
        [str(layout_probe), *map(str, (width, height, ld, box_width, box_height))],
        check=True,
        capture_output=True,
        text=True,
    )
    dim_x, dim_y, row_stride, box_x, box_y, step_x, step_y = map(int, result.stdout.split())
    assert (dim_x, dim_y) == (width, height)
    assert (box_x, box_y) == (box_width, box_height)
    assert row_stride == ld * 4
    assert (step_x, step_y) == (1, 1)

    # NVIDIA's dimension 0 is contiguous; dimension 1 advances by row_stride.
    # Check every logical coordinate and tail box against independent row-major
    # indexing, including padding which must stay outside the logical dimensions.
    addresses = set()
    for row_origin in range(0, height, box_height):
        for col_origin in range(0, width, box_width):
            for local_row in range(box_y):
                for local_col in range(box_x):
                    x, y = col_origin + local_col, row_origin + local_row
                    driver_in_bounds = x < dim_x and y < dim_y
                    assert driver_in_bounds == (x < width and y < height)
                    if driver_in_bounds:
                        byte_offset = x * 4 + y * row_stride
                        assert byte_offset == (y * ld + x) * 4
                        addresses.add(byte_offset)
    assert len(addresses) == width * height


def test_partial_tile_store_preserves_padding_and_guards(tmp_path: Path) -> None:
    """Execute the production store helper with cooperative logical threads."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for the real store helper")
    source = tmp_path / "tail.cpp"
    binary = tmp_path / "tail"
    source.write_text(
        '#include "core/common/headers/tma_2d_layout.hpp"\n'
        "#include <vector>\n"
        "int main() {\n"
        "  constexpr int height=129, width=385, ld=400, guard=17;\n"
        "  for (int threads : {1, 32, 128, 256}) {\n"
        "    std::vector<float> storage(guard + height*ld + guard, -12345.0f);\n"
        "    float* output = storage.data() + guard;\n"
        "    for (int y=0; y<height; y+=128) for (int x=0; x<width; x+=64) {\n"
        "      float tile[128][64];\n"
        "      int rows = height-y<128 ? height-y : 128;\n"
        "      int cols = width-x<64 ? width-x : 64;\n"
        "      for (int r=0; r<128; ++r) for (int c=0; c<64; ++c)\n"
        "        tile[r][c] = float((y+r)*width+x+c);\n"
        "      for (int tid=0; tid<threads; ++tid)\n"
        "        cuda_tma::store_partial_2d_tile(tile, output, ld, y, x, rows, cols, tid, threads);\n"
        "    }\n"
        "    for (int y=0; y<height; ++y) for (int x=0; x<ld; ++x) {\n"
        "      float expected = x<width ? float(y*width+x) : -12345.0f;\n"
        "      if (output[y*ld+x] != expected) return 1;\n"
        "    }\n"
        "    for (int i=0; i<guard; ++i)\n"
        "      if (storage[i] != -12345.0f || storage[guard+height*ld+i] != -12345.0f) return 2;\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(CODE_ROOT),
         str(source), "-o", str(binary)],
        check=True, capture_output=True, text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)


def test_cuda_encoder_uses_the_production_host_metadata() -> None:
    source = (CODE_ROOT / "core/common/headers/tma_helpers.cuh").read_text(encoding="utf-8")
    encoder = source.split("inline bool make_2d_tensor_map(", 1)[1].split(
        "inline bool make_1d_tensor_map(", 1
    )[0]
    assert "make_2d_tensor_map_layout(width, height, ld, box_width, box_height)" in encoder
    for field in ("dimensions", "strides_bytes", "box", "element_strides"):
        assert f"layout.{field}" in encoder


def test_async_prefetch_2d_bounds_partial_output_tiles() -> None:
    source = (CODE_ROOT / "ch07/async_prefetch_2d_demo.cu").read_text(encoding="utf-8")
    kernel = source.split("__global__ void tma_copy_2d_kernel", 1)[1].split("#endif", 1)[0]

    assert "rows == TILE_M && cols == TILE_N" in kernel
    assert "cp_async_bulk_tensor_2d_shared_to_global" in kernel
    assert "index < rows * cols" in kernel
    assert "static_cast<std::size_t>(tile_m + local_row) * output_ld" in kernel
    assert "output[output_index] = tile[local_row][local_col]" in kernel


def test_cuda_gate_records_unavailable_compiler_without_claiming_a_pass(tmp_path: Path) -> None:
    output = tmp_path / "receipt"
    env = os.environ.copy()
    env["PATH"] = str(tmp_path)  # Real availability failure; no CUDA success is mocked.
    result = subprocess.run(
        [sys.executable, str(CODE_ROOT / "tests/cuda/run_tma_2d_layout_validation.py"),
         "--arch", "sm_100a", "--output-dir", str(output)],
        capture_output=True,
        check=False,
        text=True,
        env=env,
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert result.returncode == 3
    assert report["status"] == "SKIPPED"
    assert report["checks"] == []
    assert "no CUDA compile or execution occurred" in report["reason"]
    assert "core/common/headers/tma_2d_layout.hpp" in report["source_sha256"]


@pytest.mark.parametrize(
    ("chapter", "target"),
    [("ch07", "async_prefetch_2d_demo"), ("ch07", "optimized_tma_bulk_tensor_2d"),
     ("ch07", "optimized_tma_copy"), ("ch10", "tma_multicast_cluster"),
     ("ch10", "tma_2d_pipeline_blackwell")],
)
def test_tma_build_graph_tracks_the_shared_layout_header(chapter: str, target: str) -> None:
    for suffix in ("_sm100", "_verify_sm100"):
        name = target + suffix
        result = subprocess.run(
            ["make", "-np", "ARCH=sm_100", name],
            cwd=CODE_ROOT / chapter,
            capture_output=True,
            check=True,
            text=True,
        )
        rule = re.search(rf"^{name}: (.+)$", result.stdout, re.MULTILINE)
        assert rule is not None
        prerequisites = rule.group(1).split()
        assert prerequisites[0] == target + ".cu"
        assert "../core/common/headers/tma_2d_layout.hpp" in prerequisites
