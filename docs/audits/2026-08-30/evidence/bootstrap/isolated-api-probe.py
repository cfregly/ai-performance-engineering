"""Verify the pinned release APIs in an isolated env, without claiming CUDA coverage."""

import json
import platform
import sys
from pathlib import Path

import torch
import torchao
from torchao.float8 import Float8LinearConfig, convert_to_float8_training
from torchao.quantization import quantize_

assert torch.__version__ == "2.9.1"
assert torchao.__version__ == "0.15.0"
assert torchao.skip_loading_so_files is False
model = convert_to_float8_training(
    torch.nn.Sequential(torch.nn.Linear(128, 128, bias=False)),
    config=Float8LinearConfig(),
)
assert type(model[0]).__name__ == "Float8Linear"
assert callable(quantize_)
receipt = {
    "platform": platform.platform(),
    "python": sys.version,
    "torch": torch.__version__,
    "torchao": torchao.__version__,
    "cuda_available": torch.cuda.is_available(),
    "torchao_skip_loading_so_files": torchao.skip_loading_so_files,
    "torchao_native_libraries": sorted(
        str(path) for path in Path(torchao.__file__).parent.glob("*.so")
    ),
    "converted_module": type(model[0]).__name__,
    "imported_api": ["Float8LinearConfig", "convert_to_float8_training", "quantize_"],
    "boundary": (
        "Isolated macOS Python API import and conversion only; "
        "CUDA native ABI, FP8 execution, and Linux bootstrap are untested."
    ),
}
Path(sys.argv[1]).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt, indent=2))
