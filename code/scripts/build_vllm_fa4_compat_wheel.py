#!/usr/bin/env python3
"""Build a guarded vLLM 0.16 wheel with the upstream FA4 import fix."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile


@dataclass(frozen=True)
class WheelContract:
    wheel_filename: str
    wheel_sha256: str
    distribution: str
    version: str
    common_path: str
    common_before_sha256: str
    common_after_sha256: str


CONTRACT = WheelContract(
    wheel_filename="vllm-0.16.0+cu130-cp38-abi3-manylinux_2_35_x86_64.whl",
    wheel_sha256="bda6ff19ead743fb30c6271cdeb7daf62d5bd5f7a53cb6c2e7d987d53ea3d49f",
    distribution="vllm",
    version="0.16.0+cu130",
    common_path="vllm/model_executor/layers/rotary_embedding/common.py",
    common_before_sha256="ec02c70d14fcc2ee56a590102ba97bd0b746854e17d51c2ff103ab130e7bae63",
    common_after_sha256="03642283c38fe28498064ca1e8a4f9af475a8142cdfe7f95992497f45a376da6",
)
PIN_FILE = Path(__file__).resolve().parents[1] / "vllm_no_deps.pin"
REQUIREMENTS_FILE = Path(__file__).resolve().parents[1] / "requirements_latest.txt"
RUNTIME_VERSIONS = {
    "torch": "2.9.1+cu130",
    "vllm": CONTRACT.version,
    "flash-attn-4": "4.0.0b19",
    "flashinfer-python": "0.6.3",
}

UPSTREAM = {
    "issue": "https://github.com/vllm-project/vllm/issues/42675",
    "pull_request": "https://github.com/vllm-project/vllm/pull/42679",
    "merge_commit": "1f60771c744811e027f1309b9093cded7521d953",
}
RETIREMENT = (
    "Remove this backport after vllm_no_deps.pin selects a published wheel that "
    "contains upstream merge 1f60771c744811e027f1309b9093cded7521d953 and "
    "that wheel passes the FA4 import and full model verification gates."
)

OLD_IMPORT = b"import math\nfrom importlib.util import find_spec\n"
NEW_IMPORT = b"import math\nfrom contextlib import suppress\nfrom importlib import import_module\n"
# vLLM 0.16 predates the later current_platform.is_cpu() guard. This applies
# the merged direct-import fallback to that release without changing its other
# platform behavior.
OLD_INIT = b"""        self.apply_rotary_emb_flash_attn = None
        if find_spec("flash_attn") is not None:
            from flash_attn.ops.triton.rotary import apply_rotary

            self.apply_rotary_emb_flash_attn = apply_rotary
"""
NEW_INIT = b"""        self.apply_rotary_emb_flash_attn = None
        with suppress(ModuleNotFoundError):
            self.apply_rotary_emb_flash_attn = import_module(
                "flash_attn.ops.triton.rotary"
            ).apply_rotary
"""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_pin(pin_file: Path, contract: WheelContract = CONTRACT) -> dict[str, Any]:
    active = [
        line.strip()
        for line in pin_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    expected = f"vllm=={contract.version}"
    if active != [expected]:
        raise RuntimeError(f"Expected {pin_file} to contain only {expected}")
    return {
        "path": str(pin_file.resolve()),
        "sha256": sha256_file(pin_file),
        "requirement": expected,
    }


def inspect_requirements(requirements_file: Path) -> dict[str, Any]:
    active = [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    selected: dict[str, str] = {}
    for name in ("torch", "flashinfer-python"):
        matches = [line for line in active if line.startswith(f"{name}==")]
        expected = f"{name}=={RUNTIME_VERSIONS[name]}"
        if matches != [expected]:
            raise RuntimeError(f"Expected {requirements_file} to contain only one {expected}")
        selected[name] = RUNTIME_VERSIONS[name]
    return {
        "path": str(requirements_file.resolve()),
        "sha256": sha256_file(requirements_file),
        "selected_versions": selected,
    }


def _record_digest(payload: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
    return "sha256=" + digest.rstrip(b"=").decode("ascii")


def _validate_member(info: Any) -> None:
    name = info.filename
    path = PurePosixPath(name)
    if not name or name.startswith("/") or "\\" in name or ".." in path.parts:
        raise RuntimeError(f"Unsafe wheel member: {name!r}")
    mode = info.external_attr >> 16
    if mode and stat.S_ISLNK(mode):
        raise RuntimeError(f"Symlink wheel member is unsupported: {name}")


def _find_dist_info_member(names: list[str], filename: str) -> str:
    suffix = f".dist-info/{filename}"
    matches = [name for name in names if name.startswith("vllm-") and name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one vLLM {filename}, found {len(matches)}")
    return matches[0]


def _validate_record(record: bytes, common_path: str, common: bytes) -> None:
    rows = list(csv.reader(io.StringIO(record.decode("utf-8"))))
    matches = [row for row in rows if row and row[0] == common_path]
    if len(matches) != 1 or len(matches[0]) != 3:
        raise RuntimeError("vLLM RECORD lacks one complete rotary source row")
    expected = [common_path, _record_digest(common), str(len(common))]
    if matches[0] != expected:
        raise RuntimeError("vLLM RECORD does not match the rotary source payload")


def _patch_record(record: bytes, common_path: str, common: bytes) -> bytes:
    rows = list(csv.reader(io.StringIO(record.decode("utf-8"))))
    matches = [row for row in rows if row and row[0] == common_path]
    if len(matches) != 1:
        raise RuntimeError("vLLM RECORD lacks one rotary source row")
    matches[0][:] = [common_path, _record_digest(common), str(len(common))]
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    return output.getvalue().encode("utf-8")


def patch_common(common: bytes, contract: WheelContract = CONTRACT) -> bytes:
    if sha256_bytes(common) != contract.common_before_sha256:
        raise RuntimeError("vLLM rotary source hash does not match the qualified input")
    if common.count(OLD_IMPORT) != 1 or common.count(OLD_INIT) != 1:
        raise RuntimeError("Upstream backport anchors are not unique")
    patched = common.replace(OLD_IMPORT, NEW_IMPORT).replace(OLD_INIT, NEW_INIT)
    if sha256_bytes(patched) != contract.common_after_sha256:
        raise RuntimeError("Patched vLLM rotary source hash is unexpected")
    return patched


def inspect_wheel(
    wheel: Path,
    *,
    patched: bool,
    contract: WheelContract = CONTRACT,
) -> dict[str, Any]:
    if wheel.name != contract.wheel_filename:
        raise RuntimeError(f"Expected wheel filename {contract.wheel_filename}, got {wheel.name}")
    wheel_hash = sha256_file(wheel)
    if not patched and wheel_hash != contract.wheel_sha256:
        raise RuntimeError("vLLM wheel hash does not match the qualified input")

    with ZipFile(wheel) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise RuntimeError("Wheel contains duplicate members")
        for info in infos:
            _validate_member(info)
        if names.count(contract.common_path) != 1:
            raise RuntimeError("Wheel lacks one vLLM rotary source")
        metadata_name = _find_dist_info_member(names, "METADATA")
        record_name = _find_dist_info_member(names, "RECORD")
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        if metadata.get("Name") != contract.distribution:
            raise RuntimeError("Wheel distribution name is not vllm")
        if metadata.get("Version") != contract.version:
            raise RuntimeError("Wheel version does not match the qualified vLLM version")
        common = archive.read(contract.common_path)
        expected_hash = contract.common_after_sha256 if patched else contract.common_before_sha256
        if sha256_bytes(common) != expected_hash:
            state = "patched" if patched else "upstream"
            raise RuntimeError(f"Wheel {state} rotary source hash is unexpected")
        _validate_record(archive.read(record_name), contract.common_path, common)
        member_hashes = {
            info.filename: sha256_bytes(archive.read(info)) for info in infos if not info.is_dir()
        }
    return {
        "path": str(wheel.resolve()),
        "sha256": wheel_hash,
        "size": wheel.stat().st_size,
        "record_path": record_name,
        "member_hashes": member_hashes,
    }


def build_backport(
    input_wheel: Path,
    output_dir: Path,
    *,
    contract: WheelContract = CONTRACT,
    pin_file: Path = PIN_FILE,
    requirements_file: Path = REQUIREMENTS_FILE,
) -> tuple[Path, dict[str, Any]]:
    pin = inspect_pin(pin_file, contract)
    requirements = inspect_requirements(requirements_file)
    before = inspect_wheel(input_wheel, patched=False, contract=contract)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_wheel = output_dir / contract.wheel_filename
    manifest_path = output_dir / "vllm-fa4-backport.manifest.json"
    if output_wheel.exists() or manifest_path.exists():
        raise RuntimeError("Refusing to overwrite an existing backport wheel or manifest")

    temporary_dir = Path(tempfile.mkdtemp(prefix=".vllm-fa4-backport-", dir=output_dir))
    temporary = temporary_dir / contract.wheel_filename
    try:
        with ZipFile(input_wheel) as source:
            common = source.read(contract.common_path)
            patched_common = patch_common(common, contract)
            record_name = str(before["record_path"])
            patched_record = _patch_record(
                source.read(record_name), contract.common_path, patched_common
            )
            with ZipFile(temporary, "w", allowZip64=True) as target:
                target.comment = source.comment
                for info in source.infolist():
                    if info.filename == contract.common_path:
                        target.writestr(info, patched_common)
                    elif info.filename == record_name:
                        target.writestr(info, patched_record)
                    elif info.is_dir():
                        target.writestr(info, b"")
                    else:
                        with (
                            source.open(info) as reader,
                            target.open(info, "w") as writer,
                        ):
                            shutil.copyfileobj(reader, writer, length=1024 * 1024)
        after = inspect_wheel(temporary, patched=True, contract=contract)
        changed = {contract.common_path, str(before["record_path"])}
        before_members = before["member_hashes"]
        after_members = after["member_hashes"]
        if before_members.keys() != after_members.keys():
            raise RuntimeError("Backport changed the wheel member inventory")
        unexpected = [
            name
            for name in before_members
            if name not in changed and before_members[name] != after_members[name]
        ]
        if unexpected:
            raise RuntimeError(f"Backport changed unrelated wheel members: {unexpected}")
        temporary.replace(output_wheel)
        after["path"] = str(output_wheel.resolve())
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)

    manifest = {
        "schema": "aisp.vllm-fa4-wheel-backport.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": asdict(contract),
        "pin": pin,
        "requirements": requirements,
        "input": {key: before[key] for key in ("path", "sha256", "size")},
        "output": {key: after[key] for key in ("path", "sha256", "size")},
        "changed_members": sorted(changed),
        "upstream": UPSTREAM,
        "retirement": RETIREMENT,
        "tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output_wheel, manifest


def _runtime_probe(python: Path, *, patched: bool) -> dict[str, Any]:
    code = r"""
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import torch

expected = json.loads(os.environ["AISP_VLLM_FA4_EXPECTED"])
if sys.prefix == sys.base_prefix:
    raise RuntimeError("Installation target must be a virtual environment")
versions = {
    name: metadata.version(name)
    for name in ("torch", "flash-attn-4", "flashinfer-python")
}
if torch.__version__ != expected["torch"]:
    raise RuntimeError(f"imported torch version mismatch: {torch.__version__}")
for name, version in versions.items():
    if version != expected[name]:
        raise RuntimeError(f"{name} version mismatch: {version}")
try:
    versions["vllm"] = metadata.version("vllm")
except metadata.PackageNotFoundError:
    versions["vllm"] = None
if versions["vllm"] not in (None, expected["vllm"]):
    raise RuntimeError(f"vllm version mismatch: {versions['vllm']}")
fa4_cute = importlib.util.find_spec("flash_attn.cute.interface")
if fa4_cute is None or fa4_cute.origin is None:
    raise RuntimeError("FlashAttention 4 CuTe interface is unavailable")
fa4_root = Path(metadata.distribution("flash-attn-4").locate_file("")).resolve()
if fa4_root not in Path(fa4_cute.origin).resolve().parents:
    raise RuntimeError("FlashAttention module does not come from flash-attn-4")
try:
    legacy = importlib.util.find_spec("flash_attn.ops.triton.rotary")
except ModuleNotFoundError:
    legacy = None
if legacy is not None:
    raise RuntimeError("Legacy FlashAttention rotary namespace unexpectedly exists")

receipt = {
    "versions": versions,
    "fa4_cute_retained": True,
    "fa4_cute_origin": fa4_cute.origin,
    "legacy_rotary_absent": True,
}
if os.environ["AISP_VLLM_FA4_PATCHED"] == "1":
    import vllm
    import vllm._C
    from vllm.config import VllmConfig
    from vllm.config.vllm import set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

    dist = metadata.distribution("vllm")
    common = Path(dist.locate_file(expected["common_path"]))
    common_hash = hashlib.sha256(common.read_bytes()).hexdigest()
    if common_hash != expected["common_after_sha256"]:
        raise RuntimeError("Installed vLLM rotary source is not the guarded backport")
    dist_root = Path(dist.locate_file("")).resolve()
    if dist_root not in Path(vllm.__file__).resolve().parents:
        raise RuntimeError("Imported vLLM does not come from the installed distribution")
    before = torch.cuda.is_initialized()
    with set_current_vllm_config(VllmConfig()):
        op = ApplyRotaryEmb()
    if op.apply_rotary_emb_flash_attn is not None:
        raise RuntimeError("FA4 incorrectly selected the legacy rotary path")
    receipt.update({
        "installed_common": str(common),
        "installed_common_sha256": common_hash,
        "vllm_file": vllm.__file__,
        "bundled_extension_imported": True,
        "cuda_initialized_before_constructor": before,
        "cuda_initialized_after_constructor": torch.cuda.is_initialized(),
    })
print("AISP_VLLM_FA4_PROBE=" + json.dumps(receipt, sort_keys=True))
"""
    expected = {
        **RUNTIME_VERSIONS,
        "common_path": CONTRACT.common_path,
        "common_after_sha256": CONTRACT.common_after_sha256,
    }
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(
        PYTHONNOUSERSITE="1",
        AISP_VLLM_FA4_EXPECTED=json.dumps(expected, sort_keys=True),
        AISP_VLLM_FA4_PATCHED="1" if patched else "0",
    )
    result = subprocess.run(
        [str(python), "-c", code],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Runtime probe failed ({result.returncode}):\n{result.stdout}")
    marker = "AISP_VLLM_FA4_PROBE="
    receipts = [line for line in result.stdout.splitlines() if line.startswith(marker)]
    if len(receipts) != 1:
        raise RuntimeError(f"Runtime probe receipt missing:\n{result.stdout}")
    return {"receipt": json.loads(receipts[0][len(marker) :]), "stdout": result.stdout}


def install_backport(python: Path, wheel: Path) -> dict[str, Any]:
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError("--install-python must name an absolute virtual-environment Python")
    before = _runtime_probe(python, patched=False)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Backport wheel installation failed ({result.returncode}):\n{result.stdout}"
        )
    after = _runtime_probe(python, patched=True)
    return {"python": str(python), "before": before, "pip_stdout": result.stdout, "after": after}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-wheel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--install-python",
        type=Path,
        help="Optionally install with --no-deps into this explicit virtual-environment Python",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    wheel, manifest = build_backport(args.input_wheel, args.output_dir)
    manifest_path = args.output_dir / "vllm-fa4-backport.manifest.json"
    if args.install_python is not None:
        try:
            manifest["installation"] = install_backport(args.install_python, wheel)
        except BaseException as exc:
            manifest["installation"] = {"error": f"{type(exc).__name__}: {exc}"}
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            raise
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
