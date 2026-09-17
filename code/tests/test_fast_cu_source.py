"""Real file-system checks for the pinned fast.cu source supply chain."""

from __future__ import annotations

import json
import shutil

import pytest

from labs.fast_cu.source import (
    MANIFEST_PATH,
    UPSTREAM_DIR,
    UPSTREAM_REVISION,
    verify_upstream,
)


def test_complete_pinned_source_tree():
    manifest = verify_upstream()
    assert manifest["revision"] == UPSTREAM_REVISION
    assert len(manifest["files"]) == 34
    assert "MIT License" in (UPSTREAM_DIR / "LICENSE").read_text()
    for rung in range(10):
        assert f"gb300/nvfp4/gemm{rung}.cuh" in manifest["files"]
    for rung in range(1, 13):
        assert f"h100/matmul/matmul_{rung}.cuh" in manifest["files"]


@pytest.fixture
def copied_source(tmp_path):
    root = tmp_path / "upstream"
    shutil.copytree(UPSTREAM_DIR, root)
    manifest_path = tmp_path / "manifest.json"
    shutil.copyfile(MANIFEST_PATH, manifest_path)
    return root, manifest_path


def test_modified_source_rejected(copied_source):
    root, manifest_path = copied_source
    (root / "h100" / "sum.cu").write_text("// modified\n")
    with pytest.raises(ValueError, match="integrity mismatch: h100/sum.cu"):
        verify_upstream(root, manifest_path)


def test_missing_license_rejected(copied_source):
    root, manifest_path = copied_source
    (root / "LICENSE").rename(root / "LICENSE.moved")
    with pytest.raises(ValueError, match="file set changed"):
        verify_upstream(root, manifest_path)


def test_extra_header_rejected(copied_source):
    root, manifest_path = copied_source
    (root / "injected.cuh").write_text("// unpinned include\n")
    with pytest.raises(ValueError, match="extra=.*injected.cuh"):
        verify_upstream(root, manifest_path)


def test_symlink_rejected(copied_source):
    root, manifest_path = copied_source
    (root / "linked.cuh").symlink_to(root / "h100" / "sum.cu")
    with pytest.raises(ValueError, match="Symlink"):
        verify_upstream(root, manifest_path)


@pytest.mark.parametrize("field,value", [("revision", "0" * 40), ("schema_version", 99)])
def test_changed_manifest_identity_rejected(copied_source, field, value):
    root, manifest_path = copied_source
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest schema|source revision"):
        verify_upstream(root, manifest_path)
