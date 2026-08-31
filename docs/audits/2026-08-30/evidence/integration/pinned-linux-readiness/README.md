This package repairs active dependency pins and prepares Linux acceptance for W1-005, W1-006 and W1-124. It does not qualify the complete Linux/CUDA environment.

The five source files listed in `current-source-manifest.json` now use Triton 3.5.1, matching the official Torch 2.9.1 CUDA wheel's Linux dependency. The CPU workflow explicitly installs requests 2.34.2 and the existing tokenizers 0.22.2 pin, both needed during ordinary test collection. Torch remains 2.9.1+cu130. Earlier source repairs and dated GB300 incident notes remain intact.

`epoch-source-diff.txt` shows only this continuation's changes. The five original source snapshots and earlier evidence are retained. Final host checks passed 36 tests with no skips, Bash syntax validation and two Python AST parses. These tests ran in the existing Darwin CPU environment, not a newly installed Linux environment.

| Retained target-resolution attempt | Result and boundary |
| --- | --- |
| Original and corrected CUDA pair, and old CPU graph, attempt 1 | Three CLI authoring errors: uv 0.9.18 rejects combining `--no-build` and `--only-binary`. No resolution occurred. |
| Torch 2.9.1+cu130 / Triton 3.5.0, attempt 2 | Expected version conflict: the Torch wheel requires Triton 3.5.1 on Linux. |
| Torch 2.9.1+cu130 / Triton 3.5.1, attempt 2 | 26 packages resolved. This checks the core pair, not every repository pin. |
| CPU workflow before tokenizers addition | 19 direct pins, 55 resolved packages. Historical result only. |
| CPU workflow with tokenizers | 20 direct pins, 58 resolved packages. The `--torch-backend cpu` routing differs from the literal workflow for Triton. |
| First CPU package transaction, PyPI only | 19 non-Torch direct pins, 51 resolved packages. |
| Exact official CPU Torch URL plus PyPI packages, constrained by the previous result | 58 packages resolved. Every previous version was retained, including the actual PyPI Triton wheel hash. |
| Unmodified full requirements | uv rejected the relative local find-links directive before solving. The referenced local wheel directory is absent. This is not proof that pip/setup fails. |
| Public-only full requirements diagnostic | All 90 package specifications retained; only the absent local find-links directive omitted. Stopped because GPUtil 1.4.0 has only a source distribution and this probe forbids builds. No full graph verdict. |
| Torch / vLLM pair, generic Linux target | vLLM wheel unavailable for that target. This is not a Torch version conflict. |
| Torch / vLLM pair, explicit manylinux 2.31 | 172 packages resolved. Published vLLM 0.16.0 metadata requires Torch 2.9.1, which accepts the +cu130 local version. Other repository pins were not constraints in this pair. |
| Torch / torchtitan pair | 73 packages resolved. This does not establish Python API, native ABI or import compatibility. |

Each attempt directory preserves its exact input, command, stdout, stderr and any resolver output. Successful resolution uses only binary metadata and official public indexes; it does not install, import or execute the selected Linux packages. The first three argument failures were corrected by retaining `--only-binary :all:` alone. No dependency was discarded and no `--no-deps` option was used to obtain a resolver success.

The CPU provenance check is in `cpu-source-policy-comparison.json`. Earlier CPU-backend outputs selected a different Triton wheel than PyPI; matching package versions alone did not establish matching artifacts. The final constrained union uses the exact Torch CPU wheel link observed in the official index and PyPI for the other packages. This is a dependency compatibility check, not execution of the workflow's two pip install transactions. The generated resolver outputs are evidence, not repository lockfiles intended for blind pip replay.

The initial `dependency-contract.json` still says the real Linux resolver was pending because it predates the uv attempts. It also retains the old 19-pin input. The later attempts extend that evidence without rewriting it. Similarly, original test failures and one test-authoring failure remain preserved.

Actual Linux installation and full collection remain pending. The parent reported that Docker's public-image pull was blocked by the macOS Keychain credential helper; this task did not bypass it or run a container. CUDA compilation, device linking, imports, execution, sanitizers and performance remain pending. ARM bootstrap also remains unsupported by the current prebuilt TorchAO CUDA wheel; the existing explicit guard and previously documented source-build requirement are unchanged. An actual validated ARM artifact is still needed.

The full requirements probes do not reproduce setup.sh's staged installation, excluded packages, later no-deps installs or import checks. GPUtil's source-only release is a limitation of this binary-only probe, not evidence of an inherent setup defect. No direct-install documentation was withdrawn solely because uv rejected the local find-links input.

Official sources are captured in `primary-source-fetches.json`, `additional-primary-source-fetches.json` and their referenced files. uv's target-resolution behavior is described in the [official CLI reference](https://docs.astral.sh/uv/reference/cli/#uv-pip-compile--python-platform). `receipt.json` records scope and dispositions; `final-artifact-manifest.json` binds all retained files in this evidence directory except itself.
