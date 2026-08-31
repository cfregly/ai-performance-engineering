# Current attention and prefill acceptance package

This package advances the operational gate to the current reviewed source epoch. It does not update, replace or recertify the original `evidence/attention/` receipt. That older gate correctly rejects four later source/README hash changes. Its entire evidence directory remains unchanged, and the new driver diff and manifest delta document this transition.

After obtaining separately authorized target hardware and its required software stack, run from the repository root with the target Python environment:

```bash
python docs/audits/2026-08-30/evidence/integration/attention-final-epoch/run_cuda_acceptance.py --output-dir /absolute/path/to/a/fresh/attention-attempt
```

The output directory must not already exist. The driver creates it without overwriting previous attempts. Source or selection changes require another reviewed manifest; editing a receipt to make a mismatch pass is not qualification.

Ordinary `pytest --collect-only -k test_real_` selected exactly **25 original attention cases and 7 additional prefill cases**. Their complete node identities are frozen in `expected_cuda_cases.json`. The driver invokes those exact identities, one case per subprocess, rather than accepting a numerical minimum. Acceptance requires all 32 exact cases, return code zero, no failures, no errors, no skips and matching source hashes before and after the run. It also rejects source drift between cases.

Each case retains its command, JUnit result, log and pytest temporary artifacts. The outer command has a 600-second process-group timeout; the existing prefill test has a separate 300-second child timeout. Failed attempts retain completed case results. `gpu_executed` is false for preflight, unknown after an unsuccessful dispatch, and true only after all real CUDA cases pass.

The combined gate requires an actual CUDA device, nvcc, a CUDA 13+ Torch build and the existing extension targets sm_100/103/120/121. The original cases also require functioning Triton, FlashAttention CuTe and the applicable cuDNN runtime; absence is not substituted by another backend. A skip prevents acceptance. Installed package versions are recorded, but environment availability is not a declaration that the stack is pinned or qualified.

The manifest retains all original source paths at their current hashes and adds the four prefill wrappers, their test, local execution/build/verification dependencies, relevant CUDA headers, pytest configuration, the current persistent-decode README and this gate's driver/selection files. Resolvable static Python dependencies are included conservatively. This source identity inventory does not qualify third-party packages, dynamic environment state or hardware.

CPU checks of this package exercise only real filesystem integrity and report/selection logic. Parser fixtures are explicitly not GPU results. The real CPU capability preflight returns **HOLD** with no CUDA cases dispatched. Existing source reviews and the final prefill receipt support the source epoch; they do not establish CUDA correctness.

Even a future gate pass is bounded correctness evidence only. Sanitizer runs, full-size coverage, numerical-budget calibration and fresh benchmark timing remain separate requirements. The inherited decode tolerance remains rtol=0.1/atol=1.0 and is not newly calibrated. The added prefill gate's exact comparisons use fixed dyadic FP32 inputs and do not calibrate arbitrary or lower-precision inputs. Historical performance artifacts remain unchanged and do not establish performance for this revision.
