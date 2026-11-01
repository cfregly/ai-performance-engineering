# Historical numerical count record: b200-run2

Source run `b200-run2`, recorded 2026-10-24, reports **1,403,693,404 one-step differences out of 4,211,081,216 finite float32 inputs with |x| >= 2^-123**. The default CUDA reciprocal-multiply row is compared with inline `div.rn.f32`. The maximum difference is one representable step, away from zero within this cohort.

The recorded environment is NVIDIA B200, driver 580.173.02, CUDA and nvcc 13.0, PyTorch 2.13.0+cu130 and Triton 3.7.1.

Original JSON SHA-256: `5ad5ed54c3b63f1f576c061266601e0276fd69625ab3043801440466afcdd931`. Original Markdown SHA-256: `814b4c8037e4ed485fc256d912d56765699917f0b260933b4ed0fefb47824ad5`. Measurement author and archived source identity are preserved in the [provenance metadata](provenance.json).

This is a numerical-only reading entry. The maintained successor did not produce this historical count, and the original result has no embedded executed-source digest. The [combined numerical census](numerical-census.md) explains the cohort correction, separate tolerance observation and computed percentages. No timing or profiler data is reproduced here.
