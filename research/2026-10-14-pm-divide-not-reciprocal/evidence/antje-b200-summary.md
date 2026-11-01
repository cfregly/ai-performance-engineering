# Historical numerical count record: b200

Source run `b200`, recorded 2026-10-24, reports **1,403,693,404 one-step differences out of 4,211,081,216 finite float32 inputs with |x| >= 2^-123**. The default CUDA reciprocal-multiply row is compared with inline `div.rn.f32`. The maximum difference is one representable step, away from zero within this cohort.

The recorded environment is NVIDIA B200, driver 580.173.02, CUDA and nvcc 13.0, PyTorch 2.13.0+cu130 and Triton 3.7.1.

Original JSON SHA-256: `71e0888fa0df11c32c178dd9571c59244045ab821291e88c6481eab334c9431b`. Original Markdown SHA-256: `06b77eaa6e9663cf15862e1a090af38283a6e7cd454e8be0c63fcae34def0766`. Measurement author and archived source identity are preserved in the [provenance metadata](provenance.json).

This is a numerical-only reading entry. The maintained successor did not produce this historical count, and the original result has no embedded executed-source digest. The [combined numerical census](numerical-census.md) explains the cohort correction, separate tolerance observation and computed percentages. No timing or profiler data is reproduced here.
