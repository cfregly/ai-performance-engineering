# Division-by-six numerical census

The retained source records report **1,403,693,404 one-step differences among 4,211,081,216 inputs** when multiplying by the rounded float32 reciprocal of six instead of using the `div.rn.f32` reference. The cohort is finite float32 **|x| >= 2^-123**, with exponent fields 4–254 for both signs. Every recorded difference in this cohort is one representable step away from zero.

This is a uniform census of input bit patterns. It does not estimate the frequency of differences in a model's activations or justify an application's error budget.

## Two recorded source runs

| Source record | Recorded date | Inputs in the displayed cohort | Different | Maximum representable steps |
|---|---|---:|---:|---:|
| [b200](antje-b200-summary.md) | 2026-10-24 | 4,211,081,216 | 1,403,693,404 | 1 |
| [b200-run2](antje-b200-run2-summary.md) | 2026-10-24 | 4,211,081,216 | 1,403,693,404 | 1 |

Both records identify NVIDIA B200, driver 580.173.02, CUDA and nvcc 13.0, PyTorch 2.13.0+cu130 and Triton 3.7.1. The displayed CUDA row uses the default build and expression `x * (1.0f / 6.0f)` against inline `div.rn.f32`.

The [machine-readable census](numerical-census.json) retains each original JSON and Markdown SHA-256, run identity, measurement attribution and exact field selection. The original reports do not embed an executed-source digest. Archived source identity is not an execution-time source attestation. The maintained successor did not produce these historical counts.

## Derived chart values

These values are computed from the retained counts, rather than newly measured:

- Inputs: `2 * 251 * 2^23 = 4,211,081,216`.
- Different: `2 * 251 * 2,796,202 = 1,403,693,404`.
- Same: `4,211,081,216 - 1,403,693,404 = 2,807,387,812`.
- Rounded to one decimal place: **33.3% different**, **66.7% same**. The exact different share is slightly below one third.

## Why a tolerance check can still pass

The separate recorded Triton reciprocal-multiply comparison reported zero finite-input failures of `torch.isclose` at the recorded `torch.allclose` defaults, `rtol=1e-5` and `atol=1e-8`. Its reference was `torch.div(x, six)` with a float32 tensor divisor. That finite-input coverage includes the displayed cohort. This is a distinct source observation from the CUDA bit census, not a new compiler-lowering or GPU result.

The historical `normal_quotient` label names this whole-binade census. It omits some smaller inputs whose quotients are also normal. Near-zero and signed-zero direction labels outside the displayed cohort do not support a general direction claim.

This companion includes no timing or profiler results. It makes no speedup, serving-performance or task-quality claim. See the [methods and setup](../code/README.md), [source index](source-index.json) and [provenance](provenance.json).
