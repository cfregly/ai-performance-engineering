# Torch 2.14.1 DLPack adapter

The hash-locked stack uses the included `torch-c-dlpack-ext` wheel derived from
upstream 0.1.5. It retains the Apache 2.0 license and NOTICE in the wheel.

The patch uses Torch 2.14.1's built-in `Tensor.__dlpack_c_exchange_api__` and
requires that API. It removes the unused top-level `build_backend.py` from the
installed payload because FlashInfer owns a different module with that name.
The wheel contains no native fallback. Older Torch releases need their own
compatible adapter.

`torch_c_dlpack_ext.provenance.json` records the verified upstream archive,
source patches, builder and contract hashes, complete wheel payload, and build
tools. Two fresh offline Linux builds produced byte-identical wheels. No CUDA
work was performed during packaging.

To reproduce from `code/` in the required isolated Linux build environment:

```bash
CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  python scripts/build_torch_dlpack_wheel.py --output-dir /absolute/new-output
```

Run the command again with a different fresh output directory and compare wheel
SHA-256 hashes. The builder rejects an existing output directory and validates
the upstream source and resulting pure-Python wheel. Runtime interoperability
and the wider GPU stack require their separate functional checks.
