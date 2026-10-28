# Reproduce the legal-row projection benchmark

This benchmark measures a synthetic output-head projection/filter/legal-argmax stage. It uses eager PyTorch CUDA `torch.mm` with preallocated output buffers. Both paths share synthetic weights, hidden states, legal IDs and score-buffer shapes. It does not run a language model or a server.

The retained run used Linux, Python 3.12.3, PyTorch 2.14.1+cu130, CUDA 13.0, NVIDIA driver 580.173.02 and one NVIDIA B200 reporting a 1000 W power limit. FP32 matmul precision was `highest` and TF32 was disabled. Dimensions and the fixed decision rule are in [plan.json](plan.json).

From the benchmark directory, recompute the six cells and run the evidence tests without importing PyTorch or using a GPU:

```sh
python3 code/analyze.py
python3 -m unittest discover -s code -p 'test_analyze.py' -v
python3 code/screen.py --self-check
```

To run a new screen, use an existing matching CUDA environment on an available B200. Select the allocated CUDA ordinal and a new absolute output directory:

```sh
python3 code/screen.py --device cuda:0 --output /tmp/legal-row-projection-new-run
python3 code/analyze.py --raw /tmp/legal-row-projection-new-run/raw.jsonl --plan code/plan.json --output /tmp/legal-row-analysis.json
```

The supervisor gives the worker 155 seconds and enforces a 170-second outer window. It retains partial output and writes a terminal handback on failure. There is no automatic rerun. A COMPLETE run means every cell finished; each cell's fixed timing criteria are reported separately.

The three paired blocks use AB, BA, AB order after two warmup blocks per path. Each measured block contains 16 projection/filter/selection steps followed by CUDA synchronization. A is dense projection, legal-score gather and argmax; B is projection through legal weight rows and the same argmax. Changing mode includes row gathering inside each timed step. Static mode gathers once during separately reported setup and reuses those rows. The changing sequence contains one shared legal set per step, reused across warmup and paired blocks.

The benchmark checks every legal logit against an FP64 projection and requires exact selected-token agreement. Sorted IDs make the first maximum choose the smallest legal ID on exact ties. Separate all-equal and two-way-maximum fixtures exercise that rule. The retained raw log records the generated legal IDs, per-step errors, selected tokens, setup intervals and wall/CUDA-event samples.

`screen.py` and `plan.json` preserve the executed bytes. The analyzer and tests preserve the retained analysis source. This guide is maintained documentation and its bytes differ from the guide that accompanied the original run. [File hashes](../evidence/hashes.json) identify every shared scientific file.

The evidence tests compare the raw-log calculation with the native worker summary and reject a missing pair or nonfinite timing. They do not rerun numerical correctness on the GPU. The raw correctness records report assertions made during the retained GPU run.
