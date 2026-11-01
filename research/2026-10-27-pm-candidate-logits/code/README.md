# Reproduce the allowed-row projection benchmark

A token ID is the model's numeric label for a token and its output-weight row. Allowed token IDs identify the tokens the app permits at the next step.

This benchmark measures a synthetic output-head projection/filter/allowed-argmax stage. It uses eager PyTorch CUDA `torch.mm` with preallocated output buffers. Both paths use the same synthetic weights, hidden states and allowed IDs. Each has its own `[8, K]` allowed-score buffer, where `K` is the allowed-set size. The dense path also uses an `[8, 131072]` full-vocabulary buffer. It does not run a language model or a server. In code identifiers and evidence fields, `legal` means the article's allowed IDs.

The catch: the allowed set must be ready before scoring, so its weights can be gathered first. The usual path needs the set only when filtering scores afterward. Both timers start with the allowed set already on the GPU. Creating, sorting and transferring it were excluded from both paths.

My advice: use full scoring if the caller needs every token's score, or if making the set ready early and gathering its weights erases the saving. Both tested paths return allowed scores and their greedy winner. Floating-point near-ties remain a separate precision-validation question.

The retained run used Linux, Python 3.12.3, PyTorch 2.14.1+cu130, CUDA 13.0, NVIDIA driver 580.173.02 and one NVIDIA B200 reporting a 1000 W power limit. FP32 matmul precision was `highest` and TF32 was disabled. Dimensions and the fixed decision rule are in [plan.json](plan.json).

From the package root, `research/2026-10-27-pm-candidate-logits/` (not its `code/` directory), recompute the six cells and run the evidence tests without importing PyTorch or using a GPU:

```sh
python3 code/analyze.py
python3 -m unittest discover -s code -p 'test_analyze.py' -v
python3 code/screen.py --self-check
```

A successful `analyze.py` run prints JSON with six cells and `all_cells_pass: true`. Without `--output`, it writes no file. The unit tests finish with `OK`. The self-check prints `Source and frozen-plan checks passed. No torch import or GPU work ran.` These commands inspect retained evidence and source only.

To run a new screen, use an existing matching CUDA environment on an available B200. Select the allocated CUDA ordinal and a new absolute output directory:

```sh
python3 code/screen.py --device cuda:0 --output /tmp/allowed-row-projection-new-run
python3 code/analyze.py --raw /tmp/allowed-row-projection-new-run/raw.jsonl --plan code/plan.json --output /tmp/allowed-row-analysis.json
```

The retained historical supervisor starts the separate GPU worker, enforces its recorded time limit and records final status, source hashes and cleanup in `HANDBACK.json`. The worker performs the measurements and writes the raw records. Its work window is 155 seconds within a 170-second outer window. Partial evidence is retained on failure. The terminal status is `COMPLETE` or `PARTIAL`. `PARTIAL` keeps the error, worker output, logs and cleanup fields, and returns a nonzero exit. Inspect those records before another authorized run. There is no automatic rerun. A COMPLETE run means every cell finished. Each cell's fixed timing criteria are reported separately.

The three paired blocks use AB, BA, AB order after two warmup blocks per path. Each measured block contains 16 projection/filter/selection steps followed by CUDA synchronization. A is dense projection, allowed-score gather and argmax. B is projection through allowed weight rows and the same argmax. Changing mode includes row gathering inside each timed step. Static mode gathers once during separately reported setup and reuses those rows. The changing sequence contains one shared allowed set per step, reused across warmup and paired blocks.

The benchmark checks every allowed logit against an FP64 projection and requires exact selected-token agreement. Sorted IDs make the first maximum choose the smallest allowed ID on exact ties. Separate all-equal and two-way-maximum fixtures exercise that rule. The retained raw log records the generated allowed IDs, per-step errors, selected tokens, setup intervals and wall/CUDA-event samples.

`screen.py` and `plan.json` preserve the executed bytes. The current analyzer and tests add stricter validation of the retained records. This guide is maintained documentation and its bytes differ from the guide that accompanied the original run. [File hashes](../evidence/hashes.json) identify every shared scientific file.

The current analyzer rejects malformed or incomplete records, including invalid allowed sets, step coverage, tie winners and paired timings. CPU tests compare the six-cell raw-log calculation with the native worker summary and reject a missing pair or nonfinite timing. The raw correctness records report assertions made during the retained GPU run. The [original analyzer](executed-analyze.py) retains the exact historical analysis bytes. These checks use the same captured data and do not rerun numerical correctness or timing on the GPU.
