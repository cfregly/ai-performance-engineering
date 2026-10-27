# Compiler miss-policy illustration assets

These illustrations show compiler-routing choices after representative warmup. They contain no new performance measurement, latency result or speedup claim. The A/B sequences and route counts are hypothetical.

![Choose your compile-miss policy](images/fun.png)

The cover shows the policy choice when recompilation is needed. Valid cached compiled code remains usable. The application decides how to handle a raised error.

Source: [PyTorch compiler-stance API](https://docs.pytorch.org/docs/2.14/generated/torch.compiler.set_stance.html).

Runnable source: [chapter 14 routing diagnostic](https://github.com/cfregly/ai-performance-engineering/blob/fca05006eeb9562996479cbaf781b1c7a168ab0c/code/ch14/recompilation_demo.py). Its default CPU/eager backend exercises routing without generating optimized GPU kernels. No workload was run for these illustrations.

Files include three explainers, six audience variants and a Gamma Whiteboard cover. Each family has a 1600 x 900 PNG, a 1080 x 1080 square PNG, a 400 x 225 preview, editable SVG exports and a text description. The cover's original vector PDF is retained as images/gamma-cover.pdf.

The kernel-engineer example uses fullgraph=True. These images are conceptual drawings, with no captured profiler trace or execution-time scale.

## Image formats

| Family | Landscape | Square | Phone preview |
| --- | --- | --- | --- |
| a | ![a landscape](images/a.png) | ![a square](images/a-square.png) | ![a phone preview](images/a-400.png) |
| b | ![b landscape](images/b.png) | ![b square](images/b-square.png) | ![b phone preview](images/b-400.png) |
| c | ![c landscape](images/c.png) | ![c square](images/c-square.png) | ![c phone preview](images/c-400.png) |
| aud-curious | ![aud-curious landscape](images/aud-curious.png) | ![aud-curious square](images/aud-curious-square.png) | ![aud-curious phone preview](images/aud-curious-400.png) |
| aud-ml-engineer | ![aud-ml-engineer landscape](images/aud-ml-engineer.png) | ![aud-ml-engineer square](images/aud-ml-engineer-square.png) | ![aud-ml-engineer phone preview](images/aud-ml-engineer-400.png) |
| aud-performance-engineer | ![aud-performance-engineer landscape](images/aud-performance-engineer.png) | ![aud-performance-engineer square](images/aud-performance-engineer-square.png) | ![aud-performance-engineer phone preview](images/aud-performance-engineer-400.png) |
| aud-kernel-engineer | ![aud-kernel-engineer landscape](images/aud-kernel-engineer.png) | ![aud-kernel-engineer square](images/aud-kernel-engineer-square.png) | ![aud-kernel-engineer phone preview](images/aud-kernel-engineer-400.png) |
| aud-platform-operations | ![aud-platform-operations landscape](images/aud-platform-operations.png) | ![aud-platform-operations square](images/aud-platform-operations-square.png) | ![aud-platform-operations phone preview](images/aud-platform-operations-400.png) |
| aud-business-capacity | ![aud-business-capacity landscape](images/aud-business-capacity.png) | ![aud-business-capacity square](images/aud-business-capacity-square.png) | ![aud-business-capacity phone preview](images/aud-business-capacity-400.png) |
| fun | ![fun landscape](images/fun.png) | ![fun square](images/fun-square.png) | ![fun phone preview](images/fun-400.png) |
