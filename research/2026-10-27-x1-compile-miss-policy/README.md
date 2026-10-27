# Compiler miss-policy diagrams

Choose what happens when a call needs recompilation: raise an error or run it eagerly. Valid cached compiled code remains usable. The application decides how to handle a raised error.

![Compile-miss policy](images/fun.png)

[PyTorch compiler-stance reference](https://docs.pytorch.org/docs/2.14/generated/torch.compiler.set_stance.html) · [Runnable routing diagnostic](https://github.com/cfregly/ai-performance-engineering/blob/fca05006eeb9562996479cbaf781b1c7a168ab0c/code/ch14/recompilation_demo.py)

The input examples assume warmup cached a variant for A, while B requires recompilation. In the four-call examples, strict errors are caught so every call is attempted. The GPU-routing example uses fullgraph=True.

Each diagram has landscape, square and phone-width exports, editable SVGs and a text description. The cover's vector PDF is images/gamma-cover.pdf.

| Diagram | Landscape | Square | Phone preview |
| --- | --- | --- | --- |
| Policy choice | ![Policy choice, landscape](images/a.png) | ![Policy choice, square](images/a-square.png) | ![Policy choice, phone preview](images/a-400.png) |
| Input example | ![Input example, landscape](images/b.png) | ![Input example, square](images/b-square.png) | ![Input example, phone preview](images/b-400.png) |
| Inspection steps | ![Inspection steps, landscape](images/c.png) | ![Inspection steps, square](images/c-square.png) | ![Inspection steps, phone preview](images/c-400.png) |
| Curious readers | ![Curious readers, landscape](images/aud-curious.png) | ![Curious readers, square](images/aud-curious-square.png) | ![Curious readers, phone preview](images/aud-curious-400.png) |
| ML engineers | ![ML engineers, landscape](images/aud-ml-engineer.png) | ![ML engineers, square](images/aud-ml-engineer-square.png) | ![ML engineers, phone preview](images/aud-ml-engineer-400.png) |
| Performance engineers | ![Performance engineers, landscape](images/aud-performance-engineer.png) | ![Performance engineers, square](images/aud-performance-engineer-square.png) | ![Performance engineers, phone preview](images/aud-performance-engineer-400.png) |
| GPU engineers | ![GPU engineers, landscape](images/aud-kernel-engineer.png) | ![GPU engineers, square](images/aud-kernel-engineer-square.png) | ![GPU engineers, phone preview](images/aud-kernel-engineer-400.png) |
| Platform operations | ![Platform operations, landscape](images/aud-platform-operations.png) | ![Platform operations, square](images/aud-platform-operations-square.png) | ![Platform operations, phone preview](images/aud-platform-operations-400.png) |
| Capacity planning | ![Capacity planning, landscape](images/aud-business-capacity.png) | ![Capacity planning, square](images/aud-business-capacity-square.png) | ![Capacity planning, phone preview](images/aud-business-capacity-400.png) |
| Cover | ![Cover, landscape](images/fun.png) | ![Cover, square](images/fun-square.png) | ![Cover, phone preview](images/fun-400.png) |
