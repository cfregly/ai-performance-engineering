# AI Performance Engineering

_**Update:** Are you interested in a hands-on course for this material?_

_If so, fill out this [**form**](https://docs.google.com/forms/d/e/1FAIpQLSf4TMDLsPcfuoLhaDktXu-hhKIGntQm550BY-ov6bRT_VMJhQ/viewform?usp=sharing&ouid=111382272947765737941) to express interest and be notified._

## About This Repo

AI Systems Performance Engineering code, tooling, and resources for the O'Reilly book covering GPU optimization, distributed training, inference scaling, and full-stack performance tuning for modern AI workloads.

[**Chat**](https://chatgpt.com/g/g-691a6b188d808191b16cdd2b7732cf11-ai-systems-performance-engineering) with this book directly!

Just take me to the [**code**](code/).

[![O'Reilly Book](img/ai_sys_perf_engg_cover_cheetah_sm.png)](https://www.amazon.com/Systems-Performance-Engineering-Optimizing-Algorithms/dp/B0F47689K8/)

> **O'Reilly Book – November 2025**  
> [Available on Amazon](https://www.amazon.com/Systems-Performance-Engineering-Optimizing-Algorithms/dp/B0F47689K8/)

### AI Systems Performance Engineering Book
Modern AI systems demand more than raw FLOPs—they need goodput‑driven, profile‑first engineering across hardware, software, and algorithms. This hands‑on guide shows how to turn GPUs, interconnects, and runtime stacks into efficient, reliable training and inference pipelines. 

You’ll learn to diagnose real bottlenecks with Nsight and PyTorch profilers, squeeze bandwidth and memory, and use compiler stacks (PyTorch + OpenAI Triton) to craft high‑impact kernels. On the serving side, master high‑throughput inference with vLLM/SGLang, TensorRT‑LLM, and NVIDIA Dynamo—including disaggregated prefill/decode and paged KV cache—then scale across racks without blowing the budget.

Using a hands‑on, empirical methodology with case studies, profiling data, this book is useful for AI/ML engineers, systems engineers, researchers, and platform teams building or operating training/inference at scale. The book contains thousands of lines of PyTorch and CUDA C++ code examples for modern NVIDIA GPUs.

* Profile for goodput, not just utilization—use Nsight Systems/Compute and the PyTorch profiler to find the real stall points. 

* Exploit memory & bandwidth—optimize layouts, caching, and data movement to feed the GPU continuously. 

* Tune with compilers—leverage the PyTorch compiler stack and Triton to generate high‑impact kernels without C++ boilerplate. 

* Scale training sanely—apply parallelism strategies (DP, FSDP, TP, PP, CP, and MoE) and overlap computation/communication to minimize bubbles. 

* Serve trillion parameter models efficiently—use vLLM, SGLang, TensorRT‑LLM and NVIDIA Dynamo with disaggregated prefill/decode and KV‑cache movement.

* Reduce cost per token—engineer for performance‑per‑watt and throughput per dollar, not just peak speed.

* Adopt AI‑assisted optimization—let AI help synthesize and tune kernels as systems outgrow manual tweaking
 
* Ship with confidence—apply the 200+ item [checklist](docs/appendix.md) to reproduce wins and prevent regressions across teams.

### Author Bio

Chris Fregly is a performance engineer and AI product leader who has driven innovations at Netflix, Databricks, and Amazon Web Services (AWS). He has led performance‑focused engineering teams that built AI/ML products, scaled go‑to‑market initiatives, and reduced cost for large‑scale generative‑AI and analytics workloads. 

Chris is the author of two other O’Reilly books: Data Science on AWS and Generative AI on AWS. He's also the creator of the O’Reilly course “High‑Performance AI in Production with NVIDIA GPUs" and the DeepLearning.ai course "Generative AI with Large-Language Models" with Andrew Ng.

His work spans kernel‑level tuning, compiler‑driven acceleration, distributed training, and high‑throughput inference. Chris hosts a monthly meetup called [AI Performance Engineering](https://www.meetup.com/ai-performance-engineering).

### 200+ Item Performance [checklist](docs/appendix.md)

The book ships with a **200+ item performance [checklist](docs/appendix.md)** that captures field‑tested optimizations covering the entire lifecycle. You can apply these immediately:

- ✅ Performance tuning mindset and cost optimization
- ✅ Reproducibility and documentation best practices
- ✅ System architecture and hardware planning
- ✅ Operating system and driver optimizations
- ✅ GPU programming and CUDA tuning
- ✅ Distributed training and network optimization
- ✅ Efficient inference and serving
- ✅ Power and thermal management
- ✅ Latest profiling tools and techniques
- ✅ Architecture-specific optimizations

### Links

- **Book**: [AI Systems Performance Engineering on Amazon](https://www.amazon.com/Systems-Performance-Engineering-Optimizing-Algorithms/dp/B0F47689K8/)
- **Meetup**: [AI Performance Engineering](https://www.meetup.com/ai-performance-engineering)
- **YouTube**: [AI Performance Engineering Channel](https://www.youtube.com/@AIPerformanceEngineering)

> *Built in San Francisco for the AI performance engineering community*

### Key Focus Areas

- **GPU Architecture, PyTorch, CUDA, and OpenAI Triton Programming**
- **Distributed Training & Inference**
- **Memory Optimization & Profiling**
- **PyTorch Performance Tuning**
- **Multi-Node Scaling Strategies**

## Book Chapters

### Chapter 1: Introduction and AI System Overview

- The AI Systems Performance Engineer
- Benchmarking and Profiling
- Scaling Distributed Training and Inference
- Managing Resources Efficiently
- Cross-Team Collaboration
- Transparency and Reproducibility

### Chapter 2: AI System Hardware Overview

- The CPU and GPU "Superchip"
- NVIDIA Grace CPU & Blackwell GPU
- NVIDIA GPU Tensor Cores and Transformer Engine
- Streaming Multiprocessors, Threads, and Warps
- Ultra-Scale Networking
- NVLink and NVSwitch
- Multi-GPU Programming

### Chapter 3: OS, Docker, and Kubernetes Tuning

- Operating System Configuration
- GPU Driver and Software Stack
- NUMA Awareness and CPU Pinning
- Container Runtime Optimizations
- Kubernetes for Topology-Aware Orchestration
- Memory Isolation and Resource Management

### Chapter 4: Tuning Distributed Networking Communication

- Overlapping Communication and Computation
- NCCL for Distributed Multi-GPU Communication
- Topology Awareness in NCCL
- Distributed Data Parallel Strategies
- NVIDIA Inference Transfer Library (NIXL)
- In-Network SHARP Aggregation

### Chapter 5: GPU-based Storage I/O Optimizations

- Fast Storage and Data Locality
- NVIDIA GPUDirect Storage
- Distributed, Parallel File Systems
- Multi-Modal Data Processing with NVIDIA DALI
- Creating High-Quality LLM Datasets

### Chapter 6: GPU Architecture, CUDA Programming, and Maximizing Occupancy

- Understanding GPU Architecture
- Threads, Warps, Blocks, and Grids
- CUDA Programming Refresher
- Understanding GPU Memory Hierarchy
- Maintaining High Occupancy and GPU Utilization
- Roofline Model Analysis

### Chapter 7: Profiling and Tuning GPU Memory Access Patterns

- Coalesced vs. Uncoalesced Global Memory Access
- Vectorized Memory Access
- Tiling and Data Reuse Using Shared Memory
- Warp Shuffle Intrinsics
- Asynchronous Memory Prefetching

### Chapter 8: Occupancy Tuning, Warp Efficiency, and Instruction-Level Parallelism

- Profiling and Diagnosing GPU Bottlenecks
- Nsight Systems and Compute Analysis
- Tuning Occupancy
- Improving Warp Execution Efficiency
- Exposing Instruction-Level Parallelism

### Chapter 9: Increasing CUDA Kernel Efficiency and Arithmetic Intensity

- Multi-Level Micro-Tiling
- Kernel Fusion
- Mixed Precision and Tensor Cores
- Using CUTLASS for Optimal Performance
- Inline PTX and SASS Tuning

### Chapter 10: Intra-Kernel Pipelining and Cooperative Thread Block Clusters

- Intra-Kernel Pipelining Techniques
- Warp-Specialized Producer-Consumer Model
- Persistent Kernels and Megakernels
- Thread Block Clusters and Distributed Shared Memory
- Cooperative Groups

### Chapter 11: Inter-Kernel Pipelining and CUDA Streams

- Using Streams to Overlap Compute with Data Transfers
- Stream-Ordered Memory Allocator
- Fine-Grained Synchronization with Events
- Zero-Overhead Launch with CUDA Graphs

### Chapter 12: Dynamic and Device-Side Kernel Orchestration

- Dynamic Scheduling with Atomic Work Queues
- Batch Repeated Kernel Launches with CUDA Graphs
- Dynamic Parallelism
- Orchestrate Across Multiple GPUs with NVSHMEM

### Chapter 13: Profiling, Tuning, and Scaling PyTorch

- NVTX Markers and Profiling Tools
- PyTorch Compiler (torch.compile)
- Profiling and Tuning Memory in PyTorch
- Scaling with PyTorch Distributed
- Multi-GPU Profiling with HTA

### Chapter 14: PyTorch Compiler, XLA, and OpenAI Triton Backends

- PyTorch Compiler Deep Dive
- Writing Custom Kernels with OpenAI Triton
- PyTorch XLA Backend
- Advanced Triton Kernel Implementations

### Chapter 15: Multi-Node Inference Parallelism and Routing

- Disaggregated Prefill and Decode Architecture
- Parallelism Strategies for MoE Models
- Speculative and Parallel Decoding Techniques
- Dynamic Routing Strategies

### Chapter 16: Profiling, Debugging, and Tuning Inference at Scale

- Workflow for Profiling and Tuning Performance
- Dynamic Request Batching and Scheduling
- Systems-Level Optimizations
- Quantization Approaches for Real-Time Inference
- Application-Level Optimizations

### Chapter 17: Scaling Disaggregated Prefill and Decode

- Prefill-Decode Disaggregation Benefits
- Prefill Workers Design
- Decode Workers Design
- Disaggregated Routing and Scheduling Policies
- Scalability Considerations

### Chapter 18: Advanced Prefill-Decode and KV Cache Tuning

- Optimized Decode Kernels (FlashMLA, ThunderMLA, FlexDecoding)
- Tuning KV Cache Utilization and Management
- Heterogeneous Hardware and Parallelism Strategies
- SLO-Aware Request Management

### Chapter 19: Dynamic and Adaptive Inference Engine Optimizations

- Adaptive Parallelism Strategies
- Dynamic Precision Changes
- Kernel Auto-Tuning
- Reinforcement Learning Agents for Runtime Tuning
- Adaptive Batching and Scheduling

### Chapter 20: AI-Assisted Performance Optimizations

- AlphaTensor AI-Discovered Algorithms
- Automated GPU Kernel Optimizations
- Self-Improving AI Agents
- Scaling Toward Multi-Million GPU Clusters

## Community Resources

Monthly meetups with 100k+ members across 20+ cities:

- [YouTube Channel](https://www.youtube.com/@AIPerformanceEngineering)
- [Meetup Group](https://www.meetup.com/ai-performance-engineering)

Recent sessions:

- [Dynamic Adaptive RL Inference CUDA Kernel Tuning](resources/Dynamic_Adaptive_RL_Inference_CUDA_Kernel_Tuning.pdf)
- [High Performance Agentic AI Inference Systems](resources/High_Performance_Agentic_AI_Inference_Systems.pdf)
- [PyTorch Model Optimization](resources/PyTorch_Model_Optimization.pdf)

### Monthly Meetup Summaries
- **Jun 15, 2026** - [YouTube](https://www.youtube.com/watch?v=1gALYQESnNU): Hacking AI Accelerators: Maxxing Tokens & Outcomes with Rob Ferguson @ Fireworks.AI
- **May 19, 2026** - [YouTube](https://www.youtube.com/watch?v=Z_L-vowv_Io): OpenAI on AWS + Cerebras vs GPU vs TPU + High-Performance KV Cache Offload
- **Apr 23, 2026** - [YouTube](https://www.youtube.com/watch?v=1i_9-4bx3Ms): Optimizing AI Inference for Heterogeneous Clusters by Natalie Serrino, Founder @ Gimlet Labs
- **Mar 23, 2026** - [YouTube](https://www.youtube.com/watch?v=qDOxTI-4pnU) & Slides: [NVIDIA GTC 2026 Performance Recap](resources/NVIDIA_GTC_2026_Performance_Recap_including_Slides_and_Workshops.pdf): NVIDIA GTC 2026 Conf Recap + Inference Engines + Scaling Disaggregated Prefill-Decode + RadixAttention
- **Feb 16, 2026** - [YouTube](https://www.youtube.com/watch?v=GdICZ0ov7Cw) & Slides: [NVFP4 Low Precision Numerics by Riccardo Mereu @ Verda](resources/NVFP4_Low_Precision_Numerics_Verda.pdf): OpenClaw/MCP for AI Systems Performance Tuning + NVFP4 Low Precision Optimizations
- **Jan 19, 2026** - [YouTube](https://www.youtube.com/watch?v=o-etY6VLHZo) & Slides: [Optimizing_Data_Transfer_With_NVIDIA_Nsight_Systems_Profiler_by_Chaim_Rand](resources/Optimizing_Data_Transfer_With_NVIDIA_Nsight_Systems_Profiler_by_Chaim_Rand.pdf)
- **Dec 16, 2025** - [YouTube: NeurIPS 2025 AI Systems Recap](https://www.youtube.com/watch?v=_8RoYhWhj1o), [YouTube: Advanced Data Curation and Visualization for LLMs with NVIDIA cuML](https://www.youtube.com/watch?v=MaUdaibdIxU) & Slides: [NVIDIA Data Curation](resources/nvidia_data_curation.pdf), and [YouTube: Automated Browser Use with Amazon AGI by Antje Barth](https://www.youtube.com/watch?v=bTRvIRfi-wk)
- **November 17, 2025** - [YouTube](https://youtu.be/2EWDG_Dxjs8) & Slides: [SpeedOfLight_Inference_Modular_Nov_17_2025_Abdul_Dakkak](resources/SpeedOfLight_Inference_Modular_Nov_17_2025_Abdul_Dakkak.pdf): Speed of Light Inference with NVIDIA and AMD GPUs and Modular Platform by Abdul Dakkak
- **October 20, 2025** - [YouTube](https://youtu.be/d3ZLodGTlAo): AI-Powered GPU Kernel Optimization + Distributed PyTorch with nbdistributed
- **September 15, 2025** – [YouTube](https://www.youtube.com/watch?v=eLnHXL1xXfM): Dynamic Adaptive RL inference kernel tuning deep dive.
- **August 18, 2025** – [YouTube](https://www.youtube.com/watch?v=SBPlOUww57I): Multi-GPU orchestration strategies and Nsight profiling case studies.
- **July 21, 2025** – [YouTube](https://youtu.be/jaiMotxv8ck): FlashMLA, ThunderMLA, and FlexDecoding kernel walkthroughs with live Nsight Compute demos.
- **June 16, 2025** – Slides: [High Performance Agentic AI Inference Systems](resources/High_Performance_Agentic_AI_Inference_Systems.pdf) covering disaggregated inference routing.
- **May 19, 2025** – [YouTube](https://youtu.be/F8jJwI9xHTE) & [PyTorch Data Loader Optimization](resources/PyTorch_Model_Optimization_Data_Loader.pdf): Torch.compile pipelines, data loader throughput tuning, and cross-architecture CUDA/ROCm kernels.
- **April 21, 2025** – [YouTube](https://youtu.be/XoZcY_fDUKA) & [AI Performance Engineering Meetup Slides](resources/AI_Performance_Engineering_Meetup_Apr_21_2025.pdf): End-to-end GPU performance playbook plus the [PyTorch Model Optimization](resources/PyTorch_Model_Optimization.pdf) workshop.

## Contributing

Contributions are welcome! See `CONTRIBUTING.md` for guidelines on code, documentation, and performance improvements.

## License

Apache 2.0 License – see `LICENSE` for details.
