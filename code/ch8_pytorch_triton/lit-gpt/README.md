<div align="center">
<img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/LitStableLM_Badge.png" alt="Lit-GPT" width="128"/>

# ⚡ Lit-GPT

Compare After Profiles
```
./compare_nsight.py  ./profile_results/nsight_report-1744318827-after-removing-excess-sync.sqlite ./profile_results/nsight_report-1744352327.nsys-rep
```

```
# Debian/Ubuntu
sudo apt update
sudo apt install nsight-systems-2025.2.1 nsight-compute

# RHEL/CentOS/Fedora
sudo yum makecache
sudo yum install nsight-systems-2025.2.1 nsight-compute
```

Starting in v2024.6+ (and in 2025.2.1), nsys can periodically sample your Python interpreter and record full Python backtraces. You need:

CPU sampling (-s cpu)

Python sampling (--python-sampling=true)

Sampling frequency (--python-sampling-frequency=<Hz>, default 1000 Hz)

(Optionally) CUDA backtraces so you see Python→CUDART→kernel call paths

```
nsys profile \
  -t cuda,nvtx,osrt,cudnn,cublas \
  -s cpu \
  --python-sampling=true \
  --python-sampling-frequency=1000 \
  --cudabacktrace=true \
  --cudabacktrace-threshold=0 \
  -o python_stack_report \
  python finetune/lora.py \
    --io.train_data_dir data/alpaca \
    --io.checkpoint_dir checkpoints/google/gemma-2b \
    --precision bf16-true \
    --train.micro_batch_size 2 \
    --profile_only nsys

```

Use backslashes (\) to split long commands across lines.

-o (or --output) names your .nsys-rep file; here it’s time-stamped.

-t cuda,nvtx,osrt,cudnn,cublas selects the timeline streams you want.

--gpu-metrics-device=all gathers hardware metrics from all GPUs.

--stats=true --force-overwrite=true (optional) forces CSV/SQLite exports on every run.

Place your Python script and its flags after the Nsight options.

Starting in v2024.6+ (and in 2025.2.1), nsys can periodically sample your Python interpreter and record full Python backtraces. You need:

CPU sampling (-s cpu)

Python sampling (--python-sampling=true)

Sampling frequency (--python-sampling-frequency=<Hz>, default 1000 Hz)

(Optionally) CUDA backtraces so you see Python→CUDART→kernel call paths


Here’s a cleaned-up, multi-line version of your Nsight Systems profiling command. Replace the path to `nsys` or any of the script arguments as needed.

```bash
# Basic LoRA fine-tuning profile
~/nsight-systems-2024.2.1/bin/nsys profile \
  -o ~/profile_results/nsight_report-$(date +%s) \
  -t cuda,nvtx,osrt,cudnn,cublas \
  --gpu-metrics-device=all \
  python finetune/lora.py \
    --io.train_data_dir data/alpaca \
    --io.checkpoint_dir checkpoints/google/gemma-2b \
    --precision bf16-true \
    --profile_only nsys
```

If you also want to sweep a micro-batch size and enable PyTorch-level tracing (autograd shapes + function calls), you can do:

```bash
~/nsight-systems-2024.2.1/bin/nsys profile \
  -o ~/profile_results/nsight_report-$(date +%s) \
  -t cuda,nvtx,osrt,cudnn,cublas \
  --gpu-metrics-device=all \
  --stats=true \
  --force-overwrite=true \
  python finetune/lora.py \
    --io.train_data_dir  data/alpaca \
    --io.checkpoint_dir checkpoints/google/gemma-2b \
    --precision         bf16-true \
    --train.micro_batch_size 2 \
    --profile_only      nsys \
    --pytorch=autograd-shapes-nvtx,functions-trace
```

**Key points**  
- Use backslashes (`\`) to split long commands across lines.  
- `-o` (or `--output`) names your `.nsys-rep` file; here it’s time-stamped.  
- `-t cuda,nvtx,osrt,cudnn,cublas` selects the timeline streams you want.  
- `--gpu-metrics-device=all` gathers hardware metrics from all GPUs.  
- `--stats=true --force-overwrite=true` (optional) forces CSV/SQLite exports on every run.  
- Place your Python script and its flags *after* the Nsight options.


```
ncu --mode=launch --target-processes all \
    -o ncu_report \
    python finetune/lora.py \
      --io.train_data_dir data/alpaca \
      --io.checkpoint_dir checkpoints/google/gemma-2b \
      --precision bf16-true
```




<!--
<p align="center">
  <a href="https://www.lightning.ai/">Lightning.ai</a> •
  <a href="https://lightning.ai/docs/pytorch/stable/">PyTorch Lightning</a> •
  <a href="https://lightning.ai/docs/fabric/stable/">Fabric</a>
</p>
-->

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytorch-lightning)
![cpu-tests](https://github.com/lightning-AI/lit-stablelm/actions/workflows/cpu-tests.yml/badge.svg) [![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lit-stablelm/blob/master/LICENSE) [![Discord](https://img.shields.io/discord/1077906959069626439?style=plastic)](https://discord.gg/VptPCZkGNa)

<img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/LitStableLM.gif" alt="Lit-GPT and pineapple pizza" width="500px"/>

</div>

&nbsp;

# ⚡ Lit-GPT

Hackable [implementation](lit_gpt/model.py) of state-of-the-art open-source large language models released under the **Apache 2.0 license**.

Supports the following popular model checkpoints:

| Model                                                                                | Model size                               | Reference                                                                                                                    |
|--------------------------------------------------------------------------------------|------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| [Code Llama](tutorials/download_code_llama.md) by Meta AI                            | 7B, 13B, 34B, 70B                        | [Rozière et al. 2023](https://arxiv.org/abs/2308.12950)                                                                      |
| [Dolly](tutorials/download_dolly.md) by Databricks                                   | 3B, 7B, 12B                              | [Conover et al. 2023](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) |
| [Falcon](tutorials/download_falcon.md) by TII UAE                                    | 7B, 40B, 180B                            | [TII 2023](https://falconllm.tii.ae)                                                                                         |
| [FreeWilly2](tutorials/download_freewilly_2.md) (Stable Beluga 2) by Stability AI    | 70B                                      | [Stability AI 2023](https://stability.ai/blog/stable-beluga-large-instruction-fine-tuned-models)                             |
| [Function Calling Llama 2](tutorials/download_function_calling_llama_2.md) by Trelis | 7B                                       | [Trelis et al. 2023](https://huggingface.co/Trelis/Llama-2-7b-chat-hf-function-calling-v2)                                   |
| [Gemma](tutorials/download_gemma.md) by Google                                       | 2B, 7B                                   | [Google Team, Google Deepmind](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf)                         |
| [Llama 2](tutorials/download_llama_2.md) by Meta AI                                  | 7B, 13B, 70B                             | [Touvron et al. 2023](https://arxiv.org/abs/2307.09288)                                                                      |
| [LongChat](tutorials/download_longchat.md) by LMSYS                                  | 7B, 13B                                  | [LongChat Team 2023](https://lmsys.org/blog/2023-06-29-longchat/)                                                            |
| [Mistral and Mixtral](tutorials/download_mistral.md) by Mistral AI                   | 7B                                       | [Mistral website](https://mistral.ai/)                                                                                       |
| [Nous-Hermes](https://huggingface.co/NousResearch/Nous-Hermes-13b) by NousResearch   | 7B, 13B, 70B                             | [Org page](https://huggingface.co/NousResearch)                                                                              |
| [OpenLLaMA](tutorials/download_openllama.md) by OpenLM Research                      | 3B, 7B, 13B                              | [Geng & Liu 2023](https://github.com/openlm-research/open_llama)                                                             |
| [Phi](tutorials/download_phi.md) by Microsoft Research                               | 1.3B, 2.7B                               | [Li et al. 2023](https://arxiv.org/abs/2309.05463)                                                                           |
| [Platypus](https://huggingface.co/garage-bAInd/Platypus-30B) by Lee at el.           | 7B, 13B, 70B                             | [Lee, Hunter, and Ruiz 2023](https://arxiv.org/abs/2308.07317)                                                               |
| [Pythia](tutorials/download_pythia.md) by EleutherAI                                 | {14,31,70,160,410}M, {1,1.4,2.8,6.9,12}B | [Biderman et al. 2023](https://arxiv.org/abs/2304.01373)                                                                     |
| [RedPajama-INCITE](tutorials/download_redpajama_incite.md) by Together               | 3B, 7B                                   | [Together 2023](https://together.ai/blog/redpajama-models-v1)                                                                |
| [StableCode](https://huggingface.co/stabilityai/stable-code-3b) by Stability AI      | 3B                                       | [Stability AI 2023](https://stability.ai/blog/stablecode-llm-generative-ai-coding)                                           |
| [StableLM](tutorials/download_stablelm.md) by Stability AI                           | 3B, 7B                                   | [Stability AI 2023](https://github.com/Stability-AI/StableLM)                                                                |
| [StableLM Zephyr](tutorials/download_stablelm.md) by Stability AI                    | 3B                                       | [Stability AI 2023](https://stability.ai/blog/stablecode-llm-generative-ai-coding)                                           |
| [TinyLlama](tutorials/download_tinyllama.md) by Zhang et al.                         | 1.1B                                     | [Zhang et al. 2023](https://github.com/jzhang38/TinyLlama)                                                                   |
| [Vicuna](tutorials/download_vicuna.md) by LMSYS                                      | 7B, 13B, 33B                             | [Li et al. 2023](https://lmsys.org/blog/2023-03-30-vicuna/)                                                                  |

This implementation extends on [Lit-LLaMA](https://github.com/lightning-AI/lit-llama) and [nanoGPT](https://github.com/karpathy/nanoGPT), and it's **powered by [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) ⚡**.

&nbsp;

---

**🏆 NeurIPS 2023 Large Language Model Efficiency Challenge: 1 LLM + 1 GPU + 1 Day**

The Lit-GPT repository was the official starter kit for the [NeurIPS 2023 LLM Efficiency Challenge](https://llm-efficiency-challenge.github.io), which is a competition focused on finetuning an existing non-instruction tuned LLM for 24 hours on a single GPU.

---

&nbsp;

## Lit-GPT design principles

This repository follows the main principle of **openness through clarity**.

**Lit-GPT** is:

- **Simple:** Single-file implementation without boilerplate.
- **Correct:** Numerically equivalent to the original model.
- **Optimized:** Runs fast on consumer hardware or at scale.
- **Open-source:** No strings attached.

Avoiding code duplication is **not** a goal. **Readability** and **hackability** are.

&nbsp;

## Get involved!

[Join our Discord](https://discord.gg/VptPCZkGNa) to build high-performance, truly open-source models for the common benefit of the community.

&nbsp;

## Setup

Clone the repo:

```bash
git clone https://github.com/Lightning-AI/lit-gpt
cd lit-gpt
```

Install with all dependencies (including CLI, quantization, tokenizers for all models, etc.):

```bash
pip install -r requirements-all.txt
```

&nbsp;

## Use the model

To generate text predictions, you need to download the model weights. **If you don't have them, check out our [guide](tutorials/download_stablelm.md).**

Run inference:

```bash
python generate/base.py --prompt "Hello, my name is"
```

This will run the 3B pretrained model and require ~7 GB of GPU memory using the `bfloat16` datatype.

[Full guide for generating samples from the model](tutorials/inference.md).

You can also chat with the model interactively:

```bash
python chat/base.py
```

&nbsp;

### Run large models on smaller consumer devices

We support 4-bit quantization (as in QLoRA), (bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq) and 8-bit quantization (bnb.int8) for inference by following [this guide](tutorials/quantize.md).

&nbsp;

## Finetune the model

We provide a simple training scripts (`finetune/adapter.py`, `finetune/adapter_v2.py`, and `finetune/lora.py`) that instruction-tunes a pretrained model on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset.

1. Download the data and generate an instruction tuning dataset:

```bash
python scripts/prepare_alpaca.py
```

2. Run the finetuning script

For example, you can either use

Adapter ([Zhang et al. 2023](https://arxiv.org/abs/2303.16199)):

```bash
python finetune/adapter.py
```

or Adapter v2 ([Gao et al. 2023](https://arxiv.org/abs/2304.15010)):

```bash
python finetune/adapter_v2.py
```

or LoRA ([Hu et al. 2021](https://arxiv.org/abs/2106.09685)):

```bash
python finetune/lora.py
```

(Please see the [tutorials/finetune_adapter](tutorials/finetune_adapter.md) for details on the differences between the two adapter methods.)

The finetuning requires at least one GPU with ~12 GB memory (RTX 3060).

It is expected that you have downloaded the pretrained weights as described above.
More details about each finetuning method and how you can apply it to your own data can be found in our technical how-to guides.

&nbsp;

### Finetuning how-to guides

These technical tutorials illustrate how to run the finetuning code.

- [Finetune with Adapters](tutorials/finetune_adapter.md)
- [Finetune with LoRA or QLoRA](tutorials/finetune_lora.md)

&nbsp;

### Understanding finetuning -- conceptual tutorials

Looking for conceptual tutorials and explanations? We have some additional articles below:

- [Understanding Parameter-Efficient Finetuning of Large Language Models: From Prefix Tuning to LLaMA-Adapters](https://lightning.ai/pages/community/article/understanding-llama-adapters/)

- [Parameter-Efficient LLM Finetuning With Low-Rank Adaptation (LoRA)](https://lightning.ai/pages/community/tutorial/lora-llm/)

&nbsp;

## Pretraining

We provide simple training scripts based on Fabric if you want to venture into pretraining. Conversion scripts for our optimized streaming `PackedDataset` are included.

Follow this guide to start pretraining on

- [RedPajama, a reproduction of LLaMA's training set](tutorials/pretrain_redpajama.md)
- [OpenWeb Text, a reproduction of GPT-2's dataset](tutorials/pretrain_openwebtext.md)



&nbsp;


## Supported datasets

Lit-GPT includes a variety of dataset preparation scripts for finetuning and pretraining. Additional information about the datasets and dataset preparation is provided in the [Preparing Datasets](tutorials/prepare_dataset.md) tutorial.


&nbsp;

## XLA

Lightning AI has partnered with Google to add first-class support for [Cloud TPUs](https://cloud.google.com/tpu) in [Lightning’s frameworks](https://github.com/Lightning-AI/lightning) and Lit-GPT,
helping democratize AI for millions of developers and researchers worldwide.

Using TPUs with Lightning is as straightforward as changing one line of code.

We provide scripts fully optimized for TPUs in the [XLA directory](xla)

&nbsp;

## Get involved!

We are on a quest towards fully open source AI.

<img align="right" src="https://pl-public-data.s3.amazonaws.com/assets_lightning/LitStableLM_Illustration.png" alt="Lit-GPT" width="128"/>

Join us and start contributing, especially on the following areas:

- [ ] [Pretraining](https://github.com/Lightning-AI/lit-gpt/labels/pre-training)
- [ ] [Fine-tuning](https://github.com/Lightning-AI/lit-gpt/labels/fine-tuning)
- [ ] [Quantization](https://github.com/Lightning-AI/lit-gpt/labels/quantization)
- [ ] [Sparsification](https://github.com/Lightning-AI/lit-gpt/labels/sparsification)

We welcome all individual contributors, regardless of their level of experience or hardware. Your contributions are valuable, and we are excited to see what you can accomplish in this collaborative and supportive environment.

Unsure about contributing? Check out our [How to Contribute to Lit-GPT and Lit-LLaMA
](https://lightning.ai/pages/community/tutorial/how-to-contribute-to-litgpt/) guide.

Don't forget to [join our Discord](https://discord.gg/VptPCZkGNa)!

&nbsp;

## Acknowledgements

- [@karpathy](https://github.com/karpathy) for [nanoGPT](https://github.com/karpathy/nanoGPT)
- [@EleutherAI](https://github.com/EleutherAI) for [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) and the [Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [@TimDettmers](https://github.com/TimDettmers) for [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [@Microsoft](https://github.com/microsoft) for [LoRA](https://github.com/microsoft/LoRA)
- [@tridao](https://github.com/tridao) for [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)

&nbsp;

## Citation

If you use Lit-GPT in your research, please cite the following work:

```bibtex
@misc{lit-gpt-2023,
  author       = {Lightning AI},
  title        = {Lit-GPT},
  howpublished = {\url{https://github.com/Lightning-AI/lit-gpt}},
  year         = {2023},
}
```

&nbsp;

## License

Lit-GPT is released under the [Apache 2.0](https://github.com/Lightning-AI/lit-gpt/blob/main/LICENSE) license.
