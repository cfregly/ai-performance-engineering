# Fastest GPU kernels, written from scratch.

## H100 BF16 Matrix Multiplication

Matrix multiplication of square bf16 matrices, accumulated in fp32.

```
N=4096
Kernel: 763 TFLOPs
cuBLAS: 716 TFLOPs

N=8192
Kernel: 808 TFLOPs
cuBLAS: 795 TFLOPs
```

Explanation in https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog

##### To run:
```
make matmul && out/matmul
```
Example kernels are in [`h100/matmul/`](h100/matmul), and orchestration is
in [`h100/matmul.cu`](h100/matmul.cu).

## GB300 NVFP4 Matrix Multiplication

The NVFP4 example contains ten complete kernel snapshots, a shared correctness
and benchmark runner, and a cuBLASLt baseline.

```bash
make nvfp4
./out/nvfp4 8192 8192 8192 --rounds 5 --cooldown-sec 8

make nvfp4-ladder
./out/nvfp4-r0 8192 8192 8192 --rounds 5 --cooldown-sec 8
./out/nvfp4-r9 8192 8192 8192 --rounds 5 --cooldown-sec 8
```

This requires a GB300/B300 and CUDA 13.1. See
[`gb300/nvfp4/README.md`](gb300/nvfp4/README.md) for every optimization
rung, scheduler controls, correctness-only runs, and benchmark modes.

## H100 Sum Reduction

We compute sum of 2^30 elements.

##### To run:
```
make sum && out/sum
```

```
Kernel: 3240.11 GB/s
cub Library: 3193 GB/s
```

The kernel is in [`h100/sum.cu`](h100/sum.cu).
