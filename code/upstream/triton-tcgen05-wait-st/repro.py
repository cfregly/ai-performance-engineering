# DRAFT — internal review pending
#
# Status PROBE (not a filing reproducer — the underlying release bug is
# already filed upstream, see STATUS.md): does the sm_103 Triton
# LLVM-selection abort
#
#   LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.tcgen05.wait.st
#
# still reproduce on this stack (NGC 26.05: torch 2.12.0a0, Triton 3.7,
# GB300 sm_103)?
#
# Verified answer (GB300, 2026-06-11): NO on vanilla Triton 3.7.0 —
# probes A/B/C below are all clean. The abort reproduces ONLY when the
# Triton target arch is mis-set to plain `sm_103` instead of `sm_103a`
# (probe D, rc=134 with the exact error string): Triton plans tcgen05
# codegen from the compute capability (10.3), but the arch-conditional
# tcgen05 intrinsics are only selectable for the `sm_103a` target, and
# LLVM dies with an UNCATCHABLE fatal instead of a Python exception.
# That mis-targeting is exactly what an over-broad arch-"de-suffixing"
# integration patch does (this campaign's aborts came from one that
# rewrote every `sm_*a` except `sm_100a`).
#
# The abort is an uncatchable SIGABRT inside Triton's make_ptx, so every
# probe compile runs in a CHILD process; the parent classifies the child's
# exit code + stderr. Probes:
#   A. plain tl.dot matmul kernel JIT (control — clean),
#   B. the same kernel with a dead `num_warps: tl.constexpr` parameter
#      (suspected campaign trigger; clean standalone),
#   C. torch.compile(mode="max-autotune") on a small attention+MLP block
#      (suspected campaign trigger; clean standalone),
#   D. probe A with `sm_arch_from_capability` patched to return `sm_103`
#      (no 'a') — ABORTS with the tcgen05.wait.st LLVM fatal (mechanism).
#
# Outcome semantics: a child exiting 134/-6 with "Cannot select" +
# "tcgen05" in stderr == the abort reproduced for that probe.
#
# Tested environment: NVIDIA GB300 (sm_103), CUDA 13.2,
# torch 2.12.0a0 (NGC 26.05). No external deps.

import os
import subprocess
import sys
import tempfile

CHILD_COMMON = r"""
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    DEAD_NUM_WARPS_PARAM
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        mask_a = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)
        mask_b = (k + offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_c)

def run(extra_kwargs):
    M = N = K = 1024
    bm = bn = 128
    bk = 64
    a = torch.randn((M, K), dtype=torch.float16, device="cuda")
    b = torch.randn((K, N), dtype=torch.float16, device="cuda")
    c = torch.empty((M, N), dtype=torch.float16, device="cuda")
    grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=8, **extra_kwargs)
    torch.cuda.synchronize()
    print("CHILD_OK")
"""

CHILD_A = CHILD_COMMON.replace(
    "    DEAD_NUM_WARPS_PARAM\n", "") + "\nrun({})\n"

# Probe D: mis-target plain `sm_103` (no 'a' suffix) the way an over-broad
# arch-de-suffixing integration patch would, then JIT the SAME plain
# matmul as probe A. Triton still plans tcgen05 codegen (keyed on compute
# capability 10.3) but the arch-conditional intrinsics are not selectable
# for `sm_103` -> uncatchable `LLVM ERROR: Cannot select: intrinsic
# %llvm.nvvm.tcgen05.wait.st`.
ARCH_STRIP_PRELUDE = r"""
import triton.backends.nvidia.compiler as _tc
_orig_sm_arch = _tc.sm_arch_from_capability
def _desuffixed_sm_arch(capability, _o=_orig_sm_arch):
    arch = _o(capability)
    if arch.endswith("a") and not arch.endswith("100a"):
        arch = arch[:-1]
    return arch
_tc.sm_arch_from_capability = _desuffixed_sm_arch
"""

CHILD_D = CHILD_A.replace(
    "import triton.language as tl",
    "import triton.language as tl" + ARCH_STRIP_PRELUDE)

# Trigger 1: a dead `num_warps: tl.constexpr` kernel parameter. The launch
# kwarg num_warps=8 then binds to the KERNEL parameter, not the launch
# meta-parameter (the campaign's occupancy_tuning lab hit this).
CHILD_B = CHILD_COMMON.replace(
    "    DEAD_NUM_WARPS_PARAM\n",
    "    num_warps: tl.constexpr,\n") + "\nrun({})\n"

CHILD_C = r"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, d=512, heads=8):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, 4 * d, bias=False)
        self.down = nn.Linear(4 * d, d, bias=False)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        b, s, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q = q.view(b, s, self.heads, -1).transpose(1, 2)
        k = k.view(b, s, self.heads, -1).transpose(1, 2)
        v = v.view(b, s, self.heads, -1).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(b, s, d)
        x = x + self.proj(o)
        return x + self.down(F.gelu(self.up(self.norm2(x))))

m = Block().cuda().to(torch.bfloat16).eval()
x = torch.randn(2, 1024, 512, device="cuda", dtype=torch.bfloat16)
cm = torch.compile(m, mode="max-autotune", fullgraph=True, dynamic=False)
with torch.no_grad():
    cm(x)
torch.cuda.synchronize()
print("CHILD_OK")
"""

PROBES = [
    ("A: plain tl.dot matmul JIT (control)", CHILD_A, 300, "clean"),
    ("B: dead num_warps:tl.constexpr param", CHILD_B, 300, "clean"),
    ("C: torch.compile max-autotune attn+MLP block", CHILD_C, 1200, "clean"),
    ("D: probe A mis-targeted to sm_103 (no 'a')", CHILD_D, 300, "ABORT"),
]


def classify(rc, err):
    aborted = ("Cannot select" in err and "tcgen05" in err)
    if aborted:
        return "ABORT-REPRODUCED (LLVM cannot select tcgen05 intrinsic)"
    if rc == 0:
        return "clean"
    return f"other failure (rc={rc})"


def main():
    import torch
    import triton
    cap = torch.cuda.get_device_capability()
    print(f"torch {torch.__version__}, triton {triton.__version__}, "
          f"device {torch.cuda.get_device_name()}, sm_{cap[0]}{cap[1]}\n")

    results = {}
    tmpdir = tempfile.mkdtemp(prefix="tcgen05_probe_")
    for i, (label, code, timeout, expect) in enumerate(PROBES):
        # Triton @jit requires kernels defined in a real .py file
        # (inspect.getsourcelines), so `python -c` is not usable here.
        path = os.path.join(tmpdir, f"child_{i}.py")
        with open(path, "w") as fh:
            fh.write(code)
        try:
            res = subprocess.run([sys.executable, path],
                                 capture_output=True, text=True,
                                 timeout=timeout)
            rc, err = res.returncode, res.stderr
        except subprocess.TimeoutExpired:
            rc, err = None, "(timeout)"
        verdict = classify(rc, err)
        results[label] = (verdict, expect)
        print(f"probe {label}\n    -> {verdict} (expected: {expect})")
        for line in err.splitlines():
            if "LLVM ERROR" in line or "Cannot select" in line:
                print(f"    stderr: {line.strip()}")

    vanilla_clean = all(v == "clean" for lbl, (v, e) in results.items()
                        if e == "clean")
    mechanism = any(v.startswith("ABORT") for lbl, (v, e) in results.items()
                    if e == "ABORT")
    print(f"\nPROBE RESULT: vanilla Triton {'CLEAN' if vanilla_clean else 'ABORTS'} "
          f"on sm_103a; sm_103 (no-'a') mis-target "
          f"{'REPRODUCES the abort' if mechanism else 'did NOT abort'} "
          f"(see STATUS.md for the upstream filing state)")


if __name__ == "__main__":
    main()
