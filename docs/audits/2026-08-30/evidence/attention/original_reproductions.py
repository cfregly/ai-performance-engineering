"""Reproduce host-visible original defects; do not emulate GPU qualification."""
from __future__ import annotations
import ast
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Optional

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'code'))
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from labs.flashattention4.flashattention4_common import build_dense_attention_mask, count_nonmasked_attention_elements
from labs.cudnn_sdpa_bench.baseline_flash_sdp import _sdpa_context, _select_backend
BASE='b57e4c6a9e261c09ac09208705d040c81b03d35e'

def original(path):
    return subprocess.check_output(['git','show',f'{BASE}:{path}'],cwd=ROOT,text=True)

def functions(path,names,environment):
    tree=ast.parse(original(path))
    selected=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert len(selected)==len(names)
    exec(compile(ast.Module(body=selected,type_ignores=[]),f'{BASE}:{path}','exec'),environment)
    return environment

def flags(context):
    with context:
        return {name:getattr(torch.backends.cuda,name+'_sdp_enabled')() for name in ('cudnn','flash','math','mem_efficient')}

backend=functions('code/labs/cudnn_sdpa_bench/baseline_flash_sdp.py',
 ['_resolve_backend','_select_backend','_sdpa_context'],dict(torch=torch,Optional=Optional,
 SDPBackend=SDPBackend,sdpa_kernel=sdpa_kernel,nullcontext=nullcontext,
 _BACKEND_CHOICES=('auto','cudnn','flash','math')))
flops=functions('code/labs/flashattention4/flashattention4_common.py',['count_nonmasked_attention_elements'],{})
result=dict(base=BASE,cuda_available=torch.cuda.is_available(),cuda_compiled=False,cuda_executed=False,
 W016_evidence_kind='CPU mathematical translation of old per-thread softmax, not execution of CUDA binary',
 W066=dict(old_requested_cudnn_selection_on_CPU=backend['_select_backend']('cudnn'),
           new_requested_cudnn_selection_on_CPU=_select_backend('cudnn'),
           old_actual_torch_context_flags=flags(backend['_sdpa_context']('cudnn')),
           new_actual_torch_context_flags=flags(_sdpa_context('cudnn'))))
q=torch.tensor([1.,2.]);k=torch.tensor([[1.,-1.],[-1.,1.]]);v=torch.tensor([[10.,0.],[0.,10.]])
weights=(q*k/(2**.5)).exp()
old=(weights*v).sum(0)/weights.sum(0)
reference=F.scaled_dot_product_attention(q[None,None,None,:],k[None,None,:,:],v[None,None,:,:]).flatten()
result['W016']=dict(old_per_dimension_softmax=old.tolist(),true_SDPA=reference.tolist(),
 max_abs_difference=float((old-reference).abs().max()),
 fp16_unshifted_exp_overflows=bool(torch.isinf(torch.exp(torch.tensor([30.*30./2**.5],dtype=torch.float16))).item()))
result['W072']={}
for mode in ('alibi','softcap'):
 seq=2048
 mask=build_dense_attention_mask(mode,seq_len=seq,window_size=0,device=torch.device('cpu'))
 old_count=flops['count_nonmasked_attention_elements'](mode,q_seq_len=seq,kv_seq_len=seq)
 actual=int(mask.count_nonzero())
 new_count=count_nonmasked_attention_elements(mode,q_seq_len=seq,kv_seq_len=seq)
 assert new_count==actual and old_count!=actual
 result['W072'][mode]=dict(old_count=old_count,actual_mask_count=actual,new_count=new_count,old_numerator_overcount=old_count/actual)
with torch.random.fork_rng():
 torch.manual_seed(740)
 q,k,v=[torch.randn(2,3,7,5) for _ in range(3)]
 expected=F.scaled_dot_product_attention(q,k,v)
 misinterpreted=F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2)
 result['W074']=dict(evidence_kind='Real CPU SDPA under the two axis interpretations; no CuTe kernel executed',
 public_BHSD=list(q.shape),correct_CuTe_BSHD=list(q.transpose(1,2).shape),
 wrong_sequence_length=3,right_sequence_length=7,max_abs_output_difference=float((expected-misinterpreted).abs().max()))
# Exact originals retained as hashes and small quotes for source-only race/claim review.
source_checks={
 'W022':('code/core/common/async_input_pipeline.py','batch_gpu = batch_cpu.to(self.device, non_blocking=self.cfg.non_blocking)'),
 'W039':('code/labs/persistent_decode/persistent_decode_ext.cu','return smem[0];'),
 'W088':('code/labs/persistent_decode/optimized_persistent_decode_triton.py','self._num_items = min(self.batch, self.num_programs)'),
 'W089':('code/labs/persistent_decode/paged_kv_offload_common.py','target[..., :slice_len, :].copy_(self._host_page_view(start, slice_len))'),
 'W090':('code/labs/persistent_decode/tma_extension.py','cuda::memcpy_async(&smem[threadIdx.x], &src[global_idx], sizeof(float), pipe);'),
 'W118':('code/labs/decode_optimization/baseline_decode_warp_specialized.py','Baseline for warp-specialized Triton decode')}
result['source_only_original_checks']={}
for finding,(path,quote) in source_checks.items():
 text=original(path)
 assert quote in text,(finding,path)
 result['source_only_original_checks'][finding]=dict(path=path,sha256=hashlib.sha256(text.encode()).hexdigest(),quote=quote,
 gpu_race_reproduced=False)
result['W071']=dict(source='HANDOFF.md:47',classification='Historical 2026-08-17 recorded result; not a fresh GPU run',
 baseline_ms=3.429079,optimized_ms=3.572818,ratio=3.429079/3.572818,gate=1.05,provider_fix_requalified=False)
result['W073']=dict(classification='Original source has zero-filled tail loads and no negative-infinity score mask; CUDA gate pending',
 public_API_tail_and_causal_affected=True,shipped_1024_noncausal_pair_affected_by_these_two_faults=False)
print(json.dumps(result,indent=2))
