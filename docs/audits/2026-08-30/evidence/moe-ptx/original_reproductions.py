"""Execute actual original Torch CPU layer/capture paths from the reviewed base."""
import hashlib,json,subprocess,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'code'))
import torch
BASE='b57e4c6a9e261c09ac09208705d040c81b03d35e'
PATH='code/labs/moe_cuda_ptx/moe_cuda_ptx_common.py'
source=subprocess.check_output(['git','show',BASE+':'+PATH],cwd=ROOT,text=True)
module=types.ModuleType('audit_original_moe_ptx');module.__file__=str(ROOT/PATH)
sys.modules[module.__name__]=module
exec(compile(source,BASE+':'+PATH,'exec'),module.__dict__)
report=dict(base=BASE,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
 evidence_kind='Actual Torch CPU operations and original benchmark/capture methods; no GPU execution',rows=[])
for dtype in (torch.bfloat16,torch.float16):
 bench=module.MoECudaPtxBenchmark(target='moe_layer',backend='cuda',label='original-cpu-repro')
 bench.device=torch.device('cpu')
 bench.workload=module.MoECudaPtxWorkload(num_tokens=65,num_experts=4,hidden_dim=48,
     expert_ffn_dim=32,capacity_factor=1.5,dtype=dtype)
 bench.setup();bench.benchmark_fn();bench.capture_verification_payload()
 original=bench.outputs.clone();payload=bench.get_verify_output().clone()
 bench.outputs[-1,-1]+=1;bench.capture_verification_payload()
 ignored=bool(torch.equal(payload,bench.get_verify_output()))
 bench.outputs.zero_();bench.capture_verification_payload()
 report['rows'].append(dict(dtype=str(dtype),logical_shape=list(original.shape),payload_shape=list(payload.shape),
 max_abs_reference=float(original.abs().max()),last_element_corruption_ignored=ignored,
 all_zero_passes_original_tolerance=bool(torch.allclose(torch.zeros_like(original),original,rtol=.05,atol=.2)),
 all_zero_passes_suggested_tolerance=bool(torch.allclose(torch.zeros_like(original),original,rtol=.02,atol=.02)),
 zero_validate_result=bench.validate_result()))
 bench.teardown()
print(json.dumps(report,indent=2))
