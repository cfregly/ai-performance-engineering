"""Collect actual CUDA MoE layer errors; no policy or acceptance is generated.

Run only after the root task has authorized hardware custody. Every attempted
run writes a new result, including unavailable CUDA, partial records or failures.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'code'))
import torch
from labs.moe_cuda_ptx.moe_cuda_ptx_common import (
    MoECudaPtxWorkload, build_state, measure_layer_output_errors,
    reference_layer_forward, run_layer_baseline, run_layer_cuda, snapshot_layer_inputs,
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=314159)
    parser.add_argument('--num-tokens',type=int,default=32768)
    parser.add_argument('--hidden-dim',type=int,default=7168)
    parser.add_argument('--expert-ffn-dim',type=int,default=2048)
    parser.add_argument('--num-experts',type=int,default=8)
    parser.add_argument('--histogram',choices=('balanced','skewed'),default='balanced')
    parser.add_argument('--dtype',choices=('bf16','fp16'),default='bf16')
    parser.add_argument('--capacity-factor',type=float,default=1.25)
    args=parser.parse_args()
    # Reserve the output before attempting CUDA work; never overwrite an attempt.
    with args.output.open('x') as destination:
        report=dict(schema_version=1,status='HOLD',accepted=False,gpu_executed=False,
            captured_at=datetime.now(timezone.utc).isoformat(),config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
            source_sha256=hashlib.sha256((ROOT/'code/labs/moe_cuda_ptx/moe_cuda_ptx_common.py').read_bytes()).hexdigest(),
            records=[],performance_measured=False)
        exit_code=3
        try:
            workload=MoECudaPtxWorkload(num_tokens=args.num_tokens,num_experts=args.num_experts,
                hidden_dim=args.hidden_dim,expert_ffn_dim=args.expert_ffn_dim,
                capacity_factor=args.capacity_factor,histogram=args.histogram,
                dtype=torch.bfloat16 if args.dtype=='bf16' else torch.float16)
            workload.validate()
            if not torch.cuda.is_available():
                report['reason']='Actual CUDA unavailable; no numerical evidence collected'
            else:
                report.update(device=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),gpu_executed=True)
                torch.manual_seed(args.seed)
                state=build_state(workload,torch.device('cuda'))
                snapshot=snapshot_layer_inputs(state)
                base=state.x.clone()
                with torch.inference_mode():
                    for amplitude in (1,4,16):
                        state.x.copy_(base*amplitude)
                        snapshot['x']=state.x.detach().to('cpu',copy=True)
                        expected=reference_layer_forward(snapshot,torch.device('cuda'))
                        for backend,fn in (('baseline',run_layer_baseline),('cuda_bmm',run_layer_cuda)):
                            actual=fn(state,workload)
                            torch.cuda.synchronize()
                            report['records'].append(dict(backend=backend,input_amplitude=amplitude,
                                **measure_layer_output_errors(actual,expected)))
                report.update(status='CALIBRATION_ONLY_NOT_ACCEPTED',reason='One shape/seed/routing and three amplitudes; no threshold inferred')
                exit_code=0
        except Exception as exc:
            report.update(status='FAILED_NOT_ACCEPTED',exception_type=type(exc).__name__,reason=str(exc))
            exit_code=2
        finally:
            report['finished_at']=datetime.now(timezone.utc).isoformat()
            json.dump(report,destination,indent=2);destination.write('\n');destination.flush()
    print(json.dumps({k:report.get(k) for k in ('status','accepted','gpu_executed','reason')}))
    return exit_code

if __name__=='__main__':
    raise SystemExit(main())
