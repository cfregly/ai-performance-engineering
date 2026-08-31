import ast,json,pathlib,subprocess,tempfile
import torch
from core.harness import validity_checks as vc
from core.benchmark import verify_runner as vr
from tests.protection_test_utils import TensorWork,make_runner,preserve_rng_state
REPO=pathlib.Path('/Users/admin/dev/ai-perf/ai-performance-engineering')
BASE='b57e4c6a9e261c09ac09208705d040c81b03d35e'
def original_function(relative,name,namespace):
 text=subprocess.check_output(['git','show',BASE+':'+relative],cwd=REPO,text=True)
 tree=ast.parse(text)
 node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
 ns=dict(namespace)
 exec(compile(ast.Module(body=[node],type_ignores=[]),'<original '+relative+'>','exec'),ns)
 return ns[name]
old_compile=original_function('code/core/harness/validity_checks.py','get_compile_state',vars(vc))
old_jitter=original_function('code/core/benchmark/verify_runner.py','_run_jitter_check',vars(vr))
vc.clear_compile_cache()
compile_evidence=[]
fn=torch.compile(lambda value,flag:value*2 if flag else value+1,backend='eager',dynamic=False)
value=torch.arange(8,dtype=torch.float32)
for stage,args in [('before',None),('compile',(value,True)),('cache_hit',(value+1,True)),('guard_recompile',(value,False))]:
 if args is not None: fn(*args)
 compile_evidence.append({'stage':stage,'original':old_compile(),'current':vc.get_compile_state(),'actual_counters':{str(k):dict(v) for k,v in torch._dynamo.utils.counters.items()}})
vc.clear_compile_cache()
class Preallocated(TensorWork):
 def __init__(self,behavior):
  super().__init__();self.behavior=behavior;self.calls=0
 def setup(self):
  super().setup();self.storage=torch.empty(8,4);self.output=self.storage.T
 def benchmark_fn(self):
  self.calls+=1
  if self.behavior!='cached' or self.calls==1:self.output.copy_(self.input*2)
 def get_verify_output(self):
  if self.behavior=='none' and self.calls>1:return None
  if self.behavior in {'nested','nested_cached'}:return {'output':[self.output]}
  return self.output
 def capture_verification_payload(self):
  if self.behavior=='capture_error':raise RuntimeError('injected refresh error')
jitter=[]
with tempfile.TemporaryDirectory(prefix='aisp-protection-mechanisms-') as temporary, preserve_rng_state():
 runner=make_runner(pathlib.Path(temporary))
 for behavior in ['real','cached','nested','none','capture_error']:
  case={'behavior':behavior}
  for version,method in [('original',old_jitter),('current',vr.VerifyRunner._run_jitter_check)]:
   torch.manual_seed(42)
   work=Preallocated(behavior);work.setup();work.benchmark_fn();before=work.input.clone()
   result=method(runner,work,work.get_input_signature(),vr.VerifyConfig())
   case[version]={'passed':result[0],'reason':result[1],'input_restored':torch.equal(before,work.input)}
  jitter.append(case)
record={'baseline':BASE,'scope':'real CPU Dynamo and verification behavior; not CUDA performance evidence','compile':compile_evidence,'jitter':jitter}
output=REPO/'docs/audits/2026-08-30/evidence/validation/protection-coverage-original-mechanisms.json'
with output.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
print(json.dumps(record,indent=2))
