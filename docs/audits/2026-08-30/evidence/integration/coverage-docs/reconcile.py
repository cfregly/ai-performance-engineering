"""Correct factual threat-inventory claims without changing operational rules."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[6]
out=Path(__file__).resolve().parent
rows={}
def add(names,mechanism,status='Scoped check'):
 for name in names.split(';'):rows[name.strip()]=(mechanism,status)
add('Invalid Ground Truth','Selected-reference caching/comparison; does not validate dataset labels')
add('Uninitialized Memory','No uninitialized-memory provenance detector','Unsupported')
add('Train/Test Overlap;Test Data Leakage;Missing Holdout Sets','No dataset provenance, leakage or holdout enforcement','Unsupported')
add('CPU Spillover','Wall/CUDA timing cross-check; no per-operation CPU placement detector','Scoped timing check')
add('Background Thread;Priority Elevation;Background Process Noise','Subprocess execution does not prohibit threads, lock priority or isolate host processes','Unsupported policy')
add('Fragmentation Effects','Allocator cleanup/memory-growth diagnostics; no fragmentation parity')
add('Page Fault Timing;Unified Memory Faults','No page-fault or managed-memory event detector','Unsupported')
add('Swap Interference','Detect enabled swap; does not disable swap or lock memory','Environment gate')
add('Host Callback Escape;Workspace Pre-compute;Persistent Kernel;Undeclared Multi-GPU;Context Switch Overhead;Driver Overhead;Cooperative Launch Abuse;Dynamic Parallelism Hidden','No corresponding execution/provenance inspector','Unsupported')
add('Mode Inconsistency;Inductor Asymmetry;Autotuning Variance','No general compiler-mode/backend parity or autotuning-variance guard','Unsupported')
add('Guard Failure Hidden','Process-cumulative Dynamo graph counts with source metadata; not resident cache or compile parity')
add('Topology Mismatch','Compare declared topology; no ring/tree algorithm field','Scoped signature check')
add('Barrier Timing;Gradient Bucketing Mismatch;Async Gradient Timing','No barrier-timing, gradient-bucket parity or async-gradient completion detector','Unsupported')
add('Pipeline Bubble Hiding','Declared rank workload and timing cross-checks; no bubble classifier')
add('Device Mismatch','Environment inventory lacks expected/observed GPU identity parity; separate Tier-1 preflight attests target','Unsupported generic identity policy')
add('Frequency Boost','Application-clock lock; actual observed-NVML integration requires GPU','Implemented; runtime pending')
add('Memory Overcommit','Memory-growth diagnostic; no overcommit policy','Scoped diagnostic')
add('NUMA Inconsistency','Advisory affinity diagnostics; no pinning or cross-node rejection','Advisory')
add('CPU Governor Mismatch','Strict environment gate rejects non-performance governor; does not set or lock it','Environment gate')
add('Thermal Throttling','NVML temperature/clock-drop/throttling diagnostics','Scoped; hardware pending')
add('Power Limit Difference','Power draw captured; configured power-limit parity absent','Unsupported')
add('Driver Version Mismatch;Library Version Mismatch','Available RunManifest provenance; no cross-run version lock','Unsupported version parity')
add('Virtualization Overhead','Runtime virtualization notice is advisory; separate repository policy still applies','Advisory')
add('Cherry-picking;Outlier Injection;Variance Gaming;Percentile Selection','Preserve supplied samples/statistics; no upstream omission/injection/selection detector','Scoped reporting')
add('Insufficient Samples','Duration-driven adaptive iterations with maximum; no power/variance guarantee','Scoped timing')
add('Self-Modifying Tests','Config-value immutability; no test-source immutability','Unsupported source policy')
add('Benchmark Overfitting;Benchmark Memorization','Fresh-input/jitter cached-output checks; no general dataset-overfitting detector')
add('Timer Granularity','Adaptive measurement duration; no timer-resolution guarantee','Scoped; CUDA pending')
add('Warmup Bleed;Warmup Computation','L2 clearing after warmup; not general warmup-work detector','Scoped; eviction pending')
add('Profiler Overhead','Harness timing disables its profiler; no nested-profiler rejection')
incident={
 'Invalid Ground Truth':rows['Invalid Ground Truth'],
 'Benchmark Overfitting':rows['Benchmark Overfitting'],
 'Data Contamination':rows['Test Data Leakage'],
 'Cherry-picking':rows['Cherry-picking'],
 'Train/Test Overlap':rows['Train/Test Overlap'],
 'Missing Holdout Sets':rows['Missing Holdout Sets'],
 'Reproducibility':rows['Driver Version Mismatch'],
 'Evaluation Integrity':('Contract/config checks; no test-source immutability','Scoped checks'),
}
notice='This threat inventory includes scoped checks and unsupported policies. It is not a guarantee that every listed attack is detected. Unsupported cases and missing CUDA runs are explicit skips, never passing coverage. See the [protection-test disposition](../docs/audits/2026-08-30/evidence/validation/protection-coverage-receipt.json) and [clock-lock follow-up](../docs/audits/2026-08-30/evidence/validation/clock-lock-followup/receipt.json). Unmapped rows below retain an advertised mechanism but remain individually unaudited; no independent coverage or runtime qualification is asserted.'
report=[]
for rel in ('code/core/scripts/refresh_readmes.py','code/AGENTS.md'):
 p=root/rel;s=p.read_text();original=s
 backup=out/(p.name+'.before.txt')
 if not backup.exists():backup.write_text(s)
 # Only rows in the two coverage inventories are changed, never incident facts or rules.
 lines=[];count=0
 for line in s.splitlines():
  indent=line[:len(line)-len(line.lstrip())]
  cells=[c.strip() for c in line.strip().split('|')[1:-1]]
  if len(cells)==6 and cells[4] in ('OK','✅'):
   key=cells[1].strip('*')
   cells[3],cells[4]=rows.get(key,(cells[3],'Inventory; not re-audited'))
   line=indent+'| '+' | '.join(cells)+' |';count+=1
  elif len(cells)==4 and cells[3] in ('OK','✅'):
   key=cells[0].strip('*')
   cells[2],cells[3]=incident.get(key,(cells[2],'Inventory; not re-audited'))
   line=indent+'| '+' | '.join(cells)+' |'
  lines.append(line)
 s='\n'.join(lines)+'\n'
 s=s.replace('Note: All 95 validity issues are protected by the harness.',notice)
 s=s.replace('**✅ All 95 validity issues are now protected by our harness**',notice)
 s=s.replace('Below is the reference list of validity issues we explicitly protect against, plus real-world incidents that','Below is a threat inventory with scoped and unsupported checks, plus real-world incidents that')
 s=s.replace('Total: 11 categories, 95 validity issues - all protected by the harness.',f'Total: 11 categories, {count} inventory rows. These are threats and advertised mechanisms, not a count of verified protections.')
 s=s.replace('**Total: 11 categories, 95 validity issues — ✅ ALL PROTECTED by our harness (20 linked to real-world incidents with citations)**',f'**Total: 11 categories, {count} inventory rows; no blanket coverage claim.**')
 s=s.replace('All 95 validity protections are implemented in the following modules:','Implementation entrypoints are listed below; this does not establish that every inventory policy is implemented or verified:')
 s=s.replace('its 95 validity protections','its implemented validity checks (see the scoped inventory below)')
 # Keep the explicit bare-metal operational restriction; correct only the factual checker description.
 s=s.replace('**Virtualization Note:** `validate_environment()` treats virtualization (hypervisor present) as invalid. Benchmarks are supported only on bare metal.','**Virtualization Note:** The runtime `validate_environment()` checker currently warns on virtualization rather than rejecting it. The repository operational restriction remains: Benchmarks are supported only on bare metal. A warning-only implementation does not waive that rule.')
 assert count == 95, 'One-time migration expects the original95inventoryrows; do not rerun after applying'
 p.write_text(s)
 report.append({'path':rel,'before_sha256':hashlib.sha256(original.encode()).hexdigest(),'after_sha256':hashlib.sha256(s.encode()).hexdigest(),'inventory_rows':count,'unchanged_first_400_lines':original.splitlines()[:400]==s.splitlines()[:400]})
(out/'source-changes.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
