"""Write review summaries only beside this new review; never modify inputs."""
from pathlib import Path
import collections
from datetime import datetime, timezone
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / '.git').exists())
E = ROOT / 'docs/audits/2026-08-30/evidence'
P05 = 'code/labs/nanochat_fullstack/audit_wave1/'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
rel = lambda path: str(path.relative_to(ROOT))
scan = json.loads((HERE / 'snapshot-2/scan.json').read_text())
refs = json.loads((HERE / 'snapshot-2/all-hash-references.json').read_text())
prior = json.loads((E / 'integration/artifact-review/scan.json').read_text())
prior_changes = [dict(path=p, recorded_sha256=h, current_sha256=sha(ROOT/p) if (ROOT/p).is_file() else None)
                 for p,h in prior['manifest_snapshot_sha256'].items()
                 if not (ROOT/p).is_file() or sha(ROOT/p) != h]
anchors = {
'integration/hygiene-collection/validation-receipts.json': 'cf15abcf9faa8b07009d80e682d9064b9d3991e551ff7541c5f06a89dd2504f7',
'integration/optional-dependency-collection/receipt.json': '865ad97d0773143c33bcde6ad8d1f14f36db079479c898ed293791c820590f19',
'zero2-parity/cuda-gate/validation-receipts.json': 'e10a2809d6a952518baa3a944bd0eadbd7b6c4ce9b60a3c6c88ede451df2075a',
'torchrun-verification/receipt.json': 'ba528b7a5540497eaa5d9506093862dc95c69b6599d2b758c2e94a9fcab0f736',
'integration/prefill-full-output/receipt.json': '5be4ba0cd1d1c1aad288842587bc889671484917dbc8de63a2945745208e2907',
'integration/prefill-full-output/final-artifact-manifest.json': '8a45beaba8ca607c2028168ee880c0981b5997e10d88c2effac1cbd92b4f2659',
'integration/artifact-review/scan.json': 'fd4a921687bbf050e379df1e6ed88549e8a4f7f2355b074461e53cb8543e00c4',
}
anchor_checks = [dict(path=rel(E/p), recorded_sha256=h, current_sha256=sha(E/p), matches=sha(E/p)==h)
                 for p,h in anchors.items()]
tracked = subprocess.check_output(['git','diff','--name-only','HEAD','--'],cwd=ROOT,text=True).splitlines()
untracked = subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
paths = sorted(p for p in set(tracked+untracked) if (ROOT/p).is_file() and
               not p.startswith(('docs/audits/',P05)) and p != 'AUDIT_REMEDIATION_PLAN.md')
source_coverage=[]
for p in paths:
    current=sha(ROOT/p)
    matching=sorted(set(ref['receipt'] for ref in refs if ref['resolved_path']==p and
                        ref.get('expected_sha256')==current and not ref.get('historical_archive')))
    source_coverage.append(dict(path=p,sha256=current,current_covered_by=matching))
missing_current=[item for item in source_coverage if not item['current_covered_by']]
p05path=ROOT/P05
p05=json.loads((p05path/'receipt.json').read_text())
p05_logs=[dict(path=P05+name,recorded_sha256=digest,current_sha256=sha(p05path/name),
               matches=sha(p05path/name)==digest,
               current_git_ignore_rules=[x['rule'] for x in scan['scoped_log_exceptions'] if x['path']==P05+name])
          for name,digest in p05['evidence_sha256'].items() if name.endswith('.log')]
portability = {
 'discovery':'P05 evidence is package-local, outside the earlier scanner canonical evidence tree.',
 'before_observation':{'kind':'Transcription of read-only shell output observed in this agent turn before parent fix; not a fresh before-state rerun.',
  'command':'git check-ignore -v code/labs/nanochat_fullstack/audit_wave1/before.log code/labs/nanochat_fullstack/audit_wave1/final-strict-tests.log',
  'sample_output':['code/.gitignore:226:*.log\tcode/labs/nanochat_fullstack/audit_wave1/before.log','code/.gitignore:226:*.log\tcode/labs/nanochat_fullstack/audit_wave1/final-strict-tests.log'],
  'receipt_log_count':15,
  'canonical_byte_match_observation':'12 logs had no byte-identical canonical artifact; two empty logs and lint log shared bytes with unrelated package artifacts, not explicit lineage-preserving archives.'},
 'parent_resolution':{'path':P05+'.gitignore','sha256':sha(p05path/'.gitignore'),'contents':(p05path/'.gitignore').read_text(),
  'mechanism':'New directory-scoped !*.log keeps all original receipt paths; no original log or receipt rewritten.'},
 'after_actual_scan':p05_logs,
}
ignored_gaps=[x for x in scan['ignored_files'] if not x['disposable'] and not x['byte_identical_unignored_copies']]
changed_after_scan=[p for p,h in scan['manifest_snapshot_sha256'].items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
report={
 'status':'PASS_EVIDENCE_INTEGRITY_AT_SNAPSHOT__FINAL_SOURCE_AND_RUNTIME_GATES_PENDING',
 'created_utc':datetime.now(timezone.utc).isoformat(), 'snapshot_utc':scan['snapshot_utc'], 'head':scan['head'],
 'scope':'Read-only artifact/receipt/hash/path/inventory audit; only this new final-artifact-review directory written. No tests, GPU, build, installation, Git mutation, source change or previous evidence edit.',
 'scanner_lineage':'Original C review preserved; snapshot-1 uses extended original scanner. snapshot-2 additionally recognizes .mk and audits package-local P05 evidence.',
 'counts':{'evidence_files':scan['evidence_files_scanned'],'hash_bearing_metadata':scan['manifests_scanned'],
  'explicit_hash_references':scan['explicit_hash_references'],'previous_metadata_rechecked':len(prior['manifest_snapshot_sha256']),
  'historical_source_drift_paths':len(scan['source_hash_drift']), 'current_changed_source_doc_test_paths':len(source_coverage),
  'current_source_manifest_gaps':len(missing_current),'markdown_file_links':len(scan['markdown_file_links']),
  'portable_log_exceptions':len(scan['scoped_log_exceptions']), 'ignored_capture_files_with_portable_copies':sum(not x['disposable'] for x in scan['ignored_files']),
  'disposable_ignored_python_bytecode':sum(x['disposable'] for x in scan['ignored_files'])},
 'integrity':{'artifact_hash_mismatches':scan['artifact_hash_mismatches'],'missing_hash_references':scan['missing_hash_references'],
  'parse_errors':scan['parse_errors'],'unresolved_file_path_strings':scan['unresolved_path_strings_for_manual_triage'],
  'missing_markdown_links':[x for x in scan['markdown_file_links'] if not x['exists']],
  'metadata_changed_during_scan':scan['manifest_changed_during_scan'], 'metadata_changed_after_scan_before_review':changed_after_scan,
  'prior_metadata_changed':prior_changes,'focused_receipt_anchor_checks':anchor_checks,
  'ignored_non_disposable_without_portable_copy':ignored_gaps},
 'inventory':scan['ledger'],
 'source_epoch':{'current_manifest_gaps':missing_current,
  'mutable_control_document_excluded':'AUDIT_REMEDIATION_PLAN.md is parent-owned ongoing planning state, not a frozen source result; source inventory ledger was read at the recorded snapshot hash.',
  'historical_drift_rule':'Earlier source hashes and original snapshots remain valid records of earlier epochs. A later matching hash covers current source identity only; it does not rerun earlier tests or transfer GPU qualification.',
  'examples':['Hygiene final source matches its latest receipt; earlier import-only and integration receipts remain unchanged.',
   'Torchrun toy verification gap in the earlier ZeRO CUDA-gate receipt is historical: the later Torchrun receipt explicitly withdraws the generic wrapper and preserves unsupported actual-training acceptance.',
   'Optional-dependency lazy imports and full-output prefill changes have later matching receipts; prior kernel/attention/dependency hashes remain historical.',
   'The restored retired CUDA stub has a distinct follow-up receipt; the original retirement record was not rewritten.'],
  'handoff':'Parent interim source-before-full-cpu.json covers 318 changed/new source/doc/test paths. Parent final source manifest, any subsequent attention gate, documentation changes and whole-suite results are outside this snapshot and require later receipts.'},
 'p05_portability_followup':portability,
 'acceptance_limits':['Hash/path/inventory consistency is not independent reproduction of numerical or performance results.',
  'No full pytest, CUDA compilation, device-link/import, GPU/NCCL/multi-GPU training, profiler capture, sanitizer or performance run performed in this audit.',
  'Existing CPU results remain tied to their recorded source/env; CPU-only skips and prepared CUDA gates remain HOLD.',
  'Original 128 findings preserved; second wave is still required and awaiting report, so whole goal is not complete.',
  'These audit artifacts are untracked; include them and both scoped evidence .gitignore files in the eventual handoff. No staging or commit was performed.'],
}
assert not any(report['integrity'][k] for k in ['artifact_hash_mismatches','missing_hash_references','parse_errors','unresolved_file_path_strings','missing_markdown_links','metadata_changed_during_scan','metadata_changed_after_scan_before_review','prior_metadata_changed','ignored_non_disposable_without_portable_copy'])
assert all(x['matches'] for x in anchor_checks)
assert all(x['matches'] and x['current_git_ignore_rules'] for x in p05_logs) and len(p05_logs)==15
assert not missing_current
for name,data in [('review.json',report),('current-source-coverage.json',source_coverage),('p05-portability-followup.json',portability)]:
    path=HERE/name
    if path.exists():raise RuntimeError(f'Refusing overwrite: {path}')
    path.write_text(json.dumps(data,indent=2)+'\n')
summary=f'''# Final artifact review snapshot

Evidence integrity passes at {scan['snapshot_utc']}. This review ran no tests or GPU work and changed only this new review directory.

- {scan['evidence_files_scanned']} files, {scan['manifests_scanned']} hash-bearing metadata records, and {scan['explicit_hash_references']} explicit hash references checked; no changed or missing recorded evidence, unresolved file references, or missing Markdown file links.
- All {len(prior['manifest_snapshot_sha256'])} metadata files recorded by the earlier artifact review remain byte-identical. The latest hygiene, optional-dependency, ZeRO CUDA-gate, Torchrun withdrawal and prefill receipts match their recorded hashes.
- All original 128 IDs remain exactly once in source, ledger and package assignments, with titles, locations and severities preserved: 5 critical, 37 high, 58 medium, 28 low. Source hash remains `474779ab49b67c5c888e5f689b2400204b6d0f46f304219cedd0773694b6e1ba`.
- {len(scan['source_hash_drift'])} paths have legitimate historical source hashes. All {len(source_coverage)} current changed source/doc/test paths have matching current hash coverage; none lacks a manifest at this snapshot. The parent still owns the final source manifest after remaining integration and documentation work.
- Found package-local P05 logs outside the earlier scanner scope. The parent added a narrow `.gitignore` beside that receipt. All 15 original logs now remain portable at their original paths and match their receipt hashes; no log or receipt was rewritten. Six ignored CPU-profile captures retain portable byte-identical archives; six other ignored files are disposable Python bytecode.

The earlier ZeRO receipt's toy-wrapper gap is historical. The later Torchrun receipt withdraws generic wrapper verification; it does not qualify actual training. Prepared CUDA and multi-GPU gates remain HOLD. The second wave is still required and awaiting its report.

See [review.json](review.json), [full hash observations](snapshot-2/all-hash-references.json), [current source coverage](current-source-coverage.json), and [P05 portability follow-up](p05-portability-followup.json). Snapshot-1 and the earlier artifact-review files are preserved. This snapshot does not cover subsequent parent reports, source edits or a final full-suite result.
'''
(HERE/'review.md').write_text(summary)
files={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file()}
(HERE/'review-files-sha256.json').write_text(json.dumps(files,indent=2)+'\n')
print(json.dumps({'status':report['status'],'counts':report['counts'],'review_sha256':sha(HERE/'review.json'),'manifest_sha256':sha(HERE/'review-files-sha256.json')},indent=2))
