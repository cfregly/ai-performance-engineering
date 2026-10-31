#!/usr/bin/env python3
"""Recompute the six-cell stage screen from retained, unprofiled samples."""
import argparse, collections, hashlib, json, math, statistics
from pathlib import Path

def analyze(raw,plan):
    rows=[json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    expected={(size,mode) for size in plan['legal_set_sizes'] for mode in plan['set_modes']}
    pairs=collections.defaultdict(list)
    configs={}
    correctness=collections.defaultdict(list)
    for row in rows:
        key=(row.get('size'),row.get('mode'))
        if row['kind']=='pair': pairs[key].append(row)
        if row['kind']=='cell_config': configs[key]=row
        if row['kind']=='correctness': correctness[key].append(row)
    assert set(pairs)==expected and set(configs)==expected and set(correctness)==expected
    cells=[]
    for size,mode in sorted(expected):
        samples=sorted(pairs[size,mode],key=lambda x:x['pair'])
        assert [x['pair'] for x in samples]==list(range(plan['matched_pairs']))
        assert [x['order'] for x in samples]==plan['pair_orders']
        assert len(correctness[size,mode])==plan['steps_per_block']
        for row in samples:
            for name in ['baseline_wall_ms','candidate_wall_ms','baseline_cuda_ms','candidate_cuda_ms']:
                assert math.isfinite(row[name]) and row[name]>0
        base=[x['baseline_wall_ms'] for x in samples];opt=[x['candidate_wall_ms'] for x in samples]
        bm,om=statistics.median(base),statistics.median(opt)
        rank=math.ceil(.95*len(samples))-1
        all_faster=all(o<b for b,o in zip(base,opt))
        passed=(1-om/bm>=plan['minimum_median_reduction'] and all_faster and sorted(opt)[rank]<=sorted(base)[rank]*(1+plan['maximum_p95_block_regression']))
        cells.append({'legal_tokens':size,'set_mode':mode,'baseline_median_block_ms':bm,'optimized_median_block_ms':om,'speedup':bm/om,'median_reduction':1-om/bm,'baseline_samples_ms':base,'optimized_samples_ms':opt,'row_setup_ms':configs[size,mode]['static_row_setup_wall_ms'],'every_pair_faster':all_faster,'screen_pass':passed,'block_steps':plan['steps_per_block'],'pairs':len(samples)})
    assert len([x for x in rows if x['kind']=='explicit_ties'])==2
    return {'scope':plan['scope'],'source_raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'plan_canonical_json_sha256':hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest(),'counts':dict(collections.Counter(x['kind'] for x in rows)),'checked_logits_per_path':sum(x.get('checked_logits',0) for x in rows),'checked_token_rows_per_path':sum(len(x.get('selected_tokens',[])) for x in rows),'max_abs_error_baseline':max(x.get('baseline_max_abs_error',0) for x in rows),'max_abs_error_optimized':max(x.get('candidate_max_abs_error',0) for x in rows),'cells':cells,'all_cells_pass':all(x['screen_pass'] for x in cells),'interpretation':'Observed synthetic strict-FP32 output-head stage timing. Three paired blocks in one process. No model, serving or request-tail measurement.'}
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,default=Path(__file__).resolve().parents[1]/'evidence/raw.jsonl')
    parser.add_argument('--plan',type=Path,default=Path(__file__).with_name('plan.json'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=analyze(args.raw,json.loads(args.plan.read_text()))
    result['plan_file_sha256']=hashlib.sha256(args.plan.read_bytes()).hexdigest()
    text=json.dumps(result,indent=2)+'\n'
    if args.output:args.output.write_text(text)
    print(text,end='')
if __name__=='__main__':main()
