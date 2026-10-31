#!/usr/bin/env python3
"""Recompute the six-cell stage screen from retained, unprofiled samples."""
import argparse
import collections
import hashlib
import json
import math
import statistics
from pathlib import Path


def require(condition, message):
    """Keep evidence validation active under python -O as well."""
    if not condition:
        raise ValueError(message)


def finite_number(value, name, *, positive=False):
    require(
        type(value) in (int, float)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0),
        f"{name} must be a finite {'positive' if positive else 'nonnegative'} number",
    )


def analyze(raw, plan):
    rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    expected = {
        (size, mode)
        for size in plan['legal_set_sizes']
        for mode in plan['set_modes']
    }
    steps = plan['steps_per_block']
    pair_count = plan['matched_pairs']
    require(type(steps) is int and steps > 0, 'steps_per_block must be positive')
    require(type(pair_count) is int and pair_count > 0, 'matched_pairs must be positive')
    require(len(plan['pair_orders']) == pair_count, 'pair_orders length mismatch')
    require(all(order in ('AB', 'BA') for order in plan['pair_orders']), 'invalid pair order')
    require(set(plan['set_modes']) == {'static', 'changing'}, 'unexpected set modes')
    pairs = collections.defaultdict(list)
    configs = {}
    correctness = collections.defaultdict(list)
    paths = collections.defaultdict(list)
    ties = []
    for row in rows:
        require(isinstance(row, dict), 'raw rows must be objects')
        kind = row.get('kind')
        if kind == 'explicit_ties':
            ties.append(row)
            continue
        key = (row.get('size'), row.get('mode'))
        require(key in expected, f'unexpected cell {key}')
        if kind == 'pair':
            pairs[key].append(row)
        elif kind == 'cell_config':
            require(key not in configs, f'duplicate cell_config for {key}')
            configs[key] = row
        elif kind == 'correctness':
            correctness[key].append(row)
        elif kind == 'path_sample':
            paths[key].append(row)
        else:
            raise ValueError(f'unknown row kind {kind!r}')
    require(
        set(pairs) == set(configs) == set(correctness) == set(paths) == expected,
        'missing or unexpected evidence cells',
    )
    require(len(ties) == 2, 'expected exactly two explicit tie rows')
    require({row.get('mode') for row in ties} == {'static', 'changing'}, 'tie modes mismatch')
    for row in ties:
        tokens = row.get('token_ids')
        require(
            tokens == [1, 1] and all(type(token) is int for token in tokens),
            'tie token_ids must be [1, 1]',
        )
        require(
            row.get('all_equal_row') == 0 and row.get('two_way_maximum_row') == 1,
            'tie fixture row labels mismatch',
        )

    cells = []
    for size, mode in sorted(expected):
        key = (size, mode)
        config = configs[key]
        legal_sets = config.get('legal_ids')
        require(isinstance(legal_sets, list), f'legal_ids must be a list for {key}')
        require(
            len(legal_sets) == (1 if mode == 'static' else steps),
            f'legal set count mismatch for {key}',
        )
        for ids in legal_sets:
            require(isinstance(ids, list) and len(ids) == size, f'legal set size mismatch for {key}')
            require(
                all(type(token) is int and 0 <= token < plan['vocab_size'] and token % 2 == 0 for token in ids),
                f'legal IDs must be in-range even integers for {key}',
            )
            require(ids == sorted(set(ids)), f'legal IDs must be sorted and unique for {key}')
        if mode == 'changing':
            require(
                len({tuple(ids) for ids in legal_sets}) == steps,
                f'changing sets must be distinct for {key}',
            )
        finite_number(config.get('static_row_setup_wall_ms'), f'{key} row setup')

        checks = correctness[key]
        require(
            len(checks) == steps
            and all(type(row.get('step')) is int for row in checks)
            and sorted(row['step'] for row in checks) == list(range(steps)),
            f'correctness must contain unique steps 0..{steps - 1} for {key}',
        )
        for row in checks:
            require(
                row.get('checked_logits') == plan['batch_size'] * size,
                f'checked-logit count mismatch for {key}',
            )
            tokens = row.get('selected_tokens')
            allowed = legal_sets[0 if mode == 'static' else row['step']]
            require(
                isinstance(tokens, list)
                and len(tokens) == plan['batch_size']
                and all(type(token) is int and token in allowed for token in tokens),
                f'selected-token rows mismatch for {key}',
            )
            finite_number(row.get('baseline_max_abs_error'), f'{key} baseline error')
            finite_number(row.get('candidate_max_abs_error'), f'{key} candidate error')

        samples = pairs[key]
        require(
            all(type(row.get('pair')) is int for row in samples)
            and sorted(row['pair'] for row in samples) == list(range(pair_count)),
            f'pair indices mismatch for {key}',
        )
        samples = sorted(samples, key=lambda row: row['pair'])
        require(
            [row.get('order') for row in samples] == plan['pair_orders'],
            f'pair order mismatch for {key}',
        )
        path_samples = {}
        for row in paths[key]:
            path_key = (row.get('pair'), row.get('path'))
            require(path_key not in path_samples, f'duplicate path sample for {key}')
            path_samples[path_key] = row
        require(
            set(path_samples) == {(pair, path) for pair in range(pair_count) for path in ('A', 'B')},
            f'path samples mismatch for {key}',
        )
        for row in samples:
            require(row.get('steps') == steps, f'block step count mismatch for {key}')
            for name in ('baseline_wall_ms', 'candidate_wall_ms', 'baseline_cuda_ms', 'candidate_cuda_ms'):
                finite_number(row.get(name), f'{key} {name}', positive=True)
            for path, prefix in (('A', 'baseline'), ('B', 'candidate')):
                path_row = path_samples[row['pair'], path]
                for clock in ('wall', 'cuda'):
                    require(
                        path_row.get(f'{clock}_ms') == row[f'{prefix}_{clock}_ms'],
                        f'path sample and pair timing mismatch for {key}',
                    )
        base = [row['baseline_wall_ms'] for row in samples]
        opt = [row['candidate_wall_ms'] for row in samples]
        bm, om = statistics.median(base), statistics.median(opt)
        rank = math.ceil(.95 * len(samples)) - 1
        baseline_p95, optimized_p95 = sorted(base)[rank], sorted(opt)[rank]
        all_faster = all(o < b for b, o in zip(base, opt))
        passed = (
            1 - om / bm >= plan['minimum_median_reduction']
            and all_faster
            and optimized_p95 <= baseline_p95 * (1 + plan['maximum_p95_block_regression'])
        )
        cells.append({
            'legal_tokens': size, 'set_mode': mode,
            'baseline_median_block_ms': bm, 'optimized_median_block_ms': om,
            'baseline_p95_block_ms': baseline_p95, 'optimized_p95_block_ms': optimized_p95,
            'speedup': bm / om, 'median_reduction': 1 - om / bm,
            'baseline_samples_ms': base, 'optimized_samples_ms': opt,
            'row_setup_ms': config['static_row_setup_wall_ms'],
            'every_pair_faster': all_faster, 'screen_pass': passed,
            'block_steps': steps, 'pairs': len(samples),
        })
    checks = [row for row in rows if row['kind'] == 'correctness']
    return {
        'scope': plan['scope'],
        'source_raw_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
        'plan_canonical_json_sha256': hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
        'counts': dict(collections.Counter(row['kind'] for row in rows)),
        'checked_logits_per_path': sum(row['checked_logits'] for row in checks),
        'checked_token_rows_per_path': sum(len(row['selected_tokens']) for row in checks),
        'max_abs_error_baseline': max(row['baseline_max_abs_error'] for row in checks),
        'max_abs_error_optimized': max(row['candidate_max_abs_error'] for row in checks),
        # Errors <= atol suffice for the numerical tolerance check.
        # screen.py checked FP64 projection and argmax agreement at run time;
        # this retained-error analysis does not re-execute those checks.
        'all_correctness_checks_pass': all(
            row[name] <= plan['correctness_atol']
            for row in checks
            for name in ('baseline_max_abs_error', 'candidate_max_abs_error')
        ),
        'cells': cells,
        'all_cells_pass': all(cell['screen_pass'] for cell in cells),
        'interpretation': 'Observed synthetic strict-FP32 output-head stage timing. Three paired blocks in one process. No model, serving or request-tail measurement.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path(__file__).resolve().parents[1] / 'evidence/raw.jsonl')
    parser.add_argument('--plan', type=Path, default=Path(__file__).with_name('plan.json'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = analyze(args.raw, json.loads(args.plan.read_text()))
    result['plan_file_sha256'] = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
