"""Synthetic populated CPU reporting fixtures, never GPU or profiler evidence."""
from __future__ import annotations

import json

import pytest

from divide_not_reciprocal import exhaustive, report
from divide_not_reciprocal.ulp import Tally as Comparison


def populated_exhaustive():
    # Distinct classes at boundaries and both signs catch region misrouting
    # which an empty dictionary or all-zero table cannot detect.
    hist = [[0] * exhaustive.NCLASS for _ in range(exhaustive.NBIN)]
    max_ulp = [0] * exhaustive.NBIN
    for sign in (0, 1):
        for exponent, counts, largest in [
            (0, (2, 3, 5, 7, 0, 0), 19),
            (3, (11, 13, 17, 23, 0, 0), 29),
            (4, (31, 37, 41, 43, 0, 0), 47),
            (254, (53, 59, 61, 67, 0, 0), 71),
            (255, (0, 0, 0, 0, 73, 79), 83),
        ]:
            index = (sign << 8) | exponent
            hist[index] = list(counts)
            max_ulp[index] = largest
    return exhaustive.Tally('synthetic', 0, 'synthetic expression',
                            'synthetic PTX', 'synthetic SASS', hist,
                            max_ulp, 0.125, {(0, 4): 121, (1, 254): 187})


def comparison():
    return Comparison(inputs=8, same=3, away1=2, toward1=1, far=0,
                      nan_both=1, nan_one=1, max_ulp=1,
                      isclose_fail=1, max_gap_tol=None, chunks=1).as_dict()


def populated_results(state='COMPLETE'):
    row = comparison()
    return {
        'date': 'synthetic fixture date', 'state': state,
        'stages': {'synthetic': state},
        'exhaustive': [exhaustive.summarize(populated_exhaustive())],
        'triton': {'reference': 'synthetic tensor reference',
                   'coverage': 'synthetic patterns, not a run',
                   'modes': {'synthetic': row},
                   'pairs_differ': {'synthetic pair': 3}},
        'hardsigmoid': {'coverage': 'synthetic finite groups',
                       'inside': {'synthetic': row},
                       'moved': {'synthetic': row},
                       'outside': {'synthetic': row}},
        'timing': {'n': 32, 'bytes_moved': 256, 'warmup': 0, 'iters': 1,
                   'cases': {'synthetic': {'launch': 'synthetic',
                       'median_ms': 1.0, 'p10_ms': 0.9, 'p90_ms': 1.1,
                       'tb_per_s': 0.0}}},
        'ncu': {'n': 32, 'launches': [{
            'case': 'synthetic', 'kernel': 'synthetic_kernel',
            'grid': [2, 1, 1], 'block': [128, 1, 1],
            'inst_executed': 64, 'warp_instructions_per_32_elements': 64.0,
            'dram_bytes': 256, 'time_s': 0.0001}],
            'errors': [{'case': 'synthetic failed capture',
                        'message': 'injected missing report'}]},
        'conditions': 'Synthetic CPU fixture. No hardware was invoked.',
        'errors': [] if state == 'COMPLETE' else [{
            'stage': 'synthetic', 'type': 'Injected',
            'message': 'synthetic failure after completed data'}],
    }


def test_populated_summary_preserves_distinct_regions_and_json_schema():
    summary = exhaustive.summarize(populated_exhaustive())
    assert summary['census']['inputs'] == 784
    assert summary['census']['differ'] == 616
    assert summary['census']['max_ulp'] == 71
    assert summary['census']['per_binade_min'] == 121
    assert summary['census']['per_binade_max'] == 187
    assert summary['bottom_binades']['inputs'] == 128
    assert summary['bottom_binades']['differ'] == 106
    assert summary['subnormal_inputs']['inputs'] == 34
    assert summary['subnormal_inputs']['differ'] == 30
    assert summary['subnormal_inputs']['max_ulp'] == 19
    assert summary['specials'] == {
        'inputs': 304, 'nan_both': 146, 'nan_one': 158, 'differ': 0}
    assert summary['normal_inputs_differ'] == 722
    assert summary['total'] == 1250
    assert summary['all_differ'] == 752
    # Summary uses scalar extrema rather than leaking tuple-key source maps.
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary


@pytest.mark.parametrize('state', ['COMPLETE', 'PARTIAL'])
def test_populated_all_stage_checkpoint_and_markdown(tmp_path, state):
    data = populated_results(state)
    report.write(tmp_path, 'synthetic', data)
    retained = json.loads((tmp_path / 'synthetic.json').read_text())
    assert retained == data
    text = (tmp_path / 'synthetic.md').read_text()
    for table in range(1, 6):
        assert f'Table {table}:' in text
    assert f'State: **{state}**' in text
    assert 'undefined (nonfinite output)' in text
    assert 'synthetic_kernel' in text
    assert '(2, 1, 1)' in text and '(128, 1, 1)' in text
    assert 'paired tiny launch is retained for identity' in text
    assert 'elements per reported large launch' in text
    assert 'docs-form expressions divide before adding 0.5 and can still differ' in text
    assert 'zeros and subnormal inputs (field 0)' in text
    assert 'infinities and NaNs (field 255)' in text
    assert 'Comparison roles: p is torch.nn.functional.hardsigmoid(x)' in text
    assert 'q is the row expression' in text
    assert 'injected missing report' in text
    assert ('## Failures' in text) == (state == 'PARTIAL')


def test_populated_markdown_failure_keeps_completed_json(tmp_path, monkeypatch):
    data = populated_results('PARTIAL')
    original = report.render

    def reject_populated_render(name, results):
        assert 'Table 5:' in original(name, results)
        raise ValueError('injected final Markdown failure')

    monkeypatch.setattr(report, 'render', reject_populated_render)
    with pytest.raises(ValueError, match='injected final Markdown failure'):
        report.write(tmp_path, 'synthetic', data)
    assert json.loads((tmp_path / 'synthetic.json').read_text()) == data
    assert not (tmp_path / 'synthetic.md').exists()
