"""Check evidence integrity without importing torch or using a GPU."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from analyze import analyze
from screen import summarize

P = Path(__file__).resolve().parents[1]


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((P / 'code/plan.json').read_text())
        self.rows = [
            json.loads(line)
            for line in (P / 'evidence/raw.jsonl').read_text().splitlines()
            if line.strip()
        ]

    def altered(self, rows):
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp) / 'raw.jsonl'
            raw.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
            return analyze(raw, self.plan)

    def first(self, kind, **fields):
        return next(
            row for row in self.rows
            if row['kind'] == kind and all(row.get(key) == value for key, value in fields.items())
        )

    def test_native_six_cells(self):
        result = self.altered(self.rows)
        native = json.loads((P / 'evidence/worker_result.json').read_text())
        expected = {(row['size'], row['mode']): row for row in native['cells']}
        self.assertEqual(len(result['cells']), 6)
        self.assertTrue(result['all_cells_pass'])
        for cell in result['cells']:
            row = expected[cell['legal_tokens'], cell['set_mode']]
            self.assertEqual(cell['baseline_median_block_ms'], row['baseline_median_wall_ms'])
            self.assertEqual(cell['optimized_median_block_ms'], row['candidate_median_wall_ms'])
            self.assertEqual(cell['screen_pass'], row['screen_pass'])

    def test_published_summary_matches_raw(self):
        """Bind every published numeric cell and correctness total to raw rows."""
        result = analyze(P / 'evidence/raw.jsonl', self.plan)
        summary = json.loads((P / 'evidence/summary.json').read_text())
        cells = {(row['legal_tokens'], row['set_mode']): row for row in result['cells']}
        self.assertEqual(
            {(row['size'], row['mode']) for row in summary['cells']},
            set(cells),
        )
        field_map = {
            'baseline_median_wall_ms': 'baseline_median_block_ms',
            'candidate_median_wall_ms': 'optimized_median_block_ms',
            'baseline_p95_block_wall_ms': 'baseline_p95_block_ms',
            'candidate_p95_block_wall_ms': 'optimized_p95_block_ms',
            'speedup': 'speedup',
            'median_reduction': 'median_reduction',
            'row_setup_wall_ms': 'row_setup_ms',
            'every_pair_faster': 'every_pair_faster',
            'screen_pass': 'screen_pass',
            'sample_count_per_path': 'pairs',
        }
        for row in summary['cells']:
            cell = cells[row['size'], row['mode']]
            for published, recomputed in field_map.items():
                with self.subTest(cell=(row['size'], row['mode']), field=published):
                    self.assertEqual(row[published], cell[recomputed])
            self.assertEqual(cell['block_steps'], self.plan['steps_per_block'])
        self.assertEqual(summary['pairs_per_cell'], self.plan['matched_pairs'])
        self.assertEqual(summary['sample_orders'], self.plan['pair_orders'])
        self.assertEqual(summary['checked_legal_logits_per_path'], result['checked_logits_per_path'])
        self.assertEqual(summary['selected_token_rows_per_path'], result['checked_token_rows_per_path'])
        self.assertEqual(summary['maximum_absolute_error']['baseline'], result['max_abs_error_baseline'])
        self.assertEqual(summary['maximum_absolute_error']['candidate'], result['max_abs_error_optimized'])
        self.assertEqual(summary['all_correctness_checks_pass'], result['all_correctness_checks_pass'])
        self.assertEqual(result['checked_logits_per_path'], 86016)
        self.assertEqual(result['checked_token_rows_per_path'], 768)
        self.assertEqual(result['counts'], {
            'explicit_ties': 2, 'cell_config': 6, 'correctness': 96,
            'path_sample': 36, 'pair': 18,
        })
        self.assertFalse(summary['serving_measured'])
        self.assertFalse(summary['model_measured'])

    def test_correctness_flag_uses_recorded_error_threshold(self):
        for field in ('baseline_max_abs_error', 'candidate_max_abs_error'):
            with self.subTest(field=field):
                rows = copy.deepcopy(self.rows)
                check = next(row for row in rows if row['kind'] == 'correctness')
                check[field] = self.plan['correctness_atol']
                self.assertTrue(self.altered(rows)['all_correctness_checks_pass'])
                check[field] = math.nextafter(self.plan['correctness_atol'], math.inf)
                self.assertFalse(self.altered(rows)['all_correctness_checks_pass'])

    def test_missing_pair_rejected(self):
        self.rows.remove(self.first('pair', size=16, mode='static', pair=2))
        with self.assertRaisesRegex(ValueError, 'pair indices mismatch'):
            self.altered(self.rows)

    def test_nonfinite_timing_rejected(self):
        self.first('pair')['baseline_wall_ms'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite positive'):
            self.altered(self.rows)

    def test_pair_order_mismatch_rejected(self):
        self.first('pair', pair=1)['order'] = 'AB'
        with self.assertRaisesRegex(ValueError, 'pair order mismatch'):
            self.altered(self.rows)

    def test_missing_correctness_step_rejected(self):
        self.rows.remove(self.first('correctness', size=16, mode='static', step=3))
        with self.assertRaisesRegex(ValueError, 'unique steps'):
            self.altered(self.rows)

    def test_duplicate_correctness_step_rejected(self):
        self.first('correctness', size=16, mode='static', step=3)['step'] = 2
        with self.assertRaisesRegex(ValueError, 'unique steps'):
            self.altered(self.rows)

    def test_tie_row_count_rejected(self):
        for extra in (False, True):
            with self.subTest(extra=extra):
                rows = copy.deepcopy(self.rows)
                tie = next(row for row in rows if row['kind'] == 'explicit_ties')
                if extra:
                    rows.append(copy.deepcopy(tie))
                else:
                    rows.remove(tie)
                with self.assertRaisesRegex(ValueError, 'exactly two explicit tie rows'):
                    self.altered(rows)

    def test_tie_modes_rejected(self):
        self.first('explicit_ties', mode='changing')['mode'] = 'static'
        with self.assertRaisesRegex(ValueError, 'tie modes mismatch'):
            self.altered(self.rows)

    def test_wrong_tie_winner_rejected(self):
        self.first('explicit_ties')['token_ids'] = [1, 7]
        with self.assertRaisesRegex(ValueError, 'tie token_ids'):
            self.altered(self.rows)

    def test_invalid_legal_ids_rejected(self):
        config = self.first('cell_config', size=16, mode='static')
        original = copy.deepcopy(config['legal_ids'][0])
        cases = {
            'unsorted': list(reversed(original)),
            'duplicate': [original[0]] + original[:-1],
            'out_of_range': original[:-1] + [self.plan['vocab_size']],
            'odd': [original[0] + 1] + original[1:],
            'wrong_size': original[:-1],
        }
        for label, ids in cases.items():
            with self.subTest(label=label):
                config['legal_ids'][0] = ids
                with self.assertRaisesRegex(ValueError, 'legal (IDs|set size)'):
                    self.altered(self.rows)

    def test_changing_legal_set_count_rejected(self):
        self.first('cell_config', mode='changing')['legal_ids'].pop()
        with self.assertRaisesRegex(ValueError, 'legal set count mismatch'):
            self.altered(self.rows)

    def test_repeated_changing_set_rejected(self):
        sets = self.first('cell_config', mode='changing')['legal_ids']
        sets[1] = list(sets[0])
        with self.assertRaisesRegex(ValueError, 'changing sets must be distinct'):
            self.altered(self.rows)

    def test_duplicate_config_rejected(self):
        self.rows.append(copy.deepcopy(self.first('cell_config')))
        with self.assertRaisesRegex(ValueError, 'duplicate cell_config'):
            self.altered(self.rows)

    def test_checked_logit_count_rejected(self):
        self.first('correctness')['checked_logits'] -= 1
        with self.assertRaisesRegex(ValueError, 'checked-logit count mismatch'):
            self.altered(self.rows)

    def test_selected_token_count_rejected(self):
        self.first('correctness')['selected_tokens'].pop()
        with self.assertRaisesRegex(ValueError, 'selected-token rows mismatch'):
            self.altered(self.rows)

    def test_token_outside_legal_set_rejected(self):
        self.first('correctness')['selected_tokens'][0] = -1
        with self.assertRaisesRegex(ValueError, 'selected-token rows mismatch'):
            self.altered(self.rows)

    def test_nonfinite_error_rejected(self):
        self.first('correctness')['candidate_max_abs_error'] = float('inf')
        with self.assertRaisesRegex(ValueError, 'finite nonnegative'):
            self.altered(self.rows)

    def test_invalid_setup_interval_rejected(self):
        self.first('cell_config')['static_row_setup_wall_ms'] = -1
        with self.assertRaisesRegex(ValueError, 'finite nonnegative'):
            self.altered(self.rows)

    def test_wrong_block_step_count_rejected(self):
        self.first('pair')['steps'] = 15
        with self.assertRaisesRegex(ValueError, 'block step count mismatch'):
            self.altered(self.rows)

    def test_path_sample_mismatch_rejected(self):
        self.first('path_sample')['wall_ms'] += 1
        with self.assertRaisesRegex(ValueError, 'path sample and pair timing mismatch'):
            self.altered(self.rows)

    def test_missing_path_sample_rejected(self):
        self.rows.remove(self.first('path_sample'))
        with self.assertRaisesRegex(ValueError, 'path samples mismatch'):
            self.altered(self.rows)


    def wall_cohort(self, baseline, candidate):
        """Copy one cell and keep its pair/path evidence mutually consistent."""
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if (row.get('size'), row.get('mode')) != (16, 'static'):
                continue
            if row['kind'] == 'pair':
                row['baseline_wall_ms'] = baseline[row['pair']]
                row['candidate_wall_ms'] = candidate[row['pair']]
            elif row['kind'] == 'path_sample':
                values = baseline if row['path'] == 'A' else candidate
                row['wall_ms'] = values[row['pair']]
        samples = [
            row for row in rows
            if row['kind'] == 'pair' and (row['size'], row['mode']) == (16, 'static')
        ]
        live_summary = summarize(samples, self.plan)
        result = self.altered(rows)
        cell = next(
            cell for cell in result['cells']
            if (cell['legal_tokens'], cell['set_mode']) == (16, 'static')
        )
        return live_summary, cell, result

    def test_gain_threshold_at_neighboring_float_timings(self):
        # At mathematical 10%, 1 - 9 / 10 is just below the stored 0.10.
        # Record the existing conservative float behavior without changing it.
        cases = (
            ('just_more_gain', math.nextafter(9.0, -math.inf), True),
            ('mathematical_ten_percent', 9.0, False),
            ('just_less_gain', math.nextafter(9.0, math.inf), False),
        )
        for label, candidate, expected in cases:
            with self.subTest(case=label):
                native, cell, result = self.wall_cohort([10.0] * 3, [candidate] * 3)
                for actual in (native, cell):
                    self.assertTrue(actual['every_pair_faster'])
                    self.assertEqual(actual['screen_pass'], expected)
                self.assertEqual(result['all_cells_pass'], expected)
                self.assertEqual(sum(c['screen_pass'] for c in result['cells']), 5 + expected)

    def test_every_pair_must_be_strictly_faster_despite_median_gain(self):
        # All cases retain 20% median gain and meet the 2% p95 allowance.
        # Only the third matched pair changes across the strict-win boundary.
        cases = (
            ('just_faster', math.nextafter(10.0, -math.inf), True),
            ('equal', 10.0, False),
            ('slower_within_p95_allowance', 10.1, False),
        )
        for label, last_candidate, expected in cases:
            with self.subTest(case=label):
                native, cell, result = self.wall_cohort(
                    [10.0] * 3, [8.0, 8.0, last_candidate],
                )
                for actual in (native, cell):
                    self.assertEqual(actual['every_pair_faster'], expected)
                    self.assertEqual(actual['screen_pass'], expected)
                self.assertEqual(result['all_cells_pass'], expected)
                self.assertEqual(sum(c['screen_pass'] for c in result['cells']), 5 + expected)

    def test_summary_requires_exact_matched_pair_count(self):
        for count, expected in ((2, False), (3, True), (4, False)):
            with self.subTest(count=count):
                samples = [
                    {'baseline_wall_ms': 10.0, 'candidate_wall_ms': 8.0}
                    for _ in range(count)
                ]
                result = summarize(samples, self.plan)
                self.assertTrue(result['every_pair_faster'])
                self.assertEqual(result['sample_count_per_path'], count)
                self.assertEqual(result['screen_pass'], expected)
        self.rows.append(copy.deepcopy(self.first('pair', size=16, mode='static')))
        with self.assertRaisesRegex(ValueError, 'pair indices mismatch'):
            self.altered(self.rows)

    def test_three_sample_p95_reports_nearest_rank_maximum(self):
        native, cell, result = self.wall_cohort(
            [20.0, 10.0, 15.0], [16.0, 8.0, 12.0],
        )
        self.assertEqual(native['baseline_p95_block_wall_ms'], 20.0)
        self.assertEqual(native['candidate_p95_block_wall_ms'], 16.0)
        self.assertEqual(cell['baseline_p95_block_ms'], 20.0)
        self.assertEqual(cell['optimized_p95_block_ms'], 16.0)
        self.assertTrue(result['all_cells_pass'])

    def test_setup_is_reported_separately_from_block_screen(self):
        # The plan reports setup separately. It defines no setup-size cutoff.
        for mode in ('static', 'changing'):
            for setup_ms in (0.0, 1_000_000.0):
                with self.subTest(mode=mode, setup_ms=setup_ms):
                    rows = copy.deepcopy(self.rows)
                    config = next(
                        row for row in rows
                        if row['kind'] == 'cell_config'
                        and (row['size'], row['mode']) == (16, mode)
                    )
                    config['static_row_setup_wall_ms'] = setup_ms
                    result = self.altered(rows)
                    cell = next(
                        cell for cell in result['cells']
                        if (cell['legal_tokens'], cell['set_mode']) == (16, mode)
                    )
                    self.assertEqual(cell['row_setup_ms'], setup_ms)
                    self.assertTrue(cell['screen_pass'])
                    self.assertTrue(result['all_cells_pass'])


if __name__ == '__main__':
    unittest.main()
