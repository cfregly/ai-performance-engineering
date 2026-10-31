"""Check evidence integrity without importing torch or using a GPU."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from analyze import analyze

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


if __name__ == '__main__':
    unittest.main()
