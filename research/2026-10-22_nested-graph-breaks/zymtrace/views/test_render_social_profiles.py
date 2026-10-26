"""Check scientific weights, event retention, and uncropped social geometry."""
import json
import math
from pathlib import Path
import unittest

import render_social_profiles as renderer


class SocialProfileTests(unittest.TestCase):
    def test_focus_retains_every_event_and_selected_frame(self):
        for word in ('without', 'with'):
            arm = renderer.load_arm(word)
            self.assertEqual(len(arm['focused']), len(arm['doc']['records']))
            self.assertAlmostEqual(math.fsum(r['gpu_time_fraction'] for r in arm['doc']['records']), 1)
            for record, (frames, weight) in zip(arm['doc']['records'], arm['focused']):
                expected = [(f['name'], f['kind']) for f in record['frames'] if f['kind'].startswith('cuda') or any(t in f['name'] for t in renderer.source.FOCUS)]
                self.assertEqual(frames, expected)
                self.assertEqual(weight, record['exported_time_weight'])
            self.assertAlmostEqual(renderer.source.tree(arm['focused'])['weight'], arm['total'])

    def test_per_arm_sidecars_are_exact_copies(self):
        for word in ('without', 'with'):
            arm = renderer.load_arm(word)
            copied = renderer.OUTPUT / f'zymtrace-{word}-nested-graph-breaks-square.stacks.json'
            self.assertEqual(copied.read_bytes(), arm['path'].read_bytes())

    def test_every_text_and_bar_fits_canvas_and_profile(self):
        doc = json.loads((renderer.OUTPUT / 'layout-manifest.json').read_text())
        self.assertEqual(len(doc['cards']), 3)
        for card in doc['cards'].values():
            width, height = card['canvas']
            boxes = [t['bbox'] for t in card['text_boxes']] + card['bar_boxes']
            for x, y, w, h in boxes:
                self.assertGreaterEqual(min(x, y, w, h), -1e-6)
                self.assertLessEqual(x + w, width + 1e-6)
                self.assertLessEqual(y + h, height + 1e-6)
            for x, y, w, h in card['bar_boxes']:
                self.assertTrue(any(px - 1e-6 <= x and py <= y and x + w <= px + pw + 1e-6 and y + h <= py + ph for px, py, pw, ph in (p['bounds'] for p in card['profiles'])))
            for profile in card['profiles']:
                self.assertAlmostEqual(profile['weight_fraction'], 1)
        for item in doc['inputs'].values():
            self.assertEqual(renderer.source.digest(renderer.source.PACKAGE / item['path']), item['sha256'])


if __name__ == '__main__':
    unittest.main()
