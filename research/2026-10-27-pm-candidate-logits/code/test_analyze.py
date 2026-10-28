"""Check evidence integrity without importing torch or using a GPU."""
import json,tempfile,unittest
from pathlib import Path
from analyze import analyze
P=Path(__file__).resolve().parents[1]
class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.plan=json.loads((P/'code/plan.json').read_text())
        self.rows=[json.loads(x) for x in (P/'evidence/raw.jsonl').read_text().splitlines()]
    def altered(self,rows):
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/'raw.jsonl';file.write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
            return analyze(file,self.plan)
    def test_native_six_cells(self):
        result=self.altered(self.rows)
        native=json.loads((P/'evidence/worker_result.json').read_text())
        expected={(x['size'],x['mode']):x for x in native['cells']}
        self.assertEqual(len(result['cells']),6)
        self.assertEqual(result['checked_logits_per_path'],86016)
        for cell in result['cells']:
            row=expected[cell['legal_tokens'],cell['set_mode']]
            self.assertEqual(cell['baseline_median_block_ms'],row['baseline_median_wall_ms'])
            self.assertEqual(cell['optimized_median_block_ms'],row['candidate_median_wall_ms'])
            self.assertEqual(cell['screen_pass'],row['screen_pass'])
    def test_missing_pair_rejected(self):
        rows=[x for x in self.rows if not(x['kind']=='pair' and x['size']==16 and x['mode']=='static' and x['pair']==2)]
        with self.assertRaises(AssertionError):self.altered(rows)
    def test_nonfinite_timing_rejected(self):
        rows=[dict(x,baseline_wall_ms=float('nan')) if x['kind']=='pair' else x for x in self.rows]
        with self.assertRaises(AssertionError):self.altered(rows)
if __name__=='__main__':unittest.main()
