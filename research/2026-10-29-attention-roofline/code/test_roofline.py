"""Check exact arithmetic, boundary conditions, and toy reduction equivalence."""

import csv
import hashlib
import json
import subprocess
import sys
import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from roofline import (
    AttentionSpec,
    MIB,
    batch_partial_bytes,
    build_results,
    iter_toy_partials,
    parse_bandwidth,
    reduce_toy,
    rendered_outputs,
    time_floor_ms,
    toy_comparison,
    traffic_floor_bytes,
)


class AnalyticalModelTests(unittest.TestCase):
    def test_main_example_exact_integer_arithmetic(self):
        spec = AttentionSpec()
        self.assertEqual(spec.tiles, 32)
        self.assertEqual(spec.causal_tile_pairs, 8448)
        self.assertEqual(spec.partial_bytes_per_pair, 131072)
        self.assertEqual(spec.total_partial_bytes, 1107296256)
        self.assertEqual(spec.total_partial_bytes // MIB, 1056)
        self.assertEqual(traffic_floor_bytes(spec.total_partial_bytes, 128 * MIB), 1946157056)

    def test_hypothetical_time_floors(self):
        self.assertEqual(time_floor_ms(1946157056, "8"), Decimal("0.243269632"))
        self.assertEqual(time_floor_ms(1946157056, "6.4"), Decimal("0.30408704"))

    def test_decimal_bandwidth_and_binary_capacity_units(self):
        self.assertEqual(time_floor_ms(10**12, "1"), Decimal("1000"))
        self.assertEqual(time_floor_ms(10**9, "1"), Decimal("1"))
        self.assertEqual(time_floor_ms(0, "8"), Decimal("0"))
        self.assertEqual(128 * MIB, 134217728)
        self.assertEqual(parse_bandwidth("6.4"), Decimal("6.4"))

    def test_zero_sequence_and_empty_batches(self):
        spec = replace(AttentionSpec(), sequence_length=0)
        self.assertEqual(spec.tiles, 0)
        self.assertEqual(spec.causal_tile_pairs, 0)
        self.assertEqual(spec.total_partial_bytes, 0)
        self.assertEqual(traffic_floor_bytes(spec.total_partial_bytes, 0), 0)
        self.assertEqual(batch_partial_bytes(spec, 1), [])

    def test_capacity_equal_to_or_larger_than_partials(self):
        self.assertEqual(traffic_floor_bytes(127, 128), 0)
        self.assertEqual(traffic_floor_bytes(128, 128), 0)
        self.assertEqual(traffic_floor_bytes(129, 128), 2)
        self.assertEqual(traffic_floor_bytes(128, 0), 256)

    def test_nondivisible_sequence_counts_padded_key_tiles(self):
        spec = AttentionSpec(5, 4, 2, 2, 1, 2)
        self.assertEqual(spec.tiles, 3)
        self.assertEqual(spec.causal_tile_pairs, 24)
        self.assertEqual(spec.partial_bytes_per_pair, 24)
        self.assertEqual(spec.total_partial_bytes, 576)
        self.assertEqual(batch_partial_bytes(spec, 2), [288, 288])
        self.assertEqual(sum(batch_partial_bytes(spec, 2)), spec.total_partial_bytes)
        # A final single-row key tile still gets two allocated rows per partial.
        last_key_partials = [rows for _, key, rows in iter_toy_partials(spec, False) if key == 2]
        self.assertEqual(len(last_key_partials), 4)
        self.assertTrue(all(len(rows) == 2 and rows[1] == (0, 0, 0) for rows in last_key_partials))

    def test_single_tile_and_one_element_sequence(self):
        spec = AttentionSpec(1, 1, 1, 1, 1, 128)
        self.assertEqual(spec.tiles, 1)
        self.assertEqual(spec.causal_tile_pairs, 1)
        self.assertEqual(spec.total_partial_bytes, 1024)

    def test_batching_preserves_total_partial_volume_but_changes_boundary(self):
        spec = AttentionSpec()
        expected_max_mib = {1: 64, 2: 126, 4: 244, 8: 456, 16: 784, 32: 1056}
        for size, maximum_mib in expected_max_mib.items():
            with self.subTest(size=size):
                sizes = batch_partial_bytes(spec, size)
                self.assertEqual(sum(sizes), spec.total_partial_bytes)
                self.assertEqual(max(sizes), maximum_mib * MIB)
        self.assertEqual(batch_partial_bytes(spec, 100), [spec.total_partial_bytes])
        self.assertEqual(spec.final_dk_dv_bytes, 4 * MIB)

    def test_final_short_batch_uses_actual_consecutive_groups(self):
        sizes = batch_partial_bytes(AttentionSpec(), 3)
        self.assertEqual(len(sizes), 11)
        self.assertEqual(sizes[-1], 126 * MIB)
        self.assertEqual(max(sizes), 174 * MIB)
        self.assertEqual(sum(sizes), AttentionSpec().total_partial_bytes)

    def test_grouped_heads_do_not_merge_unique_partials(self):
        self.assertEqual(AttentionSpec(kv_heads=1).total_partial_bytes, AttentionSpec(kv_heads=4).total_partial_bytes)
        self.assertEqual(AttentionSpec(kv_heads=4).final_dk_dv_bytes, 16 * MIB)

    def test_reject_invalid_specification(self):
        for field, invalid in (
            ("sequence_length", -1),
            ("sequence_length", 1.5),
            ("sequence_length", True),
            ("query_heads", 0),
            ("kv_heads", 0),
            ("key_dimension", -1),
            ("value_dimension", 0),
            ("tile_rows", 0),
            ("tile_rows", "128"),
        ):
            with self.subTest(field=field, invalid=invalid):
                with self.assertRaises(ValueError):
                    replace(AttentionSpec(), **{field: invalid})
        with self.assertRaises(ValueError):
            AttentionSpec(query_heads=3, kv_heads=2)

    def test_reject_invalid_traffic_capacity_batch_and_bandwidth(self):
        for value in (-1, True, 1.5, "4"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    traffic_floor_bytes(value, 0)
                with self.assertRaises(ValueError):
                    traffic_floor_bytes(0, value)
                with self.assertRaises(ValueError):
                    time_floor_ms(value, "8")
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                batch_partial_bytes(AttentionSpec(), value)
        for value in ("0", "-1", "NaN", "Infinity", "-Infinity", "abc", "", True, 6.4, None):
            with self.subTest(bandwidth=value):
                with self.assertRaises(ValueError):
                    parse_bandwidth(value)

    def test_toy_materialized_and_streamed_paths_match_expected_values(self):
        spec = AttentionSpec(5, 4, 2, 2, 1, 2)
        expected_integer = [
            [[0, 0, 0] for _ in range(5)] for _ in range(2)
        ]
        # Compute the oracle by visiting output cells, independently of either reducer.
        for kv_head in range(2):
            for position in range(5):
                key_tile, row = divmod(position, 2)
                for feature in range(3):
                    expected_integer[kv_head][position][feature] = sum(
                        ((query_head + 1) * 11 + (query_tile + 1) * 7 + (key_tile + 1) * 5 + (row + 1) * 3 + feature) % 17 - 8
                        for query_head in range(kv_head * 2, kv_head * 2 + 2)
                        for query_tile in range(key_tile, 3)
                    )
        for floating in (False, True):
            expected = (
                [[[value / 8.0 for value in row] for row in head] for head in expected_integer]
                if floating else expected_integer
            )
            partials = list(iter_toy_partials(spec, floating))
            self.assertEqual(reduce_toy(spec, iter(partials), floating), expected)
            self.assertEqual(reduce_toy(spec, iter_toy_partials(spec, floating), floating), expected)
        self.assertEqual(toy_comparison()["comparisons"]["integer"]["partial_count"], 24)

    def test_toy_rejects_large_materialization(self):
        with self.assertRaises(ValueError):
            list(iter_toy_partials(AttentionSpec(), False))
        with self.assertRaises(ValueError):
            list(iter_toy_partials(AttentionSpec(2, 2, 1, 10, 10, 2), False))

    def test_deterministic_outputs_and_explicit_measurement_limits(self):
        first = rendered_outputs()
        self.assertEqual(first, rendered_outputs())
        result = json.loads(first["RESULTS.json"])
        self.assertEqual(result, build_results())
        self.assertEqual(result["measurements"], {"gpu_runs": 0, "gpu_timings": [], "cpu_timings": []})
        self.assertFalse(result["toy_correctness_illustration"]["timings_collected"])
        self.assertEqual(len(result["cache_sensitivity"]), 10)
        self.assertEqual(len(result["reduction_batch_sensitivity"]), 6)

    def test_retained_evidence_values_and_hashes_match_calculator(self):
        package = Path(__file__).resolve().parents[1]
        evidence = json.loads(
            (package / "evidence/analytical-results.json").read_text(encoding="utf-8"),
            parse_float=Decimal,
        )
        result = build_results()
        derivation = result["derivation"]
        screen = result["target_screen"]
        batch_two = next(row for row in result["reduction_batch_sensitivity"] if row["query_tiles_per_batch"] == 2)
        expected = {
            "causal_partial_pairs": (derivation["causal_tile_pairs_all_query_heads"], "pairs"),
            "unique_partial_scratch": (derivation["total_partial_bytes"], "bytes"),
            "retention_allowance": (result["phase_contract"]["maximum_on_chip_retention_bytes_between_phases"], "bytes"),
            "mandatory_partial_hbm_traffic_floor": (derivation["partial_hbm_traffic_floor_bytes"], "bytes"),
            "transfer_floor_at_8_decimal_tb_per_s": (derivation["hypothetical_bandwidth_floors_ms"]["8"], "ms"),
            "transfer_floor_at_6_4_decimal_tb_per_s": (derivation["hypothetical_bandwidth_floors_ms"]["6.4"], "ms"),
            "illustrative_target": (screen["illustrative_target_ms"], "ms"),
            "peak_two_query_tile_batch_scratch": (batch_two["maximum_partial_scratch_bytes"], "bytes"),
            "logical_final_fp32_dk_dv": (batch_two["logical_final_dk_dv_bytes_residency_unspecified"], "bytes"),
            "retention_equality_at_8_tb_s": (screen["retention_at_equal_floor_bytes"]["8"], "bytes"),
            "retention_equality_at_6_4_tb_s": (screen["retention_at_equal_floor_bytes"]["6.4"], "bytes"),
            "bandwidth_equality_at_128_mib_retention": (screen["bandwidth_at_equal_floor_decimal_tb_per_s"], "decimal TB/s"),
        }
        self.assertEqual(len(evidence["values"]), len(expected))
        self.assertEqual({row["name"] for row in evidence["values"]}, set(expected))
        for row in evidence["values"]:
            with self.subTest(value=row["name"]):
                value, unit = expected[row["name"]]
                self.assertEqual(Decimal(row["value"]), Decimal(value))
                self.assertEqual(row["unit"], unit)

        for field, relative in (("code", "code/roofline.py"), ("raw_results", "evidence/derivation/RESULTS.json")):
            with self.subTest(reference=field):
                reference = evidence[field]
                self.assertEqual(reference["path"], relative)
                data = (package / relative).read_bytes()
                self.assertEqual(reference["sha256"], hashlib.sha256(data).hexdigest())
        imported_calculator = Path(sys.modules[build_results.__module__].__file__).resolve()
        self.assertEqual(imported_calculator, (package / evidence["code"]["path"]).resolve())
        retained_results = json.loads((package / evidence["raw_results"]["path"]).read_bytes())
        self.assertEqual(retained_results, result)

    def test_reader_conversions_and_two_buffer_capacity_are_exact(self):
        result = build_results()
        derivation = result["derivation"]
        self.assertEqual(derivation["not_retained_bytes_at_phase_boundary"], 973078528)
        self.assertEqual(973078528, 928 * MIB)
        self.assertEqual(derivation["partial_hbm_traffic_floor_bytes"], 1946157056)
        self.assertEqual(1946157056, 1856 * MIB)
        capacity = result["target_screen"]["retention_at_equal_floor_bytes"]["8"]
        self.assertEqual(round(Decimal(capacity) / MIB, 2), Decimal("293.06"))
        batch_two = next(row for row in result["reduction_batch_sensitivity"] if row["query_tiles_per_batch"] == 2)
        both_buffers = batch_two["maximum_partial_scratch_bytes"] + batch_two["logical_final_dk_dv_bytes_residency_unspecified"]
        self.assertEqual(both_buffers, 130 * MIB)
        self.assertGreater(both_buffers, 128 * MIB)

    def test_target_equality_is_a_boundary_not_a_feasibility_claim(self):
        result = build_results()
        screen = result["target_screen"]
        self.assertEqual(screen["floor_exceeds_target"], {"8": True, "6.4": True})
        self.assertEqual(screen["retention_at_equal_floor_bytes"]["8"], 307296256)
        self.assertEqual(screen["retention_at_equal_floor_bytes"]["6.4"], 467296256)
        size = AttentionSpec().total_partial_bytes
        for bandwidth, capacity in (("8", 307296256), ("6.4", 467296256)):
            self.assertEqual(time_floor_ms(traffic_floor_bytes(size, capacity), bandwidth), Decimal("0.200"))
            self.assertGreater(time_floor_ms(traffic_floor_bytes(size, capacity - 1), bandwidth), Decimal("0.200"))
            self.assertLess(time_floor_ms(traffic_floor_bytes(size, capacity + 1), bandwidth), Decimal("0.200"))
        self.assertEqual(time_floor_ms(1946157056, screen["bandwidth_at_equal_floor_decimal_tb_per_s"]), Decimal("0.200"))


class CLIReproductionTests(unittest.TestCase):
    def test_cli_checks_zero_retention_csv_and_rejects_modified_output(self):
        script = Path(__file__).with_name("roofline.py").resolve()
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            command = [sys.executable, str(script), "--output-dir", str(output)]
            written = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertEqual(written.returncode, 0, written.stderr)
            names = {"RESULTS.json", "cache_sensitivity.csv", "reduction_batch_sensitivity.csv"}
            self.assertEqual({path.name for path in output.iterdir()}, names)

            result = json.loads((output / "RESULTS.json").read_text(encoding="utf-8"))
            zero = next(row for row in result["cache_sensitivity"] if row["retained_capacity_mib"] == 0)
            self.assertEqual(zero["retained_capacity_bytes"], 0)
            self.assertEqual(zero["partial_hbm_traffic_floor_bytes"], 2214592512)
            self.assertEqual(Decimal(zero["floor_ms_at_8_decimal_tb_per_s"]), Decimal("0.276824064"))

            csv_path = output / "cache_sensitivity.csv"
            with csv_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 10)
            zero_csv = next(row for row in rows if row["retained_capacity_mib"] == "0")
            self.assertEqual(zero_csv["retained_capacity_bytes"], "0")
            self.assertEqual(zero_csv["partial_hbm_traffic_floor_bytes"], "2214592512")
            self.assertEqual(Decimal(zero_csv["floor_ms_at_8_decimal_tb_per_s"]), Decimal("0.276824064"))

            original = {name: (output / name).read_bytes() for name in names}
            checked = subprocess.run(command + ["--check"], capture_output=True, text=True, timeout=10)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertIn("Reproduction check passed", checked.stdout)
            self.assertEqual({name: (output / name).read_bytes() for name in names}, original)

            modified = original[csv_path.name].replace(b"2214592512", b"2214592513", 1)
            self.assertNotEqual(modified, original[csv_path.name])
            csv_path.write_bytes(modified)
            rejected = subprocess.run(command + ["--check"], capture_output=True, text=True, timeout=10)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("Reproduction check failed:", rejected.stderr)
            self.assertIn(str(csv_path), rejected.stderr)
            expected_after_failure = dict(original, **{csv_path.name: modified})
            self.assertEqual({name: (output / name).read_bytes() for name in names}, expected_after_failure)


if __name__ == "__main__":
    unittest.main()
