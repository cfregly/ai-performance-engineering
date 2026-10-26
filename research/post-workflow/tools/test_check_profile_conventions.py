import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.check_profile_conventions import validate_profile_conventions


class ProfileConventionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.package = Path(self.temp.name)
        self.views = self.package / "zymtrace/views"
        self.views.mkdir(parents=True)
        self.authority = {"schema_field": "JsonEvent.c", "schema_value": "Time spent in this event.", "quantity": "gpu_time", "unit": "fraction", "normalization": "per_worker_sum", "absolute_export_unit": "undocumented"}
        self.write("authority.json", self.authority)
        self.semantics = {
            "schema_version": 1,
            "weights": {"quantity": "gpu_time_share", "unit": "fraction", "field": "c", "normalization": "per_worker_sum"},
            "unit_authority": {"verified": True, "schema_field": "JsonEvent.c", "schema_value": "Time spent in this event.", "absolute_export_unit": "undocumented", "source_url": "https://example.org/schema", "source_path": "authority.json", "sha256": self.digest("authority.json")},
            "scope": {"phases": ["setup", "warmup", "timing", "correctness"], "normalized_width": True, "disclosure": "Percentages are independently normalized to each worker's total captured GPU time."},
            "arms": {},
        }
        self.provenance = {"inputs": {}, "outputs": {}}
        for arm, pid in (("disabled", 101), ("enabled", 202)):
            self.write(f"{arm}.json", {"strings": ["cuda", "uuid"], "events": [{"pid": pid, "u": 1, "k": 0, "sid": 0, "c": 0.25, "f": [0, 1]}, {"pid": pid, "u": 1, "k": 0, "sid": 0, "c": 0.5, "f": [2, 3, 4]}, {"pid": 999, "u": 1, "k": 0, "sid": 0, "c": 99}]})
            (self.views / f"{arm}.png").write_bytes(f"retained image {arm}".encode())
            self.write(f"zymtrace/views/{arm}.stacks.json", {"pid": pid, "gpu_uuid": "GPU-uuid", "unit": "fraction of captured GPU time", "records": [{"raw_event_index": 0, "exported_time_weight": 0.25, "gpu_time_fraction": 0.25 / 0.75, "frame_ids": [0, 1]}, {"raw_event_index": 1, "exported_time_weight": 0.5, "gpu_time_fraction": 0.5 / 0.75, "frame_ids": [2, 3, 4]}]})
            self.bind_output(f"zymtrace/views/{arm}.png")
            self.bind_output(f"zymtrace/views/{arm}.stacks.json")
            totals = {"accepted_records": 2, "excluded_records": 1, "sum_exported_time_weights": 0.75, "sum_gpu_time_fractions": 1.0}
            self.semantics["arms"][arm] = {"expected_pid": pid, "gpu_uuid": "GPU-uuid", "events": {"path": f"{arm}.json", "sha256": self.digest(f"{arm}.json")}, "images": [f"zymtrace/views/{arm}.png"], **totals}
            self.provenance["inputs"][arm] = {"pid": pid, **totals}

    def tearDown(self):
        self.temp.cleanup()

    def write(self, path, value):
        (self.package / path).write_text(json.dumps(value))

    def digest(self, path):
        return hashlib.sha256((self.package / path).read_bytes()).hexdigest()

    def bind_output(self, relative):
        self.provenance["outputs"][relative] = {"sha256": self.digest(relative), "bytes": (self.package / relative).stat().st_size}

    def validate(self):
        self.write("zymtrace/views/profile-semantics.json", self.semantics)
        self.write("zymtrace/views/provenance.json", self.provenance)
        return validate_profile_conventions(self.package)

    def test_valid_filtered_time_weights(self):
        self.assertEqual(self.validate(), [])

    def test_fraction_normalization_is_independent_of_export_scale(self):
        raw = json.loads((self.package / "disabled.json").read_text())
        for row in raw["events"]:
            row["c"] *= 1000
        self.write("disabled.json", raw)
        self.semantics["arms"]["disabled"]["events"]["sha256"] = self.digest("disabled.json")
        self.semantics["arms"]["disabled"]["sum_exported_time_weights"] = 750
        self.provenance["inputs"]["disabled"]["sum_exported_time_weights"] = 750
        sidecar_path = "zymtrace/views/disabled.stacks.json"
        sidecar = json.loads((self.package / sidecar_path).read_text())
        for row in sidecar["records"]:
            row["exported_time_weight"] *= 1000
        self.write(sidecar_path, sidecar)
        self.bind_output(sidecar_path)
        self.assertEqual(self.validate(), [])

    def test_legacy_without_views_is_unaffected(self):
        self.assertEqual(validate_profile_conventions(self.package / "other"), [])

    def test_count_weights_rejected(self):
        self.semantics["weights"]["unit"] = "count"
        self.assertTrue(any("never count" in e for e in self.validate()))

    def test_old_count_provenance_rejected(self):
        self.write("zymtrace/views/provenance.json", {"width_unit": "one exported CUDA event row"})
        self.assertTrue(any("count weights" in e for e in validate_profile_conventions(self.package)))

    def test_count_renderer_metadata_rejected_even_with_time_declaration(self):
        self.provenance["width_unit"] = "one exported CUDA event row"
        self.assertTrue(any("count weights" in e for e in self.validate()))

    def test_guessed_seconds_rejected(self):
        self.semantics["weights"]["unit"] = "seconds"
        self.assertTrue(any("guessed absolute units" in e for e in self.validate()))

    def test_fraction_total_rejected(self):
        self.semantics["arms"]["disabled"]["sum_gpu_time_fractions"] = 0.25
        self.assertTrue(any("sum_gpu_time_fractions" in e for e in self.validate()))

    def test_incorrect_retained_authority_rejected(self):
        self.authority["absolute_export_unit"] = "seconds"
        self.write("authority.json", self.authority)
        self.semantics["unit_authority"]["sha256"] = self.digest("authority.json")
        self.assertTrue(any("retained unit authority" in e for e in self.validate()))

    def test_normalization_required(self):
        self.semantics["scope"]["normalized_width"] = False
        self.assertTrue(any("normalized_width=true" in e for e in self.validate()))

    def test_missing_authority_rejected(self):
        del self.semantics["unit_authority"]
        self.assertTrue(any("unit authority" in e for e in self.validate()))

    def test_wrong_pid_or_gpu_rejected(self):
        for key, value in (("expected_pid", 303), ("gpu_uuid", "wrong")):
            with self.subTest(key=key):
                old = self.semantics["arms"]["disabled"][key]
                self.semantics["arms"]["disabled"][key] = value
                self.assertTrue(any("filtered source" in e for e in self.validate()))
                self.semantics["arms"]["disabled"][key] = old

    def test_duplicate_pids_rejected(self):
        self.semantics["arms"]["enabled"]["expected_pid"] = 101
        self.assertTrue(any("distinct" in e for e in self.validate()))

    def test_hash_mismatch_rejected(self):
        self.semantics["arms"]["disabled"]["events"]["sha256"] = "0" * 64
        self.assertTrue(any("sha256 mismatch" in e for e in self.validate()))

    def test_nonfinite_weights_rejected(self):
        for weight in (float("nan"), float("inf"), -1, 0):
            with self.subTest(weight=weight):
                self.write("disabled.json", {"events": [{"pid": 101, "u": "uuid", "k": "cuda", "sid": 0, "c": weight}]})
                self.semantics["arms"]["disabled"]["events"]["sha256"] = self.digest("disabled.json")
                self.assertTrue(any("positive and finite" in e for e in self.validate()))

    def test_provenance_totals_rejected(self):
        self.provenance["inputs"]["disabled"]["sum_exported_time_weights"] = 500
        self.assertTrue(any("provenance sum_exported_time_weights" in e for e in self.validate()))

    def test_stale_swapped_image_rejected(self):
        (self.views / "disabled.png").write_bytes((self.views / "enabled.png").read_bytes())
        self.assertTrue(any("image or sidecar hash/bytes mismatch" in e for e in self.validate()))

    def test_image_provenance_required(self):
        del self.provenance["outputs"]["zymtrace/views/disabled.png"]
        self.assertTrue(any("missing from provenance.outputs" in e for e in self.validate()))

    def test_wrong_sidecar_fraction_rejected_even_with_updated_hash(self):
        relative = "zymtrace/views/disabled.stacks.json"
        sidecar = json.loads((self.package / relative).read_text())
        sidecar["records"][0]["gpu_time_fraction"] = 0.5
        self.write(relative, sidecar)
        self.bind_output(relative)
        self.assertTrue(any("sidecar gpu_time_fraction" in e for e in self.validate()))

    def test_wrong_sidecar_stack_rejected_even_with_updated_hash(self):
        relative = "zymtrace/views/disabled.stacks.json"
        sidecar = json.loads((self.package / relative).read_text())
        sidecar["records"][0]["frame_ids"] = [0]
        self.write(relative, sidecar)
        self.bind_output(relative)
        self.assertTrue(any("full source stack" in e for e in self.validate()))

    def test_wrong_sidecar_source_index_rejected(self):
        relative = "zymtrace/views/disabled.stacks.json"
        sidecar = json.loads((self.package / relative).read_text())
        sidecar["records"][0]["raw_event_index"] = 2
        self.write(relative, sidecar)
        self.bind_output(relative)
        self.assertTrue(any("raw_event_index=0" in e for e in self.validate()))

    def test_missing_active_image_rejected(self):
        (self.views / "disabled.png").unlink()
        self.assertTrue(any("active image" in e for e in self.validate()))


if __name__ == "__main__":
    unittest.main()
