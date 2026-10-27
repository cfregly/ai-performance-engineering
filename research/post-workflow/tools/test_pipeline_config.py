import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from tools.pipeline_config import load_config, main


class PipelineConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.path = self.root / "pipeline.json"
        self.config = {"schema_version": 1, "identity": {"name": "Example Author", "github_owner": "example-user"}, "repositories": {"drafts": "example-user/posts-drafts", "public": "example-user/research", "public_path": "research"}, "schedule": {"timezone": "America/Los_Angeles", "hours": [17, 5], "minute": 0}, "gpu": {"ssh_host": "gpu-host", "max_smoke_seconds": 180}, "publication": {"mode": "draft_only"}}

    def load(self):
        self.path.write_text(json.dumps(self.config), encoding="utf-8")
        return load_config(self.path)

    def invalid(self, message):
        with self.assertRaisesRegex(ValueError, message):
            self.load()

    def cli(self, *args):
        self.load()
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = main([str(self.path), *map(str, args)])
        return result, stdout.getvalue(), stderr.getvalue()

    def test_normalized_config_adds_drafts_path_and_sorts_hours(self):
        loaded = self.load()
        self.assertEqual(loaded["repositories"]["drafts_path"], "posts")
        self.assertEqual(loaded["schedule"]["hours"], [5, 17])
        self.assertEqual(loaded["publication"], {"mode": "draft_only"})

    def test_generic_example_is_valid(self):
        example = Path(__file__).resolve().parents[1] / "pipeline.example.json"
        self.assertEqual(load_config(example)["identity"]["github_owner"], "example-user")

    def test_root_path_and_custom_drafts_subdir(self):
        self.config["repositories"].update(public_path=".", drafts_path="research/drafts")
        loaded = self.load()
        self.assertEqual(loaded["repositories"]["public_path"], ".")
        result, output, errors = self.cli("--drafts-root", self.root)
        self.assertEqual((result, errors), (0, ""))
        self.assertEqual(Path(output.strip()), self.root / "research/drafts")

    def test_public_root_resolves_subdirectory(self):
        result, output, errors = self.cli("--public-root", self.root)
        self.assertEqual((result, errors), (0, ""))
        self.assertEqual(Path(output.strip()), self.root / "research")

    def test_schedule_is_timezone_aware_json(self):
        result, output, errors = self.cli("--schedule")
        self.assertEqual((result, errors), (0, ""))
        self.assertEqual(json.loads(output), {"timezone": "America/Los_Angeles", "hours": [5, 17], "minute": 0})

    def test_cli_default_prints_normalized_config(self):
        result, output, errors = self.cli()
        self.assertEqual((result, errors), (0, ""))
        self.assertEqual(json.loads(output)["repositories"]["drafts_path"], "posts")

    def test_get_prints_string_and_numeric_scalars(self):
        for key, expected in (("repositories.public", "example-user/research\n"), ("repositories.public_path", "research\n"), ("gpu.max_smoke_seconds", "180\n")):
            with self.subTest(key=key):
                result, output, errors = self.cli("--get", key)
                self.assertEqual((result, output, errors), (0, expected, ""))

    def test_get_rejects_missing_key(self):
        result, output, errors = self.cli("--get", "identity.unknown")
        self.assertEqual((result, output), (1, ""))
        self.assertIn("configuration key not found", errors)

    def test_get_rejects_object_or_list(self):
        for key in ("repositories", "schedule.hours"):
            with self.subTest(key=key):
                result, output, errors = self.cli("--get", key)
                self.assertEqual((result, output), (1, ""))
                self.assertIn("must identify a scalar", errors)

    def test_owner_must_match_drafts_repository(self):
        self.config["repositories"]["drafts"] = "another-owner/posts-drafts"
        self.invalid("owner must match")

    def test_case_insensitive_owner_match(self):
        self.config["identity"]["github_owner"] = "Example-User"
        self.assertEqual(self.load()["identity"]["github_owner"], "Example-User")

    def test_repositories_must_differ_ignoring_case(self):
        self.config["repositories"]["public"] = "EXAMPLE-USER/POSTS-DRAFTS"
        self.invalid("must differ")

    def test_invalid_repository_syntax(self):
        for value in ("https://github.com/example-user/research", "example-user", "example-user/../research", "example-user/..", "bad--owner/research", "-owner/research", "owner/repo name"):
            with self.subTest(value=value):
                self.config["repositories"]["public"] = value
                self.invalid("repositories.public")

    def test_unsafe_repository_paths(self):
        for field in ("public_path", "drafts_path"):
            for value in ("/Users/example/research", "../research", "research/../private", "C:\\Users\\example", "~/research", ".git/hooks", "research;echo-forbidden", "research/$(command)", "research folder", "research/'quoted'", " research", "research ", ""):
                with self.subTest(field=field, value=value):
                    self.config["repositories"][field] = value
                    self.invalid(f"repositories.{field}")
            self.config["repositories"][field] = "posts"

    def test_symlink_cannot_escape_checkout(self):
        with tempfile.TemporaryDirectory() as outside:
            (self.root / "research").symlink_to(outside, target_is_directory=True)
            result, output, errors = self.cli("--public-root", self.root)
            self.assertEqual((result, output), (1, ""))
            self.assertIn("outside the checkout", errors)

    def test_invalid_timezone(self):
        self.config["schedule"]["timezone"] = "Not/A_Real_Zone"
        self.invalid("IANA timezone")

    def test_invalid_hours(self):
        for value in ([], [5, 5], [24], [-1], [True], [5.0], "5,17"):
            with self.subTest(value=value):
                self.config["schedule"]["hours"] = value
                self.invalid("schedule.hours")

    def test_invalid_minute(self):
        for value in (-1, 60, True, 5.0):
            with self.subTest(value=value):
                self.config["schedule"]["minute"] = value
                self.invalid("schedule.minute")

    def test_smoke_limit_is_one_through_300_seconds(self):
        for value in (0, 301, True, 180.0):
            with self.subTest(value=value):
                self.config["gpu"]["max_smoke_seconds"] = value
                self.invalid("max_smoke_seconds")

    def test_ssh_host_is_not_command_or_credentials(self):
        for value in ("user@host", "ssh gpu-host", "/Users/example/.ssh/id_rsa", "gpu-host; publish", "-oProxyCommand=other"):
            with self.subTest(value=value):
                self.config["gpu"]["ssh_host"] = value
                self.invalid("gpu.ssh_host")

    def test_publication_must_remain_draft_only(self):
        self.config["publication"]["mode"] = "publish"
        self.invalid("must be draft_only")

    def test_unknown_fields_at_every_level(self):
        original = copy.deepcopy(self.config)
        for section in (None, "identity", "repositories", "schedule", "gpu", "publication"):
            with self.subTest(section=section):
                self.config = copy.deepcopy(original)
                (self.config if section is None else self.config[section])["surprise"] = True
                self.invalid("unknown field")

    def test_secret_fields_are_rejected(self):
        self.config["gpu"]["private_key"] = "never-saved"
        self.invalid("secret and credential fields are forbidden")

    def test_invalid_config_does_not_print_secret_value(self):
        self.config["gpu"]["password"] = "value-that-must-not-be-printed"
        self.path.write_text(json.dumps(self.config), encoding="utf-8")
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = main([str(self.path)])
        self.assertEqual((result, stdout.getvalue()), (1, ""))
        self.assertNotIn("value-that-must-not-be-printed", stderr.getvalue())

    def test_optional_theme_is_a_name(self):
        self.config["theme"] = "whiteboard"
        self.assertEqual(self.load()["theme"], "whiteboard")
        self.config["theme"] = {"api_key": "forbidden"}
        self.invalid("theme")

    def test_schema_version_is_exact_integer(self):
        for value in (True, 1.0, 2):
            with self.subTest(value=value):
                self.config["schema_version"] = value
                self.invalid("schema_version must be 1")

    def test_missing_required_field(self):
        del self.config["publication"]
        self.invalid("config.publication is required")

    def test_duplicate_json_fields_are_rejected(self):
        self.path.write_text('{"schema_version": 1, "schema_version": 2}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate configuration field"):
            load_config(self.path)

    def test_invalid_json_and_missing_file_are_actionable(self):
        self.path.write_text("not JSON", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "cannot read pipeline config"):
            load_config(self.path)
        self.path.unlink()
        with self.assertRaisesRegex(ValueError, "cannot read pipeline config"):
            load_config(self.path)


if __name__ == "__main__":
    unittest.main()
