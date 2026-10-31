from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from tools.typefully_analytics import COUNTS, FLAGS, summarize


class AnalyticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "posts.csv"
        self.row = {
            "tweet_id": "123", "created_at": "2026-10-26T03:30:46Z", "text": "A thread part.",
            "url": "https://x.com/example/status/123", **{field: "0" for field in COUNTS},
            **{field: "false" for field in FLAGS}, "is_thread_part": "true", "engagement_rate": "0.03",
            "impression_count": "1000", "retweet_count": "2", "bookmark_count": "3",
            "conversation_length": "", "article": "",
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self, rows: list[dict], fields: list[str] | None = None) -> None:
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields or list(self.row))
            writer.writeheader()
            writer.writerows(rows)

    def summary(self, start: str = "2026-10-25", end: str = "2026-10-26") -> dict:
        return summarize(self.path, start, end, "2026-10-26T12:00:00-07:00")

    def test_actual_schema_rates_counts_and_thread_parts(self) -> None:
        self.write([self.row])
        result = self.summary()
        post = result["posts"][0]
        self.assertEqual(result["sample_size"], 1)
        self.assertEqual(post["reposts_per_1000_impressions"], 2)
        self.assertEqual(post["bookmarks_per_1000_impressions"], 3)
        self.assertEqual(post["rate_denominator_impressions"], 1000)
        self.assertEqual(post["source_engagement_rate"], 0.03)
        self.assertIs(post["is_thread_part"], True)

    def test_blank_is_null_but_zero_remains_zero(self) -> None:
        self.row.update(url_link_clicks="", user_profile_clicks="0")
        self.write([self.row])
        result = self.summary()
        self.assertIsNone(result["posts"][0]["counts"]["url_link_clicks"])
        self.assertEqual(result["posts"][0]["counts"]["user_profile_clicks"], 0)
        self.assertIsNone(result["aggregate"]["totals"]["url_link_clicks"])
        self.assertEqual(result["aggregate"]["missing_counts"]["url_link_clicks"], 1)

    def test_missing_numerator_prevents_aggregate_rate(self) -> None:
        self.write([self.row, {**self.row, "tweet_id": "124", "retweet_count": ""}])
        result = self.summary()
        self.assertIsNone(result["aggregate"]["reposts_per_1000_impressions"])
        self.assertEqual(result["rankable_sample_size"], 1)

    def test_zero_impressions_is_not_a_zero_rate(self) -> None:
        self.row["impression_count"] = "0"
        self.write([self.row])
        result = self.summary()
        self.assertIsNone(result["posts"][0]["reposts_per_1000_impressions"])
        self.assertIsNone(result["aggregate"]["bookmarks_per_1000_impressions"])
        self.assertEqual(result["rankable_sample_size"], 0)

    def test_duplicate_id_rejected_even_outside_window(self) -> None:
        self.write([self.row, self.row])
        with self.assertRaisesRegex(ValueError, "duplicate tweet_id"):
            self.summary("2026-10-01", "2026-10-02")

    def test_timezone_creation_window_boundaries(self) -> None:
        self.write([{**self.row, "tweet_id": str(i), "created_at": created} for i, created in enumerate(
            ("2026-10-25T06:59:59Z", "2026-10-25T07:00:00Z", "2026-10-26T06:59:59Z", "2026-10-26T07:00:00Z"))])
        result = self.summary("2026-10-25", "2026-10-25")
        self.assertEqual([post["tweet_id"] for post in result["posts"]], ["1", "2"])

    def test_invalid_counts_required_columns_and_window(self) -> None:
        for value in ("-1", "NaN", "Infinity", "0.5"):
            with self.subTest(value=value):
                self.write([{**self.row, "retweet_count": value}])
                with self.assertRaises(ValueError):
                    self.summary()
        self.write([{key: value for key, value in self.row.items() if key != "retweet_count"}],
                   [key for key in self.row if key != "retweet_count"])
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            self.summary()
        with self.assertRaisesRegex(ValueError, "ascending order"):
            self.summary("2026-10-26", "2026-10-25")

    def test_ranking_is_descending_without_sample_cutoff(self) -> None:
        self.write([self.row, {**self.row, "tweet_id": "124", "retweet_count": "3"}])
        result = self.summary()
        self.assertEqual([post["tweet_id"] for post in result["ranked_by_reposts_per_1000_impressions"]], ["124", "123"])


if __name__ == "__main__":
    unittest.main()
