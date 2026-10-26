import csv
import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("quality_index", ROOT / "tools/corpus_quality_index.py")
index = importlib.util.module_from_spec(spec)
spec.loader.exec_module(index)


class IndexTests(unittest.TestCase):
    def test_persistence_idempotency_review_history_and_read_only_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp)
            db = report / "index.sqlite"
            row = dict(match_id="synthetic-match", map_name="Synthetic", game_mode="squad",
                match_type="official", created_at="2026-01-01", locked_not_inspected="False",
                anomaly_class="source_field_conflict", classification="usable_after_correction_or_masking",
                trajectory_status="needs_correction_or_masking", rank_status="needs_reconciliation",
                combat_status="not_checked", issue_codes="outcome_needs_reconciliation",
                rank_issue_codes="participant_rank_disagree", context_notes="",
                participant_rank_disagree="2", human_rank_disagree="2", bot_rank_disagree="0")
            with (report / "match_quality.csv").open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader(); writer.writerow(row)
            (report / "summary.json").write_text(json.dumps(dict(
                captured_at="2026-01-01", inventory_matches=1, classification_code_sha256="synthetic")))
            first = index.build(report, db)
            with sqlite3.connect(db) as conn:
                conn.execute("INSERT INTO issue_reviews VALUES(NULL,?,?,?,?,?,?,?)", (
                    first["run_id"], "synthetic-match", "participant_rank_disagree",
                    "2026-01-02", "synthetic-reviewer", "confirmed", "Synthetic test only"))
            second = index.build(report, db)
            self.assertEqual(first["run_id"], second["run_id"])
            self.assertEqual(second["issues"], 2)
            self.assertEqual(second["issue_reviews"], 1)
            result = index.query(db, issue="participant_rank_disagree")
            issue = next(i for i in result[0]["issues"] if i["code"] == "participant_rank_disagree")
            self.assertEqual(issue["review_status"], "confirmed")
            self.assertEqual(issue["evidence"]["participant_rank_disagree"], "2")
            self.assertEqual(index.query(db, match="' OR 1=1 --"), [])
            with index.read_only(db) as conn:
                with self.assertRaises(sqlite3.OperationalError):
                    conn.execute("DELETE FROM issues")
            with sqlite3.connect(db) as conn:
                with self.assertRaises(sqlite3.IntegrityError):
                    conn.execute("DELETE FROM issue_reviews")
            # A new audit appends a separate version rather than overwriting old evidence/reviews.
            summary = json.loads((report / "summary.json").read_text())
            summary["captured_at"] = "2026-01-03"
            (report / "summary.json").write_text(json.dumps(summary))
            third = index.build(report, db)
            self.assertNotEqual(first["run_id"], third["run_id"])
            self.assertEqual(third["issue_reviews"], 0)
            with index.read_only(db) as conn:
                self.assertEqual(conn.execute("SELECT count(*) FROM audit_runs").fetchone()[0], 2)
                self.assertEqual(conn.execute("SELECT count(*) FROM issue_reviews").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
