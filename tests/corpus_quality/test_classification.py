"""Synthetic rule checks only; never access PUBG observations or train models."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("corpus_audit", ROOT / 'src/pubg_lab/data/quality/classify_collected_data.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ClassificationTests(unittest.TestCase):
    def run_case(self, overrides=None, meta=None, failed=False):
        inv = dict(match_id="synthetic", map_name="Baltic_Main", game_mode="squad",
                   match_type="official", telemetry_fetched=True)
        inv.update(meta or {})
        values = dict(match_id="synthetic",participant_rows=4,participant_accounts=4,
                      roster_rows=2,distinct_team_ids=2,distinct_ranks=2,max_rank=2,
                      rank1_rosters=1,won_rosters=1,start_time="start",end_time="end",
                      event_duration_s=100,position_rows=20,ingame_position_rows=20,
                      basic_valid_position_rows=20,roster_actors_with_positions=4,
                      positive_zone_pair_rows=10,damage_rows=3)
        values.update(overrides or {})
        records = {name:{"match_ids":["synthetic"],"ok":True,"rows":[]} for name in audit.QUERIES}
        records["structure"]["rows"]=[values]
        if failed:
            records["positions"]["ok"]=False
        return audit.classify([inv],[records],set())[0]

    def test_clean_checked_case(self):
        self.assertEqual(self.run_case()["classification"],"no_issue_detected_in_checked_rules")

    def test_rank_conflict_does_not_mean_whole_match_unusable(self):
        row=self.run_case({"participant_rank_disagree":1})
        self.assertEqual(row["classification"],"usable_after_correction_or_masking")
        self.assertEqual(row["rank_status"],"needs_reconciliation")
        self.assertEqual(row["anomaly_class"],"source_field_conflict")

    def test_clock_alignment_is_preprocessing_not_source_corruption(self):
        row=self.run_case({"game_clock_offset_median_s":8,"game_clock_offset_min_s":7,
                          "game_clock_offset_max_s":9})
        self.assertEqual(row["classification"],"usable_after_correction_or_masking")
        self.assertEqual(row["anomaly_class"],"no_anomaly_detected_in_checked_rules")

    def test_non_br_absent_rosters_not_corruption(self):
        row=self.run_case({"participant_rows":0,"participant_accounts":0,"roster_rows":0,
                          "distinct_team_ids":0,"roster_actors_with_positions":0},
                         {"map_name":"Range_Main","game_mode":"clansolo"})
        self.assertEqual(row["classification"],"no_issue_detected_in_checked_rules")
        self.assertEqual(row["rank_status"],"not_applicable")

    def test_multiple_lives_not_duplicate_error(self):
        row=self.run_case({"actors_with_multiple_death_times":2})
        self.assertEqual(row["classification"],"no_issue_detected_in_checked_rules")
        self.assertIn("possible_reentry",row["context_notes"])

    def test_bad_join_quarantines_core_tasks(self):
        self.assertEqual(self.run_case({"participant_cross_match_roster":1})["classification"],
                         "quarantine_for_core_tasks")

    def test_query_failure_never_becomes_missing_data_evidence(self):
        row=self.run_case(failed=True)
        self.assertEqual(row["classification"],"insufficient_evidence")
        self.assertEqual(row["issue_codes"],"query_incomplete")

    def test_locked_match_never_queried(self):
        inv=[dict(match_id="locked",map_name="Baltic_Main",game_mode="squad",telemetry_fetched=True)]
        row=audit.classify(inv,[],{"locked"})[0]
        self.assertEqual(row["classification"],"protected_not_inspected")
        with self.assertRaises(AssertionError):
            audit.classify(inv,[{"positions":{"match_ids":["locked"],"ok":True,"rows":[]}}],{"locked"})


if __name__ == "__main__":
    unittest.main()
