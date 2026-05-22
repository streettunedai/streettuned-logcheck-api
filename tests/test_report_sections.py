import unittest

from app.analyzers.ls_gas import build_report_sections


class TestReportSections(unittest.TestCase):
    def test_report_sections_include_richer_tuner_guidance(self):
        sections = build_report_sections(
            meta={"filename": "i75.csv", "row_count": 100, "column_count": 20},
            summary={"log_duration_sec": 120.0, "max_map_kpa": 97.0, "has_commanded_fueling_channel": True},
            trust_buckets={"confirmed_channels": ["RPM"], "missing_channels": ["WB_AFR"], "invalid_channels": [], "uncertain_channels": []},
            fueling_guidance={"can_make_wot_fueling_suggestions": False, "reason_wot_fueling_limited": "No trustworthy external wideband actual available; do not use narrowbands as actual AFR."},
            kr_events=[{"peak_kr_deg": 5.9}],
        )
        self.assertIn("what_this_log_can_be_used_for", sections)
        self.assertIn("what_this_log_cannot_prove", sections)
        self.assertIn("hard_stops", sections)
        self.assertIn("fueling_read", sections)
        self.assertIn("timing_kr_read", sections)
        self.assertIn("airflow_maf_map_read", sections)
        self.assertIn("temperature_heat_risk", sections)
        self.assertIn("what_can_be_edited", sections)
        self.assertTrue(sections["what_i_see"]["why_hard_stop_matters"])
        self.assertTrue(sections["what_i_see"]["likely_causes_to_inspect"])


if __name__ == "__main__":
    unittest.main()
