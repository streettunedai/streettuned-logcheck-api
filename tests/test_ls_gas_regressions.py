import unittest
import pandas as pd

from app.analyzers.ls_gas import extract_kr_events, map_columns


class TestLsGasRegressions(unittest.TestCase):
    def test_kr_prefers_primary_channel_over_totalkr(self):
        df = pd.DataFrame(
            {
                "KR": [0.0, 0.0, 5.8, 5.9, 0.0, 0.0],
                "Total KR": [0.0, 0.0, 10.9, 10.2, 0.0, 0.0],
            }
        )
        events = extract_kr_events(df.rename(columns={"KR": "KR_deg", "Total KR": "TotalKR_deg"}))
        self.assertTrue(events)
        self.assertLessEqual(events[0]["peak_kr_deg"], 6.0)

    def test_map_boost_same_column_does_not_double_map_as_boost(self):
        df = pd.DataFrame(
            {
                "Engine RPM": [700, 1500, 2500],
                "TPS": [2, 20, 70],
                "Boost/Vacuum": [5.0, 10.0, 14.0],
            }
        )
        matched, _trust = map_columns(df)
        self.assertIn("MAP_kPa", matched)
        self.assertNotIn("Boost_kPa", matched)


if __name__ == "__main__":
    unittest.main()
