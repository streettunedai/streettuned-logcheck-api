import unittest
import pandas as pd

from app.analyzers.ls_gas import detect_maf_frequency, detect_wideband


def s(vals):
    return pd.Series(vals)


class TestChannelDetection(unittest.TestCase):
    def test_maf_confirmed_mass_airflow_sensor_hz(self):
        df = pd.DataFrame({"Mass Airflow Sensor (Hz)": s([2000, 2200, 2500, 2300, 2700, 3000, 2800, 2600, 2900])})
        r = detect_maf_frequency(df)
        self.assertEqual(r["status"], "CONFIRMED")

    def test_maf_missing_or_rejects_calculated(self):
        self.assertEqual(detect_maf_frequency(pd.DataFrame({"Dynamic Airflow lb/h": s([10, 11, 12, 12, 13])}))['status'], 'MISSING')
        self.assertEqual(detect_maf_frequency(pd.DataFrame({"Mass Airflow Sensor g/s": s([4, 5, 6, 7, 8, 9, 10, 9, 8])}))['status'], 'SUSPECT')

    def test_maf_likely_missing_unit(self):
        df = pd.DataFrame({"Mass Airflow Sensor": s([1500, 1700, 1600, 1900, 2100, 1800, 2200, 2050, 2300])})
        self.assertEqual(detect_maf_frequency(df)["status"], "LIKELY")

    def test_wideband_confirmed_and_rejects(self):
        df = pd.DataFrame({"Wideband AFR": s([14.7, 13.8, 12.4, 11.9, 12.8, 13.2, 14.1, 14.7, 15.0])})
        self.assertEqual(detect_wideband(df)["status"], "CONFIRMED")
        self.assertEqual(detect_wideband(pd.DataFrame({"O2 B1 V": s([0.1, 0.8, 0.1, 0.9, 0.1, 0.8, 0.1, 0.9, 0.1])}))['status'], 'MISSING')

    def test_wideband_analog_voltage_is_suspect(self):
        df = pd.DataFrame({"Analog 1 V": s([0.1, 0.2, 0.3, 0.2, 0.4, 0.5, 0.45, 0.55, 0.6])})
        result = detect_wideband(df)
        self.assertEqual(result["status"], "SUSPECT")
        self.assertNotEqual(result["status"], "CONFIRMED")

    def test_wideband_eio_afr_is_confirmed(self):
        df = pd.DataFrame({"EIO Input 1 AFR": s([14.8, 14.2, 13.0, 12.2, 12.9, 13.4, 14.0, 14.6, 15.0])})
        self.assertEqual(detect_wideband(df)["status"], "CONFIRMED")

    def test_wideband_likely_when_unit_missing(self):
        df = pd.DataFrame({"Wideband": s([0.95, 0.98, 1.02, 0.92, 0.89, 1.05, 1.01, 0.97, 1.03])})
        self.assertEqual(detect_wideband(df)["status"], "LIKELY")

    def test_commanded_afr_is_not_wideband(self):
        df = pd.DataFrame({"Commanded AFR": s([14.7, 14.2, 13.5, 12.8, 12.3, 12.0, 12.6, 13.2, 13.8])})
        self.assertEqual(detect_wideband(df)["status"], "MISSING")

    def test_plain_maf_not_auto_confirmed_without_hz(self):
        df = pd.DataFrame({"MAF": s([1100, 1300, 1500, 1250, 1600, 1800, 1700, 1900, 2000])})
        self.assertNotEqual(detect_maf_frequency(df)["status"], "CONFIRMED")


if __name__ == '__main__':
    unittest.main()
