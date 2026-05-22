import importlib
import os
import unittest
from unittest.mock import patch


class _Req:
    def __init__(self, platform: str):
        self.query_params = {"platform": platform}


class TestPlatformRouting(unittest.TestCase):
    def test_non_strict_allows_ls_and_duramax(self):
        with patch.dict(os.environ, {}, clear=False):
            import app.main as main_mod
            importlib.reload(main_mod)
            self.assertEqual(main_mod.resolve_platform(_Req("ls")), "ls")
            self.assertEqual(main_mod.resolve_platform(_Req("duramax")), "duramax")

    def test_strict_ls_blocks_non_ls(self):
        with patch.dict(os.environ, {"STRICT_PLATFORM": "ls"}, clear=False):
            import app.main as main_mod
            importlib.reload(main_mod)
            self.assertEqual(main_mod.resolve_platform(_Req("ls")), "ls")
            with self.assertRaises(Exception):
                main_mod.resolve_platform(_Req("duramax"))
            with self.assertRaises(Exception):
                main_mod.resolve_platform(_Req("auto"))


if __name__ == "__main__":
    unittest.main()
