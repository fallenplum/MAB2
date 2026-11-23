from pathlib import Path
import unittest

class P2PermissionTest(unittest.TestCase):
    def test_old_cache_disables_new_interface_suffix(self):
        text=Path('p2_service_runner.py').read_text(encoding='utf-8')
        self.assertIn('interfaces = [""] * 4 if old else row.get("new_relay_interface", [""] * 4)', text)

if __name__ == '__main__': unittest.main()
