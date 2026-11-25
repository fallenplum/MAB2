from itertools import product
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.e0_certificate import fixed_threshold_certificate, full_resolution_status


class CertificateTests(unittest.TestCase):
    def test_empty_task_is_invalid_and_not_an_empty_decision_set(self):
        result = fixed_threshold_certificate([])
        self.assertEqual(result.status, "INVALID")
        self.assertIsNone(result.decision)

    def test_all_small_ternary_cases_are_conservative(self):
        for labels in product((0, 1, None), repeat=4):
            result = fixed_threshold_certificate(labels, threshold=0.75)
            self.assertIn(result.status, {"CERTIFIED", "UNRESOLVED", "INVALID"})
            if result.status == "CERTIFIED":
                if result.decision == "PASS":
                    self.assertGreaterEqual(result.supported, result.required_support)
                else:
                    self.assertLess(result.supported + result.unknown, result.required_support)

    def test_threshold_boundary_and_partial_state(self):
        self.assertEqual(fixed_threshold_certificate([1, 1, 1], threshold=0.75).decision, "PASS")
        self.assertEqual(fixed_threshold_certificate([1, 1, 1, None]).status, "UNRESOLVED")
        unresolved = fixed_threshold_certificate([1, 1, None, None])
        self.assertEqual(unresolved.status, "UNRESOLVED")
        self.assertIsNone(unresolved.decision)
        self.assertEqual(full_resolution_status([1, 0, 1]), "FULL_RESOLVED")
        self.assertEqual(full_resolution_status([1, None, 1]), "PARTIAL")

    def test_illegal_values_fail_closed(self):
        result = fixed_threshold_certificate([1, 2, None])
        self.assertEqual(result.status, "INVALID")
        self.assertEqual(result.reason, "illegal_label")


if __name__ == "__main__":
    unittest.main()


