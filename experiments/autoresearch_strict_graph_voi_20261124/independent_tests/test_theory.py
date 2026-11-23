import unittest


class StrictCertificateTheoryTests(unittest.TestCase):
    def test_strict_margin_covers_all_completions(self):
        known_a, known_b, unknown = 2.0, 0.5, 0.25
        self.assertGreater(known_a, known_b + unknown)

    def test_equality_is_not_safe(self):
        self.assertFalse(1.0 > 0.0 + 1.0)

    def test_no_drift_has_empty_dirty_cone(self):
        old = (0, 1, 0)
        new = (0, 1, 0)
        self.assertEqual([i for i, (a, b) in enumerate(zip(old, new)) if a != b], [])


if __name__ == "__main__":
    unittest.main()

