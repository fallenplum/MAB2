import itertools
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.r1_certificate import assert_invariant, certificate, required_support
from src.r1_math import clairvoyant_cost, uniform_expected_cost, uniform_expected_count
from src.r1_parsers import parse_doc_verdict, parse_fact_verdict


class R1CertificateTests(unittest.TestCase):
    def test_n33_boundaries(self):
        self.assertEqual(required_support(33, 80), 27)
        self.assertEqual(certificate([1] * 27 + [None] * 6).action, "PASS")
        self.assertEqual(certificate([0] * 7 + [None] * 26).action, "FAIL")
        self.assertEqual(certificate([0] * 6 + [None] * 27).status, "UNRESOLVED")

    def test_exhaustive_small_ternary_contract(self):
        for n in range(1, 9):
            for pct in (40, 50, 60, 70, 80, 90):
                for labels in itertools.product((0, 1, None), repeat=n):
                    cert = certificate(labels, pct)
                    assert_invariant(cert)
                    self.assertIn(cert.status, {"CERTIFIED", "FULL_RESOLVED", "UNRESOLVED", "INVALID"})

    def test_parser_separation_and_mutations(self):
        self.assertEqual(parse_fact_verdict("UN SUPPORTED"), 0)
        self.assertEqual(parse_fact_verdict("SUPPORT"), 1)
        self.assertIsNone(parse_fact_verdict("NOT SUPPORT"))
        self.assertEqual(parse_doc_verdict("PASS"), "PASS")
        self.assertEqual(parse_doc_verdict("FAIL"), "FAIL")
        self.assertIsNone(parse_doc_verdict("SUPPORT"))

    def test_exact_random_math(self):
        self.assertAlmostEqual(uniform_expected_count(2, 10, 4), 4.4)
        self.assertAlmostEqual(uniform_expected_cost(2, [1, 2, 3, 4], [5, 6]), 9.4)
        self.assertEqual(clairvoyant_cost(2, [4, 1, 3]), 4)


if __name__ == "__main__":
    unittest.main()

