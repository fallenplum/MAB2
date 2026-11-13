from decimal import Decimal
import unittest
from reference import vote,vote_text
class Tests(unittest.TestCase):
    def test_missing_votes(self):self.assertIsNone(vote([None]*4))
    def test_tie_earliest(self):self.assertEqual(vote([Decimal('2'),Decimal('1'),Decimal('1'),Decimal('2')]),Decimal('2'))
    def test_majority_over_earliest(self):self.assertEqual(vote([Decimal('2'),Decimal('1'),None,Decimal('1')]),Decimal('1'))
    def test_canonical_numeric(self):self.assertEqual(vote([Decimal('1.0'),Decimal('1.00'),Decimal('2'),None]),Decimal('1'))
    def test_decimal_no_scientific_notation(self):self.assertEqual(vote_text(Decimal('0.000000001')),'0.000000001')
if __name__=='__main__':unittest.main()
