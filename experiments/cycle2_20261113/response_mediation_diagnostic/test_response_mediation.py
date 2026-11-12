from decimal import Decimal
import unittest
from response_mediation_scan import canonical_number

class CanonicalNumberTest(unittest.TestCase):
    def test_exact_decimal_equivalence_without_context_rounding(self):
        for a,b in [('1','1.0000'),('0','-0.000'),('100000','1E5'),('1.25','1.2500'),('123456789012345678901234567890123456789.1','123456789012345678901234567890123456789.10')]:
            self.assertEqual(canonical_number(Decimal(a)),canonical_number(Decimal(b)))
        self.assertNotEqual(canonical_number(Decimal('123456789012345678901234567890123456789.1')),canonical_number(Decimal('123456789012345678901234567890123456789.2')))
        self.assertNotEqual(canonical_number(None),canonical_number(Decimal('0')))

if __name__=='__main__':unittest.main()
