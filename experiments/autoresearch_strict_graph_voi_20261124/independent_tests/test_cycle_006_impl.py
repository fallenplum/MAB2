from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1]
class Cycle006ImplementationTests(unittest.TestCase):
    def test_cycle006_implementation_identity(self):
        src=(ROOT/'cycles'/'h2_fast.py').read_text(encoding='utf-8')
        self.assertIn('label_orders',src)
        self.assertIn('float(np.mean(completion))',src)
        self.assertIn('terminal in current',src)
        self.assertIn('min(options,key=lambda item:(item[0],item[1],item[2]))',src)
        self.assertNotIn('best_value + 1e-9',src)
if __name__=='__main__': unittest.main()

