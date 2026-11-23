import unittest
import numpy as np
from p2_logic import terminal_answer

class P2DecisionTest(unittest.TestCase):
    def test_all_cells_weighted_by_matching_relay_row(self):
        source=np.array([0.1,0.7,0.1,0.1])
        relay=[np.array([0.9,0.1,0,0]),np.array([0.0,0.0,0.0,1.0]),np.array([0.25]*4),np.array([0.25]*4)]
        labels={(i,j):(5 if (i,j)==(1,3) else 0) for i in range(4) for j in range(4)}
        self.assertEqual(terminal_answer(source,relay,labels),5)

if __name__ == '__main__': unittest.main()
