import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'cycles'))
from cycle_005_topology_h2 import run_h2
from strict_graph_voi_cpu import FAMILY_SPECS, make_case, run_static


class Cycle005H2Tests(unittest.TestCase):
    def test_deterministic_action_soundness_on_smoke_matrix(self):
        for seed in (20261124, 20261125):
            rng = np.random.default_rng(seed)
            for family in FAMILY_SPECS:
                for _ in range(3):
                    case = make_case(rng, family, 0.4, 'heterogeneous', 2)
                    result = run_h2(case)
                    self.assertTrue(result['certificate'])
                    self.assertEqual(result['action'], result['target'])

    def test_candidate_uses_no_lower_cost_than_pairwise_on_fixed_fixture(self):
        rng = np.random.default_rng(20261124)
        h2 = []
        pair = []
        for family in FAMILY_SPECS:
            for _ in range(8):
                case = make_case(rng, family, 0.4, 'heterogeneous', 2)
                h2.append(run_h2(case)['cost'])
                pair.append(run_static(case, 'masked_pairwise')['cost'])
        # This is a fixed smoke fixture, not the confirmatory gate. It catches
        # accidental policy regressions while allowing stochastic aggregate
        # outcomes to be evaluated by the cycle ledger.
        self.assertLessEqual(float(np.mean(h2)), float(np.mean(pair)))


if __name__ == '__main__':
    unittest.main()

