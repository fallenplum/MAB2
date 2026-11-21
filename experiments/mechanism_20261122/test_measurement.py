"""Artificial qualification of the LP, common seed calculation, and rank cancellation."""
import itertools,unittest
import numpy as np
from crossed import bounds
from choice_score import shared,independent
from natural_score import score_question,ORDERS
class Checks(unittest.TestCase):
    def test_all_degenerate_laws(self):
        for v in itertools.product((0,1),repeat=4):
            lo,hi=bounds(v,(0,1,2,3));target=float(v[0]-v[1]-v[2]+v[3]!=0);self.assertAlmostEqual(lo,target);self.assertAlmostEqual(hi,target)
    def test_equivalent_requests(self):
        self.assertEqual(bounds((.3,.7,.3,.7),(0,1,0,1)),(0,0))
    def test_shared_vs_independent(self):
        p=np.full((4,4),.25);self.assertAlmostEqual(shared(p,2)[0],0);self.assertGreater(independent(p,2),0)
    def test_common_order_reflection(self):
        r=np.random.default_rng(391).dirichlet(np.ones(4),4)
        for y in range(4):self.assertAlmostEqual(shared(r,y)[0],shared(r[:,::-1],3-y)[0])
    def test_full_graph_placebo(self):
        rng=np.random.default_rng(40);p=rng.dirichlet(np.ones(4),16);q=p.copy()
        for i,o in enumerate(ORDERS):
            if len(o)==3:q[i]=rng.dirichlet(np.ones(4))
        before=score_question(p,0);after=score_question(q,0)
        for b,a in zip(before,after):
            for key in ('ambiguous','canonical_flip','margin_range','new_margin_range','canonical_worst_regret'):self.assertEqual(a[key],b[key])
    def test_no_order_effect(self):
        p=np.tile([.1,.2,.3,.4],(16,1));self.assertTrue(all(r['canonical_worst_regret']==0 and r['new_margin_range']==0 for r in score_question(p,1)))
if __name__=='__main__':unittest.main()
