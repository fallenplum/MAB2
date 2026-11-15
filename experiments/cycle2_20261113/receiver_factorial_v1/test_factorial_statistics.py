from pathlib import Path
import sys,unittest,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'exogenous_paths_v1'))
from factorial_statistics import summarize,ORDER
def fixture():
    x=np.zeros((96,2,29));t=np.ones_like(x);k=np.ones_like(x);shocks=['U']*48+['V']*48
    for n in ['G1__Q_FULL','G1_MINUS_EU__Q_FULL','G1_MINUS_EV__Q_FULL','INDEPENDENT3_Q']:x[:,:,ORDER.index(n)]=1
    for i in range(40):x[i,:,ORDER.index('G1_MINUS_EU__Q_FULL')]=0
    for n in ['SINGLE_Q','VOTE_Q3']:x[:64,:,ORDER.index(n)]=1
    t[:,:,ORDER.index('G1__Q_FULL')]=.5;k[:,:,ORDER.index('G1__Q_FULL')]=.5
    return x,t,k,shocks
class Tests(unittest.TestCase):
    def test_known_signal_and_quality_cost_all_pass(self):
        r=summarize(*fixture());self.assertTrue(all(r['gates'].values()));self.assertAlmostEqual(r['credit_signals_by_cell']['Q_FULL']['D'],40/96)
        self.assertEqual(r['credit_signals_by_cell']['Q_FULL']['nonzero_question_support'],40)
        self.assertEqual(r['bootstrap_runs'],0)
    def test_no_substitution_or_cost_rescue(self):
        x,t,k,s=fixture();k[:,:,ORDER.index('G1__Q_FULL')]=2
        r=summarize(x,t,k,s);self.assertFalse(r['gates']['independent_aggregation_cost_screen'])
        self.assertEqual(r['screen_action'],'DO_NOT_EXPAND_NOMINEE_WITHOUT_NEW_HYPOTHESIS')
        x[:,:,:]=0;x[:,:,ORDER.index('G1__M_FULL')]=1
        r=summarize(x,t,k,s);self.assertFalse(r['gates']['gain_over_single_Q']);self.assertEqual(r['fixed_nominee'],'G1__Q_FULL')
    def test_strong_vote_cannot_be_omitted(self):
        x,t,k,s=fixture();x[:,:,ORDER.index('VOTE_Q3')]=1
        self.assertFalse(summarize(x,t,k,s)['gates']['gain_over_cheap_Q_vote'])
    def test_incomplete_or_unmeasured_cost_rejected(self):
        x,t,k,s=fixture()
        with self.assertRaises(ValueError):summarize(x[:95],t,k,s)
        k[:,:,ORDER.index('INDEPENDENT3_Q')]=0
        with self.assertRaises(ValueError):summarize(x,t,k,s)
if __name__=='__main__':unittest.main()
