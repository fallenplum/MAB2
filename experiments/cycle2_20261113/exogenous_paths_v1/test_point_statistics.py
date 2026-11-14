import unittest,numpy as np
from point_statistics import summarize,paired
class Tests(unittest.TestCase):
    def test_known_credit_shift(self):
        x=np.ones((96,2,9));sh=['U']*48+['V']*48
        x[:48,:,4]=0;x[48:,:,5]=0
        r=summarize(x,sh)
        self.assertEqual(r['primary_D'],1);self.assertEqual(r['nonzero_question_support'],96)
        self.assertFalse(r['method_success_claim']);self.assertEqual(r['bootstrap_runs'],0)
    def test_reverse_and_zero_do_not_promote(self):
        sh=['U']*48+['V']*48;x=np.ones((96,2,9));x[:48,:,5]=0;x[48:,:,4]=0
        self.assertEqual(summarize(x,sh)['primary_D'],-1)
        for a in [x,np.zeros_like(x)]:self.assertEqual(summarize(a,sh)['screen_action'],'DO_NOT_AUTOMATICALLY_EXPAND_THIS_INSTANCE')
    def test_binary_and_pairing(self):
        with self.assertRaises(ValueError):summarize(np.ones((96,2,9))*.5,['U']*48+['V']*48)
        r=paired(np.array([1,1,0,0]),np.array([0,1,1,0]))
        self.assertEqual((r['corrections'],r['breakages'],r['paired_difference']),(1,1,0))
if __name__=='__main__':unittest.main()
