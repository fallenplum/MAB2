import unittest
import numpy as np
from analyze_cache import group_stats,bootstrap_difference,cross_seed_risk

class DiagnosticTests(unittest.TestCase):
    def test_original_cross_seed_risk_identity_negative_preserved(self):
        delta=np.array([[[1,-1,0,0,0],[1,1,1,1,1]]],dtype=float);pred=np.array([[0.,.2]])
        r=cross_seed_risk(pred,delta)
        expected=(pred-delta.mean(-1))**2-delta.var(-1,ddof=1)/5
        np.testing.assert_allclose(r,expected,rtol=0,atol=4*np.finfo(float).eps)
        self.assertLess(r[0,0],0)
    def test_support_and_saturation(self):
        lo=np.array([[1,0],[0,1]]);hi=np.array([[1,0],[1,0]]);d=hi-lo
        got=group_stats(np.ones_like(d,dtype=bool),d,lo,hi)
        self.assertEqual(got['paired_seed_observations'],4);self.assertEqual(got['questions'],2)
        self.assertEqual(got['nonzero_credit_fraction'],.5)
        self.assertEqual(got['both_correct_fraction'],.25);self.assertEqual(got['both_wrong_fraction'],.25)
        self.assertEqual(got['mean_signed_credit'],0)
    def test_empty_support_is_not_zero_effect(self):
        d=np.zeros((3,2));f=np.zeros_like(d,dtype=bool)
        self.assertIsNone(group_stats(f,d,d,d)['nonzero_credit_fraction'])
        self.assertIsNone(bootstrap_difference(f,d)['difference'])
    def test_question_cluster_resampling_keeps_paired_cells(self):
        flag=np.array([[True,False]]*4);d=np.array([[1,0]]*4)
        got=bootstrap_difference(flag,d)
        self.assertEqual(got['difference'],1);self.assertEqual(got['ci95'],[1,1])
        self.assertEqual(got['cluster'],'question')
    def test_rare_group_reports_invalid_resamples(self):
        f=np.zeros((4,2),dtype=bool);f[0,0]=True;d=f.astype(int)
        got=bootstrap_difference(f,d)
        self.assertGreater(got['valid_resamples'],0);self.assertLess(got['valid_resamples'],2000)
        self.assertIsNone(got['ci95']);self.assertIn('Fewer than two',got['interval_not_reported_reason'])

if __name__=='__main__':unittest.main(verbosity=2)
