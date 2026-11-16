import unittest
import numpy as np
from evaluate import summarize
NAMES=['G0','G0_MINUS_EU','G0_MINUS_EV','G1','G1_MINUS_EU','G1_MINUS_EV','FLAT_ANNOTATION','SINGLE_Q','SINGLE_M','INDEPENDENT3_Q','VOTE_Q3']
SEEDS=[26111671,26111672]
def fixture():
    e=np.ones((96,2,11));t=np.ones_like(e)*1000;e[:24,:,0]=0;e[:24,:,4]=0;e[:20,:,7]=0;return e,t
class Tests(unittest.TestCase):
    def test_positive_artificial_control(self):
        e,t=fixture();self.assertTrue(summarize(e,e,t,NAMES,SEEDS)['admission'])
    def test_constant_credit_has_no_reliability(self):
        e,t=fixture();e[:,:,4]=1;self.assertFalse(summarize(e,e,t,NAMES,SEEDS)['gates']['cross_seed_reliability'])
    def test_equal_quality_extra_cost_rejected(self):
        e,t=fixture();e[:,:,7]=1;t[:,:,3]=2000;self.assertFalse(summarize(e,e,t,NAMES,SEEDS)['gates']['service_economic_signal'])
    def test_seed_loss_not_hidden_in_mean(self):
        e,t=fixture();e[:2,0,3]=0;e[:,:,6]=1;self.assertFalse(summarize(e,e,t,NAMES,SEEDS)['gates']['fixed_G1_quality'])
if __name__=='__main__':unittest.main()
