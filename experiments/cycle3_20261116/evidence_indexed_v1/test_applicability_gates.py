import unittest
import numpy as np
from evaluate_v2 import summarize
NAMES=['G0','G0_MINUS_EU','G0_MINUS_EV','G1','G1_MINUS_EU','G1_MINUS_EV','FLAT_EVIDENCE','SINGLE_Q','SINGLE_M','INDEPENDENT3_Q','VOTE_Q3']
class GateTests(unittest.TestCase):
 def result(self,em,violations=0,valid=None):
  return summarize(em,em,np.ones_like(em),np.ones((96,2,2,2),bool),violations,[True]*100 if valid is None else valid,NAMES,[1,2])
 def fixture(self):
  x=np.ones((96,2,11));x[:24,:,4]=0;return x
 def test_constructed_pass(self):self.assertTrue(self.result(self.fixture())['admission'])
 def test_certificate_failure_overrides_good_quality(self):self.assertFalse(self.result(self.fixture(),violations=1)['admission'])
 def test_one_seed_signal_is_insufficient(self):
  x=self.fixture();x[:,1,4]=1;self.assertFalse(self.result(x)['gates']['credit_shift_support'])
 def test_quality_failure_is_not_hidden_by_other_condition(self):
  x=self.fixture();x[:,:,3]=0;self.assertFalse(self.result(x)['gates']['fixed_G1_quality'])
 def test_invalid_selection_rate_is_gated(self):self.assertFalse(self.result(self.fixture(),valid=[True]*97+[False]*3)['gates']['selector_valid'])
if __name__=='__main__':unittest.main()
