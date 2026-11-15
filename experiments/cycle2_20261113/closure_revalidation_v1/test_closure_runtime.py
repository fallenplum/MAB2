import unittest,copy
import closure_runtime_v2 as runtime
def plan(n):return {'source_pins':{},'mode':'formal','gpus':list(range(n)),'prefix_per_worker':2,'worker_cap_seconds':171000/n,'assignments':{str(i):list(range(i,640,n)) for i in range(n)},'seeds':[26111571,26111572,26111573,26111574,26111575]}
class Tests(unittest.TestCase):
    def test_fixed_aggregate_cap_for_each_allowed_placement(self):
        for n in [1,2,4]:runtime.verify(plan(n))
    def test_duplicate_missing_question_rejected(self):
        p=plan(2);p['assignments']['0'][0]=1
        with self.assertRaises(ValueError):runtime.verify(p)
    def test_expanded_cap_or_changed_seeds_rejected(self):
        p=plan(4);p['worker_cap_seconds']+=1
        with self.assertRaises(ValueError):runtime.verify(p)
        p=plan(2);p['seeds'][0]+=999
        with self.assertRaises(ValueError):runtime.verify(p)
if __name__=='__main__':unittest.main()
