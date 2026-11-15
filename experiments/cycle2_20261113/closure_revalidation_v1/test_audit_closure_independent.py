"""Artificial-data auditor checks; no private outputs, labels, or GPU calls."""
import copy,json,unittest
from test_evaluate_closure import Tests as Fixtures
from evaluate_closure import evaluate
from audit_closure_independent import audit

class IndependentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        f=list(Fixtures().synthetic())
        # Deterministic, varied initial and purchased gains, with opposing
        # service behavior. Exercise acquisition without exposing real data.
        for i,perseed in enumerate(f[2]):
            for s,preds in enumerate(perseed):
                for j,name in enumerate(preds):
                    preds[name]='Cedar' if (i*7+s*3+j*5)%11<5 else 'Birch'
        summary,private=evaluate(*f)
        cls.selection=f[0];cls.summary=json.loads(json.dumps(summary))
        cls.private=json.loads(json.dumps(private))
    def test_varied_full_reconstruction(self):
        result=audit(self.private,self.summary,self.selection)
        self.assertGreater(result['checks'],500000)
        self.assertTrue(any(t['choices']['closure_EVSI']['decision']['pairs']>0 for t in self.private['policy_traces']))
    def test_changed_cost_rejected(self):
        p=copy.deepcopy(self.private);p['rows'][0]['complete_cost']['tokens']+=1
        with self.assertRaises(AssertionError):audit(p,self.summary,self.selection)
    def test_changed_action_rejected(self):
        p=copy.deepcopy(self.private)
        d=p['policy_traces'][0]['choices']['closure_EVSI']['decision'];d['action']=(d['action']+1)%3
        with self.assertRaises(AssertionError):audit(p,self.summary,self.selection)
    def test_changed_summary_rejected(self):
        s=copy.deepcopy(self.summary);s['results']['change']['closure_EVSI']['accuracy']+=.01
        with self.assertRaises(AssertionError):audit(self.private,s,self.selection)

if __name__=='__main__':unittest.main()
