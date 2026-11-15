import unittest,numpy as np
import closure_policy as p
class Tests(unittest.TestCase):
    def test_simplex_and_nonnegative_information_value(self):
        rng=np.random.default_rng(1)
        for _ in range(100):
            a=rng.integers(0,10,(2,3))+1/3
            for e in [0,1]:self.assertGreaterEqual(p.evsi(a,[6000,4200,5800],e),0)
    def test_graph_cost_changes_threshold_without_label_changes(self):
        a=np.array([[1,8,1],[1,8,1]],float)
        self.assertEqual(p.action(a,[6000,6000,6000]),0)
        self.assertEqual(p.action(a,[6000,4000,6000]),1)
        self.assertEqual(p.action(a,[6000,4000,6000],protect=0),0)
    def test_keep_stale_protect_buy_nothing(self):
        def forbidden(*x):self.fail('Unrequested feedback accessed')
        for m in ['keep','stale','protect_changed']:
            r=p.run(np.zeros((8,2)),[6000,4000,5900],[6000,5900,5900],[1500,1500],0,m,1,forbidden)
            self.assertEqual(r['pairs'],0)
    def test_every_feedback_is_unique_paid_and_bounded(self):
        I=np.zeros((8,2))
        for m in p.METHODS:
            seen=[]
            def buy(e,q):
                self.assertNotIn((e,q),seen);self.assertIn(e,[0,1]);self.assertIn(q,range(8));seen.append((e,q));return {'gain':(-1 if (e+q)%3==0 else 1),'charged_tokens':7000}
            r=p.run(I,[6000,4100,5900],[6000,5900,5900],[1500,1500],0,m,17,buy)
            self.assertEqual(len(seen),r['pairs']);self.assertLessEqual(r['pairs'],8)
            self.assertEqual([(x['edge'],x['feedback_row']) for x in r['trace']],seen)
    def test_periodic_fixed_schedule(self):
        seen=[]
        def buy(e,q):seen.append((e,q));return {'gain':0,'charged_tokens':500}
        p.run(np.zeros((8,2)),[6000,4000,5900],[6000,5900,5900],[1000,1000],0,'periodic',1,buy)
        self.assertEqual(seen,[(e,q) for q in [0,2,4,6] for e in [0,1]])
    def test_expensive_feedback_stops(self):
        def forbidden(*x):self.fail('Negative predicted net value purchased')
        r=p.run(np.zeros((8,2)),[6000,5900,5900],[6000,5900,5900],[1e9,1e9],0,'closure_EVSI',1,forbidden)
        self.assertEqual(r['pairs'],0)
    def test_invalid_feedback_and_initial_schema_rejected(self):
        with self.assertRaises(ValueError):p.run(np.zeros((8,3)),[1,1,1],[1,1,1],[1,1],0,'random',1,lambda *a:{})
        with self.assertRaises(ValueError):p.run(np.zeros((8,2)),[1,1,1],[1,1,1],[1,1],0,'random',1,lambda *a:{'gain':0,'charged_tokens':0,'hidden_service_answer':1})
if __name__=='__main__':unittest.main()
