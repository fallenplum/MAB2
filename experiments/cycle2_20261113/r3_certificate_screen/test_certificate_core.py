import itertools
import unittest
import numpy as np
from certificate_core import certificate,structural_cost,oracle,run_policy

def independent_completion_winner(observed):
    n,m=observed.shape; missing=list(zip(*np.where(observed<0)));winners=set()
    for values in itertools.product((0,1),repeat=len(missing)):
        z=observed.copy()
        for cell,value in zip(missing,values):z[cell]=value
        totals=[sum(int(z[q,a]) for q in range(n)) for a in range(m)]
        winners.add(totals.index(max(totals)))
    return next(iter(winners)) if len(winners)==1 else None

def independent_cost(mask,dirty):
    total=0
    for row in mask:
        arms=[a for a in range(len(row)) if row[a]]
        if not arms:continue
        if len(arms)==1:total+=4
        else:total+=3+sum(2 if dirty[a] else 1 for a in arms)
    return total

class Tests(unittest.TestCase):
    def test_interval_certificate_necessary_sufficient(self):
        for raw in itertools.product((-1,0,1),repeat=6):
            x=np.array(raw).reshape(2,3)
            self.assertEqual(certificate(x),independent_completion_winner(x))

    def test_cost_all_subsets(self):
        d=np.array([0,1,0,1,0,1,0])
        for raw in itertools.product((0,1),repeat=7):
            x=np.array(raw).reshape(1,7)
            self.assertEqual(structural_cost(x,d),independent_cost(x,d))
        self.assertEqual(structural_cost(np.ones((12,7)),d),156)

    def test_milp_matches_exhaustive_certificates(self):
        d=np.array([0,1,0]);masks=[np.array(raw).reshape(2,3) for raw in itertools.product((0,1),repeat=6)]
        for raw in itertools.product((0,1),repeat=6):
            y=np.array(raw).reshape(2,3);winner=int(np.argmax(y.sum(axis=0)))
            candidates=[independent_cost(mask,d) for mask in masks
                        if independent_completion_winner(np.where(mask,y,-1))==winner]
            r=oracle(y,d)
            self.assertEqual(r['status'],'OPTIMAL_CERTIFICATE_VERIFIED')
            self.assertEqual(r['cost'],min(candidates))

    def test_policies_query_unique_cells_and_get_correct_certificate(self):
        d=np.array([0,1,0]);rng=np.random.default_rng(44)
        for _ in range(30):
            y=rng.integers(2,size=(4,3));old=rng.integers(2,size=(4,3))
            for method in ('cheapest','random','gap_unit','gap_bundle'):
                seen=set()
                def query(q,a):
                    self.assertNotIn((q,a),seen);seen.add((q,a));return int(y[q,a])
                r=run_policy(old,d,method,query,19)
                self.assertEqual(r['action'],int(np.argmax(y.sum(axis=0))))
                self.assertEqual(r['revealed'],len(seen))
                self.assertGreaterEqual(r['cost'],oracle(y,d)['cost'])

    def test_unpurchased_values_do_not_change_a_prefix(self):
        old=np.zeros((3,3),int);d=np.array([0,1,0])
        for method in ('cheapest','random','gap_unit','gap_bundle'):
            y1=np.zeros((3,3),int);seen1=[]
            def query1(q,a):seen1.append((q,a));return int(y1[q,a])
            run_policy(old,d,method,query1,7)
            self.assertGreaterEqual(len(seen1),3)
            y2=np.ones((3,3),int)
            for cell in seen1[:3]:y2[cell]=0
            seen2=[]
            def query2(q,a):seen2.append((q,a));return int(y2[q,a])
            run_policy(old,d,method,query2,7)
            self.assertEqual(seen1[:3],seen2[:3])

    def test_bad_feedback_rejected(self):
        with self.assertRaises(ValueError):
            run_policy(np.zeros((2,3),int),[0,1,0],'cheapest',lambda q,a:2)

if __name__=='__main__':unittest.main()
