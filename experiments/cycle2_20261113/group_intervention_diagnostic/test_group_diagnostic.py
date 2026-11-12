"""Independent enumerations and counterexamples; no historical data access."""
import itertools,json,hashlib,time,unittest
from pathlib import Path
import numpy as np
from group_math import four_world_metrics,cross_square,crossfit,summarize_faces
from inventory_metadata import faces_for_masks

class Tests(unittest.TestCase):
    def test_all_binary_corners(self):
        for y in itertools.product((0,1),repeat=4):
            m=four_world_metrics(np.repeat(np.array(y)[:,None],5,axis=1))
            self.assertEqual(m['L'][0],m['a'][0]+m['b'][0]-m['h'][0])
            self.assertEqual(cross_square(m['h']),float(m['h'][0]**2))

    def test_cross_product_independent_enumeration(self):
        for x in itertools.product((-1,0,1),repeat=5):
            expected=sum(x[i]*x[j] for i in range(5) for j in range(i+1,5))/10
            self.assertAlmostEqual(cross_square(np.array(x)),expected,places=14)
        self.assertLess(cross_square([-1,1,0,0,0]),0)

    def test_redundancy_and_opposite_synergy(self):
        redundant=np.repeat(np.array([0,1,1,1])[:,None],5,axis=1)
        r=summarize_faces([{'q':0,'pair':[0,1],'quality':redundant}])
        self.assertEqual(r['zero_single_screen_crossfit_question_weighted']['L']['point'],1)
        self.assertEqual(r['all_faces_question_weighted']['u_h']['point'],1)
        self.assertEqual(r['investment_screen']['status'],'INCONCLUSIVE_INSUFFICIENT_SUPPORT')
        synergistic=np.repeat(np.array([0,0,0,1])[:,None],5,axis=1)
        self.assertFalse(any(x['selected'] for x in crossfit(synergistic)))

    def test_selection_cannot_see_joint_or_heldout(self):
        y=np.zeros((4,5));z=y.copy();z[0,:]=1
        self.assertEqual([x['selected'] for x in crossfit(y)],[x['selected'] for x in crossfit(z)])
        z=y.copy();z[3,2:]=1
        self.assertEqual(crossfit(y)[0]['selected'],crossfit(z)[0]['selected'])
        self.assertNotEqual(crossfit(y)[0]['heldout'],crossfit(z)[0]['heldout'])

    def test_no_corner_imputation(self):
        self.assertEqual(len(faces_for_masks([[0,0],[0,1],[1,0]])[0]),0)
        faces,_=faces_for_masks(list(itertools.product((0,1),repeat=3)))
        self.assertEqual(len(faces),6)
        self.assertEqual(len({(tuple(x['edges']),x['background']) for x in faces}),6)

    def test_question_weight_not_face_weight(self):
        y1=np.repeat(np.array([0,1,1,1])[:,None],5,axis=1)
        y0=np.zeros((4,5))
        rows=[{'q':0,'pair':[0,1],'quality':y1}]+[{'q':1,'pair':[1,2],'quality':y0} for _ in range(9)]
        r=summarize_faces(rows)
        self.assertEqual(r['all_faces_question_weighted']['L']['point'],.5)
        self.assertEqual(r['zero_single_screen_crossfit_question_weighted']['L']['point'],.5)
        self.assertEqual(r['complete_questions'],2)

    def test_invalid_quality(self):
        for y in [np.zeros((3,5)),np.full((4,5),np.nan),np.full((4,5),.3)]:
            with self.assertRaises(ValueError):four_world_metrics(y)

    def test_empty_support_is_undefined(self):
        r=summarize_faces([])
        self.assertEqual(r['investment_screen']['status'],'NOT_IDENTIFIABLE_NO_COMPLETE_FOUR_WORLDS')
        self.assertEqual(r['all_faces_question_weighted'],{})

if __name__=='__main__':
    started=time.process_time();suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    path=Path(__file__).parent/'SYNTHETIC_TEST_RECEIPT_ROOT_001.json'
    with path.open('x',encoding='utf-8') as f:
        json.dump({'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,
                   'cpu_seconds':time.process_time()-started,'true_data_read':False,
                   'sources':{name:hashlib.sha256((path.parent/name).read_bytes()).hexdigest()
                              for name in ['group_math.py','inventory_metadata.py','test_group_diagnostic.py']}},f,indent=2)
    raise SystemExit(not result.wasSuccessful())
