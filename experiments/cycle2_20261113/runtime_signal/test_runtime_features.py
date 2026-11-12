import itertools,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from graph_credit.scoring import numeric
from runtime_features import first_round,features_for_edge,topk_weights,question_yields

def nodes():
    return {f'r1:w{i}':{'parsed':{'answer':['1000','1,000.00','2',''][i],'uncertainty':''},
        'generation':{'output_tokens':12},'message_metadata':{'format_ok':i!=3,'message_tokens':20}}
        for i in range(4)}

class Tests(unittest.TestCase):
    def test_numeric_equivalence_and_missing_not_equal(self):
        f,e,d=first_round(nodes(),numeric)
        self.assertEqual(e[0,1],1);self.assertEqual(d[0,2],1)
        self.assertEqual(e[3].sum(),0);self.assertEqual(d[3].sum(),0)
        self.assertEqual(f[3,1],0)

    def test_stage_boundary_rejects_later_nodes(self):
        x=nodes();x['aggregate']={'answer':'private test label'}
        with self.assertRaises(ValueError):first_round(x,numeric)

    def test_target_edge_context_and_shapes(self):
        f,e,d=first_round(nodes(),numeric);mask=np.zeros(12,int);mask[7]=1
        g,r,h=features_for_edge(mask,0,f,e,d)
        self.assertEqual(g.shape,(26,));self.assertEqual(r.shape,(40,));self.assertEqual(h.shape,(5,))
        self.assertEqual(h[0],0);self.assertEqual(h[1],1);self.assertEqual(h[3],.5)
        mask[0]=1
        with self.assertRaises(ValueError):features_for_edge(mask,0,f,e,d)

    def test_topk_tie_expectation_vs_enumeration(self):
        s=np.array([3,2,2,2,1,0]);target=np.array([0,1,0,1,0,1]);w=topk_weights(s,3)
        results=[target[[0,*pair]].mean() for pair in itertools.combinations((1,2,3),2)]
        self.assertAlmostEqual(np.dot(w,target)/3,np.mean(results))
        self.assertTrue(np.array_equal(topk_weights(np.arange(24)),np.r_[np.zeros(18),np.ones(6)]))

    def test_uniform_means_no_index_order_advantage(self):
        for target in itertools.product((0,1),repeat=6):
            self.assertAlmostEqual(np.dot(topk_weights(np.zeros(6),3),target)/3,np.mean(target))

    def test_question_and_seed_weighting(self):
        s=np.zeros((2,2,12,5));y=np.zeros_like(s);y[0,:,:,0]=1;y[1]=1
        np.testing.assert_allclose(question_yields(s,y),[.2,1],atol=1e-12)

    def test_invalid_scores_rejected(self):
        with self.assertRaises(ValueError):topk_weights(np.array([np.nan,1]),1)
        with self.assertRaises(ValueError):question_yields(np.zeros((2,24,5)),np.zeros((2,24,5)))

if __name__=='__main__':unittest.main()
