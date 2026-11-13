from pathlib import Path
import hashlib, tempfile, unittest
import numpy as np
from screen import classify, reconstruct_anchors, receiver_change, theta_crossseed, summarize, selected_npy_rows, EMPTY

def digest(s):return hashlib.sha256(s.encode()).hexdigest()

class ScreenTests(unittest.TestCase):
    def test_literal_not_numeric_and_unknown(self):
        a=np.array([[digest('2')]*4,[digest('2'),digest('2.0'),digest('2'),digest('2')],
                    [EMPTY]*4,[digest('2')]*4])
        np.testing.assert_array_equal(classify(a,np.array([0,0,0,1],dtype=np.uint16)),[0,1,-1,-1])

    def test_anchor_reconstruction_and_receiver_other_edges(self):
        anchors=np.array([[[0]*12,[0]*12]],dtype=np.int8)
        anchors[0,1,0]=1 # 0->1
        rows=[];left=np.empty((1,2,12),dtype=int)
        for a in range(2):
            for e in range(12):
                row=anchors[0,a].copy();row[e]=0
                left[0,a,e]=len(rows);rows.append(row)
        rebuilt=reconstruct_anchors(np.array(rows),left)
        np.testing.assert_array_equal(rebuilt,anchors)
        # same receiver=1 other incoming edges 7(2->1),10(3->1), but e0 excludes itself
        np.testing.assert_array_equal(np.flatnonzero(receiver_change(rebuilt)[0]),[7,10])

    def test_crossseed_not_naive_square_and_negative_kept(self):
        x=np.broadcast_to([1,-1,1,-1],(3,12,4)).copy()
        np.testing.assert_allclose(theta_crossseed(x),-1/3)
        x[:]=2
        np.testing.assert_allclose(theta_crossseed(x),4)

    def test_zero_variance_not_positive_or_negative_evidence(self):
        z=np.array([0]*20+[1]*20);dirty=np.zeros((40,12),bool);dirty[:,:6]=True
        out=summarize(z,dirty,np.zeros((40,12)),97531)
        self.assertEqual(out['primary_interaction'],0)
        self.assertFalse(out['inferential_support_guard'])
        self.assertEqual(out['status'],'INSUFFICIENT_SUPPORT_FOR_FIXED_LITERAL_PROBE')

    def test_single_stratum_kept_not_redefined(self):
        z=np.ones(30,dtype=int);dirty=np.zeros((30,12),bool);dirty[:,:6]=True
        out=summarize(z,dirty,np.zeros((30,12)),97531)
        self.assertIsNone(out['primary_interaction'])
        self.assertIsNone(out['ci95_descriptive'])
        self.assertEqual(out['primary_support_z1'],30)

    def test_paired_question_equal_weight_and_repeatable(self):
        z=np.array([0]*20+[1]*20);dirty=np.zeros((40,12),bool)
        theta=np.empty((40,12))
        for q in range(40):
            dirty[q,:1+q%10]=True
            theta[q]=0;theta[q,dirty[q]]=(q%20)/20+z[q]
        out=summarize(z,dirty,theta,97531)
        self.assertAlmostEqual(out['primary_interaction'],1)
        self.assertTrue(out['inferential_support_guard'])
        self.assertEqual(out['status'],'EXPLORATORY_SIGNAL_ONLY')
        self.assertEqual(out,summarize(z,dirty,theta,97531))

    def test_stream_rows_do_not_decode_excluded_nan_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=np.arange(30,dtype=float).reshape(6,5);a[[1,3,4]]=np.nan
            p=Path(tmp)/'fixture.npz';np.savez_compressed(p,quality_by_seed=a)
            selected=selected_npy_rows(p,'quality_by_seed',[0,2,5])
            np.testing.assert_array_equal(selected,a[[0,2,5]])
            self.assertTrue(np.isfinite(selected).all())

    def test_unknown_retained_in_support_not_consensus(self):
        z=np.array([-1,0,1]);d=np.tile(np.arange(12)%2,(3,1)).astype(bool)
        out=summarize(z,d,np.zeros((3,12)),97531)
        self.assertEqual(out['z_counts'],{'-1':1,'0':1,'1':1})
        self.assertEqual(out['cells']['z-1_d1']['question_edge_cells'],6)
        self.assertEqual(out['primary_support_z0'],1)

if __name__=='__main__':unittest.main(verbosity=2)
