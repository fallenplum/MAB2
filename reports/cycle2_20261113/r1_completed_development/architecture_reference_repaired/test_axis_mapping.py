import unittest
import numpy as np
from axis_mapping import verify_stays, BACKGROUND_AXIS


class AxisRegression(unittest.TestCase):
    def fixture(self):
        score = np.zeros((8,2,24,7)); cost = np.zeros((8,2,7,5))
        score[:,0,:,0] = 1  # change means B
        cost[:,0,0,:3] = [200,30,216]
        cost[:,1,0,:3] = [100,20,216]
        q = {'A':np.zeros((8,24)), 'B':np.ones((8,24))}
        c = {'A':np.tile([100,20,216],(8,1)), 'B':np.tile([200,30,216],(8,1))}
        return q,c,score,cost

    def test_explicit_mapping(self):
        self.assertEqual(BACKGROUND_AXIS, {'B':0,'A':1})
        self.assertEqual(verify_stays(*self.fixture())['checked_aggregate_graphs'],384)

    def test_old_AB_positional_bug_is_rejected(self):
        q,c,s,v = self.fixture()
        old_q = {'A':s[:,0,:,0], 'B':s[:,1,:,0]}
        old_c = {'A':v[:,0,0,:3], 'B':v[:,1,0,:3]}
        with self.assertRaises(ValueError): verify_stays(old_q,old_c,s,v)

    def test_correct_labels_wrong_cost_is_rejected(self):
        q,c,s,v = self.fixture()
        with self.assertRaises(ValueError): verify_stays(q,{'A':c['B'],'B':c['A']},s,v)

    def test_one_of_384_score_mismatches_is_rejected(self):
        q,c,s,v = self.fixture(); q['A'][7,23] = 1
        with self.assertRaises(ValueError): verify_stays(q,c,s,v)

    def test_one_episode_cost_mismatch_is_rejected(self):
        q,c,s,v = self.fixture(); c['B'][7,0] += 1
        with self.assertRaises(ValueError): verify_stays(q,c,s,v)

    def test_missing_background_is_rejected(self):
        q,c,s,v = self.fixture(); del q['B']
        with self.assertRaises(ValueError): verify_stays(q,c,s,v)


if __name__ == '__main__': unittest.main()
