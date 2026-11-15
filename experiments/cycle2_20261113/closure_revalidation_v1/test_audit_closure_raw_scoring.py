import unittest
from audit_closure_raw_scoring import metrics,vote
class RawTests(unittest.TestCase):
    def test_articles_and_punctuation(self):self.assertEqual(metrics('The Cedar!','cedar'),(1,1.))
    def test_overlap_with_repetition(self):self.assertAlmostEqual(metrics('York York city','York city city')[1],2/3)
    def test_partial_answer(self):self.assertAlmostEqual(metrics('New York','York')[1],2/3)
    def test_yes_no_guard(self):self.assertEqual(metrics('yes indeed','yes'),(0,0.))
    def test_empty_official_convention(self):self.assertEqual(metrics('',''),(1,0.))
    def test_raw_vote_tie(self):self.assertEqual(vote(['Cedar','Birch','Oak']),0)
    def test_normalized_vote(self):self.assertEqual(vote(['Oak','The Cedar!','cedar']),1)
if __name__=='__main__':unittest.main()
