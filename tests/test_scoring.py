import unittest
from graph_credit.scoring import score_item


class ScoringTests(unittest.TestCase):
    def test_explicit_answer(self):
        self.assertEqual(score_item('{"answer":"1,234", "evidence":"2 times 617"}', 'explanation\n#### 1234'), 1)

    def test_no_last_number_heuristic(self):
        self.assertEqual(score_item('I tried 12 then 42', '#### 42'), 0)

    def test_ambiguous_answers(self):
        self.assertEqual(score_item('Answer: 12\nFinal answer: 42', '#### 42'), 0)

    def test_numeric_and_decimal(self):
        self.assertEqual(score_item('42.00', '#### 42'), 1)
        self.assertEqual(score_item('Answer: nan', '#### 42'), 0)

    def test_malformed_grouping_is_not_reinterpreted(self):
        self.assertEqual(score_item('Answer: 1,,234', '#### 1234'), 0)
        self.assertEqual(score_item('Answer: 12,34', '#### 1234'), 0)


if __name__ == '__main__':
    unittest.main()
