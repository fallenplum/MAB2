import importlib.util,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('f10score',Path(__file__).with_name('score.py'));score=importlib.util.module_from_spec(spec);spec.loader.exec_module(score)
def call(fp,tokens=10):return dict(request_fingerprint=fp,model='mistral',generation=dict(input_tokens=tokens,output_tokens=2,gpu_seconds=.3,elapsed_seconds=.4))
def program(fp,passed=1):return dict(program_sha256=fp,visible=dict(passed=passed,seconds=.2))
class Tests(unittest.TestCase):
 def test_shared_prefix_paid_once_and_all_branches_paid(self):
  a=dict(condition='G1',nodes={'U':call('prefix'),'M':call('terminal1')},programs=[program('p1')]);b=dict(condition='G1_MINUS_EU',nodes={'U':call('prefix'),'M':call('terminal2')},programs=[program('p2')]);c=score.cost([a,b]);self.assertEqual(c['tokens'],36);self.assertEqual(c['unique_generation_calls'],3);self.assertEqual(c['visible_execution_seconds'],.4)
 def test_single_does_not_pay_unused_feedback(self):
  c=score.cost([dict(condition='SINGLE_C',nodes={'M':call('x')},programs=[program('p')])]);self.assertEqual(c['tokens'],12);self.assertEqual(c['visible_execution_seconds'],0)
 def test_tie_rule_and_exhaustive_feedback(self):
  rows=[dict(selected_program=0,programs=[program('a',1)]),dict(selected_program=0,programs=[program('b',2)]),dict(selected_program=0,programs=[program('c',2)])];self.assertEqual(score.choice(rows),1)
 def test_prefix_stopping_keeps_same_selected_program(self):
  import itertools
  for outcomes in itertools.product(range(3),repeat=3):
   full=max(range(3),key=lambda i:(outcomes[i],-i));end=next((i+1 for i,v in enumerate(outcomes) if v==2),3);early=max(range(end),key=lambda i:(outcomes[i],-i));self.assertEqual(full,early)
if __name__=='__main__':unittest.main()
