import sys,unittest,copy
from pathlib import Path
HERE=Path(__file__).resolve().parent
for d in ['exogenous_paths_v1','receiver_factorial_v1','hotpot_support']:sys.path.insert(0,str(HERE.parent/d))
import evaluate_closure as e
from test_closure_replay_policies import records
from closure_operator import SCHEDULE
class Tests(unittest.TestCase):
    def synthetic(self):
        selection={'questions':[{'id':f'artificial-{i}'} for i in range(640)],'episode_shocks':['U','V']*4,'generation_seeds':[26111571,26111572,26111573,26111574,26111575]}
        rs=[[records(i,shock=(i//80)%2) for s in range(5)] for i in range(640)]
        preds=[[{n:'Cedar' if n=='SINGLE_Q' else 'Birch' for n in SCHEDULE} for s in range(5)] for i in range(640)]
        vault={r['id']:'Cedar' for r in selection['questions']};return selection,rs,preds,vault
    def test_full_synthetic_raw_answer_scoring(self):
        result,private=e.evaluate(*self.synthetic())
        self.assertEqual(result['results']['change']['SINGLE_Q']['accuracy'],1)
        self.assertEqual(result['results']['change']['closure_EVSI']['accuracy'],0)
        self.assertFalse(result['gates']['single_Q_quality_and_utility'])
        self.assertEqual(result['screen_action'],'STOP_FORMAL5_CANDIDATE')
        self.assertEqual(len(private['rows']),1360);self.assertEqual(len(private['policy_traces']),80)
    def test_missing_gold_fails_closed(self):
        x=list(self.synthetic());x[-1].pop('artificial-0')
        with self.assertRaises(AssertionError):e.evaluate(*x)
if __name__=='__main__':unittest.main()
