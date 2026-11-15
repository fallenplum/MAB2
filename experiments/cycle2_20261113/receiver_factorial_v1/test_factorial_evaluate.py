import sys,unittest,copy
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'exogenous_paths_v1'))
sys.path.insert(0,str(HERE.parent/'hotpot_support'))
from test_operator import Backend,payload
from factorial_operator import FactorialPathOperator
from exogenous_path_operator import digest
from evaluate_factorial import evaluate
def fixture():
    rows=[];matrices=[];vault={}
    template=FactorialPathOperator(Backend('q'),Backend('m')).run_matrix(payload(),[26111541,26111542],'U')
    for i in range(96):
        p=payload();p['id']='fabricated_'+str(i);shock='U' if i<48 else 'V';m=copy.deepcopy(template)
        m['question_fingerprint']=digest(p);m['removed_upstream']=shock
        rows.append({'index':i,'id':p['id'],'removed_upstream':shock});matrices.append(({'index':i,'payload':p,'shock':shock},m));vault[p['id']]='fabricated'
    return {'questions':rows,'generation_seeds':[26111541,26111542]},matrices,vault
class Tests(unittest.TestCase):
    def test_complete_costs_and_raw_answers(self):
        s,m,v=fixture();out,private=evaluate(s,m,lambda:v)
        self.assertEqual(out['fixed_nominee'],'G1__Q_FULL');self.assertEqual(out['bootstrap_runs'],0)
        self.assertEqual(len(private['cost_rows']),5568)
        self.assertGreater(sum(out['tokens_by_condition_seed']['VOTE_Q3']),0)
    def test_incomplete_never_reads_labels(self):
        s,m,v=fixture();m[-1][1]['conditions'].pop();opened=[]
        with self.assertRaises(AssertionError):evaluate(s,m,lambda:opened.append(1) or v)
        self.assertEqual(opened,[])
    def test_bad_vote_selection_never_reads_labels(self):
        s,m,v=fixture();m[0][1]['conditions'][-1]['selected_node']='invalid';opened=[]
        with self.assertRaises(AssertionError):evaluate(s,m,lambda:opened.append(1) or v)
        self.assertEqual(opened,[])
if __name__=='__main__':unittest.main()
