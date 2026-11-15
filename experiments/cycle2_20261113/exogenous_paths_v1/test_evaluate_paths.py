import sys,unittest,copy
sys.path.insert(0,sys.argv.pop(1))
sys.path.insert(0,sys.argv.pop(1))
from evaluate_paths import evaluate
from exogenous_path_operator import digest
from point_statistics import ORDER
def fixture():
 rows=[];matrices=[];vault={}
 for i in range(96):
  qid=f'synthetic_{i}';shock='U' if i<48 else 'V';vault[qid]='Cedar'
  payload={'id':qid,'question':'Fabricated token?','context':{'title':[str(j) for j in range(10)],'sentences':[['Fictional'] for j in range(10)]}}
  rows.append({'index':i,'id':qid,'removed_upstream':shock});conditions=[]
  for seed in [26111311,26111312]:
   for name in ORDER:
    answer='Birch' if name==('G1_MINUS_EU' if shock=='U' else 'G1_MINUS_EV') else 'Cedar'
    if name=='SINGLE_Q':answer='Cedar tree'
    costs={'qwen':{'calls':1 if name=='SINGLE_Q' else 0 if name=='SINGLE_M' else 2},'mistral':{'calls':0 if name=='SINGLE_Q' else 1 if name=='SINGLE_M' else 2}}
    conditions.append({'key':f'{seed}/{name}','status':'COMPLETE','answer':answer,
     'nodes':{'M':{'generation':{'text':' '+answer+' ','finish_reason':'eos'}}},'logical_service_cost_by_model':costs})
  m={'status':'COMPLETE_UNSCORED','complete_terminal_conditions':18,'execution_seeds':[26111311,26111312],
   'question_fingerprint':digest(payload),'removed_upstream':shock,'conditions':conditions,
   'logical_completed_node_cost_by_model':{'qwen':{'calls':30},'mistral':{'calls':30}}}
  matrices.append(({'index':i,'payload':payload,'shock':shock},m))
 return {'questions':rows,'generation_seeds':[26111311,26111312]},matrices,vault
class Tests(unittest.TestCase):
 def test_raw_text_to_official_metric_and_cost(self):
  s,m,v=fixture();public,private=evaluate(s,m,lambda:v)
  self.assertEqual(public['primary_D'],1)
  self.assertEqual(public['accuracy_by_condition_seed']['SINGLE_Q'],[0,0])
  self.assertEqual(public['comparisons_by_seed']['G1_minus_SINGLE_Q'][0]['corrections'],96)
  self.assertEqual(public['full_logical_cost_by_condition_model']['26111311/G0/qwen']['calls'],192)
  self.assertEqual(public['bootstrap_runs'],0)
 def test_incomplete_archive_never_reads_gold(self):
  s,m,v=fixture();m[95][1]['conditions'].pop();opened=[]
  with self.assertRaises(AssertionError):evaluate(s,m,lambda:opened.append(True) or v)
  self.assertEqual(opened,[])
 def test_gold_coverage_exact(self):
  s,m,v=fixture();v['unselected']='secret'
  with self.assertRaises(AssertionError):evaluate(s,m,lambda:v)
if __name__=='__main__':unittest.main()
