import dataclasses,json,unittest
from unittest.mock import patch
import exogenous_path_operator as base
from test_operator import Backend,Generation
import function_operator as fn

class Fake(Backend):
 def generate(self,messages,seed,cap,config):
  text='def f(a): return a+1' if messages[0]['content'] in (fn.CODE,fn.REPAIR) else 'Check boundaries '+str(seed)
  g=Generation(text,10,min(cap,len(text)),.2,.1,100,'eos',self.identity['model']+str(len(self.physical_calls)))
  self.physical_calls.append({**dataclasses.asdict(g),'status':'SUCCESS','seed':seed,'prompt_sha256':base.digest(messages)});return g
def payload():return dict(id='artificial',question='Implement addition',function_name='f',visible_cases=[['[1]','2'],['[2]','3']])
def evaluate(code,name,cases,root):return dict(passed=2,total=2,seconds=.4,cases=[dict(passed=True,status='OK',stdout='',stderr='',elapsed_seconds=.2,typed_return=['atom',i]) for i in [2,3]])
class Tests(unittest.TestCase):
 def test_graph_cache_and_complete_cost(self):
  with patch.object(fn,'visible_eval',evaluate):
   for shock in 'UV':
    r=fn.CodeOperator(Fake('q'),Fake('m'),'unused').run_matrix(payload(),[1,2,3],shock);self.assertEqual(len(r['conditions']),30)
    for row in r['conditions']:
     if row['condition']=='REPAIR3':self.assertEqual(list(row['nodes']),['M'])
     if row['condition']=='BOK3':self.assertEqual(len(row['nodes']),3)
     if row['condition']=='G1_MINUS_E'+shock:self.assertNotIn(shock,row['nodes'])
     for node,rec in row['nodes'].items():
      prompt=json.loads(rec['request']['messages'][1]['content']);self.assertEqual(prompt['question'],payload()['question']);self.assertEqual(prompt['function_name'],'f')
      expected=[dict(source=n,text=row['nodes'][n]['wire_message']) for n in base.NODES if [n,node] in row['edges']]
      self.assertEqual(prompt['peer_notes'],expected)
    self.assertEqual(sum(x['calls'] for x in r['logical_completed_node_cost_by_model'].values()),87)
    self.assertEqual(r['physical_visible_execution_seconds'],.4)
 def test_hidden_schema_rejected(self):
  p=payload();p['hidden_cases']=[['[3]','4']]
  with self.assertRaises(AssertionError):fn.CodeOperator(Fake('q'),Fake('m'),'unused').run_matrix(p,[1,2,3],'U')
 def test_repairs_full_budget_and_feedback(self):
  def fail(*args):
   x=evaluate(*args);x['passed']=0
   for z in x['cases']:z['passed']=False
   return x
  with patch.object(fn,'visible_eval',fail):r=fn.CodeOperator(Fake('q'),Fake('m'),'unused').run_matrix(payload(),[1,2,3],'U')
  for row in r['conditions']:
   if row['condition']=='REPAIR3':
    self.assertEqual(list(row['nodes']),['M','T1','T2']);self.assertEqual(len(json.loads(row['nodes']['T1']['request']['messages'][1]['content'])['execution_feedback']),2)
  self.assertEqual(sum(x['calls'] for x in r['logical_completed_node_cost_by_model'].values()),93)
 def test_extraction_and_ties(self):
  self.assertEqual(fn.extract('```python\ndef f(a):return a\n```'),'def f(a):return a');self.assertEqual(fn.extract('text\n```python\nx\n```'),'text\n```python\nx\n```');self.assertEqual(fn.select([{'visible':{'passed':1}},{'visible':{'passed':1}}]),0)
if __name__=='__main__':unittest.main()
