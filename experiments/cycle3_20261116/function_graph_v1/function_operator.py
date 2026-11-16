"""Candidate applicability operator. Visible executions only; hidden labels forbidden."""
import copy,hashlib,json,re,time,uuid
from pathlib import Path
import exogenous_path_operator as base
from function_adapter import run_function

PLAN='Propose a concise algorithm and its complexity for the complete programming task. Do not write a full program. State pitfalls and boundary cases.'
CRITIC='Independently analyze the programming task. Give a concise algorithm or counterexample to common incorrect approaches. Do not write a full program.'
RELAY='Review the programming task and the received peer notes. Produce one concise corrected implementation plan. Treat peer notes as fallible.'
CODE='Solve the programming task in Python 3.8 using only the standard library. Implement the requested function as a module function or method of class Solution. Return only a complete program, optionally inside one python code fence. Do not hardcode example outputs.'
REPAIR='Repair the supplied Python program using the task and execution feedback. Return only a complete Python 3.8 standard-library program, optionally in one python code fence. Implement the requested function as a module function or method of class Solution. Generalize beyond visible cases.'
SCHEDULE=base.GRAPH_CONDITIONS+('SINGLE_Q','SINGLE_C','BOK3','REPAIR3')

def extract(text):
 s=text.strip();match=re.fullmatch(r'```(?:python|Python|py)?\s*\n(.*?)\n```',s,re.S)
 return match.group(1) if match else s
def compare(actual,expected):return actual.split()==expected.split()
def visible_eval(code,name,cases,root):
 result=[]
 for x,y in cases:
  z=run_function(code,name,json.loads(x),json.loads(y),root);result.append(z)
 return dict(passed=sum(v['passed'] for v in result),total=len(cases),cases=result,seconds=sum(v['elapsed_seconds'] for v in result))
def select(records):return max(range(len(records)),key=lambda i:(records[i]['visible']['passed'],-i))
def live(edges):
 s={'M'}
 while True:
  n=s|{u for u,v in edges if v in s}
  if n==s:return [v for v in base.NODES if v in s]
  s=n
def prompt(p,instruction,notes=None,extra=None):
 d=dict(question=p['question'],function_name=p['function_name'],peer_notes=notes or [])
 if extra:d.update(extra)
 return [dict(role='system',content=instruction),dict(role='user',content=base.canonical(d))]

class CodeOperator(base.PathOperator):
 def __init__(self,qwen,coder,sandbox_root,before_request=None):
  super().__init__(qwen,coder,config=base.PathConfig(message_tokens=192,final_tokens=1024,max_input_tokens=12288),before_request=before_request)
  self.sandbox_root=sandbox_root
  self.protocol=dict(operator='function_feedback_applicability_v1',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),base_source_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),model_identities=self.identities,config=vars(self.config),schedule=SCHEDULE,G0_edges=base.G0_EDGES,route={'U':'qwen','V':'qwen','R':'qwen','M':'mistral'},model_alias={'mistral':'Qwen2.5-Coder-7B-Instruct'},instructions=[PLAN,CRITIC,RELAY,CODE,REPAIR],code_extraction='one whole fenced block or entire stripped output; no repair',execution_compare='author call-based return comparison; qualified function adapter',feedback='two fixed visible cases; repair sees first512 characters of input/expected/actual and256stderr per case',selection='maximum number of visible cases passed, first-generated tie',hidden_test_access=False)
  self.protocol_hash=base.digest(self.protocol)
 def _program(self,record,p):
  code=extract(record['generation']['text']);key=hashlib.sha256(code.encode()).hexdigest()
  if key not in self._execution_cache:self._execution_cache[key]=visible_eval(code,p['function_name'],p['visible_cases'],self.sandbox_root)
  return dict(code=code,source_request_fingerprint=record['request_fingerprint'],visible=copy.deepcopy(self._execution_cache[key]),execution_reused=key in self._execution_seen,program_sha256=key)
 def _condition(self,p,seed,c,shock):
  graph=c in base.GRAPH_CONDITIONS;edges=base.condition_edges(c,shock) if graph else ();nodes={};programs=[]
  row=dict(key=f'{seed}/{c}',condition=c,execution_seed=seed,edges=[list(e) for e in edges],nodes=nodes,status='STARTED',programs=programs);self._conditions.append(row)
  order=live(edges) if graph else ['M'] if c.startswith('SINGLE_') else ['M','B1','B2'] if c=='BOK3' else ['M','T1','T2']
  for node in order:
   final=node not in ['U','V','R'] or not graph;model='qwen' if c=='SINGLE_Q' or not final else 'mistral'
   parents=[n for n in base.NODES if (n,node) in edges] if graph else []
   notes=[dict(source=n,text=nodes[n]['wire_message']) for n in parents]
   ins=CODE if final else PLAN if node=='U' else CRITIC if node=='V' else RELAY
   extra=None
   if c=='REPAIR3' and programs:
    if programs[-1]['visible']['passed']==len(p['visible_cases']):break
    ins=REPAIR;prev=programs[-1];fb=[]
    for (x,y),z in zip(p['visible_cases'],prev['visible']['cases']):fb.append(dict(input=x[:512],expected=y[:512],actual=json.dumps(z.get('typed_return'),ensure_ascii=False)[:512],stderr=z['stderr'][:256],status=z['status'],passed=z['passed']))
    extra=dict(previous_program=prev['code'],execution_feedback=fb)
   record=self._request(p,seed,c,node,model,prompt(p,ins,notes,extra),sum(len(self.backends[model].encode(n['text'])) for n in notes),final);nodes[node]=record
   if final:
    program=self._program(record,p);programs.append(program);self._execution_seen.add(program['program_sha256'])
  selected=select(programs);row.update(status='COMPLETE',executed_nodes=list(nodes),selected_program=selected,answer=programs[selected]['code'],logical_service_cost_by_model=base._logical_cost(nodes.values()),cache_reuse_count=sum(n['reused'] for n in nodes.values()))
 def run_matrix(self,payload,seeds,removed_upstream):
  if not self._lock.acquire(blocking=False):raise RuntimeError('Concurrent operator')
  start=time.perf_counter()
  try:
   assert not self._poisoned and set(payload)=={'id','question','function_name','visible_cases'} and len(payload['visible_cases'])==2
   assert all(isinstance(x,str) and isinstance(y,str) for x,y in payload['visible_cases'])
   assert len(seeds)==3 and len(set(seeds))==3 and removed_upstream in ['U','V'];p=copy.deepcopy(payload)
   self._matrix_id=uuid.uuid4().hex;self._cache={};self._call_ids=set();self._traces=[];self._conditions=[];self._execution_cache={};self._execution_seen=set();self._native_starts={k:len(b.physical_calls) for k,b in self.backends.items()}
   try:
    for seed in seeds:
     for c in SCHEDULE:self._condition(p,seed,c,removed_upstream)
    r=self._snapshot(p,seeds,removed_upstream,{},start,'COMPLETE_VISIBLE_ONLY');r.update(planned_terminal_conditions=30,planned_logical_calls='variable: repair early stopping',scores_computed=True,scoring_scope='visible_only',hidden_test_access=False,physical_visible_execution_seconds=sum(x['seconds'] for x in self._execution_cache.values()))
    assert r['complete_terminal_conditions']==30;return r
   except Exception:
    self._poisoned=True;raise base.PathExecutionError(self._snapshot(p,seeds,removed_upstream,{},start,'INCOMPLETE_OR_INVALID'))
  finally:self._lock.release()
