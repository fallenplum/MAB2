"""Separate implementation of synthetic output/accounting checks; no model calls."""
from pathlib import Path
import argparse,hashlib,json,math

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
 assert not a.receipt.exists()
 def load(name):return json.loads((a.input/name).read_bytes())
 def lines(name):return [json.loads(x) for x in (a.input/name).read_bytes().splitlines()]
 status=load('RESIDENT_STATUS.json');q=load('QUALIFICATION.json');identity=load('IDENTITY.json')
 assert status['status']=='CLOSED' and status['returncode']==0 and status['reason']=='CHILD_EXIT'
 assert not status['reservation_exceeded'] and 0<status['gpu_hours']<.5
 assert math.isclose(status['gpu_hours']*3600,status['resident_seconds'],rel_tol=1e-12)
 assert status['plan_sha256']=='279bec01c0a3516f3293990c8464deda78a240a6aed7bafb9ceafcd36016089c'
 routes=lines('route_calls.jsonl');physical={};models={}
 for model in ['qwen','mistral']:
  for row in lines(model+'_physical_calls.jsonl'):
   assert row['call_id'] not in physical and row['status']=='SUCCESS'
   physical[row['call_id']]=row;models[row['call_id']]=model
 assert len(routes)==len(physical)==64 and len({r['call_id'] for r in routes})==64
 route={r['call_id']:r for r in routes};assert set(route)==set(physical)
 for cid,r in route.items():
  expected='mistral' if r['node']=='aggregate' or int(r['node'].split('w')[1])%2 else 'qwen'
  assert r['model']==models[cid]==expected and r['status']=='SUCCESS'
  for field in ['seed','prompt_sha256','input_tokens','output_tokens']:assert r[field]==physical[cid][field]
  b=identity['backend']['model_identities'][expected]
  digest=hashlib.sha256(json.dumps(b,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
  assert r['model_identity_sha256']==digest
 expected_closure={'r2:w1','r3:w0','r3:w1','r3:w2','r3:w3','aggregate'}
 counted=[];costs=[];replay_fields=0;graph_checks=0;format_counts={m:{'n':0,'bad':0} for m in ['qwen','mistral']}
 for name in ['synthetic_hetero_arithmetic','synthetic_hetero_reading']:
  d={t:load(name+'_'+t+'.json') for t in ['base','changed','fresh','cache']}
  assert d['base']['mask']==[1]*12 and d['changed']['mask']==[0]+[1]*11
  assert d['changed']['mask']==d['fresh']['mask']==d['cache']['mask']
  assert len({z['protocol_hash'] for z in d.values()})==1 and len({z['question_fingerprint'] for z in d.values()})==1
  assert set(n for n,r in d['changed']['nodes'].items() if not r['reused'])==expected_closure
  assert all(r['reused'] for r in d['cache']['nodes'].values())
  for node,r in d['changed']['nodes'].items():
   f=d['fresh']['nodes'][node];cache=d['cache']['nodes'][node]
   for field in ['messages','input_fingerprint','seed','wire_message','parsed','message_metadata']:
    assert r[field]==f[field]==cache[field];replay_fields+=1
   for field in ['text','input_tokens','output_tokens','finish_reason']:
    assert r['generation'][field]==f['generation'][field]==cache['generation'][field];replay_fields+=1
   assert cache['generation']['call_id']==r['generation']['call_id']
   assert cache['output_digest']==r['output_digest']
   if node not in expected_closure:assert r['generation']['call_id']==d['base']['nodes'][node]['generation']['call_id']
  for tag,z in d.items():
   for r in z['nodes'].values():
    payload={k:v for k,v in r.items() if k not in ['output_digest','reused','source_execution_id']}
    assert hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()==r['output_digest']
   ns=[r for r in z['nodes'].values() if not r['reused']]
   assert len(ns)=={'base':13,'changed':6,'fresh':13,'cache':0}[tag]
   for r in ns:
    gen=r['generation'];cid=gen['call_id'];assert cid in physical;counted.append(cid)
    digest=hashlib.sha256(json.dumps(r['messages'],sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest();assert digest==physical[cid]['prompt_sha256']
    for field in ['input_tokens','output_tokens','gpu_seconds','elapsed_seconds','gpu_peak_allocated_bytes','finish_reason']:
     assert gen[field]==physical[cid][field]
    fc=format_counts[models[cid]];fc['n']+=1;fc['bad']+=int(not r['message_metadata']['format_ok'])
   cost={'calls':len(ns),'input_tokens':sum(r['generation']['input_tokens'] for r in ns),'output_tokens':sum(r['generation']['output_tokens'] for r in ns),'gpu_seconds':math.fsum(r['generation']['gpu_seconds'] for r in ns)}
   for field,value in cost.items():assert math.isclose(value,z['physical_cost'][field],abs_tol=1e-10,rel_tol=1e-12)
   assert z['logical_service_cost']['calls']==13
   costs.append(cost);graph_checks+=1
 assert len(counted)==len(set(counted))==64 and set(counted)==set(physical)
 assert format_counts==q['format_by_model'] and sum(x['bad'] for x in format_counts.values())==q['format_failures']==0
 assert q['inference_calls']==64 and q['benchmark_questions_read']==0 and q['formal_candidate_rounds_started']==0
 for cost,reported in zip(costs,q['costs'],strict=True):
  for field,value in cost.items():assert math.isclose(value,reported[field],abs_tol=1e-10,rel_tol=1e-12)
 result={'status':'PASS_INDEPENDENT_IMPLEMENTATION_SYNTHETIC_ENGINEERING_REPLAY','reviewer':'root separate arithmetic implementation; not external human review','synthetic_questions':2,'physical_calls':64,'models':format_counts,'replay_field_equalities':replay_fields,'graph_cost_records_checked':graph_checks,'resident_gpu_hours':status['gpu_hours'],'CUDA_event_seconds':math.fsum(r['gpu_seconds'] for r in physical.values()),'input_tokens':sum(r['input_tokens'] for r in physical.values()),'output_tokens':sum(r['output_tokens'] for r in physical.values()),'source_pins':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.input.iterdir()) if p.is_file() and p.suffix in {'.json','.jsonl'}},'review_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'limitations':['Only two synthetic tasks and one seed; no benchmark quality claim','Models and prompts unchanged; no real mixed-task scoring validation','Format success here does not estimate population format error','Receiver tokenizer runtime uses frozen executor; this check does not independently tokenize raw strings'],'GPU_calls_added_by_review':0,'formal_rounds_increment':0}
 result['review_implementation_correction']='Initial reviewer incorrectly required whole output_digest equality across fresh calls; it includes call_id and physical runtime. Corrected to compare semantic outputs and independently recompute all per-node integrity digests; reuse digest equality still required. No production output or protocol changed.'
 a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:result[k] for k in ['status','physical_calls','resident_gpu_hours','CUDA_event_seconds','replay_field_equalities']}))
if __name__=='__main__':main()
