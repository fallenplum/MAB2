"""Independent archive consistency and visible-execution accounting audit."""
import argparse,hashlib,json,math,collections
from pathlib import Path

def canon(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(canon(x).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit(root,tokenizers=None):
 root=Path(root);manifest=json.loads((root/'ARCHIVE_MANIFEST.json').read_bytes());checks=0
 for rel,h in manifest.items():assert sha(root/rel)==h;checks+=1
 close=json.loads((root/'CLOSURE.json').read_bytes());assert close['all_workers_exited'] and close['status']=='COMPLETE_UNSCORED';assert math.isclose(close['actual_GPU_hours'],sum(close['residence_seconds_by_worker'])/3600,abs_tol=1e-12)
 from function_adapter import author_compare,unpack
 from function_operator import extract,CODE,REPAIR,PLAN,CRITIC,RELAY
 tokenizers=tokenizers or {};physical=0;logical=0;programs_count=0;cpu=0;matrices=sorted(root.glob('worker_*/unit_*/MATRIX.json'))
 for f in matrices:
  m=json.loads(f.read_bytes());unit=json.loads((f.parent/'INPUT.json').read_bytes());p=unit['payload'];assert m['question_fingerprint']==digest(p);assert set(p)=={'id','question','function_name','visible_cases'}
  assert m['execution_seeds']==[261117101,261117102,261117103] and len(m['conditions'])==30 and m['hidden_test_access'] is False
  assert m['protocol_sha256']==digest(m['protocol']);traces={t['request_fingerprint']:t for t in m['request_trace']};assert len(traces)==len(m['request_trace']);used=set();program_seen={};conditions={x['key']:x for x in m['conditions']}
  for row in m['conditions']:
   c=row['condition'];s=row['execution_seed'];assert row['key']==f'{s}/{c}' and row['status']=='COMPLETE';graph=c.startswith('G');expected_edges=[['U','R'],['V','R'],['R','M'],['U','M'],['V','M']] if graph else []
   if c.startswith('G1'):expected_edges.remove([unit['shock'],'R'])
   if c.endswith('_MINUS_EU'):expected_edges.remove(['U','M'])
   if c.endswith('_MINUS_EV'):expected_edges.remove(['V','M'])
   assert row['edges']==expected_edges
   alive={'M'}
   while True:
    new=alive|{u for u,v in expected_edges if v in alive}
    if new==alive:break
    alive=new
   order=[n for n in 'UVRM' if n in alive] if graph else ['M'] if c.startswith('SINGLE') else ['M','B1','B2'] if c=='BOK3' else list(row['nodes'])
   assert set(order)==set(row['nodes']);terminal=[]
   for node in order:
    rec=row['nodes'][node];req=rec['request'];g=rec['generation'];fp=rec['request_fingerprint'];assert fp==digest(req) and fp in traces
    t=traces[fp];assert t['status']=='GENERATION_RETURNED' and t['request']==req and t['generation']==g;assert len(t['native_events'])==1
    native=t['native_events'][0];assert native['status']=='SUCCESS'
    for k in ['call_id','input_tokens','output_tokens','elapsed_seconds','gpu_seconds','gpu_peak_allocated_bytes','finish_reason']:assert native[k]==g[k]
    assert native['prompt_sha256']==digest(req['messages']) and native['seed']==req['seed'];assert req['seed']==int(digest(['exogenous_path_rng_v1',p['id'],s,node])[:16],16)%(2**31-1)
    final=not graph or node=='M';model='qwen' if c=='SINGLE_Q' or not final else 'mistral';assert rec['model']==req['model']==model and req['max_new_tokens']==(1024 if final else 192)
    assert bool(rec['reused'])==(fp in used);used.add(fp)
    msg=req['messages'];data=json.loads(msg[1]['content']);assert data['question']==p['question'] and data['function_name']==p['function_name']
    parents=[n for n in 'UVRM' if [n,node] in expected_edges];assert data['peer_notes']==[dict(source=n,text=row['nodes'][n]['generation']['text'].strip()) for n in parents]
    instruction=CODE if final else PLAN if node=='U' else CRITIC if node=='V' else RELAY
    if c=='REPAIR3' and node!='M':
     instruction=REPAIR;idx=order.index(node);prev=row['programs'][idx-1];assert prev['visible']['passed']<2 and data['previous_program']==prev['code']
     fb=[dict(input=x[:512],expected=y[:512],actual=json.dumps(z.get('typed_return'),ensure_ascii=False)[:512],stderr=z['stderr'][:256],status=z['status'],passed=z['passed']) for (x,y),z in zip(p['visible_cases'],prev['visible']['cases'])];assert data['execution_feedback']==fb
    else:assert set(data)=={'question','function_name','peer_notes'}
    assert msg[0]['content']==instruction
    if model in tokenizers:
     encoded=tokenizers[model].apply_chat_template(msg,tokenize=True,add_generation_prompt=True,return_dict=False);assert isinstance(encoded,list) and all(type(v) is int for v in encoded);assert len(encoded)==g['input_tokens']==t['capacity']['input_tokens']
    assert g['output_tokens']<=req['max_new_tokens'] and g['input_tokens']<=12288
    if final:terminal.append(rec)
    logical+=1;checks+=20
   assert len(terminal)==len(row['programs'])
   for rec,pr in zip(terminal,row['programs']):
    assert pr['code']==extract(rec['generation']['text']) and pr['program_sha256']==hashlib.sha256(pr['code'].encode()).hexdigest();assert pr['source_request_fingerprint']==rec['request_fingerprint'];v=pr['visible'];assert len(v['cases'])==v['total']==2
    for (_,gold),case in zip(p['visible_cases'],v['cases']):
     expected=case['status']=='OK' and case.get('returned') and author_compare(unpack(case['typed_return']),json.loads(gold));assert bool(case['passed'])==bool(expected)
    assert v['passed']==sum(x['passed'] for x in v['cases']) and math.isclose(v['seconds'],sum(x['elapsed_seconds'] for x in v['cases']),abs_tol=1e-12)
    key=pr['program_sha256'];assert bool(pr['execution_reused'])==(key in program_seen)
    if key in program_seen:assert v==program_seen[key]
    program_seen[key]=v;checks+=9;programs_count+=1
   index=max(range(len(row['programs'])),key=lambda i:(row['programs'][i]['visible']['passed'],-i));assert row['selected_program']==index and row['answer']==row['programs'][index]['code']
   if c=='REPAIR3':assert len(row['programs'])==3 or row['programs'][-1]['visible']['passed']==2
  assert used==set(traces);physical+=len(traces);seconds=sum(v['seconds'] for v in program_seen.values());assert math.isclose(seconds,m['physical_visible_execution_seconds'],abs_tol=1e-10);cpu+=seconds
  for model in ['qwen','mistral']:
   native=m['native_events_by_model'][model];assert len(native)==sum(t['model']==model for t in traces.values());assert {x['call_id'] for x in native}=={t['generation']['call_id'] for t in traces.values() if t['model']==model}
 return dict(status='PASS_RAW_FUNCTION_EXECUTION_AUDIT',units=len(matrices),scalar_checks=checks,physical_requests=physical,logical_calls=logical,program_records=programs_count,physical_visible_execution_seconds=cpu,actual_GPU_hours=close['actual_GPU_hours'],local_tokenizer_counts_checked=bool(tokenizers),hidden_labels_accessed=False,archive_sha256=sha(root/'ARCHIVE_MANIFEST.json'),source_sha256=sha(Path(__file__)))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--archive',required=True);a.add_argument('--out',required=True);a.add_argument('--tokenizers',nargs=2);a=a.parse_args();tokens={}
 if a.tokenizers:
  from transformers import AutoTokenizer
  tokens={key:AutoTokenizer.from_pretrained(value,local_files_only=True) for key,value in zip(['qwen','mistral'],a.tokenizers)}
 result=audit(a.archive,tokens)
 with open(a.out,'x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
