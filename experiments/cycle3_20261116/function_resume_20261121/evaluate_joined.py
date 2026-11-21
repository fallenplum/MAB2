"""CPU-only hidden-case scoring after immutable model collection closure."""
from pathlib import Path
import argparse,json,hashlib,time,os
from function_adapter import run_function

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def write(path,value):
 with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
def validate_joint_closure(closure):
 assert closure['kind']=='verified_two_segment_scorer_view' and closure['all_workers_exited']
 assert closure['status']=='COMPLETE_UNSCORED' and closure['quality_seen'] is False
 parent, continuation = closure['source_segments']
 assert parent['indices']==list(range(6)) and parent['status']=='PARTIAL_OR_FAILED' and parent['exit_codes']==[-15,-15]
 assert continuation['indices']==list(range(6,16)) and continuation['status']=='COMPLETE_UNSCORED' and continuation['exit_codes']==[0]*4
 assert closure['exit_codes']==parent['exit_codes']+continuation['exit_codes']

def main():
 a=argparse.ArgumentParser();a.add_argument('--archive',required=True);a.add_argument('--hidden',required=True);a.add_argument('--hidden-sha',required=True);a.add_argument('--archive-sha',required=True);a.add_argument('--output',required=True);a=a.parse_args();root=Path(a.archive);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 assert sha(root/'ARCHIVE_MANIFEST.json')==a.archive_sha and sha(a.hidden)==a.hidden_sha
 closure=json.loads((root/'CLOSURE.json').read_bytes());validate_joint_closure(closure)
 manifest=json.loads((root/'ARCHIVE_MANIFEST.json').read_bytes())
 for name,h in manifest.items():assert sha(root/name)==h
 labels=json.loads(Path(a.hidden).read_bytes());assert len(labels)==16;by_id={x['id']:x for x in labels};assert len(by_id)==16
 results={};actual_cases=0;programs=0;seen_ids=set()
 for matrix in sorted(root.glob('worker_*/unit_*/MATRIX.json')):
  unit=json.loads((matrix.parent/'INPUT.json').read_bytes());payload=unit['payload'];identity=payload['id'];assert identity in by_id and identity not in seen_ids;seen_ids.add(identity);label=by_id[identity];m=json.loads(matrix.read_bytes());fingerprint=hashlib.sha256(canonical(payload).encode()).hexdigest();assert fingerprint==m['question_fingerprint'] and len(m['conditions'])==30
  unique={}
  for row in m['conditions']:
   for pr in row['programs']:
    assert hashlib.sha256(pr['code'].encode()).hexdigest()==pr['program_sha256'];unique.setdefault(pr['program_sha256'],pr['code'])
  results[fingerprint]={}
  for codehash,code in unique.items():
   cases=[]
   for encoded_args,encoded_expected in label['hidden_cases']:
    assert time.monotonic()-start<4*3600,'Hidden CPU scoring wall cap reached'
    # Expected answers stay in this parent. The sandbox receives generated source and arguments only.
    z=run_function(code,payload['function_name'],json.loads(encoded_args),json.loads(encoded_expected),out/'sandbox_cases');cases.append(z);actual_cases+=1
   results[fingerprint][codehash]=dict(cases=cases,program_sha256=codehash);programs+=1
  write(out/(fingerprint+'.json'),results[fingerprint])
  print(json.dumps(dict(completed_tasks=len(seen_ids),unique_programs=programs,case_executions=actual_cases,elapsed_seconds=time.monotonic()-start)),flush=True)
 assert seen_ids==set(by_id)
 write(out/'HIDDEN_RESULTS.json',dict(results=results,metadata=dict(tasks=16,unique_programs=programs,case_executions=actual_cases,elapsed_seconds=time.monotonic()-start,actual_execution_seconds=sum(c['elapsed_seconds'] for v in results.values() for pr in v.values() for c in pr['cases']),source_sha256=sha(__file__),archive_sha256=a.archive_sha,hidden_sha256=a.hidden_sha,model_calls=0,GPU_hours=0,expected_values_sent_to_sandbox=False)))
 write(out/'COMPLETE.json',dict(status='COMPLETE_HIDDEN_EXECUTION',result_sha256=sha(out/'HIDDEN_RESULTS.json'),source_sha256=sha(__file__)))
 write(out/'ARCHIVE_MANIFEST.json',{str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()})
if __name__=='__main__':main()
