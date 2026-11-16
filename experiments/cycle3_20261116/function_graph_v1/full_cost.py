"""Actual full-collection costs, distinct from hypothetical per-service costs."""
from pathlib import Path
import argparse,json,hashlib,collections,math

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def summarize(root):
 root=Path(root);c=read(root/'CLOSURE.json');assert c['all_workers_exited'] and c['status']=='COMPLETE_UNSCORED'
 for name,h in read(root/'ARCHIVE_MANIFEST.json').items():assert sha(root/name)==h
 by_model={k:collections.Counter() for k in ['qwen','mistral']};nominal={k:collections.Counter() for k in by_model};physical_ids=set();programs=0;visible_seconds=0;logical_programs=0;units=0;conditions=0;statuses=collections.Counter()
 for f in sorted(root.glob('worker_*/unit_*/MATRIX.json')):
  m=read(f);units+=1;conditions+=len(m['conditions']);unique={}
  for t in m['request_trace']:
   g=t['generation'];k=t['model'];assert g['call_id'] not in physical_ids;physical_ids.add(g['call_id']);by_model[k]['calls']+=1
   for field in ['input_tokens','output_tokens','elapsed_seconds','gpu_seconds']:assert math.isfinite(g[field]) and g[field]>=0;by_model[k][field]+=g[field]
  for row in m['conditions']:
   for rec in row['nodes'].values():
    k=rec['model'];nominal[k]['calls']+=1;nominal[k]['exact_request_reuse_events']+=int(rec['reused'])
    for field in ['input_tokens','output_tokens']:nominal[k][field]+=rec['generation'][field]
   for pr in row['programs']:unique.setdefault(pr['program_sha256'],pr);logical_programs+=1
  programs+=len(unique);sec=sum(p['visible']['seconds'] for p in unique.values());assert math.isclose(sec,m['physical_visible_execution_seconds'],abs_tol=1e-9);visible_seconds+=sec
  for pr in unique.values():statuses.update(x['status'] for x in pr['visible']['cases'])
 journal={k:collections.Counter() for k in by_model}
 for f in root.glob('worker_*/*_requests.jsonl'):
  model=f.name.split('_requests')[0]
  for line in f.read_bytes().splitlines():journal[model][json.loads(line).get('event','UNKNOWN')]+=1
 for model in by_model:assert journal[model]['START']==journal[model]['RETURNED']==by_model[model]['calls'];assert set(journal[model])<={'START','RETURNED','HEADER'}
 movement_seconds=0;movements=0
 for f in root.glob('worker_*/PLACEMENT.jsonl'):
  for line in f.read_bytes().splitlines():movement_seconds+=json.loads(line)['seconds'];movements+=1
 load_seconds=sum(read(f)['seconds'] for f in root.glob('worker_*/LOAD_*.json'))
 assert math.isclose(c['actual_GPU_hours']*3600,sum(c['residence_seconds_by_worker']),abs_tol=1e-8)
 return dict(status='PASS_COMPLETE_COLLECTION_COST',units=units,terminal_conditions=conditions,physical_model_costs={k:dict(v) for k,v in by_model.items()},physical_generation_calls=len(physical_ids),physical_tokens=sum(v['input_tokens']+v['output_tokens'] for v in by_model.values()),sum_nominal_condition_services={k:dict(v) for k,v in nominal.items()},physical_visible_program_executions=programs,physical_visible_case_executions=sum(statuses.values()),visible_execution_status_counts=dict(statuses),physical_visible_execution_seconds=visible_seconds,logical_program_records=logical_programs,actual_GPU_residence_hours=c['actual_GPU_hours'],worker_residence_seconds=c['residence_seconds_by_worker'],loading_wall_seconds_sum=load_seconds,model_placement_wall_seconds_sum=movement_seconds,model_placement_events=movements,actual_failed_generation_calls=0,model_aliases={'qwen':'Qwen2.5-3B-Instruct','mistral':'Qwen2.5-Coder-7B-Instruct'},archive_sha256=sha(root/'ARCHIVE_MANIFEST.json'),source_sha256=sha(__file__),limitation='Full diagnostic collection costs; not a claimed deployment speedup. Loading, placement and generation timing overlap the worker-residence ledger and must not be added to it. Nominal service costs reuse measured calls and are not remeasured end-to-end latency. Qualification and hidden CPU evaluation are separate cost entries. Runtime occupancy includes task loading, guards and prefix wait.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--out',required=True);a=p.parse_args();result=summarize(a.archive)
 with open(a.out,'x') as f:json.dump(result,f,indent=2)
 print(json.dumps({k:v for k,v in result.items() if k not in ['limitation','sum_nominal_condition_services']}))
