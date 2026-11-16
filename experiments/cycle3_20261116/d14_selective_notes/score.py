"""Fixed selective-note development scoring after complete execution and cost closure."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--repo',required=True);a=ap.parse_args();p=Path(a.root);r=Path(a.repo);run=p/'d14_collection_v1';inputs=p/'d14_selective_input_v1';out=run/'scoring_v1';assert not out.exists()
 assert read(run/'AUDIT.json')['status']=='PASS_COMPLETE_TERMINAL_EXECUTION';assert read(run/'AUDIT.json')['requests']==669;cost=read(run/'COST.json')
 receipt=read(inputs/'PLAN_RECEIPT.json');assert sha(inputs/'ENTRIES.json')==receipt['entries_sha256'];assert sha(inputs/'REQUESTS.jsonl')==receipt['requests_sha256']
 entries=read(inputs/'ENTRIES.json');new={row['job_id']:row['generation'] for f in (run/'archive').glob('worker_*/RESULTS.jsonl') for row in map(json.loads,f.read_bytes().splitlines())};assert len(new)==669
 variants=['G1_SELECTIVE','G1_RANDOM_MATCH','FLAT_SELECTIVE','INDEPENDENT_SELECTIVE'];seeds=[26111671,26111672];tasks={};private={}
 for task,parent,d13name in [('hotpot','annotation_formal7_v2','d13_hotpot_v1'),('2wiki','wiki_annotation_formal8_v2','d13_2wiki_v1')]:
  f=p/parent;s=read(f/'scoring_v1/SUMMARY.json');old=np.load(f/'scoring_v1/PRIVATE_ARRAYS.npz');ix={name:i for i,name in enumerate(s['conditions'])};approval=read(f/'SCORING_RELEASE.json')
  for path in reversed(approval['import_paths']):sys.path.insert(0,path)
  from hotpot_support import answer_metrics as single_metric
  vaultpath=Path(approval['pins']['vault']['path']);assert sha(vaultpath)==approval['pins']['vault']['sha256'];vault=read(vaultpath)
  units={read(u/'INPUT.json')['index']:u for u in (f/'monitor/archive').glob('worker_*/unit_*')};em=np.full((96,2,4),np.nan);f1=em.copy();tokens=em.copy();seen=set()
  for e in entries:
   if e['task']!=task:continue
   i=e['index'];j=seeds.index(e['seed']);k=variants.index(e['variant']);assert (i,j,k) not in seen;seen.add((i,j,k));unit=units[i];inp=read(unit/'INPUT.json');gold=vault[inp['payload']['id']];g=e['cached']['generation'] if e['cache_hit'] else new[e['generation_equivalence_sha256']]
   if e['cache_hit']:
    matrix=read(unit/'MATRIX.json');row=next(v for v in matrix['conditions'] if v['condition']==e['cached']['source_condition'] and v['execution_seed']==e['seed']);assert row['nodes'][e['cached']['source_node']]['generation']==g
   scores=[single_metric(g['text'].strip(),answer) for answer in (gold if task=='2wiki' else [gold])];em[i,j,k]=max(v['em'] for v in scores);f1[i,j,k]=max(v['f1'] for v in scores);tokens[i,j,k]=e['prefix_tokens']+g['input_tokens']+g['output_tokens']
  assert len(seen)==768 and np.isfinite(em).all();u=em-5e-6*tokens;table={};qarrays={};uarrays={}
  for k,name in enumerate(variants):
   table[name]=dict(EM=float(em[:,:,k].mean()),F1=float(f1[:,:,k].mean()),service_tokens=float(tokens[:,:,k].mean()),service_utility=float(u[:,:,k].mean()),EM_by_seed=em[:,:,k].mean(axis=0).tolist(),utility_by_seed=u[:,:,k].mean(axis=0).tolist());qarrays[name]=em[:,:,k];uarrays[name]=u[:,:,k]
  for name in ['SINGLE_Q','VOTE_Q3','INDEPENDENT3_Q','G1','FLAT_ANNOTATION']:
   table[name]=s['conditions'][name];qarrays[name]=old['em'][:,:,ix[name]];uarrays[name]=qarrays[name]-5e-6*old['tokens'][:,:,ix[name]]
  d13=read(p/d13name/'SUMMARY.json');masks=read(p/d13name/'PRIVATE_MASKS.json')
  for name in ['DROP_ALL_IF_ANY_ABSTAINS','IGNORE_ALL_NOTES']:
   table['D13_'+name]=d13['policies'][name];mask=np.array(masks[name],bool);qarrays['D13_'+name]=np.where(mask,old['em'][:,:,ix['SINGLE_Q']],old['em'][:,:,ix['G1']])
   prefix=np.zeros((96,2))
   for i,unit in units.items():
    matrix=read(unit/'MATRIX.json')
    for j,seed in enumerate(seeds):
     row=next(v for v in matrix['conditions'] if v['condition']=='G1' and v['execution_seed']==seed);prefix[i,j]=sum(n['generation']['input_tokens']+n['generation']['output_tokens'] for key,n in row['nodes'].items() if key!='M')
   t=np.where(mask,prefix+old['tokens'][:,:,ix['SINGLE_Q']],old['tokens'][:,:,ix['G1']]);uarrays['D13_'+name]=qarrays['D13_'+name]-5e-6*t;assert abs(uarrays['D13_'+name].mean()-table['D13_'+name]['service_utility'])<1e-12
  comparisons={}
  for name in ['SINGLE_Q','VOTE_Q3','INDEPENDENT3_Q','INDEPENDENT_SELECTIVE','FLAT_SELECTIVE','G1_RANDOM_MATCH','D13_DROP_ALL_IF_ANY_ABSTAINS','D13_IGNORE_ALL_NOTES']:
   delta=uarrays['G1_SELECTIVE']-uarrays[name];dq=qarrays['G1_SELECTIVE']-qarrays[name];comparisons[name]=dict(utility_delta=float(delta.mean()),utility_delta_by_seed=delta.mean(axis=0).tolist(),EM_delta=float(dq.mean()),corrections=int((dq==1).sum()),destructions=int((dq==-1).sum()),passed=bool(delta.mean()>=.005 and (delta.mean(axis=0)>0).all()))
  types={i:read(unit/'INPUT.json')['payload'].get('type','unknown') for i,unit in units.items()};breakdown={group:dict(questions=sum(v==group for v in types.values()),conditions={name:dict(EM=float(em[[types[i]==group for i in range(96)],:,k].mean()),utility=float(u[[types[i]==group for i in range(96)],:,k].mean())) for k,name in enumerate(variants)}) for group in set(types.values())}
  tasks[task]=dict(conditions=table,comparisons=comparisons,passed=all(v['passed'] for v in comparisons.values()),questions=96,seeds=seeds,type_breakdown=breakdown);private[task+'_em']=em;private[task+'_f1']=f1;private[task+'_tokens']=tokens
 result=dict(status='COMPLETE_EXPOSED_DEVELOPMENT_DIAGNOSTIC',tasks=tasks,passed_both_tasks=all(t['passed'] for t in tasks.values()),physical_new_requests=669,logical_entries=1536,collection_GPU_hours=cost['actual_GPU_hours'],source_sha256=sha(__file__),entries_sha256=sha(inputs/'ENTRIES.json'),confidence_intervals=False,bootstrap_runs=0,independent_confirmation=False,formal_round_increment=0,limitations=['Exposed development questions, two shared generation seeds, one matched-random policy seed.','Full original prefix cost retained, including removed notes; physical collection cost reported separately.','A pass is not active credit-policy evidence and requires fresh confirmation.'])
 out.mkdir();np.savez_compressed(out/'PRIVATE_ARRAYS.npz',**private);(out/'SUMMARY.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(dict(passed_both_tasks=result['passed_both_tasks'],tasks={k:dict(passed=v['passed'],selective=v['conditions']['G1_SELECTIVE'],comparisons=v['comparisons']) for k,v in tasks.items()})))
if __name__=='__main__':main()
