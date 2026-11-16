"""Scalar independent audit of raw factorial answers, costs, oracle choices and credit gates."""
import argparse,json,hashlib,math,statistics,sys
from pathlib import Path
import numpy as np
def read(p):return json.loads(Path(p).read_bytes())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--private-root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();p=Path(a.private_root);run=p/'d15_star_collection_v1';s=read(run/'scoring_v1/SUMMARY.json');arrays=np.load(run/'scoring_v1/PRIVATE_ARRAYS.npz');entries=read(p/'d15_star_input_v1/ENTRIES.json');new={v['job_id']:v['generation'] for f in (run/'archive').glob('worker_*/RESULTS.jsonl') for v in map(json.loads,f.read_bytes().splitlines())};checked=0
 def near(x,y):
  nonlocal checked
  assert math.isclose(x,y,rel_tol=0,abs_tol=1e-12),(x,y);checked+=1
 def correlation(x,y):return statistics.correlation(x,y) if len(set(x))>1 and len(set(y))>1 else None
 for task,folder in [('hotpot','annotation_formal7_v2'),('2wiki','wiki_annotation_formal8_v2')]:
  formal=p/folder;approval=read(formal/'SCORING_RELEASE.json')
  for path in reversed(approval['import_paths']):sys.path.insert(0,path)
  from hotpot_support import answer_metrics
  golds=read(approval['pins']['vault']['path']);ids={read(u/'INPUT.json')['index']:read(u/'INPUT.json')['payload']['id'] for u in (formal/'monitor/archive').glob('worker_*/unit_*')};raw={};seeds=[26111671,26111672]
  for e in (x for x in entries if x['task']==task):
   i=e['index'];j=seeds.index(e['seed']);k=['RAW','FILTERED'].index(e['mode']);m=e['mask'];g=e['cached']['generation'] if e['cache_hit'] else new[e['job_id']];gold=golds[ids[i]];scores=[answer_metrics(g['text'].strip(),v) for v in (gold if task=='2wiki' else [gold])];y=max(v['em'] for v in scores);f=max(v['f1'] for v in scores);t=e['prefix_tokens']+g['input_tokens']+g['output_tokens'];raw[i,j,k,m]=(y,f,t,y-5e-6*t)
   for field,value in [('em',y),('f1',f),('tokens',t)]:near(value,float(arrays[task+'_'+field][i,j,k,m]))
  old=np.load(formal/'scoring_v1/PRIVATE_ARRAYS.npz');ix={n:i for i,n in enumerate(read(formal/'scoring_v1/SUMMARY.json')['conditions'])}
  for k,mode in enumerate(['RAW','FILTERED']):
   result=s['tasks'][task][mode];means=[]
   for m in range(8):
    rows=[raw[i,j,k,m] for i in range(96) for j in range(2)];means.append(sum(v[3] for v in rows)/192)
    for f,field in enumerate(['EM','F1','tokens','utility']):near(sum(v[f] for v in rows)/192,result['graphs'][str(m)][field])
   best=max(range(8),key=lambda m:means[m]);assert best==result['retrospective_best_fixed_mask'];choices={(i,j):max(range(8),key=lambda m:raw[i,j,k,m][3]) for i in range(96) for j in range(2)};nom={(i,j):raw[i,j,k,choices[i,1-j]] for i in range(96) for j in range(2)};near(sum(v[3] for v in nom.values())/192,result['opposite_seed_oracle_utility']);near(sum(raw[i,j,k,choices[i,j]][3] for i,j in choices)/192,result['same_seed_oracle_utility'])
   passes=[]
   for name,comparison in result['comparisons'].items():
    other={}
    for i,j in choices:
     if name in ['MATCHED_GRAPH0','RETROSPECTIVE_BEST_FIXED']:other[i,j]=raw[i,j,k,0 if name=='MATCHED_GRAPH0' else best]
     else:y=float(old['em'][i,j,ix[name]]);t=float(old['tokens'][i,j,ix[name]]);other[i,j]=(y,0,t,y-5e-6*t)
    delta={(i,j):nom[i,j][3]-other[i,j][3] for i,j in choices};avg=sum(delta.values())/192;by=[sum(delta[i,j] for i in range(96))/96 for j in range(2)];near(avg,comparison['utility_delta'])
    for j in range(2):near(by[j],comparison['by_seed'][j])
    assert sum(nom[key][0]>other[key][0] for key in choices)==comparison['corrections'];assert sum(nom[key][0]<other[key][0] for key in choices)==comparison['destructions'];passed=avg>=.005 and min(by)>0;assert passed==comparison['passed'];passes.append(passed)
   c=[[],[]];shift=[[],[]];changed=set()
   for i in range(96):
    h=int(arrays[task+'_shock'][i]);current=7^(1<<h)
    for e in range(3):
     if e==h:continue
     for j in range(2):
      oldc=raw[i,j,k,7][0]-raw[i,j,k,7^(1<<e)][0];newc=raw[i,j,k,current][0]-raw[i,j,k,current^(1<<e)][0];c[j].append(newc);shift[j].append(newc-oldc)
      if newc!=oldc:changed.add(i)
   rc=correlation(*c);rs=correlation(*shift);credit=result['credit']
   for value,name in [(rc,'current_credit_seed_correlation'),(rs,'shift_seed_correlation')]:
    if value is None:assert credit[name] is None
    else:near(value,credit[name])
   frac=sum(v!=0 for row in shift for v in row)/384;near(frac,credit['shift_fraction']);assert len(changed)==credit['distinct_shift_questions'];creditpass=rc is not None and rs is not None and rc>=.30 and rs>=.20 and frac>=.03 and len(changed)>=20;assert creditpass==result['credit_gate'];assert all(passes)==result['headroom_gate'];assert (creditpass and all(passes))==result['necessary_screen_passed']
 assert s['filtered_necessary_screen_both_tasks']==all(v['FILTERED']['necessary_screen_passed'] for v in s['tasks'].values())
 result=dict(status='PASS_INDEPENDENT_RAW_SCALAR_FACTORIAL_AUDIT',numeric_checks=checked,logical_raw_outcomes=6144,shared_official_normalizer=True,summary_sha256=hashlib.sha256((run/'scoring_v1/SUMMARY.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 with Path(a.output).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if __name__=='__main__':main()
