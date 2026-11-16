"""Independent raw-generation vote, cache, token and scalar-score audit."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
from collections import Counter
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--private-root',required=True);ap.add_argument('--repo',required=True);args=ap.parse_args();p=Path(args.private_root);r=Path(args.repo);run=p/'d19_counterfactual_consistency_v1';read=lambda f:json.loads(f.read_bytes());summary=read(run/'SUMMARY.json');receipt=read(run/'POLICY_RECEIPT.json');trace=[json.loads(v) for v in (run/'PRIVATE_POLICY_TRACE.jsonl').read_bytes().splitlines()];entries=read(p/'d16_crossed_input_v1/ENTRIES.json');gens={v['job_id']:v['generation'] for f in (p/'d16_crossed_collection_v1/archive').glob('worker_*/RESULTS.jsonl') for v in map(json.loads,f.read_bytes().splitlines())};lookup={};shock={};ts=[26111671,26111672,261116161,261116162]
for e in entries:
 key=(e['task'],e['index'],[26111671,26111672].index(e['source_seed']),e['mask'],ts.index(e['terminal_seed']));lookup[key]=e['job_id'];shock[e['task'],e['index']]=e['shock']
 if e['cache_hit']:gens[e['job_id']]=e['cached']['generation']
spec=importlib.util.spec_from_file_location('official',r/'experiments/cycle2_20261113/hotpot_support/hotpot_support.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);features=np.load(p/'d18_features_v1/FEATURES.npz');raw=np.load(p/'d16_crossed_collection_v1/scoring_v1/PRIVATE_ARRAYS.npz');fixed=np.load(p/'d18_message_decisions_v1/PRIVATE_PREDICTIONS.npz');measured={};checks=0
def eq(x,y):
 global checks
 assert abs(float(x)-float(y))<1e-10,(x,y);checks+=1
for row in trace:
 task=row['task'];i=row['index'];s=row['source'];anchor=row['anchor'];method=row['method'];b=7^(1<<shock[task,i]);edits=[b^(1<<e) for e in range(3) if b&(1<<e)];metadata=lambda m,t:gens[lookup[task,i,s,m,t]];ordered=sorted(edits,key=lambda m:(metadata(m,anchor)['input_tokens'],m));coin=int(hashlib.sha256(json.dumps(['D19',task,i,s,anchor],separators=(',',':')).encode()).hexdigest(),16)%2;random_order=edits[coin:]+edits[:coin];fixedmask=int(fixed[task+'_choices_fixed'][i,s]);schedule={'ADAPT_GRAPH':[(b,anchor)]+[(m,anchor) for m in ordered],'ALL_GRAPH_VOTE':[(b,anchor)]+[(m,anchor) for m in ordered],'RANDOM_ORDER_GRAPH':[(b,anchor)]+[(m,anchor) for m in random_order],'KEEP_CURRENT':[(b,anchor)],'SAME_GRAPH_SC':[(b,(anchor+k)%4) for k in range(3)],'NO_NOTE_SC':[(0,(anchor+k)%4) for k in range(3)],'NO_NOTE_SINGLE':[(0,anchor)],'FIXED':[(fixedmask,anchor)]}[method];expected=[];keys=[];seen=set()
 for m,t in schedule:
  job=lookup[task,i,s,m,t]
  if job in seen:continue
  seen.add(job);g=gens[job];answer=module.normalize_answer(g['text']);keys.append(answer if answer else 'EMPTY_REQUEST_'+job);expected.append((m,t,job))
  if method!='ALL_GRAPH_VOTE' and answer and keys.count(answer)>=2:break
 assert [(q['mask'],q['terminal'],q['job_id']) for q in row['queries']]==expected
 costs=0
 for q in row['queries']:
  g=gens[q['job_id']];eq(q['input_tokens'],g['input_tokens']);eq(q['output_tokens'],g['output_tokens']);costs+=g['input_tokens']+g['output_tokens']
 expectedprefix=0 if method.startswith('NO_NOTE') else float(features[task+'_prefix_tokens'][i,s,fixedmask if method=='FIXED' else 7]);eq(expectedprefix,row['prefix_tokens']);costs+=expectedprefix;eq(costs,row['total_tokens']);winner=max(range(len(keys)),key=lambda k:keys.count(keys[k]));m,t,job=expected[winner];assert row['choice']['job_id']==job;em=float(raw[task+'_em'][i,s,t,m]);measured[task,i,s,anchor,method]=(em,costs,len(expected))
for task,rr in summary['tasks'].items():
 for method,report in rr['methods'].items():
  values=[measured[task,i,s,a,method] for i in range(96) for s in range(2) for a in range(4)];n=len(values);eq(sum(v[0] for v in values)/n,report['EM']);eq(sum(v[1] for v in values)/n,report['tokens']);eq(sum(v[2] for v in values)/n,report['unique_requests']);eq(sum(v[0]-5e-6*v[1] for v in values)/n,report['utility'])
 gates=[]
 for method,report in rr['comparisons'].items():
  deltas=[];by_source=[[],[]];by_anchor=[[],[],[],[]];correct=destroy=0
  for i in range(96):
   for s in range(2):
    for a in range(4):
     x=measured[task,i,s,a,'ADAPT_GRAPH'];y=measured[task,i,s,a,method];v=x[0]-y[0]-5e-6*(x[1]-y[1]);deltas.append(v);by_source[s].append(v);by_anchor[a].append(v);correct+=x[0]==1 and y[0]==0;destroy+=x[0]==0 and y[0]==1
  mean=sum(deltas)/len(deltas);ss=[sum(v)/len(v) for v in by_source];aa=[sum(v)/len(v) for v in by_anchor];eq(mean,report['utility_gain']);eq(correct,report['corrections']);eq(destroy,report['destructions']);gate=mean>=.005 and min(ss)>0 and sum(v>0 for v in aa)>=3;assert gate==report['comparison_passed']
  for x,y in zip(ss,report['gain_by_source']):eq(x,y)
  for x,y in zip(aa,report['gain_by_anchor']):eq(x,y)
  if method in ['NO_NOTE_SINGLE','NO_NOTE_SC','SAME_GRAPH_SC','FIXED']:gates.append(gate)
 assert all(gates)==rr['necessary_screen_passed']
assert all(v['necessary_screen_passed'] for v in summary['tasks'].values())==summary['necessary_screen_both_tasks']
result=dict(status='PASS_INDEPENDENT_D19_RAW_POLICY_AND_COST_AUDIT',policy_records=len(trace),numeric_checks=checks,shared_official_normalizer=True,all_requested_cells_and_votes_replayed=True,summary_sha256=hashlib.sha256((run/'SUMMARY.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
with (run/'AUDIT.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
