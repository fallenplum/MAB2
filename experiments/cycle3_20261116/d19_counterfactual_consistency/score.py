import argparse,hashlib,json
from pathlib import Path
import numpy as np
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--private-root',required=True);a=ap.parse_args();p=Path(a.private_root);run=p/'d19_counterfactual_consistency_v1';read=lambda f:json.loads(f.read_bytes());receipt=read(run/'POLICY_RECEIPT.json');trace=run/'PRIVATE_POLICY_TRACE.jsonl';assert hashlib.sha256(trace.read_bytes()).hexdigest()==receipt['trace_sha256'];assert receipt['status']=='COMPLETE_GOLD_FREE_POLICY_REPLAY';raw=np.load(p/'d16_crossed_collection_v1/scoring_v1/PRIVATE_ARRAYS.npz');rows=[json.loads(v) for v in trace.read_bytes().splitlines()];methods=receipt['methods'];tasks={};private={}
 for task in ['hotpot','2wiki']:
  em=np.full((96,2,4,8),np.nan);tokens=em.copy();calls=em.copy();seen=set()
  for row in rows:
   if row['task']!=task:continue
   i=row['index'];s=row['source'];a=row['anchor'];j=methods.index(row['method']);key=(i,s,a,j);assert key not in seen;seen.add(key);c=row['choice'];em[key]=raw[task+'_em'][i,s,c['terminal'],c['mask']];tokens[key]=row['total_tokens'];calls[key]=len(row['queries'])
  assert len(seen)==6144 and np.isfinite(em).all();u=em-5e-6*tokens;reported={}
  for j,name in enumerate(methods):reported[name]=dict(EM=float(em[:,:,:,j].mean()),utility=float(u[:,:,:,j].mean()),tokens=float(tokens[:,:,:,j].mean()),unique_requests=float(calls[:,:,:,j].mean()),utility_by_source=u[:,:,:,j].mean(axis=(0,2)).tolist(),utility_by_anchor=u[:,:,:,j].mean(axis=(0,1)).tolist())
  comparisons={};gates=[]
  for name in methods[1:]:
   j=methods.index(name);delta=u[:,:,:,0]-u[:,:,:,j];by_source=delta.mean(axis=(0,2));by_anchor=delta.mean(axis=(0,1));gate=delta.mean()>=.005 and min(by_source)>0 and (by_anchor>0).sum()>=3;comparisons[name]=dict(utility_gain=float(delta.mean()),gain_by_source=by_source.tolist(),gain_by_anchor=by_anchor.tolist(),corrections=int(((em[:,:,:,0]==1)&(em[:,:,:,j]==0)).sum()),destructions=int(((em[:,:,:,0]==0)&(em[:,:,:,j]==1)).sum()),comparison_passed=bool(gate))
   if name in ['NO_NOTE_SINGLE','NO_NOTE_SC','SAME_GRAPH_SC','FIXED']:gates.append(gate)
  tasks[task]=dict(methods=reported,comparisons=comparisons,necessary_screen_passed=bool(all(gates)))
  for name,array in [('em',em),('tokens',tokens),('calls',calls)]:private[task+'_'+name]=array
 summary=dict(status='COMPLETE_EXPOSED_GOLD_FREE_CONSISTENCY_SCREEN',tasks=tasks,necessary_screen_both_tasks=all(v['necessary_screen_passed'] for v in tasks.values()),new_model_calls=0,new_GPU_hours=0,confidence_intervals=False,bootstrap_runs=0,independent_confirmation=False,all_probes_charged=True,trace_sha256=receipt['trace_sha256'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 with (run/'SUMMARY.json').open('x',encoding='utf-8') as f:json.dump(summary,f,indent=2,allow_nan=False)
 np.savez_compressed(run/'PRIVATE_SCORES.npz',**private);print(json.dumps(dict(passed=summary['necessary_screen_both_tasks'],tasks={k:dict(methods=v['methods'],passed=v['necessary_screen_passed']) for k,v in tasks.items()})))
if __name__=='__main__':main()
