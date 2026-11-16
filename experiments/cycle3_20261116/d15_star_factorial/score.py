"""Frozen complete-star exposed-development headroom diagnostic; oracles are not deployable."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def corr(a,b):
 a=np.asarray(a).ravel();b=np.asarray(b).ravel();return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None
def credits(y,shock):
 old=np.empty((len(y),2,2));new=old.copy()
 for i,h in enumerate(shock):
  current=7^(1<<int(h));edges=[e for e in range(3) if e!=h]
  for j,e in enumerate(edges):old[i,:,j]=y[i,:,7]-y[i,:,7^(1<<e)];new[i,:,j]=y[i,:,current]-y[i,:,current^(1<<e)]
 shift=new-old;rc=corr(new[:,0],new[:,1]);rs=corr(shift[:,0],shift[:,1]);return dict(current_credit_seed_correlation=rc,shift_seed_correlation=rs,shift_fraction=float((shift!=0).mean()),distinct_shift_questions=int((shift!=0).any(axis=(1,2)).sum()),old_nonzero=int((old!=0).sum()),current_nonzero=int((new!=0).sum()),same_nonzero_sign=int(((new[:,0]!=0)&(new[:,1]!=0)&(np.sign(new[:,0])==np.sign(new[:,1]))).sum()),opposite_nonzero_sign=int((new[:,0]*new[:,1]<0).sum()))
def summarize(em,f1,tokens,shock,old_baselines):
 assert em.shape==f1.shape==tokens.shape==(96,2,2,8);u=em-5e-6*tokens;result={}
 for k,mode in enumerate(['RAW','FILTERED']):
  y=em[:,:,k];util=u[:,:,k];bestfixed=int(np.argmax(util.mean(axis=(0,1))));choices=np.argmax(util,axis=2);other=choices[:,::-1];same_u=np.take_along_axis(util,choices[:,:,None],axis=2)[:,:,0];other_u=np.take_along_axis(util,other[:,:,None],axis=2)[:,:,0];other_y=np.take_along_axis(y,other[:,:,None],axis=2)[:,:,0]
  table={str(mask):dict(EM=float(y[:,:,mask].mean()),F1=float(f1[:,:,k,mask].mean()),tokens=float(tokens[:,:,k,mask].mean()),utility=float(util[:,:,mask].mean()),EM_by_seed=y[:,:,mask].mean(axis=0).tolist(),utility_by_seed=util[:,:,mask].mean(axis=0).tolist()) for mask in range(8)}
  controls={**old_baselines,'MATCHED_GRAPH0':(y[:,:,0],util[:,:,0]),'RETROSPECTIVE_BEST_FIXED':(y[:,:,bestfixed],util[:,:,bestfixed])};comparisons={}
  for name,(cy,cu) in controls.items():
   delta=other_u-cu;comparisons[name]=dict(utility_delta=float(delta.mean()),by_seed=delta.mean(axis=0).tolist(),corrections=int(((other_y==1)&(cy==0)).sum()),destructions=int(((other_y==0)&(cy==1)).sum()),passed=bool(delta.mean()>=.005 and min(delta.mean(axis=0))>0))
  credit=credits(y,shock);creditpass=credit['current_credit_seed_correlation'] is not None and credit['shift_seed_correlation'] is not None and credit['current_credit_seed_correlation']>=.30 and credit['shift_seed_correlation']>=.20 and credit['shift_fraction']>=.03 and credit['distinct_shift_questions']>=20
  result[mode]=dict(graphs=table,retrospective_best_fixed_mask=bestfixed,same_seed_oracle_utility=float(same_u.mean()),opposite_seed_oracle_utility=float(other_u.mean()),opposite_seed_oracle_EM=float(other_y.mean()),opposite_seed_choices_counts=np.bincount(other.ravel(),minlength=8).tolist(),comparisons=comparisons,credit=credit,credit_gate=bool(creditpass),headroom_gate=all(v['passed'] for v in comparisons.values()),necessary_screen_passed=bool(creditpass and all(v['passed'] for v in comparisons.values())))
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--private-root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();p=Path(a.private_root);run=p/'d15_star_collection_v1';out=Path(a.output);assert not out.exists();assert read(run/'AUDIT.json')['status']=='PASS_COMPLETE_TERMINAL_EXECUTION';cost=read(run/'COST.json');inp=p/'d15_star_input_v1';receipt=read(inp/'RECEIPT.json');assert sha(inp/'ENTRIES.json')==receipt['entries_sha256'];entries=read(inp/'ENTRIES.json');new={v['job_id']:v['generation'] for f in (run/'archive').glob('worker_*/RESULTS.jsonl') for v in map(json.loads,f.read_bytes().splitlines())};assert len(new)==2181
 result={};arrays={}
 for task,folder in [('hotpot','annotation_formal7_v2'),('2wiki','wiki_annotation_formal8_v2')]:
  formal=p/folder;approval=read(formal/'SCORING_RELEASE.json')
  for path in reversed(approval['import_paths']):sys.path.insert(0,path)
  from hotpot_support import answer_metrics
  vaultpath=Path(approval['pins']['vault']['path']);assert sha(vaultpath)==approval['pins']['vault']['sha256'];vault=read(vaultpath);inputs={read(u/'INPUT.json')['index']:read(u/'INPUT.json')['payload'] for u in (formal/'monitor/archive').glob('worker_*/unit_*')};em=np.full((96,2,2,8),np.nan);f1=em.copy();tokens=em.copy();shock=np.full(96,-1);seeds=[26111671,26111672];seen=set()
  for e in entries:
   if e['task']!=task:continue
   i=e['index'];j=seeds.index(e['seed']);k=['RAW','FILTERED'].index(e['mode']);m=e['mask'];assert (i,j,k,m) not in seen;seen.add((i,j,k,m));g=e['cached']['generation'] if e['cache_hit'] else new[e['job_id']];gold=vault[inputs[i]['id']];scores=[answer_metrics(g['text'].strip(),a) for a in (gold if task=='2wiki' else [gold])];em[i,j,k,m]=max(v['em'] for v in scores);f1[i,j,k,m]=max(v['f1'] for v in scores);tokens[i,j,k,m]=e['prefix_tokens']+g['input_tokens']+g['output_tokens'];shock[i]=e['shock']
  assert len(seen)==3072 and np.isfinite(em).all() and (shock>=0).all();assert np.array_equal(em[:,:,0,0],em[:,:,1,0]);assert np.array_equal(tokens[:,:,0,0],tokens[:,:,1,0]);prior=np.load(formal/'scoring_v1/PRIVATE_ARRAYS.npz');s=read(formal/'scoring_v1/SUMMARY.json');ix={k:i for i,k in enumerate(s['conditions'])};baselines={k:(prior['em'][:,:,ix[k]],prior['em'][:,:,ix[k]]-5e-6*prior['tokens'][:,:,ix[k]]) for k in ['SINGLE_Q','VOTE_Q3','INDEPENDENT3_Q']}
  assert np.array_equal(em[:,:,0,7],prior['em'][:,:,ix['INDEPENDENT3_Q']]);d14=np.load(p/'d14_collection_v1/scoring_v1/PRIVATE_ARRAYS.npz');assert np.array_equal(em[:,:,1,7],d14[task+'_em'][:,:,3]);assert np.array_equal(tokens[:,:,1,7],d14[task+'_tokens'][:,:,3]);result[task]=summarize(em,f1,tokens,shock,baselines)
  for name,v in [('em',em),('f1',f1),('tokens',tokens),('shock',shock)]:arrays[task+'_'+name]=v
 summary=dict(status='COMPLETE_EXPOSED_STAR_FACTORIAL_DIAGNOSTIC',tasks=result,filtered_necessary_screen_both_tasks=all(v['FILTERED']['necessary_screen_passed'] for v in result.values()),new_requests=2181,logical_outcomes=6144,questions_per_task=96,seeds=[26111671,26111672],physical_GPU_hours=cost['actual_GPU_hours'],source_sha256=sha(__file__),confidence_intervals=False,bootstrap_runs=0,oracle_feedback_cost_excluded=True,deployable_policy=False,independent_confirmation=False,limits=['Oracles use unavailable labels and zero priced selection feedback; only a necessary headroom diagnostic.','Best fixed graph is selected retrospectively and is an optimistic comparator.','Two generation seeds share96 questions per task.','Only three-leaf star subsets; no upstream dependency or arbitrary DAG generalization.','No raw-mode substitution if filtered screen fails.'])
 out.mkdir();np.savez_compressed(out/'PRIVATE_ARRAYS.npz',**arrays);(out/'SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(dict(filtered_necessary_screen_both_tasks=summary['filtered_necessary_screen_both_tasks'],tasks={k:v['FILTERED'] for k,v in result.items()})))
if __name__=='__main__':main()
