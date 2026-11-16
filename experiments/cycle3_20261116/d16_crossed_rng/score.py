"""Frozen crossed-RNG expected-credit diagnostic. Repeats and oracles are not free policies."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def corr(a,b):
 a=np.asarray(a).ravel();b=np.asarray(b).ravel();return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None
def credit_arrays(y,shock):
 old=np.empty((len(y),2,4,2));current=old.copy()
 for i,h in enumerate(shock):
  mask=7^(1<<int(h))
  for j,e in enumerate(e for e in range(3) if e!=h):old[i,:,:,j]=y[i,:,:,7]-y[i,:,:,7^(1<<e)];current[i,:,:,j]=y[i,:,:,mask]-y[i,:,:,mask^(1<<e)]
 return old,current,current-old
def components(x):
 a=x.transpose(0,3,1,2);grand=a.mean(axis=(2,3),keepdims=True);source=a.mean(axis=3,keepdims=True);terminal=a.mean(axis=2,keepdims=True);ss_source=float((4*(source-grand)**2).sum());ss_terminal=float((2*(terminal-grand)**2).sum());ss_interaction=float(((a-source-terminal+grand)**2).sum());ss_total=float(((a-grand)**2).sum());assert abs(ss_source+ss_terminal+ss_interaction-ss_total)<1e-8
 return dict(source_SS=ss_source,terminal_SS=ss_terminal,interaction_SS=ss_interaction,total_within_grid_SS=ss_total,fractions={k:v/ss_total if ss_total else None for k,v in [('source',ss_source),('terminal',ss_terminal),('interaction',ss_interaction)]},population_variance_components=False)
def summarize(em,f1,tokens,shock):
 assert em.shape==f1.shape==tokens.shape==(96,2,4,8);old,current,shift=credit_arrays(em,shock);utility=em-5e-6*tokens;correlations={};noise={};pairwise={}
 for name,x in [('old',old),('current',current),('shift',shift)]:
  correlations[name]=dict(terminal_half=corr(x[:,:,:2].mean(axis=(1,2)),x[:,:,2:].mean(axis=(1,2))),source_half=corr(x[:,0].mean(axis=1),x[:,1].mean(axis=1)));noise[name]=components(x);pairwise[name]=dict(fixed_source_terminal_pairs=[dict(source=s,terminal_pair=[a,b],correlation=corr(x[:,s,a],x[:,s,b])) for s in range(2) for a in range(4) for b in range(a+1,4)],fixed_terminal_source_pairs=[dict(terminal=t,correlation=corr(x[:,0,t],x[:,1,t])) for t in range(4)])
 halves=[utility[:,:,:2].mean(axis=(1,2)),utility[:,:,2:].mean(axis=(1,2))];qhalves=[em[:,:,:2].mean(axis=(1,2)),em[:,:,2:].mean(axis=(1,2))];transfers=[]
 for train,test in [(0,1),(1,0)]:
  choices=np.argmax(halves[train],axis=1);fixed=int(np.argmax(halves[test].mean(axis=0)));selected=halves[test][np.arange(96),choices];delta=selected-halves[test][:,fixed];transfers.append(dict(training_half=train,evaluation_half=test,best_fixed_mask=fixed,best_fixed_utility=float(halves[test][:,fixed].mean()),oracle_utility=float(selected.mean()),utility_gain=float(delta.mean()),oracle_EM=float(qhalves[test][np.arange(96),choices].mean()),choices=np.bincount(choices,minlength=8).tolist()))
 gains=[v['utility_gain'] for v in transfers];creditpass=all(correlations[name][axis] is not None and correlations[name][axis]>=threshold for name,threshold in [('current',.30),('shift',.20)] for axis in ['terminal_half','source_half']);headroom=float(np.mean(gains))>=.005 and min(gains)>0
 graphs={str(mask):dict(EM=float(em[:,:,:,mask].mean()),F1=float(f1[:,:,:,mask].mean()),service_tokens=float(tokens[:,:,:,mask].mean()),utility=float(utility[:,:,:,mask].mean()),utility_by_source=utility[:,:,:,mask].mean(axis=(0,2)).tolist(),utility_by_terminal=utility[:,:,:,mask].mean(axis=(0,1)).tolist()) for mask in range(8)}
 return dict(correlations=correlations,descriptive_noise_components=noise,pairwise_correlations=pairwise,transfer_oracles=transfers,mean_transfer_gain=float(np.mean(gains)),credit_gate=bool(creditpass),headroom_gate=bool(headroom),necessary_screen_passed=bool(creditpass and headroom),graphs=graphs)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--private-root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();p=Path(a.private_root);run=p/'d16_crossed_collection_v1';out=Path(a.output);assert not out.exists();assert read(run/'AUDIT.json')['status']=='PASS_COMPLETE_TERMINAL_EXECUTION';assert read(run/'INPUT_AUDIT.json')['status']=='PASS_INDEPENDENT_CROSSED_INPUT_AUDIT';cost=read(run/'COST.json');inp=p/'d16_crossed_input_v1';receipt=read(inp/'RECEIPT.json');assert sha(inp/'ENTRIES.json')==receipt['entries_sha256'];entries=read(inp/'ENTRIES.json');new={v['job_id']:v['generation'] for f in (run/'archive').glob('worker_*/RESULTS.jsonl') for v in map(json.loads,f.read_bytes().splitlines())};assert len(new)==6694
 tasks={};arrays={};source_seeds=[26111671,26111672];terminal_seeds=[26111671,26111672,261116161,261116162]
 for task,folder in [('hotpot','annotation_formal7_v2'),('2wiki','wiki_annotation_formal8_v2')]:
  formal=p/folder;approval=read(formal/'SCORING_RELEASE.json')
  for path in reversed(approval['import_paths']):sys.path.insert(0,path)
  from hotpot_support import answer_metrics
  vaultpath=Path(approval['pins']['vault']['path']);assert sha(vaultpath)==approval['pins']['vault']['sha256'];vault=read(vaultpath);ids={read(u/'INPUT.json')['index']:read(u/'INPUT.json')['payload']['id'] for u in (formal/'monitor/archive').glob('worker_*/unit_*')};em=np.full((96,2,4,8),np.nan);f1=em.copy();tokens=em.copy();shock=np.full(96,-1);seen=set()
  for e in (e for e in entries if e['task']==task):
   i=e['index'];j=source_seeds.index(e['source_seed']);k=terminal_seeds.index(e['terminal_seed']);m=e['mask'];assert (i,j,k,m) not in seen;seen.add((i,j,k,m));g=e['cached']['generation'] if e['cache_hit'] else new[e['job_id']];gold=vault[ids[i]];values=[answer_metrics(g['text'].strip(),a) for a in (gold if task=='2wiki' else [gold])];em[i,j,k,m]=max(v['em'] for v in values);f1[i,j,k,m]=max(v['f1'] for v in values);tokens[i,j,k,m]=e['prefix_tokens']+g['input_tokens']+g['output_tokens'];shock[i]=e['shock']
  assert len(seen)==6144 and np.isfinite(em).all();old=np.load(p/'d15_star_collection_v1/scoring_v1/PRIVATE_ARRAYS.npz')
  for j in range(2):assert np.array_equal(em[:,j,j,:],old[task+'_em'][:,j,1,:]) and np.array_equal(tokens[:,j,j,:],old[task+'_tokens'][:,j,1,:])
  assert np.array_equal(em[:,0,:,0],em[:,1,:,0]);tasks[task]=summarize(em,f1,tokens,shock)
  for name,v in [('em',em),('f1',f1),('tokens',tokens),('shock',shock)]:arrays[task+'_'+name]=v
 summary=dict(status='COMPLETE_EXPOSED_CROSSED_RNG_DIAGNOSTIC',tasks=tasks,necessary_screen_both_tasks=all(v['necessary_screen_passed'] for v in tasks.values()),questions_per_task=96,source_seeds=source_seeds,terminal_seeds=terminal_seeds,logical_outcomes=12288,new_requests=6694,physical_GPU_hours=cost['actual_GPU_hours'],source_sha256=sha(__file__),confidence_intervals=False,bootstrap_runs=0,independent_confirmation=False,oracle_feedback_cost_excluded=True,repeats_are_free_deployment_ensemble=False,limits=['Finite crossed grid, not independent repeated datasets or population variance estimates.','Two fixed source realizations and four terminal seeds on exposed development questions.','Oracle comparison is optimistic and does not pay feedback selection; it is only a necessary headroom screen.','D15 single-realization gates remain failed regardless of D16 outcome.'])
 out.mkdir();np.savez_compressed(out/'PRIVATE_ARRAYS.npz',**arrays);(out/'SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(dict(necessary_screen_both_tasks=summary['necessary_screen_both_tasks'],tasks={k:{f:v[f] for f in ['correlations','mean_transfer_gain','credit_gate','headroom_gate','necessary_screen_passed']} for k,v in tasks.items()})))
if __name__=='__main__':main()
