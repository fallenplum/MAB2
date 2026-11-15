"""Posthoc development-only disjoint I/C/S replay; no fresh evidence or GPU use."""
from pathlib import Path
import json,hashlib,numpy as np
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--private-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
P=args.private_root;POST=P/'path_formal_execution_v1/postclosure_v1';OUT=args.output
ORDER=['G0','G0_MINUS_EU','G0_MINUS_EV','G1','G1_MINUS_EU','G1_MINUS_EV','SINGLE_Q','SINGLE_M','INDEPENDENT3_M']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(POST/'scoring_001/PRIVATE_ARRAYS.json')=='8109daeb9a6155da6c7101f8ebc3a10a92143dd0420e0aecf018f108941e21be'
raw=json.loads((POST/'scoring_001/PRIVATE_ARRAYS.json').read_bytes());summary=json.loads((POST/'scoring_001/SUMMARY.json').read_bytes())
# Recover actual frozen order from source, never assume display order.
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'exogenous_paths_v1'))
from point_statistics import ORDER
x=np.asarray(raw['em']);assert x.shape==(96,2,9)
selection=json.loads((P/'path_screen_preparation_v1/capacity_selection_v1/SELECTION_MANIFEST.json').read_bytes());shocks=np.array([0 if q['removed_upstream']=='U' else 1 for q in selection['questions']])
tokens=np.zeros_like(x);kernel=np.zeros_like(x)
for r in raw['cost_rows']:
    seed,n=r['condition'].split('/');i=r['question_index'];s=[26111311,26111312].index(int(seed));j=ORDER.index(n)
    tokens[i,s,j]=sum(v['input_tokens']+v['output_tokens'] for v in r['by_model'].values());kernel[i,s,j]=sum(v['gpu_seconds'] for v in r['by_model'].values())
idx=lambda n:ORDER.index(n)
old=x[:,0,idx('G0')][:,None]-x[:,0,[idx('G0_MINUS_EU'),idx('G0_MINUS_EV')]]
new=x[:,0,idx('G1')][:,None]-x[:,0,[idx('G1_MINUS_EU'),idx('G1_MINUS_EV')]]
results=[]
for splitseed in [261115501,261115502,261115503,261115504,261115505]:
    rng=np.random.default_rng(splitseed);parts=[[],[],[]]
    for shock in [0,1]:
        order=rng.permutation(np.flatnonzero(shocks==shock))
        for j in range(3):parts[j].extend(order[j*16:(j+1)*16].tolist())
    I,C,S=[rng.permutation(v).tolist() for v in parts];assert not(set(I)&set(C) or set(I)&set(S) or set(C)&set(S))
    prior=old[I].mean(axis=0)
    initial_tokens=tokens[I,0,:][:,[idx(n) for n in ['G0','G0_MINUS_EU','G0_MINUS_EV']]].sum()
    initial_kernel=kernel[I,0,:][:,[idx(n) for n in ['G0','G0_MINUS_EU','G0_MINUS_EV']]].sum()
    for method in ['stale','protect_affected','always_keep','always_delete_other','random_refresh','periodic_refresh','uncertainty_refresh','affected_refresh','unaffected_refresh']:
      for budget in ([0] if method in ['stale','protect_affected','always_keep','always_delete_other'] else [4,8,16,32]):
       for policyseed in (range(20) if method=='random_refresh' else [0]):
        prng=np.random.default_rng(splitseed+policyseed*31);obs=[[[],[]],[[],[]]];feedback_tokens=0.;feedback_kernel=0.;queries=[]
        for step,i in enumerate(C[:budget]):
            shock=int(shocks[i]);count=[len(v) for v in obs[shock]]
            if method=='random_refresh':edge=int(prng.integers(2))
            elif method=='periodic_refresh':edge=(step//2)%2
            elif method=='uncertainty_refresh':
                uncertainty=[float(np.var(old[I,e],ddof=1))/(count[e]+1) if count[e]<2 else float(np.var(obs[shock][e],ddof=1))/(count[e]+1) for e in [0,1]]
                edge=int(np.argmax(uncertainty))
            elif method=='affected_refresh':edge=shock
            elif method=='unaffected_refresh':edge=1-shock
            else:raise AssertionError(method)
            # Purchase only this paired intervention, before observing its value.
            queries.append((i,edge));obs[shock][edge].append(float(new[i,edge]))
            calls=[idx('G1'),idx('G1_MINUS_EU' if edge==0 else 'G1_MINUS_EV')]
            feedback_tokens+=tokens[i,0,calls].sum();feedback_kernel+=kernel[i,0,calls].sum()
        actions=[];correct=0;service_tokens=0.;service_kernel=0.;wrong_deletions=0;missed_repairs=0
        for i in S:
            shock=int(shocks[i]);estimate=prior.copy()
            for e in [0,1]:
                if obs[shock][e]:estimate[e]=np.mean(obs[shock][e])
            if method=='protect_affected':estimate[shock]=max(0,estimate[shock])
            edge=int(np.argmin(estimate)) if min(estimate)<0 else None
            if method=='always_keep':edge=None
            if method=='always_delete_other':edge=1-shock
            name='G1' if edge is None else 'G1_MINUS_EU' if edge==0 else 'G1_MINUS_EV';j=idx(name);actions.append(name)
            correct+=x[i,1,j];service_tokens+=tokens[i,1,j];service_kernel+=kernel[i,1,j]
            if edge is not None:wrong_deletions+=int(x[i,1,j]<x[i,1,idx('G1')])
        needs_init=method not in ['always_keep','always_delete_other']
        results.append({'split_seed':splitseed,'method':method,'policy_seed':policyseed,'feedback_pairs':budget,'service_questions':32,'service_correct':int(correct),'service_accuracy':correct/32,'wrong_deletions':wrong_deletions,'action_counts':{n:actions.count(n) for n in sorted(set(actions))},'initial_tokens':float(initial_tokens if needs_init else 0),'feedback_tokens':float(feedback_tokens),'service_tokens':float(service_tokens),'total_tokens':float((initial_tokens if needs_init else 0)+feedback_tokens+service_tokens),'total_kernel_seconds':float((initial_kernel if needs_init else 0)+feedback_kernel+service_kernel),'queries_selected_before_feedback':True})
    for n in ['SINGLE_Q','SINGLE_M','INDEPENDENT3_M']:
        results.append({'split_seed':splitseed,'method':n,'policy_seed':0,'feedback_pairs':0,'service_questions':32,'service_correct':int(x[S,1,idx(n)].sum()),'service_accuracy':float(x[S,1,idx(n)].mean()),'initial_tokens':0,'feedback_tokens':0,'service_tokens':float(tokens[S,1,idx(n)].sum()),'total_tokens':float(tokens[S,1,idx(n)].sum()),'total_kernel_seconds':float(kernel[S,1,idx(n)].sum())})
aggregate=[]
for method,budget in sorted(set((r['method'],r['feedback_pairs']) for r in results)):
    rows=[r for r in results if r['method']==method and r['feedback_pairs']==budget]
    aggregate.append({'method':method,'feedback_pairs':budget,'service_accuracy_mean':float(np.mean([r['service_accuracy'] for r in rows])),'full_tokens_mean':float(np.mean([r['total_tokens'] for r in rows])),'split_accuracy_points':[float(np.mean([r['service_accuracy'] for r in rows if r['split_seed']==s])) for s in [261115501,261115502,261115503,261115504,261115505]],'policy_trajectories':len(rows)})
OUT.mkdir(exist_ok=False)
report={'status':'POSTHOC_EXPOSED_DEVELOPMENT_DIAGNOSTIC','source_arrays_sha256':sha(POST/'scoring_001/PRIVATE_ARRAYS.json'),'source_selection_sha256':sha(P/'path_screen_preparation_v1/capacity_selection_v1/SELECTION_MANIFEST.json'),'source_code_sha256':sha(Path(__file__)),'I_C_S_disjoint_within_partition':True,'partitions_overlap_not_independent_replicates':True,'generation_seed_roles':{'I_C':26111311,'S':26111312},'same_complete_cost_cap_enforced':False,'same_pair_count_diagnostic_only':True,'new_GPU_hours':0,'confidence_intervals':False,'bootstrap_runs':0,'automatic_formal_admission':False,'summary':aggregate,'rows':results}
(OUT/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':report['status'],'rows':len(results),'summary':aggregate},indent=2))
