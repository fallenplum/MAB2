"""Exploratory fixed decision diagnosis. Requires full scored and audited Formal7."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np

def diagnose(em,f1,tokens,shocks,names):
    assert em.shape==f1.shape==tokens.shape and em.shape[1:]==(2,11)
    n=len(em);assert len(shocks)==n and set(shocks)<=set('UV')
    ix={x:i for i,x in enumerate(names)};utility=em-5e-6*tokens
    graph={}
    for time in [0,1]:
        indices=np.array([[ix[f'G{time}'],ix[f'G{time}_MINUS_E'+s],ix[f'G{time}_MINUS_E'+('V' if s=='U' else 'U')]] for s in shocks])
        graph[time]={key:np.take_along_axis(value,indices[:,None,:],axis=2) for key,value in [('em',em),('f1',f1),('tokens',tokens),('u',utility)]}
    old=graph[0];new=graph[1];oldbest=old['u'].argmax(axis=2);best=new['u'].argmax(axis=2)
    choices={'KEEP':np.zeros((n,2),int),'DELETE_AFFECTED':np.ones((n,2),int),'DELETE_OTHER':np.full((n,2),2),'STALE_OLD_GRAPH_ORACLE_TRANSFER':oldbest,'CURRENT_PERFECT_INFORMATION_ORACLE':best,'ALTERNATE_SEED_ORACLE_TRANSFER':best[:,::-1]}
    rows={}
    for name,action in choices.items():
        v={k:np.take_along_axis(x,action[:,:,None],axis=2)[:,:,0] for k,x in new.items()};base=new['em'][:,:,0]
        rows[name]=dict(EM=float(v['em'].mean()),F1=float(v['f1'].mean()),service_tokens=float(v['tokens'].mean()),service_utility=float(v['u'].mean()),EM_by_seed=v['em'].mean(axis=0).tolist(),utility_by_seed=v['u'].mean(axis=0).tolist(),action_counts=np.bincount(action.ravel(),minlength=3).tolist(),corrections_vs_keep=int(((v['em']==1)&(base==0)).sum()),destructions_vs_keep=int(((v['em']==0)&(base==1)).sum()),available_without_feedback=name in ['KEEP','DELETE_AFFECTED','DELETE_OTHER'])
    direction={}
    for edge,k in [('affected',1),('other',2)]:
        oldgain=old['u'][:,:,k]-old['u'][:,:,0];gain=new['u'][:,:,k]-new['u'][:,:,0];wrong=(oldgain>0)!=(gain>0)
        c0=old['em'][:,:,0]-old['em'][:,:,k];c1=new['em'][:,:,0]-new['em'][:,:,k]
        direction[edge]=dict(old_credit={key:int(fn(c0).sum()) for key,fn in [('positive',lambda x:x>0),('zero',lambda x:x==0),('negative',lambda x:x<0)]},current_credit={key:int(fn(c1).sum()) for key,fn in [('positive',lambda x:x>0),('zero',lambda x:x==0),('negative',lambda x:x<0)]},stale_net_direction_errors=int(wrong.sum()),errors_by_seed=wrong.sum(axis=0).tolist(),wrong_delete=int(((oldgain>0)&(gain<=0)).sum()),wrong_keep=int(((oldgain<=0)&(gain>0)).sum()),mean_binary_decision_regret=float((np.abs(gain)*wrong).mean()),regret_by_seed=(np.abs(gain)*wrong).mean(axis=0).tolist())
    fixed=['KEEP','DELETE_AFFECTED','DELETE_OTHER'];oracle=rows['CURRENT_PERFECT_INFORMATION_ORACLE']['service_utility'];single=float(utility[:,:,ix['SINGLE_Q']].mean())
    result=dict(status='EXPLORATORY_DECISION_HEADROOM_NOT_CONFIRMATION',questions=n,question_seed_count=n*2,actions=['KEEP','DELETE_AFFECTED','DELETE_OTHER'],strategies=rows,net_edge_direction=direction,old_new_optimal_action_disagreements=int((oldbest!=best).sum()),oracle_headroom_vs_fixed={k:oracle-rows[k]['service_utility'] for k in fixed},oracle_headroom_vs_single_Q=oracle-single,fixed_pairwise_utility_points={a+' minus '+b:rows[a]['service_utility']-rows[b]['service_utility'] for a in fixed for b in fixed if a!=b},retrospective_best_constant_is_unavailable=True,retrospective_best_constant=max(fixed,key=lambda k:rows[k]['service_utility']),formal_gates_overridden=False,confidence_intervals=False,bootstrap_runs=0,new_GPU_calls=0,limitation='Old-graph and alternate-seed gold-dependent choices are unavailable diagnostic oracles, not deployed policies. Service utility excludes their prerequisite observations; those are reported separately. Positive headroom is not acquisition efficacy.')
    return result,choices

def main():
    a=argparse.ArgumentParser();a.add_argument('--formal',required=True);a.add_argument('--output',required=True);x=a.parse_args();f=Path(x.formal);out=Path(x.output);assert not out.exists()
    read=lambda p:json.loads(p.read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    audit=read(f/'INDEPENDENT_SCORING_AUDIT.json');assert audit['status']=='PASS_INDEPENDENT_RAW_SCORING_AND_GATES'
    summary=read(f/'scoring_v1/SUMMARY.json');arrays=np.load(f/'scoring_v1/PRIVATE_ARRAYS.npz');arc=f/'monitor/archive';manifest=read(arc/'ARCHIVE_MANIFEST.json')
    for name,h in manifest.items():assert (arc/name).resolve().is_relative_to(arc.resolve()) and sha(arc/name)==h
    units=sorted(arc.glob('worker_*/unit_*'),key=lambda p:read(p/'INPUT.json')['index']);assert len(units)==96
    shocks=[read(u/'INPUT.json')['shock'] for u in units];names=list(summary['conditions']);result,choices=diagnose(arrays['em'],arrays['f1'],arrays['tokens'],shocks,names)
    seeds=summary['seeds'];obs={name:dict(calls=0,input_tokens=0,output_tokens=0,gold_questions_per_seed=96,score_observations=0) for name in ['STALE_OLD_GRAPH_ORACLE_TRANSFER','ALTERNATE_SEED_ORACLE_TRANSFER']}
    private=[]
    for i,unit in enumerate(units):
        mat=read(unit/'MATRIX.json');rows={(c['execution_seed'],c['condition']):c for c in mat['conditions']}
        for j,seed in enumerate(seeds):
            for name,t,source_seed in [('STALE_OLD_GRAPH_ORACLE_TRANSFER',0,seed),('ALTERNATE_SEED_ORACLE_TRANSFER',1,seeds[1-j])]:
                unique={}
                for cond in [f'G{t}',f'G{t}_MINUS_EU',f'G{t}_MINUS_EV']:
                    for node in rows[source_seed,cond]['nodes'].values():
                        fp=node['request_fingerprint'];g=node['generation']
                        if fp in unique:assert unique[fp]==g
                        unique[fp]=g
                costs=dict(calls=len(unique),input_tokens=sum(v['input_tokens'] for v in unique.values()),output_tokens=sum(v['output_tokens'] for v in unique.values()))
                for key,value in costs.items():obs[name][key]+=value
                obs[name]['score_observations']+=3
                private.append(dict(index=i,seed=seed,diagnostic=name,source_seed=source_seed,source_graph=t,chosen_action=int(choices[name][i,j]),paid_observation_cost=costs,oracle_not_deployable=True))
    result.update(prerequisite_observation_costs=obs,formal7_admission=summary['admission'],formal7_gates_unchanged=summary['gates'],archive_manifest_sha256=sha(arc/'ARCHIVE_MANIFEST.json'),source_sha256=sha(__file__))
    out.mkdir(parents=True);(out/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n');(out/'PRIVATE_PROVENANCE.json').write_text(json.dumps(private,indent=2)+'\n');print(json.dumps(dict(status=result['status'],new_GPU_calls=0,formal7_admission_unchanged=summary['admission'])))

if __name__=='__main__':main()
