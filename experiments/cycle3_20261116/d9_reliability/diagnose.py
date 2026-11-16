"""Fixed, descriptive two-seed reliability diagnostic, not a deployable oracle."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

def corr(a,b):
    a=np.asarray(a).ravel();b=np.asarray(b).ravel()
    return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None

def diagnose(a,selection):
    em=a['em'];tokens=a['tokens'];assert em.shape==tokens.shape==(96,2,11)
    c0=em[:,:,[0]]-em[:,:,1:3];c1=em[:,:,[3]]-em[:,:,4:6];shift=c1-c0
    def rel(z):
        paired=z[:,0,:]*z[:,1,:]
        nonzero=(z!=0);either=nonzero.any(axis=1);both=nonzero.all(axis=1)
        return dict(realizations=int(z.size),nonzero=int(nonzero.sum()),question_edge_pairs=int(either.size),either_seed_nonzero=int(either.sum()),both_seeds_nonzero=int(both.sum()),same_nonzero_sign=int((paired>0).sum()),opposite_nonzero_sign=int((paired<0).sum()),seed_correlation=corr(z[:,0,:],z[:,1,:]))
    actions=[];costs=[]
    for i,item in enumerate(selection['questions']):
        take=[3,4,5] if item['removed_upstream']=='U' else [3,5,4]
        actions.append(em[i][:,take]);costs.append(tokens[i][:,take])
    q=np.array(actions);t=np.array(costs);u=q-5e-6*t
    same=np.argmax(u,axis=2);cross=np.flip(same,axis=1)
    # Pooled best fixed is a retrospective reference, never reported as an honest CV policy.
    fixed=int(np.argmax(u.mean(axis=(0,1))))
    ids=np.arange(96)[:,None];seeds=np.arange(2)[None,:]
    values={}
    for name,chosen in [('same_seed_oracle',same),('other_seed_oracle',cross),('pooled_best_fixed_retrospective',np.full((96,2),fixed))]:
        us=u[ids,seeds,chosen];acc=q[ids,seeds,chosen]
        values[name]=dict(EM=float(acc.mean()),service_utility=float(us.mean()),utility_by_seed=us.mean(axis=0).tolist(),EM_by_seed=acc.mean(axis=0).tolist(),action_counts=np.bincount(chosen.ravel(),minlength=3).tolist())
    base=u[:,:,fixed]
    values['other_seed_oracle']['gain_over_retrospective_fixed']=float((u[ids,seeds,cross]-base).mean())
    values['same_seed_oracle']['gain_over_retrospective_fixed']=float((u[ids,seeds,same]-base).mean())
    return dict(status='COMPLETE_EXPOSED_RELIABILITY_DIAGNOSTIC',question_groups=96,seeds=2,old_credit=rel(c0),new_credit=rel(c1),credit_shift=rel(shift),oracle_references=values,
        limitations=['Descriptive exposed-pilot results; no formal gate override or independent confirmation','Two seeds give a limited reliability view, not a variance-component estimate','Oracles consume unavailable labels; no deployment or full-cost effectiveness claim','Pooled best fixed is retrospective and optimistically selected'],new_GPU_hours=0,confidence_intervals=False,bootstrap_runs=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arrays',required=True);p.add_argument('--selection',required=True);p.add_argument('--scoring-audit',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    assert json.loads(Path(a.scoring_audit).read_bytes())['status']=='PASS_INDEPENDENT_RAW_SCORING_AND_GATES'
    result=diagnose(np.load(a.arrays),json.loads(Path(a.selection).read_bytes()))
    result['pins']={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in dict(arrays=a.arrays,selection=a.selection,audit=a.scoring_audit,source=__file__).items()}
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);(out/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'])))
