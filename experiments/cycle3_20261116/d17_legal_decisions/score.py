"""Legal post-shock oracle headroom; not a deployable policy."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def analyze(em,tokens,shock):
    assert em.shape==tokens.shape==(96,2,4,8)
    utility=em-5e-6*tokens; halves=[utility[:,:,:2].mean((1,2)),utility[:,:,2:].mean((1,2))]
    base=7^(1<<shock); n=len(base); idx=np.arange(n)
    legal=np.array([[int(b)]+[int(b)^(1<<e) for e in range(3) if b&(1<<e)] for b in base])
    assert legal.shape==(96,3) and np.all((legal&base[:,None])==legal)
    policies=np.array([[int(b)]+[int(b)&~(1<<e) for e in range(3)] for b in base])
    transfers=[]
    for tr,te in [(0,1),(1,0)]:
        train,test=halves[tr],halves[te]
        fresh=legal[idx,np.argmax(np.take_along_axis(train,legal,axis=1),axis=1)]
        stale=[]; sub=[]
        for i,b in enumerate(base):
            edges=[e for e in range(3) if b&(1<<e)]
            old_values=[train[i,7]]+[train[i,7^(1<<e)] for e in edges]
            stale.append(legal[i,int(np.argmax(old_values))])
            masks=[m for m in range(8) if (m&int(b))==m]
            sub.append(masks[int(np.argmax(train[i,masks]))])
        stale=np.array(stale);sub=np.array(sub)
        policy_values=np.take_along_axis(test,policies,axis=1);best=int(np.argmax(policy_values.mean(0)))
        fixed=policies[:,best];oof=np.empty(n,dtype=int)
        for parity in [0,1]:
            fit=idx%2!=parity;ev=~fit
            chosen=int(np.argmax(np.take_along_axis(train,policies,axis=1)[fit].mean(0)))
            oof[ev]=policies[ev,chosen]
        choices={'fresh':fresh,'stale':stale,'best_fixed':fixed,'cross_question_fixed':oof,'all_submask_oracle':sub,'keep':base}
        eval_seeds=[0,1] if te==0 else [2,3]
        stats={}
        for name,c in choices.items():
            outcomes=np.stack([em[idx,s,t,c] for s in range(2) for t in eval_seeds],axis=1)
            baseline=np.stack([em[idx,s,t,fixed] for s in range(2) for t in eval_seeds],axis=1)
            stats[name]=dict(utility=float(test[idx,c].mean()),EM=float(outcomes.mean()),mask_counts=np.bincount(c,minlength=8).tolist(),corrections_vs_fixed=int(((outcomes==1)&(baseline==0)).sum()),destructions_vs_fixed=int(((outcomes==0)&(baseline==1)).sum()),evaluation_cells=int(outcomes.size))
        transfers.append(dict(selection_half=tr,evaluation_half=te,best_fixed_policy=best,fixed_policy_utilities=policy_values.mean(0).tolist(),methods=stats,gain_vs_stale=stats['fresh']['utility']-stats['stale']['utility'],gain_vs_best_fixed=stats['fresh']['utility']-stats['best_fixed']['utility']))
    means={key:float(np.mean([t[key] for t in transfers])) for key in ['gain_vs_stale','gain_vs_best_fixed']}
    passed=all(means[key]>=.005 and min(t[key] for t in transfers)>0 for key in means)
    return dict(transfers=transfers,mean_gains=means,necessary_headroom_passed=bool(passed),legal_action_count=3)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--private-root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();p=Path(a.private_root);run=p/'d16_crossed_collection_v1'
    audit=json.loads((run/'SCORING_AUDIT.json').read_bytes());assert audit['status'].startswith('PASS')
    source=run/'scoring_v1/PRIVATE_ARRAYS.npz';arr=np.load(source)
    tasks={task:analyze(arr[task+'_em'],arr[task+'_tokens'],arr[task+'_shock']) for task in ['hotpot','2wiki']}
    result=dict(status='COMPLETE_EXPOSED_LEGAL_DECISION_DIAGNOSTIC',tasks=tasks,necessary_screen_both_tasks=all(v['necessary_headroom_passed'] for v in tasks.values()),new_model_calls=0,new_GPU_hours=0,independent_confirmation=False,oracle_feedback_cost_excluded=True,confidence_intervals=False,bootstrap_runs=0,input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out=Path(a.output);out.mkdir();(out/'SUMMARY.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
