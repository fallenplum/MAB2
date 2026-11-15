"""Descriptive anatomy only: current development data, not online policy evidence."""
import argparse,json,hashlib
from pathlib import Path
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--selection',required=True);p.add_argument('--private',required=True);p.add_argument('--audit',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    audit=read(a.audit);assert audit['status']=='PASS_INDEPENDENT_STDLIB_POINT_AUDIT' and audit['inputs_sha256']['private']==sha(a.private)
    data=read(a.private);rows=read(a.selection)['questions'];x=data['em'];assert len(x)==96
    def mean(v):return sum(v)/len(v) if v else None
    def sign(v):return (v>0)-(v<0)
    old=[[[q[s][0]-q[s][1+e] for s in range(2)] for e in range(2)] for q in x]
    new=[[[q[s][3]-q[s][4+e] for s in range(2)] for e in range(2)] for q in x]
    groups={}
    for role in ['affected','unaffected']:
        indices=[(i,(0 if r['removed_upstream']=='U' else 1) if role=='affected' else (1 if r['removed_upstream']=='U' else 0)) for i,r in enumerate(rows)]
        group={}
        for epoch,arr in [('old',old),('new',new),('change',None)]:
            pairs=[arr[i][e] if arr is not None else [new[i][e][s]-old[i][e][s] for s in range(2)] for i,e in indices]
            union=[p for p in pairs if p[0]!=0 or p[1]!=0]
            group[epoch]={'mean_by_seed':[mean([v[s] for v in pairs]) for s in range(2)],'both_zero':sum(v==[0,0] for v in pairs),
               'nonzero_union':len(union),'both_nonzero_same_sign':sum(v[0]*v[1]>0 for v in union),'opposite_signs':sum(v[0]*v[1]<0 for v in union),
               'one_zero_one_nonzero':sum((v[0]==0)!=(v[1]==0) for v in union),'seed_product_mean':mean([v[0]*v[1] for v in pairs])}
        group['mean_seed_sign_flip_nonzero']=sum(mean(old[i][e])*mean(new[i][e])<0 for i,e in indices)
        group['old_mean_zero_new_mean_nonzero']=sum(mean(old[i][e])==0 and mean(new[i][e])!=0 for i,e in indices)
        groups[role]=group
    policies={}
    for rule in ['strict_negative','nonpositive_tie_deletes']:
        for protect in [False,True]:
            selected=[];service=[];corrections=breakages=0
            for i,row in enumerate(rows):
                affected=0 if row['removed_upstream']=='U' else 1
                options=[e for e in [0,1] if not protect or e!=affected]
                edge=min(options,key=lambda e:(mean(old[i][e]),e));value=mean(old[i][edge])
                delete=value<0 if rule=='strict_negative' else value<=0
                action=4+edge if delete else 3;selected.append(action)
                for s in range(2):
                    v=x[i][s][action];base=x[i][s][3];service.append(v);corrections+=v==1 and base==0;breakages+=v==0 and base==1
            policies[rule+('_protect_affected' if protect else '_stale')]={'accuracy':mean(service),'delta_vs_G1':(corrections-breakages)/192,'corrections':corrections,'breakages':breakages,'deleted_questions':sum(v!=3 for v in selected),'questions':96,'seed_observations':192}
    oracle_same_seed=mean([max(q[s][3:6])-q[s][3] for q in x for s in range(2)])
    result={'status':'EXPLORATORY_DEVELOPMENT_ANATOMY_NOT_G3','groups':groups,'oracle_initial_credit_action_diagnostics':policies,
      'same_seed_current_menu_oracle_gain_over_G1':oracle_same_seed,'oracle_menu_is_evaluator_only':True,
      'limitations':['Old credits use both same-question G0 seeds; these are not disjoint I/C/S blocks.','No feedback acquisition or service cost is charged in the action diagnostic; it is not a budgeted policy result.','Observed zeros are noisy observations, not proof of zero true credit.','Two seeds cannot establish reliable per-question latent effects.','No threshold tuning, F1 rescue or change to the frozen primary screen.'],
      'input_pins':{'selection':sha(a.selection),'private':sha(a.private),'audit':sha(a.audit)},'source_sha256':sha(__file__),'new_GPU_calls':0,'bootstrap_runs':0,'confidence_intervals_computed':False}
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
