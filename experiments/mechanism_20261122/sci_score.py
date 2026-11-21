"""Independent reachable-influence measurement and complete-cost calibration-query allocation."""
import json
from pathlib import Path
import numpy as np
from reachability import summary
TAU=.01
TEAMS=[(0,1,2),(1,2,0),(2,0,1)]

def panel_stat(rows):
    return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]}


def allocation(panel,method,fraction,seed=0):
    C=np.array([r['C'] for r in panel]);old=np.array([r['old'] for r in panel]);K=np.array([r['K'] for r in panel])
    cost=np.array([r['cost'] for r in panel]);base=sum(r['init'] for r in panel)
    decision=(old>TAU);actual=C>TAU;eligible=np.arange(len(panel));rng=np.random.default_rng(seed)
    budget=int(np.floor(sum(cost)*fraction+1e-6));spent=0
    if method!='stale':
        decision[K<=TAU]=False
        eligible=eligible[K>TAU]
        if method=='random':order=rng.permutation(eligible)
        else:
            if method=='bound':score=np.where(old>TAU,K+TAU,np.maximum(0.,K-TAU))
            elif method=='diameter':score=np.array([r['diameter'] for r in panel])
            elif method=='uncertainty':score=np.array([r['uncertainty'] for r in panel])
            elif method=='margin':score=1/(abs(old-TAU)+1e-9)
            else:raise ValueError(method)
            order=eligible[np.argsort(-score[eligible]/cost[eligible],kind='stable')]
        for i in order:
            if spent+cost[i]<=budget:
                spent+=cost[i];decision[i]=actual[i]
    utility_off=np.array([r['off'] for r in panel])
    return dict(regret=float(np.mean(abs(C-TAU)*(decision!=actual))),accuracy=float(np.mean(utility_off+decision*C)),
                net=float(np.mean(utility_off+decision*(C-TAU))),wrong=int(sum(decision!=actual)),
                false_drop=int(sum(~decision&actual)),false_keep=int(sum(decision&~actual)),variable_tokens=float(spent),
                all_tokens=float(base+spent),available_variable_tokens=budget)


def score(folder):
    data=json.loads((folder/'INPUT.json').read_bytes());gold=json.loads((folder/'SEALED.json').read_bytes())
    ws=[]
    for i in range(3):
        raw=list(map(json.loads,(folder/f'WORKER{i}.jsonl').read_bytes().splitlines()))
        by={r['key']:r for r in raw};assert len(raw)==len(by)==len(gold)==256 and set(by)==set(gold);ws.append(by)
    result={};valid_mechanism=True;valid_budget=True
    for source,mid,reader in TEAMS:
        for condition,idx in [('no_note',0),('true_note',1),('shuffled_note',2)]:
            rows=[];effect=[];panel=[]
            for d in data:
                k=d['key'];y=gold[k]
                p=[np.asarray(w[k]['probs']) for w in ws];assert all(z.shape==(28,4) and abs(z.sum(1)-1).max()<2e-6 for z in p)
                p=[z/z.sum(1,keepdims=True) for z in p];s=p[source][idx]
                m=p[mid][3:].reshape(5,5,4);B=m[1:,0];b0=m[0,0]
                R=p[reader][3:].reshape(5,5,4);read=R[1:,1:,y];w=s[:,None]*(B-b0)
                r=summary(s,p[mid][idx],B,b0,read);e=((B-b0)*read).sum(1)
                signal=(abs(e)>=.1)&(r['natural_K']<=TAU)
                r.update(forced_correct_and_cert=bool(signal[y]),forced_wrong_and_cert=bool(np.any(signal&(np.arange(4)!=y))),
                         harmful_forced_and_cert=bool(np.any((e<=-.1)&(r['natural_K']<=TAU))),
                         source_correct_prob=float(s[y]),null_source_mass=float(s[np.abs(e)>=.1].sum()))
                rows.append(r);effect.append(r['forced_and_certified'])
                prices=[np.asarray(q[k]['input_tokens'])+1 for q in ws]
                bcost=prices[mid][3:].reshape(5,5);rcost=prices[reader][3:].reshape(5,5)
                old=np.sum(w*R[0,1:,y]);new=np.sum(w*read);K=r['natural_K']
                assert abs(new)<=K+1e-10
                panel.append(dict(C=new,old=old,K=K,diameter=r['maximal_single'],
                                  uncertainty=np.mean(R[0,1:,y]*(1-R[0,1:,y])),cost=int(rcost[1:,1:].sum()),
                                  init=int(prices[source][idx]+bcost[:,0].sum()+rcost[0,1:].sum()),
                                  off=float(np.sum(s[:,None]*b0*read))))
            summary_row=panel_stat(rows)
            curve={}
            for frac in (0.,.1,.25,.5,1.):
                curve[str(frac)]={}
                for method in ('stale','random','diameter','uncertainty','margin','bound'):
                    results=[allocation(panel,method,frac,i+20261122) for i in range(16 if method=='random' else 1)]
                    curve[str(frac)][method]=panel_stat(results)
            at=curve['0.25'];avg=at['random']['regret'];candidate=at['bound']['regret']
            win=avg>0 and candidate<=.9*avg and all(candidate<at[name]['regret'] for name in ('diameter','uncertainty','margin'))
            mech=sum(effect)>=6 and np.mean(effect)>=.05
            key=f'{source}_{mid}_{reader}_{condition}';result[key]=dict(conditions=summary_row,
                support_questions=sum(bool(v) for v in effect),mechanism_passed=bool(mech),allocation_passed=bool(win),
                budget_curves=curve,quarter_random_seed_regrets=[allocation(panel,'random',.25,i+20261122)['regret'] for i in range(16)],
                blocks=[int(sum(effect[i:i+64])) for i in range(0,256,64)])
            if condition=='true_note':
                valid_mechanism&=mech;valid_budget&=win
    return dict(cells=result,independent_mechanism_passed=bool(valid_mechanism),budget_mechanism_passed=bool(valid_budget))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);print(json.dumps(score(p.parse_args().folder),indent=2))
