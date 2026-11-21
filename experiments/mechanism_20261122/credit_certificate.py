"""Finite-support certificates for a relay edge after a change of its downstream readout."""
import hashlib
import json
from pathlib import Path
import numpy as np

POLICIES=('width_cost','directed','uncertainty','uniform')
THRESHOLD=.01


def query_order(w,prior,observe,cost,method,rng):
    remaining=np.ones(len(w),dtype=bool)
    seen_contribution=0.
    n=0;tokens=0
    while True:
        low=seen_contribution+np.minimum(w[remaining],0).sum()
        high=seen_contribution+np.maximum(w[remaining],0).sum()
        if low>THRESHOLD+1e-12 or high<THRESHOLD-1e-12 or not remaining.any():
            decision=(low>THRESHOLD+1e-12) if remaining.any() else seen_contribution>THRESHOLD
            return n,tokens,decision
        indices=np.flatnonzero(remaining)
        if method=='uniform':j=rng.choice(indices)
        else:
            score=abs(w[indices])/cost[indices]
            if method=='uncertainty':score=(prior[indices]*(1-prior[indices]))/cost[indices]
            if method=='directed':
                predict=seen_contribution+w[indices]@prior[indices]
                gains=np.where(w[indices]>0,prior[indices],1-prior[indices])
                score=score*(gains if predict>THRESHOLD else 1-gains)
            j=indices[np.argmax(score)]
        seen_contribution+=w[j]*observe(j);n+=1;tokens+=cost[j];remaining[j]=False


def item(p,c,y,a,reader):
    b=1-a;s=p[a][0];mid=p[b][1:].reshape(5,5,4);end=p[reader][1:].reshape(5,5,4)
    w=(s[:,None]*(mid[1:,0,:]-mid[0,0,:])).ravel()
    actual=end[1:,1:,y].ravel()
    old=np.tile(end[0,1:,y],4)
    # Each query returns all answer probabilities. Credit auditing here is for one known label.
    mcost=c[b][1:].reshape(5,5);rcost=c[reader][1:].reshape(5,5)
    fixed=c[a][0]+mcost[0,0]+mcost[1:,0].sum()+rcost[0,1:].sum()
    costs=rcost[1:,1:].ravel()
    assert abs(w.sum())<1e-10 and len(actual)==16
    target=w@actual;stale=w@old
    values={'fixed_tokens':float(fixed),'exhaustive_tokens':float(fixed+sum(costs)),
            'exhaustive_new_calls':16.,'stale_error':float((stale>THRESHOLD)!=(target>THRESHOLD)),
            'stale_regret':abs(target-THRESHOLD)*((stale>THRESHOLD)!=(target>THRESHOLD)),
            'always_drop_regret':max(0.,target-THRESHOLD),'always_keep_regret':max(0.,THRESHOLD-target),
            'oracle_keep':float(target>THRESHOLD),'zero_query_certified':float(np.maximum(w,0).sum()<THRESHOLD-1e-12)}
    for method in POLICIES:
        repeats=16 if method=='uniform' else 1
        records=[query_order(w,old,actual.__getitem__,costs,method,np.random.default_rng(20261122+i)) for i in range(repeats)]
        assert all(bool(r[2])==bool(target>THRESHOLD) for r in records)
        calls,spent=np.mean([r[:2] for r in records],axis=0)
        values[method+'_queries']=calls;values[method+'_total_tokens']=fixed+spent
    return values


def evaluate(folder):
    data=json.loads((folder/'INPUT.json').read_bytes());gold=json.loads((folder/'SEALED.json').read_bytes())
    ws=[{r['key']:r for r in map(json.loads,(folder/f'WORKER{i}.jsonl').read_bytes().splitlines())} for i in range(2)]
    assert all(set(w)==set(gold) for w in ws) and len(gold)==len(data)
    results={}
    for task in sorted({r['task'] for r in data}):
        for reader in (0,1):
            rows=[]
            for r in data:
                if r['task']!=task:continue
                k=r['key'];p=[np.asarray(w[k]['probs'],float) for w in ws]
                assert all(x.shape==(26,4) and np.max(abs(x.sum(1)-1))<2e-6 for x in p)
                p=[x/x.sum(1,keepdims=True) for x in p]
                c=[np.asarray(w[k]['input_tokens'])+1 for w in ws]
                rows.extend(item(p,c,gold[k],a,reader) for a in (0,1))
            stats={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
            stats['questions']=len(rows)//2
            stats['reduction_vs_exhaustive']=1-stats['directed_total_tokens']/stats['exhaustive_tokens']
            stats['reduction_vs_width']=1-stats['directed_total_tokens']/stats['width_cost_total_tokens']
            stats['passed']=stats['reduction_vs_exhaustive']>=.20 and stats['reduction_vs_width']>=.02
            results[f'{task}_{reader}']=stats
    return results


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    print(json.dumps(evaluate(parser.parse_args().folder),indent=2))
