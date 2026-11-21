"""Revalidate a paid transcript after replacing the source of a relay."""
import argparse
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np

from blind_scheduler import Audit

UNKNOWN = '__UNSEEN__'
INVALID = 'INVALID'


def intervals(audit):
    missing = audit.source[~audit.known].sum() * (1-audit.base.min())
    pending = np.where(audit.known[:,None] & ~audit.seen, audit.w, 0)
    return (min(0.,np.minimum(pending,0).sum()-missing),
            max(0.,np.maximum(pending,0).sum()+missing))


def transport_margin(audit, radius):
    lo,hi = intervals(audit)
    return all(v + hi + radius < audit.tau - 1e-12 or
               v + lo - radius > audit.tau + 1e-12 for v in [0.,*audit.observed.values()])


def insert(audit, action, value):
    audit.update(action,value)
    audit.observed.pop(INVALID,None)


def run(s, base, cond, labels, row_cost, terminal_cost, prices, old=(), order=None):
    audit=Audit(s,base,row_cost,prices,'joint')
    rank=Audit(s if order is None else order,base,row_cost,prices,'joint')
    for action,result in old:
        insert(audit,action,result)
        insert(rank,action,result)
    history=list(old)
    cost=0
    while audit.decision() is None:
        i,j=rank.next()
        value=cond[i] if j is None else labels[i,j]
        cost+=row_cost[i] if j is None else terminal_cost[i,j]
        history.append(((i,j),value))
        insert(audit,(i,j),value)
        insert(rank,(i,j),value)
    return audit,history,int(cost)


def truth(s,base,cond,labels):
    weights=s[:,None]*(cond-base)
    return {y for y in set(labels.flat)-{INVALID} if weights[labels==y].sum()>.01}


def compare(s,new,base,cond,labels,rc,tc,prices,gold,init_old,price_new,price_base):
    old,history,old_cost=run(s,base,cond,labels,rc,tc,prices)
    changed=truth(new,base,cond,labels)
    previous=old.decision()
    assert previous==truth(s,base,cond,labels)
    same,_,self_cost=run(s,base,cond,labels,rc,tc,prices,history)
    assert self_cost==0 and same.decision()==previous
    cold,_,cold_cost=run(new,base,cond,labels,rc,tc,prices)
    warm,_,warm_cost=run(new,base,cond,labels,rc,tc,prices,history)
    locked,_,locked_cost=run(new,base,cond,labels,rc,tc,prices,history,order=s)
    assert changed==cold.decision()==warm.decision()==locked.decision()
    drift=abs(new-s)
    row_tv=np.full(len(s),1-base.min())
    for (i,j),value in history:
        if j is None:row_tv[i]=abs(value-base).sum()/2
    scalar=transport_margin(old,drift.sum())
    bounded=transport_margin(old,drift@row_tv)
    if scalar:assert bounded
    if bounded:assert not warm_cost and previous==changed
    baseline=price_new+cold_cost
    return dict(changed=previous!=changed,true_changed=(gold in previous)!=(gold in changed),
        false_drop=gold in changed and gold not in previous,false_keep=gold in previous and gold not in changed,
        stable_but_needs_evidence=previous==changed and warm_cost>0,
        global_margin_skip=scalar,row_tv_skip=bounded,reweighted_skip=warm_cost==0,
        initial=init_old+old_cost,restart_full=baseline+price_base,restart_with_base=baseline,
        repair=price_new+warm_cost,old_order=price_new+locked_cost,increments=warm_cost,
        two_audit_cost=init_old+old_cost+price_new+warm_cost,
        source_tv=drift.sum()/2,all_set_agrees=True)


def cases(folder,phase):
    from passage_score import normalized
    questions=json.loads((folder/'INPUT.json').read_bytes())
    keys=json.loads((folder/'SEALED.json').read_bytes())
    models=[{r['key']:r for r in map(json.loads,(folder/f'WORKER{i}.jsonl').read_bytes().splitlines())} for i in range(3)]
    assert all(set(m)==set(keys) for m in models)
    for task in questions:
        key=task['key']; rows=[w[key] for w in models]
        for team in range(3):
            if phase=='m50':
                a=b=r=team;replacement=(a+1)%3
                pre=[np.array([x['arms']['full']['prefix'][str(j)] for j in range(6)]) for x in rows]
                tok=[x['lengths'][0] for x in rows]
                cells=[rows[r]['arms']['full']['terminal'][str(j)] for j in range(16)]
                labels=[v['label'] for v in cells]
                gold=str(Decimal(keys[key]).normalize());cap=512
            else:
                a=team;b=(team+1)%3;r=(team+2)%3;replacement=b
                pre=[np.array(x['probs']) for x in rows];tok=[x['prefix_tokens'] for x in rows]
                cells=rows[r]['terminal'];labels=[normalized(v['text']) for v in cells]
                labels=[v if v not in ('','noevidence') else INVALID for v in labels]
                gold=keys[key];cap=32
            pre=[v/v.sum(1,keepdims=True) for v in pre]
            common=[pre[a][0],pre[replacement][0],pre[b][1],pre[b][2:],np.reshape(labels,(4,4)),np.array(tok[b][2:])+1,
                    np.reshape([v['prefill']+v['decode'] for v in cells],(4,4)),
                    np.reshape([v['prefill']+cap for v in cells],(4,4)),gold,
                    tok[a][0]+tok[b][1]+2,tok[replacement][0]+1,tok[b][1]+1]
            yield team,task.get('task'),common


def score(folder,phase):
    groups=defaultdict(list)
    for team,task,common in cases(folder,phase):
        row=compare(*common)
        groups[f'team{team}'].append(row)
        if phase=='m50':groups[f'team{team}_{task}'].append(row)
    out={}
    for group,rows in groups.items():
        agg={k:float(np.mean([x[k] for x in rows])) for k in rows[0]}
        agg.update(n=len(rows),changes=sum(int(x['changed']) for x in rows),
                   safe_increment_saving=1-agg['repair']/agg['restart_with_base'],
                   new_order_saving=1-agg['repair']/agg['old_order'],
                   extra_skips=agg['reweighted_skip']-agg['row_tv_skip'])
        out[group]=agg
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['m50','m54'])
    p.add_argument('folder',type=Path);a=p.parse_args()
    print(json.dumps(score(a.folder,a.phase),indent=2))
