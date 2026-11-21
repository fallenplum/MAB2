"""Paid counterfactual replays; the scheduler receives no answer key."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from blind_scheduler import Audit
from source_transport import intervals, transport_margin

LABELS=np.arange(8)
TAU=.01


def states(a):
    lo,hi=intervals(a)
    mass=np.array([a.observed.get(y,0.) for y in LABELS])
    lower=mass+lo;upper=mass+hi
    determined=(lower>TAU+1e-12)|(upper<TAU-1e-12)
    if a.seen.all():determined[:]=True
    return lower,upper,determined,mass>TAU


class Item:
    def __init__(self, data, inherit=(), order=None):
        self.source,self.base,self.cond,self.response,self.rc,self.tc=data
        self.audit=Audit(self.source,self.base,self.rc,self.tc,'joint')
        self.rank=Audit(self.source if order is None else order,self.base,self.rc,self.tc,'joint')
        self.history=[]
        self.spent=0
        for action,val in inherit:self.observe(action,val)
        self.refresh()

    def observe(self, action, value):
        for engine in (self.audit,self.rank):engine.update(action,value)
        self.history.append((action,value))

    def refresh(self):
        self.lower,self.upper,self.decided,self.direction=states(self.audit)
        self.next=None if self.decided.all() else self.rank.next()
        if self.next is None:self.price=0
        else:
            i,j=self.next
            self.price=int(self.rc[i] if j is None else self.tc[i,j])

    def take(self):
        i,j=self.next
        self.spent+=self.price
        self.observe(self.next,self.cond[i] if j is None else self.response[i,j])
        self.refresh()

    def complete(self):
        while self.next is not None:self.take()
        return self


def decisions(data):
    s,base,cond,labels,rc,tc=data
    mass=s[:,None]*(cond-base)
    return np.array([mass[labels==y].sum() for y in LABELS])


def credit_rows(models,qs,team,kind):
    a=team;b=(team+1)%3;r=(team+2)%3
    for q in qs:
        k=q['key'];p=[np.array(w[k]['probs']) for w in models]
        p=[v/v.sum(1,keepdims=True) for v in p];t=[np.array(w[k]['tokens'])+1 for w in models]
        rc=t[b][4:8];tc=t[r][8:].reshape(4,4);base=p[b][3];cond=p[b][4:]
        response=np.array(models[r][k]['terminal']).reshape(4,4)
        old_data=(p[a][0],base,cond,response,rc,tc);new_data=(p[a][kind],*old_data[1:])
        old=Item(old_data).complete();cold=Item(new_data).complete()
        warm=Item(new_data,old.history).complete()
        locked=Item(new_data,old.history,order=p[a][0]).complete()
        assert Item(old_data,old.history).next is None
        gains=decisions(new_data);direction=gains>TAU
        assert all(np.array_equal(run.direction,direction) for run in (cold,warm,locked))
        assert np.array_equal(old.direction,decisions(old_data)>TAU)
        tvrow=np.full(4,1-base.min())
        for (i,j),val in old.history:
            if j is None:tvrow[i]=abs(val-base).sum()/2
        drift=abs(p[a][0]-p[a][kind]);radius=tvrow@drift
        lo,up,_,_=states(old.audit)
        margin=bool(((lo-radius>TAU+1e-12)|(up+radius<TAU-1e-12)).all())
        assert not margin or (warm.spent==0 and np.array_equal(direction,old.direction))
        prior=int(t[a][0]+t[b][3]+old.spent);fresh=int(t[a][kind])
        # The denominator is calculated from prompt lengths and purchased
        # history, independently of undisclosed relay or terminal values.
        not_bought=int(rc[~old.audit.known].sum()+tc[~old.audit.seen].sum())
        yield dict(key=k,data=new_data,history=old.history,old=old.direction,truth=direction,credit=gains,
                   old_cost=prior,new_source=fresh,room=not_bought,drift=drift.sum()/2,
                   truth_on=np.array([np.sum(new_data[0][:,None]*cond*(response==y)) for y in LABELS]),
                   truth_off=np.array([np.sum(new_data[0][:,None]*base*(response==y)) for y in LABELS]),
                   warm_sum=fresh+warm.spent,old_order=fresh+locked.spent,cold_sum=fresh+cold.spent,
                   cold_full=fresh+cold.spent+int(t[b][3]),margin=margin,cached=warm.spent==0)


def replay(rows,policy,seed,ratio):
    rng=np.random.default_rng(seed)
    budget=int(ratio*sum(r['room'] for r in rows))
    items=[Item(r['data'],r['history']) for r in rows]
    # Fair ties and round-robin order depend only on question identifiers.
    order=np.argsort([hashlib.sha256(('M56-order'+r['key']).encode()).digest() for r in rows])
    rank=np.argsort(order);visits=np.zeros(len(rows),int);spent=0
    while True:
        feasible=[i for i,it in enumerate(items) if it.next is not None and it.price<=budget-spent]
        if not feasible:break
        if policy=='random':ix=int(rng.choice(feasible))
        elif policy=='periodic':ix=min(feasible,key=lambda i:(visits[i],rank[i]))
        else:
            def priority(i):
                it=items[i]
                if policy=='drift':return rows[i]['drift']
                if policy=='width':return max((it.upper-it.lower)[~it.decided])/it.price
                assert policy=='regret'
                risk=np.where(rows[i]['old'],TAU-it.lower,it.upper-TAU)
                return max(0.,max(risk[~it.decided]))/it.price
            ix=max(feasible,key=lambda i:(priority(i),-rank[i]))
        spent+=items[ix].price;visits[ix]+=1;items[ix].take()
    # Return decisions first. The evaluator alone may inspect reference labels.
    return [np.where(it.decided,it.direction,r['old']) for it,r in zip(items,rows)],dict(
        available=budget,purchased=spent,audits_completed=sum(it.next is None for it in items))


def evaluate(rows,pred,keys,ct):
    actual=np.stack([r['truth'] for r in rows]);before=np.stack([r['old'] for r in rows]);pred=np.stack(pred)
    selected=(np.arange(len(rows)),[int(keys[r['key']]) for r in rows])
    mismatch=pred!=actual;old_error=actual!=before
    regret=mismatch*np.abs(np.stack([r['credit'] for r in rows])-TAU)
    return dict(**ct,all_regret=float(regret.mean()),true_regret=float(regret[selected].mean()),
                wrong_sets=int(mismatch.any(1).sum()),wrong_true=int(mismatch[selected].sum()),
                false_drop=int((mismatch&actual)[selected].sum()),false_keep=int((mismatch&~actual)[selected].sum()),
                fixed_true=int((old_error&~mismatch)[selected].sum()),broken_true=int((~old_error&mismatch)[selected].sum()),
                full_cost=sum(r['old_cost']+r['new_source'] for r in rows)+ct['purchased'])


def score(folder):
    qs=json.loads((folder/'INPUT.json').read_bytes());keys=json.loads((folder/'SEALED.json').read_bytes())
    models=[{r['key']:r for r in map(json.loads,(folder/f'WORKER{i}.jsonl').read_bytes().splitlines())} for i in range(3)]
    assert len(qs)==len(keys)==256 and all(set(w)==set(keys) for w in models)
    cells={}
    for team in range(3):
        for kind in (1,2):
            rows=list(credit_rows(models,qs,team,kind));n=len(rows);y=[int(keys[r['key']]) for r in rows]
            ch=np.array([any(r['old']!=r['truth']) for r in rows]);skip=np.array([r['cached'] for r in rows])
            means={k:float(np.mean([r[k] for r in rows])) for k in ('old_cost','new_source','room','drift','warm_sum','old_order','cold_sum','cold_full','margin','cached')}
            result=dict(means=means,n=n,changed=int(ch.sum()),stable_needs=int((~ch&~skip).sum()),
                safe_skip_gain=means['cached']-means['margin'],repair_saving=1-means['warm_sum']/means['cold_sum'],
                order_saving=1-means['warm_sum']/means['old_order'],
                true_quality_on=float(np.mean([r['truth_on'][t] for t,r in zip(y,rows)])),
                true_quality_off=float(np.mean([r['truth_off'][t] for t,r in zip(y,rows)])))
            if kind==1:
                base=dict(available=0,purchased=0,audits_completed=0)
                result['stale']=evaluate(rows,[r['old'] for r in rows],keys,base)
                pred,ct=replay(rows,'periodic',0,0)
                result['zero_cost_reweight']=evaluate(rows,pred,keys,ct)
                budgeted={}
                for ratio in (.05,.1,.25):
                    for method in ('periodic','random','drift','width','regret'):
                        results=[]
                        for seed in (11,29,47,71,101) if method=='random' else (0,):
                            pred,ct=replay(rows,method,seed,ratio)
                            results.append(evaluate(rows,pred,keys,ct))
                        budgeted[f'{ratio}_{method}']=dict(mean={k:float(np.mean([r[k] for r in results])) for k in results[0]},replicates=results)
                result['budgets']=budgeted
                wins=[]
                for k in ('true_regret','all_regret'):
                    best=min(budgeted['0.1_'+method]['mean'][k] for method in ('periodic','random','drift','width'))
                    ours=budgeted['0.1_regret']['mean'][k]
                    wins.append(best>0 and ours<=.9*best)
                result['active_gate']=all(wins)
            cells[f'{team}_{kind}']=result
    return cells


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);args=p.parse_args()
    print(json.dumps(score(args.folder),indent=2))
