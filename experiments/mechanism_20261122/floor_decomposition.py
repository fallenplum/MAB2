"""Separate hardest-label selection from incompatibility of the certificates for different labels."""
import numpy as np
from optimal_certificate import optimal
from blind_scheduler import Audit

def blind_numeric(source,base,cond,labels,row_cost,term_cost,prices,policy,tau):
    a=Audit(source,base,row_cost,prices,policy,tau);cost=0
    while a.decision() is None:
        i,j=a.next()
        if j is None:a.update((i,j),cond[i]);cost+=row_cost[i]
        else:
            a.update((i,j),labels[i,j]);a.observed.pop('INVALID',None);cost+=term_cost[i,j]
    return int(cost)


def decomposition(s,b,m,labels,rcost,tcost,prices,y,initial,tau):
    candidates={name:blind_numeric(s,b,m,labels,rcost,tcost,prices,name,tau) for name in ('joint','row_first','nested','unpriced')}
    top=min(candidates.values());targets=sorted(set(labels.flat)-{'INVALID'})+['__UNSEEN__']
    cm=optimal(s,b,m,labels,rcost,tcost,targets,top,tau=tau)
    individual={v:optimal(s,b,m,labels,rcost,tcost,[v],cm,tau=tau) for v in targets}
    cg=individual.get(y,individual['__UNSEEN__']);hard=max(individual.values())
    assert 0<=cg<=hard<=cm<=top
    if tau==.01:
        assert cm==optimal(s,b,m,labels,rcost,tcost,targets,top,tau=tau,tolerance=-1e-9)
        for v,c in individual.items():assert c==optimal(s,b,m,labels,rcost,tcost,[v],cm,tau=tau,tolerance=-1e-9)
    return dict(gold=initial+cg,hardest=initial+hard,all_numeric=initial+cm,hard_minus_gold=hard-cg,
                simultaneous_extra=cm-hard,gold_reached=bool(np.any(labels==y)),strict_extra=cm>hard,
                true_gold_easier=hard>cg,valid_categories=len(targets)-1,**{k:initial+v for k,v in candidates.items()})
