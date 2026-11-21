"""Exact cheapest retrospectively verifiable transcript, allowing terminal queries before relay queries."""
import itertools
import numpy as np
from blind_scheduler import Audit

def bounds_table(source,base,means,labels,rcost,tcost,targets):
    n=len(base);masks=np.asarray(list(itertools.product((False,True),repeat=n)))
    result=[]
    for i in range(n):
        options=[]
        observation=np.array([[label==y for y in targets] for label in labels[i]],float)
        for known in (False,True):
            direction=(means[i][None,:] if known else np.eye(n))-base
            direction*=source[i]
            for mask in masks:
                fixed=direction[:,mask]@observation[mask]
                lo=(fixed+np.minimum(direction[:,~mask],0).sum(1)[:,None]).min(0)
                hi=(fixed+np.maximum(direction[:,~mask],0).sum(1)[:,None]).max(0)
                options.append((int(known)*rcost[i]+tcost[i][mask].sum(),lo,hi))
        result.append(options)
    return result


def optimal(source,base,means,labels,rcost,tcost,targets,feasible,tau=.01):
    assert len(source)==4
    tables=bounds_table(source,base,means,labels,rcost,tcost,targets)
    w=source[:,None]*(means-base)
    yes=np.array([np.sum(w*(labels==y))>tau for y in targets])
    def combine(x,y):
        cost=[];gain=[]
        for ca,la,ua in x:
            for cb,lb,ub in y:
                if ca+cb>feasible:continue
                cost.append(ca+cb)
                gain.append(np.where(yes,la+lb,-ua-ub))
        idx=np.argsort(cost,kind='stable');return np.asarray(cost)[idx],np.asarray(gain)[idx]
    left,lval=combine(*tables[:2]);right,rval=combine(*tables[2:])
    cutoff=np.where(yes,tau,-tau);best=int(feasible)+1
    # Search all affordable left and right prefix pairs. All costs are actual nonnegative token counts.
    for i,cost in enumerate(left):
        if cost>=best:break
        cap=np.searchsorted(right,best-1-cost,side='right')
        if cap==0:continue
        val=rval[:cap]+lval[i]
        # Strict keep and non-strict drop. A numerical guard only removes borderline certificates.
        ok=np.all(np.where(yes,val>cutoff+1e-12,val>=cutoff+1e-12),axis=1)
        if np.any(ok):best=int(cost+right[np.flatnonzero(ok)[0]])
    if best>feasible:raise ArithmeticError('Feasible greedy certificate not recovered; check equality tolerance.')
    return best


def blind_replay(source,base,means,labels,rcost,tcost,prices,policy):
    engine=Audit(source,base,rcost,prices,policy);counts=0
    while engine.decision() is None:
        i,j=engine.next()
        if j is None:engine.update((i,j),means[i]);counts+=rcost[i]
        else:engine.update((i,j),labels[i,j]);counts+=tcost[i,j]
    return int(counts)
