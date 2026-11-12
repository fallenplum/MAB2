"""Finite binary bank certificate, structural costs, isolated-feedback policies."""
from __future__ import annotations
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

def bounds(observed):
    x=np.asarray(observed)
    if x.ndim!=2 or not np.isin(x,[-1,0,1]).all(): raise ValueError('invalid observed bank')
    lo=np.maximum(x,0).sum(axis=0)
    return lo,lo+(x<0).sum(axis=0)

def certificate(observed):
    lo,hi=bounds(observed)
    for a in range(len(lo)):
        if all(a==b or lo[a]>=hi[b]+int(b<a) for b in range(len(lo))): return a
    return None

def structural_cost(mask,dirty):
    m=np.asarray(mask,dtype=bool);d=np.asarray(dirty,dtype=int)
    if len(d)!=m.shape[1] or d[0]!=0 or not np.isin(d,[0,1]).all(): raise ValueError('bad dirty flags')
    singleton_dirty=((m.sum(axis=1)==1)&((m*d[None,:]).sum(axis=1)==1)).sum()
    return int(3*np.any(m,axis=1).sum()+(m*(1+d)[None,:]).sum()-singleton_dirty)

def incremental_cost(mask,cells,dirty):
    new=mask.copy()
    for q,a in cells: new[q,a]=True
    return structural_cost(new,dirty)-structural_cost(mask,dirty)

def select_cells(observed,old,dirty,method,rng):
    mask=observed>=0; n,m=observed.shape
    unknown=[(q,a) for q in range(n) for a in range(m) if not mask[q,a]]
    if not unknown: raise ValueError('Full table must be certifiable')
    if method=='random': return [unknown[int(rng.integers(len(unknown)))]]
    if method=='cheapest': return [min(unknown,key=lambda x:(incremental_cost(mask,[x],dirty),x))]
    if method not in ('gap_unit','gap_bundle'): raise ValueError(method)
    estimate=np.where(mask,observed,old).sum(axis=0)
    leader=int(np.argmax(estimate));_,hi=bounds(observed)
    challenger=min((a for a in range(m) if a!=leader),key=lambda a:(-hi[a],a))
    candidates=[x for x in unknown if x[1] in (leader,challenger)]
    # If both selected arms are complete and indistinguishable, another upper contender
    # is needed. This can occur when the point leader has a lower-priority tie.
    if not candidates:
        alternatives=[a for a in range(m) if np.any(~mask[:,a])]
        challenger=min(alternatives,key=lambda a:(-hi[a],a))
        candidates=[x for x in unknown if x[1] in (leader,challenger)]
    if method=='gap_unit': return [min(candidates,key=lambda x:(incremental_cost(mask,[x],dirty),x))]
    groups=[[x for x in candidates if x[0]==q] for q in range(n)]
    groups=[g for g in groups if g]
    return min(groups,key=lambda g:(-len(g)/incremental_cost(mask,g,dirty),g[0][0]))

def run_policy(old,dirty,method,query,seed=0):
    """Only query() sees current truth; policy state has purchased values only."""
    old=np.asarray(old,dtype=int)
    if old.ndim!=2 or old.shape[1]<2 or not np.isin(old,[0,1]).all(): raise ValueError('bad old bank')
    observed=np.full(old.shape,-1,dtype=int);rng=np.random.default_rng(seed);trace=[]
    while certificate(observed) is None:
        cells=select_cells(observed,old,dirty,method,rng)
        inc=incremental_cost(observed>=0,cells,dirty)
        values=[]
        for q,a in cells:
            if observed[q,a]>=0: raise ValueError('repurchase')
            value=query(q,a)
            if value not in (0,1): raise ValueError('nonbinary feedback')
            observed[q,a]=value;values.append(int(value))
        trace.append({'cells':[list(x) for x in cells],'values':values,'incremental_cost':inc})
    total=structural_cost(observed>=0,dirty)
    assert total==sum(t['incremental_cost'] for t in trace)
    return {'action':certificate(observed),'cost':total,'revealed':int((observed>=0).sum()),
            'trace':trace,'final_observed':observed.tolist()}

def oracle(current,dirty):
    y=np.asarray(current,dtype=int); n,m=y.shape; size=n*m+2*n
    if not np.isin(y,[0,1]).all(): raise ValueError('nonbinary oracle')
    winner=int(np.argmax(y.sum(axis=0)))
    c=np.r_[np.tile(1+np.asarray(dirty),n),np.full(n,3),np.full(n,-1)]
    rows=[];lower=[];upper=[]
    for q in range(n):
        for a in range(m):
            row=np.zeros(size);row[q*m+a]=1;row[n*m+q]=-1
            rows.append(row);lower.append(-np.inf);upper.append(0)
        # Discount is possible only for a singleton dirty target, with no base request.
        row=np.zeros(size);row[n*m+n+q]=1
        for a in range(m):row[q*m+a]=-dirty[a]
        rows.append(row);lower.append(-np.inf);upper.append(0)
        row=np.zeros(size);row[q*m:(q+1)*m]=1;row[n*m+n+q]=m-1
        rows.append(row);lower.append(-np.inf);upper.append(m)
    for a in range(m):
        if a==winner:continue
        row=np.zeros(size)
        for q in range(n):
            row[q*m+winner]=y[q,winner];row[q*m+a]=1-y[q,a]
        rows.append(row);lower.append(n+int(a<winner));upper.append(np.inf)
    res=milp(c,integrality=np.ones(size),bounds=Bounds(np.zeros(size),np.ones(size)),
        constraints=LinearConstraint(np.array(rows),np.array(lower),np.array(upper)),
        options={'mip_rel_gap':0.0,'time_limit':60.0})
    if not res.success or res.x is None or float(res.mip_gap)!=0:
        return {'status':'INCONCLUSIVE_SOLVER','message':str(res.message)}
    z=np.rint(res.x).astype(int)
    if not np.allclose(z,res.x,rtol=0,atol=1e-7):raise ValueError('nonintegral solution')
    mask=z[:n*m].reshape(n,m).astype(bool)
    obs=np.where(mask,y,-1);cost=structural_cost(mask,dirty)
    assert certificate(obs)==winner
    assert abs(float(res.fun)-cost)<1e-7
    return {'status':'OPTIMAL_CERTIFICATE_VERIFIED','action':winner,'cost':cost,
            'revealed':int(mask.sum()),'mask':mask.astype(int).tolist(),'mip_gap':float(res.mip_gap)}
