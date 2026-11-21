"""Matched stopping prefixes for one hidden target and all targets using identical query ordering."""
from pathlib import Path
import json
import numpy as np
from blind_scheduler import Audit
from floor_panels import cases

def stopping(s,base,means,response,rp,tc,prices,y,initial,tau=.01):
    n=len(s);d=response.shape[-1]
    engine=Audit(s,base,rp,prices,'joint',tau)
    mass=np.zeros(d);stops=np.full(d,np.nan);count=initial
    while True:
        missing=s[~engine.known].sum()*(1-base.min())
        pending=np.where(engine.known[:,None]&~engine.seen,engine.w,0.)
        lo=mass+np.minimum(pending,0).sum()-missing;hi=mass+np.maximum(pending,0).sum()+missing
        ready=(hi<tau-1e-12)|(lo>tau+1e-12)|engine.seen.all()
        stops[np.isnan(stops)&ready]=count
        if np.all(ready):break
        i,j=engine.next()
        if j is None:
            engine.update((i,j),means[i]);count+=rp[i]
        else:
            mass+=engine.w[i,j]*response[i,j];engine.update((i,j),'observed');count+=tc[i,j]
    assert max(stops)==count and np.isfinite(stops).all()
    return dict(gold_tokens=float(stops[y]),all_tokens=float(count),extra_tokens=float(count-stops[y]),
                extra_required=bool(count>stops[y]),gold_last=bool(stops[y]==count),first_tokens=float(min(stops)))

def evaluate(folder,phase,train=None,grader=None):
    aggregated={}
    if phase in ('m46','m50'):
        for (task,reader,key),params in cases(folder,phase,train,grader):
            s,b,m,labels,rp,tc,price,y,init=params
            cats=sorted(set(labels.flat)-{'INVALID'});cats+=['__UNSEEN__']
            resp=(labels[:,:,None]==np.array(cats)[None,None,:]).astype(float)
            target=cats.index(y) if y in cats else len(cats)-1
            aggregated.setdefault(f'{task}_{reader}',[]).append(stopping(s,b,m,resp,rp,tc,price,target,init))
    else:
        data=json.loads((folder/'INPUT.json').read_bytes());gold=json.loads((folder/'SEALED.json').read_bytes())
        models=3 if phase=='m49' else 2
        ws=[{r['key']:r for r in map(json.loads,(folder/f'WORKER{i}.jsonl').read_bytes().splitlines())} for i in range(models)]
        roles=[(a,1-a,r) for a in (0,1) for r in (0,1)] if phase=='m45' else [(0,1,2),(1,2,0),(2,0,1)]
        for row in data:
            k=row['key'];n=len(row['options'])
            v=[np.asarray(w[k]['probs']) for w in ws];v=[x/x.sum(1,keepdims=True) for x in v]
            price=[np.asarray(w[k]['input_tokens'])+1 for w in ws];offset=1 if phase=='m45' else 3
            for a,b,r in roles:
                source=v[a][0 if phase=='m45' else 1]
                mid=v[b][offset:].reshape(n+1,n+1,n);rc=v[r][offset:].reshape(n+1,n+1,n)
                bc=price[b][offset:].reshape(n+1,n+1);cost=price[r][offset:].reshape(n+1,n+1)[1:,1:]
                init=price[a][0 if phase=='m45' else 1]+bc[0,0]
                data_row=stopping(source,mid[0,0],mid[1:,0],rc[1:,1:],bc[1:,0],cost,cost,gold[k],init)
                group=f'{row["task"]}_{r}' if phase=='m45' else f'sciq_{a}_{b}_{r}'
                aggregated.setdefault(group,[]).append(data_row)
    result={}
    for k,rows in aggregated.items():
        mean={field:float(np.mean([r[field] for r in rows])) for field in rows[0]}
        mean['ratio']=mean['all_tokens']/mean['gold_tokens'];mean['instances']=len(rows)
        result[k]=mean
    return result
