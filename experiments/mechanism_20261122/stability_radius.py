"""Radius beyond which a finite-source reachability certificate can expire."""
import json
import numpy as np

def radius(p,t,tau=.01):
    current=p@t
    if current>tau:return 0.
    best=max(t)
    if best<=tau:return np.inf
    needed=tau-current;mass=0.
    for i in np.argsort(t,kind='stable'):
        gain=best-t[i]
        if gain<=0:continue
        take=min(p[i],needed/gain)
        mass+=take;needed-=take*gain
        if needed<=1e-14:return mass
    raise ArithmeticError('infeasible simplex distance')


def score(folder):
    ds=json.loads((folder/'INPUT.json').read_bytes())
    ws=[{r['key']:r for r in map(json.loads,(folder/f'WORKER{i}.jsonl').read_bytes().splitlines())} for i in range(3)]
    results={}
    for a,b in ((0,1),(1,2),(2,0)):
        rows=[]
        for r in ds:
            k=r['key'];A=np.asarray(ws[a][k]['probs']);A=A/A.sum(1,keepdims=True)
            B=np.asarray(ws[b][k]['probs']);B=B/B.sum(1,keepdims=True);tab=B[3:].reshape(5,5,4)
            t=np.abs(tab[1:,0]-tab[0,0]).sum(1)/2
            p=A[1];bound=p@t;eligible=bound<=.01;rad=radius(p,t)
            obs={'initial_certificate':eligible,'infinite_radius':eligible and np.isinf(rad),
                 'fails_within_one_point':eligible and rad<=.01,'fails_within_five_points':eligible and rad<=.05}
            for label,i in (('no_note',0),('shuffled',2)):
                dist=abs(A[i]-p).sum()/2;expired=eligible and A[i]@t>.01
                assert not expired or rad<=dist+1e-8
                obs[label+'_expired']=expired;obs[label+'_tv_distance']=dist
                obs[label+'_retained_via_radius']=eligible and dist<rad-1e-10
            for temp in (.7,1.,1.3):
                pn=p**(1/temp);pn/=pn.sum();obs['temp_'+str(temp)+'_cert']=pn@t<=.01
            rows.append(obs)
        results[f'{a}_{b}']={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
        results[f'{a}_{b}']['questions']=len(rows)
    return results
