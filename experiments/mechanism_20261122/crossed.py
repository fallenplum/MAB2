"""Finite, paired source/terminal allocation and partial identification diagnostics."""
import itertools
from functools import lru_cache
import numpy as np
from scipy.optimize import linprog

BITS=np.asarray(list(itertools.product((0,1),repeat=4)))
SIGN=np.array([1,-1,-1,1])
EVENT=(BITS@SIGN!=0).astype(float)

@lru_cache(maxsize=8000)
def bounds(probs,identity):
    x=BITS[np.all(np.stack([(BITS[:,i]==BITS[:,j]) for i in range(4) for j in range(i,4) if identity[i]==identity[j]]),axis=0)]
    event=(x@SIGN!=0).astype(float);a=np.vstack([np.ones(len(x)),x.T]);b=np.r_[1,probs]
    result=[linprog(sign*event,A_eq=a,b_eq=b,bounds=(0,None),method='highs') for sign in (1,-1)]
    assert all(v.success and np.max(abs(a@v.x-b))<1e-7 for v in result)
    return result[0].fun,-result[1].fun

def analyze(em,tokens,shock,entries):
    sources=sorted({e['source_seed'] for e in entries});terminals=sources+sorted({e['terminal_seed'] for e in entries}-set(sources))
    reg={(r['index'],sources.index(r['source_seed']),terminals.index(r['terminal_seed']),r['mask']):r for r in entries}
    assert len(reg)==em.size
    records=[];risk={name:[] for name in ('repeat_terminal','repeat_source','vary_both')}
    for q,h in enumerate(shock):
        base=7^(1<<int(h))
        for edge in range(3):
            if edge==h:continue
            masks=[7,7^(1<<edge),base,base^(1<<edge)]
            y=em[q][:,:,masks]
            for s in range(2):
                ids=[tuple(reg[q,s,t,m]['job_id'] for t in range(4)) for m in masks]
                ident=tuple(ids.index(i) for i in ids);p=y[s].mean(0)
                lo,hi=bounds(tuple(p),ident);factual=np.mean(y[s]@SIGN!=0)
                assert lo-1e-9<=factual<=hi+1e-9
                rng_mass=0.
                groups=sorted(set(ident))
                for vv in itertools.product(range(4),repeat=len(groups)):
                    vals=[y[s,vv[groups.index(i)],j] for j,i in enumerate(ident)]
                    rng_mass+=int(np.dot(vals,SIGN)!=0)/4**len(groups)
                records.append((lo,hi,factual,rng_mass,abs(p@SIGN),lo>0,hi-lo>=.25))
            credit=y[:,:,2]-y[:,:,3]
            def charge(cells):
                seen={};src={}
                for s,t in cells:
                    for m in masks[2:]:
                        r=reg[q,s,t,m];cost=tokens[q,s,t,m]-r['prefix_tokens'];key=r['job_id']
                        assert key not in seen or seen[key]==cost
                        seen[key]=cost
                        for e in range(3):
                            if m&(1<<e):src[s,e]=reg[q,s,t,1<<e]['prefix_tokens']
                return sum(seen.values())+sum(src.values())
            for t,u in itertools.combinations(range(4),2):
                target=credit[:,[t,u]].mean()
                choices={'repeat_terminal':[((0,t),(0,u)),((1,t),(1,u))],'repeat_source':[((0,t),(1,t)),((0,u),(1,u))],'vary_both':[((0,t),(1,u)),((0,u),(1,t))]}
                for name,draws in choices.items():
                    for cells in draws:
                        err=(sum(credit[s,v] for s,v in cells)/2-target)**2
                        risk[name].append((err,charge(cells)))
    z=np.asarray(records);summary={}
    for k,v in risk.items():
        riskmean,cost=np.mean(v,0);summary[k]=dict(finite_grid_MSE=riskmean,mean_tokens=cost,risk_times_mean_tokens=riskmean*cost)
    return dict(source_conditioned_edge_units=len(z),questions=len(em),lower=z[:,0].mean(),upper=z[:,1].mean(),common_seed_support=z[:,2].mean(),independent_prompt_seed_support=z[:,3].mean(),mean_absolute_marginal_interaction=z[:,4].mean(),fraction_positive_lower=z[:,5].mean(),fraction_width_at_least_quarter=z[:,6].mean(),allocation=summary)
