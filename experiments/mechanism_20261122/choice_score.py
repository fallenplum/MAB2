"""Exact answer-marginal and inverse-CDF coupling calculations on constrained model outputs."""
import itertools,json,hashlib
from pathlib import Path
import numpy as np
from crossed import bounds,SIGN

def shared_distributions(p):
    cut=np.unique(np.r_[0,1,np.clip(p.cumsum(axis=1)[:,:-1].ravel(),0,1)]);mid=(cut[:-1]+cut[1:])/2
    draw=(mid[:,None,None]>p.cumsum(axis=1)[None,:,:-1]).sum(axis=2)
    return draw,np.diff(cut)

def shared(p,y):
    labels,mass=shared_distributions(p);z=(labels==y).astype(float)
    return float(mass@((z@SIGN)!=0)),float(mass@((z[:,0]-z[:,1])**2))

def independent(p,y):
    q=p[:,y];d=(q[0]*(1-q[1]),(1-q[0])*q[1]);e=(q[2]*(1-q[3]),(1-q[2])*q[3])
    return 1-(d[0]*e[0]+d[1]*e[1]+(1-sum(d))*(1-sum(e)))

def run(inputs,gold,workers,private):
    result={};details=[]
    for role,rows in enumerate(workers):
        for task in ('arc','obqa'):
            obs=[]
            for r in inputs:
                if r['task']!=task:continue
                key=r['key'];y=gold[key];probs=[np.array(w[key]['probs'],float) for w in workers]
                assert all(p.shape==(26,4) and np.all(p>=0) and np.max(abs(p.sum(1)-1))<2e-6 for p in probs)
                for p in probs:p/=p.sum(1,keepdims=True)
                advice=[p[0] for p in probs];table=probs[role][1:].reshape(5,5,4)
                total=np.zeros(10)
                gnoise=np.random.default_rng(20261122).gumbel(size=(4096,4))
                for left,right in itertools.product(range(1,5),repeat=2):
                    p=table[[left,0,left,0],[right,right,0,0],:];weight=advice[0][left-1]*advice[1][right-1]
                    a,m2=shared(p,y);rev,_=shared(p[:,::-1],3-y);assert abs(rev-a)<1e-8
                    q=p[:,y];lo,hi=bounds(tuple(q),(0,1,2,3));b=independent(p,y);mean=q@SIGN
                    gdraw=np.argmax(gnoise[:,None,:]+np.log(p[None,:,:]+1e-300),axis=2);gc=np.mean(((gdraw==y).astype(float)@SIGN)!=0)
                    assert lo-1e-6<=min(a,b)<=max(a,b)<=hi+1e-6
                    total+=weight*np.array([a,b,gc,lo,hi,mean,abs(mean),q[0]-q[1],(q[0]-q[1])**2,m2])
                v=dict(zip(['shared','independent','gumbel','lower','upper','mean_interaction','abs_conditional_interaction','mean_credit','credit_second_source','credit_second_full'],total))
                v['source_variance']=v['credit_second_source']-v['mean_credit']**2;v['terminal_variance']=v['credit_second_full']-v['credit_second_source']
                assert min(v['source_variance'],v['terminal_variance'])>-1e-7
                obs.append(v);details.append(dict(key=key,role=role,**v))
            vals={k:float(np.mean([row[k] for row in obs])) for k in obs[0]};vals['questions']=len(obs)
            vals['gap']=vals['independent']-vals['shared'];vals['source_share']=vals['source_variance']/(vals['source_variance']+vals['terminal_variance'])
            result[f'{task}_{role}']=vals
    private.write_text(json.dumps(details),encoding='utf8')
    return dict(cells=result,fixed_screen_passed=all(abs(z['gap'])>=.05 for z in result.values()),scope='restricted categorical receiver; same reader source appears as a distinct role; no service-quality win tested')
