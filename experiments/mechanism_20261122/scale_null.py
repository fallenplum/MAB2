"""Evaluate the registered additive-evidence null on the four interventional logits."""
import numpy as np
def normalize(p):return p/p.sum(axis=-1,keepdims=True)
def clr(p):
    v=np.log(p);return v-v.mean(axis=-1,keepdims=True)
def evaluate(inputs,gold,w):
    agg={}
    for task in ('arc','obqa'):
        for role in range(2):
            obs=[]
            for r in inputs:
                if r['task']!=task:continue
                k=r['key'];y=gold[k];p=[normalize(np.array(x[k]['probs'],float)) for x in w];tab=p[role][1:].reshape(5,5,4)
                left=tab[1:,0,None,:];right=tab[None,0,1:,:];none=tab[0,0];full=tab[1:,1:];add=normalize(left*right/none);pr=normalize(np.maximum(left+right-none,1e-12));weights=np.outer(p[0][0],p[1][0])
                cr=full[...,y]-left[...,y]-right[...,y]+none[y];scale=add[...,y]-left[...,y]-right[...,y]+none[y];residual=full[...,y]-add[...,y]
                assert np.max(abs(cr-scale-residual))<1e-10
                old=left[...,y]-none[y];current=full[...,y]-right[...,y];pred=add[...,y]-right[...,y]
                alternative=pr[...,y]-right[...,y]
                def av(x):return float((weights*x).sum())
                def loss(x):return av(abs(current)*((x>0)!=(current>0)))
                obs.append(dict(zero_drift_MSE=av(cr**2),product_MSE=av(residual**2),prob_add_MSE=av((current-alternative)**2),scale_sq=av(scale**2),cross_term=av(2*scale*residual),stale_regret=loss(old),product_regret=loss(pred),prob_add_regret=loss(alternative),logit_interaction_rms=av(np.sqrt(((clr(full)-clr(add))**2).mean(-1))),absolute_drift=av(abs(cr)),abs_residual=av(abs(residual))))
            vals={k:float(np.mean([r[k] for r in obs])) for k in obs[0]};vals['risk_skill']=1-vals['product_MSE']/vals['zero_drift_MSE'];agg[f'{task}_{role}']=vals
    return agg
