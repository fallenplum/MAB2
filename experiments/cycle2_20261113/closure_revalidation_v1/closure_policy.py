"""Two-edge decision-value acquisition. No service outcomes or hidden C cache inputs."""
import numpy as np
VALUES=np.array([-1.,0.,1.])
TOKEN_WEIGHT=.005/1000
HORIZON=64
MAX_PAIRS=8
METHODS=('closure_EVSI','EVSI_fixed_price','EVSI_no_cost_ranking','EVSI_old_graph_cost',
 'random','periodic','uncertainty','graph_changed_first','stale','protect_changed','keep')

def means(alpha):return (alpha@VALUES)/alpha.sum(axis=1)
def utilities(alpha,prices):
    return np.r_[0.,means(alpha)+TOKEN_WEIGHT*(prices[0]-np.asarray(prices[1:]))]
def action(alpha,prices,protect=None):
    u=utilities(alpha,prices)
    if protect is not None:u[1+protect]=-np.inf
    # keep wins exact ties, then U, then V.
    return int(np.argmax(u))
def evsi(alpha,prices,e):
    before=float(utilities(alpha,prices).max());p=alpha[e]/alpha[e].sum();after=0.
    for j in range(3):
        a=alpha.copy();a[e,j]+=1;after+=p[j]*float(utilities(a,prices).max())
    assert after-before>=-1e-12
    return max(0.,after-before)

def run(initial_gains,service_prices,old_graph_prices,counterfactual_final_prices,changed,method,seed,buy):
    """buy(e,q) exposes ONLY a paid paired outcome, actual tokens, and cache state.

    buy must implement independent evaluator permissions/costs; no per-question
    candidate output, C outcome, or S label is accepted before selection.
    All prices are forecasts constructed from initialization I only.
    """
    I=np.asarray(initial_gains,float);prices=np.asarray(service_prices,float);old=np.asarray(old_graph_prices,float);final=np.asarray(counterfactual_final_prices,float)
    if I.shape!=(8,2) or not np.isin(I,VALUES).all():raise ValueError('Fixed eight-question initial credit matrix required')
    if prices.shape!=(3,) or old.shape!=(3,) or final.shape!=(2,) or not np.isfinite(np.r_[prices,old,final]).all() or min(np.r_[prices,old,final])<=0:raise ValueError('Positive finite initialization-only token forecasts required')
    if method not in METHODS or changed not in (None,0,1):raise ValueError('Unfrozen method or graph change')
    alpha=np.array([[(I[:,e]==v).sum()+1/3 for v in VALUES] for e in range(2)],float)
    used_prices=old if method=='EVSI_old_graph_cost' else prices
    if method=='keep':return {'action':0,'trace':[],'pairs':0}
    if method=='stale':return {'action':action(alpha,used_prices),'trace':[],'pairs':0}
    if method=='protect_changed':return {'action':action(alpha,used_prices,protect=changed),'trace':[],'pairs':0}
    counts=np.zeros(2,dtype=int);seen_base=set();seen_pairs=set();rng=np.random.default_rng(seed);trace=[]
    periodic=[(e,q) for q in (0,2,4,6) for e in (0,1)]
    for step in range(MAX_PAIRS):
        available=[e for e in (0,1) if counts[e]<8]
        if not available:break
        candidates={}
        for e in available:
            q=int(counts[e]);increment=float(final[e]+(0 if q in seen_base else prices[0]))
            fixed=float(final[e]+prices[0]);gain=HORIZON*evsi(alpha,used_prices,e)
            price=fixed if method=='EVSI_fixed_price' else increment
            candidates[e]={'q':q,'forecast_tokens':price,'expected_gain':gain,'ratio':gain/price}
        if method=='periodic':e,q=periodic[step]
        elif method=='random':e=int(rng.choice(available));q=int(counts[e])
        elif method=='graph_changed_first':
            e=min(available,key=lambda e:(counts[e],e)) if changed is None else changed if counts[changed]<8 else 1-changed
            q=int(counts[e])
        elif method=='uncertainty':
            m=means(alpha);var=(alpha@(VALUES**2))/alpha.sum(axis=1)-m*m
            e=max(available,key=lambda e:(float(var[e]/(alpha[e].sum()+1)),-e));q=int(counts[e])
        else:
            key='expected_gain' if method=='EVSI_no_cost_ranking' else 'ratio'
            e=max(available,key=lambda e:(candidates[e][key],-e));q=int(counts[e])
            if candidates[e]['expected_gain']<=TOKEN_WEIGHT*candidates[e]['forecast_tokens']:break
        assert (e,q) not in seen_pairs
        before=action(alpha,used_prices)
        # Selection is recorded before buy() can reveal any paired outcome.
        row={'step':step,'edge':e,'feedback_row':q,'before_action':before,'candidates':candidates}
        obs=buy(e,q)
        if set(obs)!={'gain','charged_tokens'} or obs['gain'] not in (-1,0,1) or not np.isfinite(obs['charged_tokens']) or obs['charged_tokens']<0:raise ValueError('Invalid paid feedback')
        seen_pairs.add((e,q));seen_base.add(q);counts[e]+=1;alpha[e,int(obs['gain'])+1]+=1
        row.update(observed_gain=int(obs['gain']),charged_tokens=float(obs['charged_tokens']),after_action=action(alpha,used_prices));trace.append(row)
    return {'action':action(alpha,used_prices),'trace':trace,'pairs':len(trace)}
