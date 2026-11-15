"""Second and final local Bayesian-family screen: evidence decides transport validity."""
from pathlib import Path
import json,hashlib,numpy as np
import bayes_revalidation_diagnostic as old
V=np.array([-1.,0.,1.])
METHODS=['graph_mixture_EVSI','no_graph_mixture_EVSI','valid_EVSI','fresh_EVSI','graph_mixture_variance','graph_mixture_random','stale','protect_dirty','stay']
def predictive(a,b,w):return w[:,None]*a/a.sum(axis=1)[:,None]+(1-w[:,None])*b/b.sum(axis=1)[:,None]
def update(a,b,w,e,y):
    aa=a.copy();bb=b.copy();ww=w.copy();po=a[e,y]/a[e].sum();pn=b[e,y]/b[e].sum();ww[e]=w[e]*po/(w[e]*po+(1-w[e])*pn);aa[e,y]+=1;bb[e,y]+=1;return aa,bb,ww
def act(a,b,w):
    m=predictive(a,b,w)@V;return int(m.argmax()) if m.max()>0 else None
def value(a,b,w,e):
    current=max(0,float((predictive(a,b,w)@V).max()));p=predictive(a,b,w)[e];value=0.
    for y in range(3):
        aa,bb,ww=update(a,b,w,e,y);value+=p[y]*max(0,float((predictive(aa,bb,ww)@V).max()))
    assert value-current>=-1e-12;return max(0.,value-current)
def run(I,C,dirty,method,seed):
    if method in ['stale','protect_dirty','stay']:return old.run(I,C,dirty,method,seed)
    a=np.array([[(I[:,e]==v).sum()+1/3 for v in V] for e in range(6)],float);b=np.full((6,3),1/3)
    w=np.where(dirty,.5,.9) if method.startswith('graph_') else np.full(6,.5 if method=='no_graph_mixture_EVSI' else 1. if method=='valid_EVSI' else 0.)
    rng=np.random.default_rng(seed);orders=[rng.permutation(6).tolist() for e in range(6)];counts=np.zeros(6,dtype=int);trace=[]
    for step in range(24):
        remaining=np.flatnonzero(counts<6)
        if method.endswith('random'):e=int(rng.choice(remaining))
        elif method.endswith('variance'):
            ma=(a@V)/a.sum(axis=1);mb=(b@V)/b.sum(axis=1);va=((a@(V**2))/a.sum(axis=1)-ma**2)/(a.sum(axis=1)+1);vb=((b@(V**2))/b.sum(axis=1)-mb**2)/(b.sum(axis=1)+1);score=w*va+(1-w)*vb+w*(1-w)*(ma-mb)**2;e=max(remaining,key=lambda j:(score[j],-int(j)))
        else:
            score={int(j):value(a,b,w,int(j)) for j in remaining};e=max(remaining,key=lambda j:(score[int(j)],-int(j)))
            if 24*score[int(e)]<=.05:break
        q=orders[e][counts[e]];counts[e]+=1;y=int(C[q,e])+1;before=act(a,b,w);a,b,w=update(a,b,w,e,y)
        trace.append({'step':step,'edge':int(e),'feedback_row':int(q),'before_action':before,'observed_gain':float(C[q,e]),'posterior_old_valid':float(w[e])})
    return act(a,b,w),trace
def tests():
    a=np.full((6,3),1/3);a[:,1]+=12;b=np.full((6,3),1/3);w=np.full(6,.5)
    assert np.allclose(predictive(a,b,w).sum(axis=1),1)
    _,_,same=update(a,b,w,0,1);_,_,contrary=update(a,b,w,0,2);assert same[0]>.5 and contrary[0]<.5
    assert value(a,b,w,0)>0
    assert np.array_equal(predictive(a,b,np.ones(6)),a/a.sum(axis=1)[:,None])
    assert np.array_equal(predictive(a,b,np.zeros(6)),b/b.sum(axis=1)[:,None])
    return {'posterior_normalization':True,'consistent_feedback_increases_validity':True,'contrary_feedback_decreases_validity':True,'value_of_information_positive_case':True,'valid_and_fresh_limits':True}
def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();old.INPUT=args.input;out=args.output;out.mkdir(exist_ok=False)
    design={'candidate':'graph_mixture_EVSI','status':'SECOND_AND_FINAL_BOUNDED_BAYES_FAMILY_CPU_DIAGNOSTIC','initial_valid_probability_dirty':.5,'initial_valid_probability_clean':.9,'conditional_models':'Dirichlet(1/3+old counts+new counts) versus Dirichlet(1/3+new counts)','learning':'Bayes update of old-evidence validity after each purchased ternary paired gain','max_pairs':24,'hypothetical_penalty':.05,'service_horizon':24,'seeds':[261115601+i for i in range(5)],'methods':METHODS,'I_C_data_already_exposed':True,'S_never_opened':True,'no_hyperparameter_sweep':True,'new_GPU_hours':0,'input_sha256':hashlib.sha256(old.INPUT.read_bytes()).hexdigest(),'tests':tests()}
    (out/'DESIGN.json').write_text(json.dumps(design,indent=2)+'\n',encoding='utf-8')
    with np.load(old.INPUT,allow_pickle=False) as f:I=f['I_delta'];C=f['C_delta'];dirty=f['dirty']
    results=[]
    for episode in range(8):
      for ci in [0,1]:
       for seed in design['seeds']:
        for method in METHODS:
            action,trace=run(I[episode],C[episode,ci,:6],dirty[episode] if ci==0 else np.zeros(6),method,seed)
            gain=0. if action is None else float(C[episode,ci,6:,action].mean())
            results.append({'episode':episode,'condition':'change' if ci==0 else 'no_change','seed':seed,'method':method,'action':action,'pairs':len(trace),'gain':gain})
    summary=[]
    for condition in ['change','no_change']:
      for method in METHODS:
        r=[x for x in results if x['condition']==condition and x['method']==method]
        summary.append({'condition':condition,'method':method,'mean_gain':float(np.mean([x['gain'] for x in r])),'mean_pairs':float(np.mean([x['pairs'] for x in r])),'episode_points':[float(np.mean([x['gain'] for x in r if x['episode']==e])) for e in range(8)]})
    report={'status':'COMPLETE_POSTHOC_MIXTURE_DIAGNOSTIC','summary':summary,'rows':results,'same_full_cost_cap':False,'hypothetical_penalty_not_measured_cost':True,'new_GPU_hours':0,'bootstrap_runs':0,'automatic_admission':False}
    (out/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({'status':report['status'],'summary':summary},indent=2))
if __name__=='__main__':main()
