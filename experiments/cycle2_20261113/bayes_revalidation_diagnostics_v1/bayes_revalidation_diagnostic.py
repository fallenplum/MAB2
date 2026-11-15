"""One bounded sequential-acquisition development diagnostic; no S array access."""
from pathlib import Path
import json,hashlib,numpy as np
INPUT=None;OUT=None
VALUES=np.array([-1.,0.,1.]);METHODS=['graph_EVSI','no_graph_EVSI','fresh_EVSI','variance','random','stale','protect_dirty','stay']
def means(alpha):return (alpha@VALUES)/alpha.sum(axis=1)
def action(alpha):
    m=means(alpha);return int(np.argmax(m)) if m.max()>0 else None
def evsi(alpha,edge):
    current=max(0,float(means(alpha).max()));p=alpha[edge]/alpha[edge].sum();value=0.
    for label in range(3):
        after=alpha.copy();after[edge,label]+=1;value+=p[label]*max(0,float(means(after).max()))
    assert value-current>=-1e-12
    return max(0.,value-current)
def run(I,C,dirty,method,seed):
    assert I.shape==(12,6) and C.shape==(6,6) and np.isin(I,VALUES).all() and np.isin(C,VALUES).all()
    prior=np.array([[(I[:,e]==v).sum() for v in VALUES] for e in range(6)],float)
    retention=np.where(dirty,.25,1.) if method=='graph_EVSI' else np.zeros(6) if method=='fresh_EVSI' else np.ones(6)
    alpha=1/3+prior*retention[:,None];rng=np.random.default_rng(seed);orders=[rng.permutation(6).tolist() for e in range(6)];counts=np.zeros(6,dtype=int);trace=[]
    if method=='stay':return None,trace
    if method=='protect_dirty':
        m=means(alpha);m[np.array(dirty,bool)]=0.;return (int(np.argmax(m)) if m.max()>0 else None),trace
    if method=='stale':return action(alpha),trace
    for step in range(24):
        remaining=np.flatnonzero(counts<6)
        if not len(remaining):break
        if method=='random':e=int(rng.choice(remaining))
        elif method=='variance':
            variance=(alpha@(VALUES**2))/alpha.sum(axis=1)-means(alpha)**2;score=variance/(alpha.sum(axis=1)+1);e=max(remaining,key=lambda j:(score[j],-int(j)))
        else:
            score={int(j):evsi(alpha,int(j)) for j in remaining};e=max(remaining,key=lambda j:(score[int(j)],-int(j)))
            # Fixed horizon 24, hypothetical 0.05 correct-answer equivalents / pair.
            # This is NOT measured token cost and cannot establish net system gain.
            if 24*score[int(e)]<=.05:break
        q=orders[e][counts[e]];counts[e]+=1
        before=action(alpha);y=C[q,e];alpha[e,int(y)+1]+=1
        trace.append({'step':step,'edge':int(e),'feedback_row':int(q),'before_action':before,'observed_gain':float(y)})
    return action(alpha),trace
def main():
    global INPUT,OUT
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();INPUT=args.input;OUT=args.output
    assert not OUT.exists();OUT.mkdir()
    contract={'status':'FROZEN_BEFORE_THIS_DIAGNOSTIC','data_role':'All R1 I/C already exposed development; never fresh confirmation','nominee':'graph_EVSI','retention_dirty':.25,'retention_clean':1,'categorical_pseudocount':1/3,'fixed_pairs_max':24,'hypothetical_pair_penalty':.05,'fixed_service_horizon':24,'C_split':'first six acquisition; last six evaluation; no S accessed','methods':METHODS,'seeds':[261115601+i for i in range(5)],'hyperparameter_search':False,'new_GPU':False,'confidence_intervals':False,'formal_admission':False,'input_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest()}
    (OUT/'DESIGN.json').write_text(json.dumps(contract,indent=2)+'\n',encoding='utf-8')
    # NPZ lazy loading opens only these three arrays, never S_quality or S_cost.
    with np.load(INPUT,allow_pickle=False) as f:I=f['I_delta'];C=f['C_delta'];dirty=f['dirty']
    results=[];traces=[]
    for episode in range(8):
      for ci in [0,1]:
       for seed in contract['seeds']:
        for method in METHODS:
            a,trace=run(I[episode],C[episode,ci,:6],dirty[episode] if ci==0 else np.zeros(6),method,seed)
            # Frozen action is evaluated only after the acquisition loop returns.
            gain=0. if a is None else float(C[episode,ci,6:,a].mean())
            results.append({'episode':episode,'condition':'change' if ci==0 else 'no_change','seed':seed,'method':method,'action':a,'feedback_pairs':len(trace),'evaluation_gain_over_stay':gain})
            traces.append({'episode':episode,'condition':ci,'seed':seed,'method':method,'trace':trace})
    summary=[]
    for ci in ['change','no_change']:
      for method in METHODS:
        rows=[r for r in results if r['condition']==ci and r['method']==method]
        summary.append({'condition':ci,'method':method,'mean_gain_over_stay':float(np.mean([r['evaluation_gain_over_stay'] for r in rows])),'mean_pairs':float(np.mean([r['feedback_pairs'] for r in rows])),'episode_points':[float(np.mean([r['evaluation_gain_over_stay'] for r in rows if r['episode']==e])) for e in range(8)]})
    out={'status':'COMPLETE_BOUNDED_POSTHOC_ACQUISITION_DIAGNOSTIC','summary':summary,'rows':results,'new_GPU_hours':0,'S_arrays_opened':False,'same_full_token_cap':False,'penalty_is_hypothetical_not_cost_evidence':True,'automatic_GPU_admission':False,'bootstrap_runs':0}
    (OUT/'RESULTS.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');(OUT/'PRIVATE_TRACES.json').write_text(json.dumps(traces,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':out['status'],'summary':summary},indent=2))
if __name__=='__main__':main()
