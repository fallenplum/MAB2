"""Prospective descriptive screen; no interval or resampling computation."""
import numpy as np
ORDER=('G0','G0_MINUS_EU','G0_MINUS_EV','G1','G1_MINUS_EU','G1_MINUS_EV','INDEPENDENT3_M','SINGLE_M','SINGLE_Q')
def paired(a,b):
    return {'accuracy_a':float(a.mean()),'accuracy_b':float(b.mean()),
            'paired_difference':float((a-b).mean()),
            'corrections':int(((a==1)&(b==0)).sum()),
            'breakages':int(((a==0)&(b==1)).sum()),'paired_observations':int(a.size)}
def summarize(em,shocks):
    x=np.asarray(em,dtype=float)
    if x.shape!=(96,2,9) or not np.isin(x,[0,1]).all():raise ValueError('Complete 96x2x9 binary EM required')
    if len(shocks)!=96 or shocks.count('U')!=48 or shocks.count('V')!=48:raise ValueError('Fixed balanced shocks required')
    old=np.stack([x[:,:,0]-x[:,:,1],x[:,:,0]-x[:,:,2]],axis=2)
    new=np.stack([x[:,:,3]-x[:,:,4],x[:,:,3]-x[:,:,5]],axis=2)
    changes=new-old
    aff=np.array([0 if v=='U' else 1 for v in shocks]);rows=np.arange(96)
    d=changes[rows,:,aff]-changes[rows,:,1-aff]
    point=float(d.mean());by_seed=d.mean(axis=0);support=int(np.count_nonzero(d.mean(axis=1)))
    retained=bool(point>=.05 and np.all(by_seed>0) and support>=20)
    return {'status':'COMPLETE_DESCRIPTIVE_POINT_SCREEN','primary_D':point,
        'D_by_seed':by_seed.tolist(),'nonzero_question_support':support,
        'D_by_shock':{k:float(d[np.array(shocks)==k].mean()) for k in ['U','V']},
        'credit_by_edge_seed':{name:arr.mean(axis=0).tolist() for name,arr in [('old',old),('new',new),('change',changes)]},
        'accuracy_by_condition_seed':{name:x[:,:,i].mean(axis=0).tolist() for i,name in enumerate(ORDER)},
        'comparisons_by_seed':{f'{g}_minus_{b}':[
            paired(x[:,s,ORDER.index(g)],x[:,s,ORDER.index(b)]) for s in range(2)]
            for g in ['G0','G1'] for b in ['INDEPENDENT3_M','SINGLE_M','SINGLE_Q']},
        'screen_action':'RETAIN_MECHANISM_FOR_NEW_CONTRACT_REVIEW' if retained else 'DO_NOT_AUTOMATICALLY_EXPAND_THIS_INSTANCE',
        'rule':'D>=0.05 AND both seed means positive AND at least20 nonzero question means',
        'rule_is_empirical_development_filter_not_significance':True,
        'method_success_claim':False,'automatic_GPU_admission':False,
        'confidence_intervals_computed':False,'bootstrap_runs':0,'project_stop':False}
