"""Fixed descriptive receiver factorial and nominated-operator filters; no intervals."""
import numpy as np
from factorial_operator import SCHEDULE
ORDER=tuple(SCHEDULE)
CELLS=('M_PARTIAL','Q_PARTIAL','M_FULL','Q_FULL')
def name(graph,cell):return graph if cell=='M_PARTIAL' else graph+'__'+cell
def paired(a,b):
    return {'mean_a':float(a.mean()),'mean_b':float(b.mean()),'difference':float((a-b).mean()),
       'corrections':int(((a==1)&(b==0)).sum()),'breakages':int(((a==0)&(b==1)).sum()),'observations':int(a.size)}
def summarize(em,tokens,kernel,shocks):
    x=np.asarray(em,float);t=np.asarray(tokens,float);k=np.asarray(kernel,float)
    if x.shape!=(96,2,29) or not np.isin(x,[0,1]).all():raise ValueError('Complete binary 96x2x29 required')
    if t.shape!=x.shape or k.shape!=x.shape or not np.isfinite(t).all() or not np.isfinite(k).all() or np.any(t<0) or np.any(k<0):raise ValueError('Complete nonnegative cost arrays required')
    if len(shocks)!=96 or shocks.count('U')!=48 or shocks.count('V')!=48:raise ValueError('Balanced frozen shocks required')
    val=lambda condition:x[:,:,ORDER.index(condition)]
    signals={}
    for cell in CELLS:
        before=np.stack([val(name('G0',cell))-val(name('G0_MINUS_EU',cell)),val(name('G0',cell))-val(name('G0_MINUS_EV',cell))],axis=2)
        after=np.stack([val(name('G1',cell))-val(name('G1_MINUS_EU',cell)),val(name('G1',cell))-val(name('G1_MINUS_EV',cell))],axis=2)
        change=after-before;aff=np.array([0 if s=='U' else 1 for s in shocks]);i=np.arange(96)
        d=change[i,:,aff]-change[i,:,1-aff]
        signals[cell]={'D':float(d.mean()),'D_by_seed':d.mean(axis=0).tolist(),'nonzero_question_support':int(np.count_nonzero(d.mean(axis=1))),
          'affected_credit_change_by_seed':change[i,:,aff].mean(axis=0).tolist(),'unaffected_credit_change_by_seed':change[i,:,1-aff].mean(axis=0).tolist()}
    contrasts={};interactions={}
    for graph in ['G0','G1']:
        for a,b,label in [('Q_PARTIAL','M_PARTIAL','receiver_Q_minus_M_partial'),('Q_FULL','M_FULL','receiver_Q_minus_M_full'),('M_FULL','M_PARTIAL','raw_access_effect_M'),('Q_FULL','Q_PARTIAL','raw_access_effect_Q')]:
            contrasts[graph+'/'+label]=[paired(val(name(graph,a))[:,s],val(name(graph,b))[:,s]) for s in range(2)]
        interaction=val(name(graph,'Q_FULL'))-val(name(graph,'Q_PARTIAL'))-val(name(graph,'M_FULL'))+val(name(graph,'M_PARTIAL'))
        interactions[graph]={'point':float(interaction.mean()),'by_seed':interaction.mean(axis=0).tolist()}
    nominated='G1__Q_FULL';a=val(nominated);comparisons={}
    for baseline in ['SINGLE_Q','SINGLE_M','INDEPENDENT3_Q','INDEPENDENT3_M','VOTE_Q3']:
        b=val(baseline);comparisons[baseline]={'pooled':paired(a,b),'by_seed':[paired(a[:,s],b[:,s]) for s in range(2)],'nonzero_question_support':int(np.count_nonzero((a-b).mean(axis=1)))}
    ni=ORDER.index(nominated);bi=ORDER.index('INDEPENDENT3_Q')
    if t[:,:,bi].sum()<=0 or k[:,:,bi].sum()<=0:raise ValueError('Measured comparator costs must be positive')
    token_ratio=float(t[:,:,ni].sum()/t[:,:,bi].sum())
    kernel_ratio=float(k[:,:,ni].sum()/k[:,:,bi].sum())
    q=comparisons['SINGLE_Q'];vote=comparisons['VOTE_Q3'];ind=comparisons['INDEPENDENT3_Q'];signal=signals['Q_FULL']
    gates={'gain_over_single_Q':q['pooled']['difference']>=.02 and all(r['difference']>0 for r in q['by_seed']) and q['nonzero_question_support']>=20,
       'gain_over_cheap_Q_vote':vote['pooled']['difference']>=.02 and all(r['difference']>0 for r in vote['by_seed']),
       'independent_aggregation_quality_screen':ind['pooled']['difference']>=-.01 and all(r['difference']>=-.03 for r in ind['by_seed']),
       'independent_aggregation_cost_screen':token_ratio<=.85 and kernel_ratio<=.9,
       'credit_signal_screen':signal['D']>=.05 and all(v>0 for v in signal['D_by_seed']) and signal['nonzero_question_support']>=20}
    return {'status':'COMPLETE_DESCRIPTIVE_FACTORIAL_SCREEN','fixed_nominee':nominated,
       'accuracy_by_condition_seed':{n:val(n).mean(axis=0).tolist() for n in ORDER},'nominated_comparisons':comparisons,
       'receiver_access_contrasts':contrasts,'receiver_access_interactions':interactions,'credit_signals_by_cell':signals,
       'tokens_by_condition_seed':{n:t[:,:,i].sum(axis=0).tolist() for i,n in enumerate(ORDER)},
       'kernel_seconds_by_condition_seed':{n:k[:,:,i].sum(axis=0).tolist() for i,n in enumerate(ORDER)},
       'nominee_cost_ratios_vs_independent3_Q':{'total_tokens':token_ratio,'kernel_seconds':kernel_ratio},'gates':gates,
       'screen_action':'RETAIN_NOMINATED_OPERATOR_FOR_COSTED_REVALIDATION_CONTRACT' if all(gates.values()) else 'DO_NOT_EXPAND_NOMINEE_WITHOUT_NEW_HYPOTHESIS',
       'other_factorial_cells_are_not_substitute_winners':True,'empirical_filter_not_population_noninferiority_or_significance':True,
       'online_credit_maintenance_tested':False,'method_success_claim':False,'project_stop':False,'automatic_GPU_admission':False,'confidence_intervals_computed':False,'bootstrap_runs':0}
