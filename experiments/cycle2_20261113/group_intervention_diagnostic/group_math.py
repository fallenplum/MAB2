"""Pure numeric grouped-intervention diagnostics; no experiment I/O."""
import itertools
import numpy as np

def four_world_metrics(quality):
    y=np.asarray(quality,dtype=float)
    if y.ndim!=2 or y.shape[0]!=4 or y.shape[1]<2 or not np.isfinite(y).all() or not np.isin(y,[0,1]).all():raise ValueError('Need binary [4,K>=2] worlds 00,01,10,11')
    f00,f01,f10,f11=y
    a=f11-f01;b=f11-f10;L=f11-f00;h=f11-f10-f01+f00
    assert np.array_equal(L,a+b-h)
    return {'a':a,'b':b,'L':L,'h':h}

def cross_square(values):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all():raise ValueError('Need finite >=2 seeds')
    return float((x.sum()**2-np.dot(x,x))/(len(x)*(len(x)-1)))

def crossfit(y):
    m=four_world_metrics(y)
    if y.shape[1]!=5:raise ValueError('Fixed five seeds required')
    out=[]
    for selection,evaluation in (([0,1],[2,3,4]),([2,3,4],[0,1])):
        selected=abs(m['a'][selection].mean())<=.05 and abs(m['b'][selection].mean())<=.05
        out.append({'selected':bool(selected),'select_positions':selection,'evaluate_positions':evaluation,
                    'heldout':{k:float(v[evaluation].mean()) for k,v in m.items()} if selected else None})
    return out

def question_ci(values):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or not np.isfinite(x).all():raise ValueError('Need finite question values')
    if not len(x):return {'point':None,'interval95':None,'interval90':None,'questions':0}
    rng=np.random.default_rng(83019)
    means=[]
    for start in range(0,10000,250):
        idx=rng.integers(0,len(x),size=(250,len(x)));means.extend(x[idx].mean(axis=1))
    return {'point':float(x.mean()),'interval95':np.quantile(means,[.025,.975]).tolist(),
            'interval90':np.quantile(means,[.05,.95]).tolist(),'questions':len(x),
            'small_cluster_warning':len(x)<30,'evidence':'EXPOSED_DEVELOPMENT_DESCRIPTIVE_BOOTSTRAP'}

def summarize_faces(face_rows):
    """Rows have q index, pair, quality[4,5], optional logical token/call matrices."""
    byq={};selected_byq={};q_zero={};support=[0,0];allpairs=set();positive_L_questions=set()
    for r in face_rows:
        q=r['q'];pair=tuple(r['pair']);allpairs.add(pair)
        y=np.asarray(r['quality'],dtype=float);m=four_world_metrics(y)
        u={k:cross_square(m[k]) for k in ('h','a','b')}
        row={**{k:float(v.mean()) for k,v in m.items()},**{'u_'+k:v for k,v in u.items()},
             'h_same_seed_square':float(np.mean(m['h']**2)),
             'h_mean_square':float(m['h'].mean()**2)}
        if 'tokens' in r:
            token=np.asarray(r['tokens']);call=np.asarray(r['calls'])
            if token.shape!=(4,5) or call.shape!=(4,5) or (token<0).any() or (call<0).any():raise ValueError('Invalid cost matrix')
            row['joint_token_saving']=float((token[3]-token[0]).mean())
            row['joint_call_saving']=float((call[3]-call[0]).mean())
        byq.setdefault(q,[]).append(row)
        selected=[]
        for f,cf in enumerate(crossfit(y)):
            if cf['selected']:
                support[f]+=1;selected.append(cf['heldout'])
        if selected:
            merged={k:float(np.mean([z[k] for z in selected])) for k in ('a','b','L','h')}
            selected_byq.setdefault(q,[]).append(merged)
            if merged['L']>0:positive_L_questions.add(q)
    def reduce(rows):
        if not rows:return {}
        keys=set.intersection(*(set(z) for values in rows.values() for z in values))
        return {k:question_ci([np.mean([z[k] for z in values]) for _,values in sorted(rows.items())]) for k in sorted(keys)}
    total=reduce(byq);selected=reduce(selected_byq)
    gate_coverage=len(byq)>=30 and len(allpairs)>=3 and len(selected_byq)>=20
    conditions={
        'coverage':gate_coverage,
        'interaction_square':bool(total and total['u_h']['point']>0 and total['u_h']['interval95'][0]>0),
        'heldout_joint_loss':bool(selected and selected['L']['point']>=.10 and selected['L']['interval95'][0]>0),
        'heldout_singles_equivalence_screen':bool(selected and all(selected[k]['interval90'][0]>=-.05 and selected[k]['interval90'][1]<=.05 for k in ('a','b')))}
    status='EXPLORATORY_SCREEN_PASS_NOT_GPU_AUTHORIZATION' if all(conditions.values()) else 'INCONCLUSIVE_OR_NO_SUPPORTED_SIGNAL'
    if not face_rows:status='NOT_IDENTIFIABLE_NO_COMPLETE_FOUR_WORLDS'
    elif not gate_coverage:status='INCONCLUSIVE_INSUFFICIENT_SUPPORT'
    return {'complete_faces':len(face_rows),'complete_questions':len(byq),'represented_pairs':len(allpairs),
            'selected_distinct_questions':len(selected_byq),'selected_face_counts_by_fold':support,
            'selected_faces':sum(len(v) for v in selected_byq.values()),'selected_questions_with_positive_heldout_L':len(positive_L_questions),
            'all_faces_question_weighted':total,'zero_single_screen_crossfit_question_weighted':selected,
            'investment_screen':{'status':status,'conditions':conditions,'project_stop':False},
            'private_question_values':{str(q):{'all_faces':byq[q],'selected_faces':selected_byq.get(q,[])} for q in sorted(byq)}}
