"""Independent D3 replay from accepted derived arrays, with no production imports.

Inputs: --d2, --attempt, --protocol. Output: --output (a new review JSON).
The program never reads original questions, model outputs, gold, or new data.
"""
import time
WALL_START = time.time()
import argparse
import hashlib
import io
import itertools
import json
import math
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

READS = {}
DIFFS = {}


def rawread(path):
    raw = path.read_bytes()
    READS[str(path)] = hashlib.sha256(raw).hexdigest()
    return raw


def jsread(path):
    return json.loads(rawread(path))


def npread(path):
    with np.load(io.BytesIO(rawread(path)), allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def close(name, actual, expected, atol=1e-12, rtol=1e-10):
    a,b=np.asarray(actual),np.asarray(expected)
    assert a.shape==b.shape,(name,a.shape,b.shape)
    DIFFS[name]=float(np.max(np.abs(a-b))) if a.size else 0.0
    np.testing.assert_allclose(a,b,atol=atol,rtol=rtol,err_msg=name)


def potential(mask, order):
    values=[1.0]+[float(x) for x in mask]
    if order==2:
        values += [float(mask[i]*mask[j]) for i,j in itertools.combinations(range(12),2)]
    return np.asarray(values)


def design_from_actual_toggles(a, order):
    # Construct two complete graph-potential rows and subtract. This does not
    # reuse the production analytic presence-credit design implementation.
    return np.asarray([potential(a['masks'][r],order)-potential(a['masks'][l],order)
                       for l,r in zip(a['left'],a['right'])]).reshape(400,2,12,-1)


def top2_scalar(scores):
    return np.asarray([sorted(range(12),key=lambda e:(-float(row[e]),e))[:2] for row in scores])


COUNTS={}
def bootstrap_count_weights(values):
    n=len(values)
    if n not in COUNTS:
        indices=np.random.default_rng(59173).integers(0,n,(10000,n))
        counts=np.zeros((10000,n),dtype=np.int16)
        np.add.at(counts,(np.repeat(np.arange(10000),n),indices.ravel()),1)
        COUNTS[n]=counts
    means=COUNTS[n]@values/n
    return {'estimate':math.fsum(float(x) for x in values)/n,
            'interval95':np.quantile(means,[.025,.975]).tolist()}


def scalar_drift(target):
    result=np.empty((len(target),12))
    for q in range(len(target)):
        for e in range(12):
            delta=[float(target[q,1,e,s]-target[q,0,e,s]) for s in range(5)]
            result[q,e]=math.fsum(x*y for x,y in itertools.combinations(delta,2))/10
    return result


def fit_all_alphas(X,y,TX,ty,alphas):
    d=X.shape[1]
    gram=X.T@X/24
    rhs=X.T@y/24
    output={}
    for alpha in alphas:
        beta=np.linalg.solve(gram+alpha*np.eye(d),rhs)
        prediction=TX@beta
        losses=[]
        for q in range(80):
            losses.append(math.fsum((float(prediction[24*q+i])-float(ty[24*q+i]))**2 for i in range(24))/24)
        output[alpha]=(beta,math.fsum(losses)/80)
    minimum=min(value[1] for value in output.values())
    chosen=max(alpha for alpha,value in output.items() if value[1]<=minimum+1e-12)
    return chosen,output,gram,rhs


def main(args):
    result=jsread(args.attempt/'D3_RESULTS.json')
    complete=jsread(args.attempt/'COMPLETE.json')
    frozen=jsread(args.attempt/'NOMINATION_FROZEN.json')
    started=jsread(args.attempt/'STARTED.json')
    access=jsread(args.attempt/'DERIVED_INPUT_ACCESS.json')
    protocol=jsread(args.protocol)
    a=npread(args.d2/'private/inputs.npz')
    d2params=npread(args.d2/'private/parameters.npz')
    d2complete=jsread(args.d2/'COMPLETE.json')
    d2results=jsread(args.d2/'D2_RESULTS.json')
    selections=npread(args.attempt/'private/FROZEN_SELECTIONS.npz')
    scored=npread(args.attempt/'private/SCORED_ARRAYS.npz')
    assert result['status']==complete['status']=='DEVELOPMENT_SCREEN_COMPLETE'
    assert complete['results_sha256']==READS[str(args.attempt/'D3_RESULTS.json')]
    assert complete['nomination_sha256']==READS[str(args.attempt/'NOMINATION_FROZEN.json')]
    assert complete['selection_sha256']==frozen['parameter_and_selection_sha256']==READS[str(args.attempt/'private/FROZEN_SELECTIONS.npz')]
    assert result['protocol_sha256']==frozen['protocol_sha256']==started['protocol_sha256']==READS[str(args.protocol)]
    assert result['source_sha256']==started['source_sha256']
    pins=protocol['input_bindings']
    for name,key in [('COMPLETE.json','complete_sha256'),('D2_RESULTS.json','results_sha256'),('private/inputs.npz','private_inputs_npz_sha256'),('private/parameters.npz','private_parameters_npz_sha256')]:
        assert READS[str(args.d2/name)]==pins[key]
    assert d2complete['results_sha256']==READS[str(args.d2/'D2_RESULTS.json')]
    assert d2complete['private_inputs_sha256']==READS[str(args.d2/'private/inputs.npz')]
    assert frozen['candidate']==protocol['unique_candidate']['name']=='D3_graph_only_order2_R3'
    assert frozen['no_fallback'] and frozen['exposed_scoring_started'] is False
    assert started['time_unix']<=access['time_unix']<=frozen['time_unix']
    assert not access['original_gold_or_questions_read'] and not access['sealed_test_read']
    assert not result['independent_confirmation'] and not result['future_service_or_full_budget_benefit']
    assert result['new_GPU_calls']==0 and result['git_cpu_seconds'] is None
    assert result['process_cpu_seconds']+result['git_cpu_upper_bound_seconds']+result['previous_attempts_cpu_upper_bound_seconds']<protocol['authorization']['max_D3_process_cpu_seconds']
    assert a['bits'].shape==(400,2,12) and a['quality_by_seed'].shape==(10394,5)
    assert len(set(a['question_ids'].tolist()))==400
    tr=a['partition']=='train';tu=a['partition']=='tune';ev=a['partition']=='test'
    assert [int(x.sum()) for x in (tr,tu,ev)]==[200,80,120]
    assert np.all(tr.astype(int)+tu.astype(int)+ev.astype(int)==1)
    for idx,(l,r) in enumerate(zip(a['left'],a['right'])):
        q,anchor,edge=idx//24,(idx%24)//12,idx%12
        assert a['row_question'][l]==a['row_question'][r]==q
        assert np.flatnonzero(a['masks'][l]!=a['masks'][r]).tolist()==[edge]
        assert a['masks'][l,edge]==0 and a['masks'][r,edge]==1
        assert any(np.array_equal(a['bits'][q,anchor],a['masks'][g]) for g in (l,r))
    target=(a['quality_by_seed'][a['right']]-a['quality_by_seed'][a['left']]).reshape(400,2,12,5)
    alphas=protocol['unique_candidate']['alpha_grid']
    all_credits={};fit_replays={};X2=None
    for order in (1,2):
        X=design_from_actual_toggles(a,order)
        if order==2:X2=X
        trainX=X[tr].reshape(4800,-1);tuneX=X[tu].reshape(1920,-1)
        trainY=target[tr].mean(axis=-1).ravel();tuneY=target[tu].mean(axis=-1).ravel()
        chosen,path,gram,rhs=fit_all_alphas(trainX,trainY,tuneX,tuneY,alphas)
        recorded=result['models'][str(order)]
        assert chosen==recorded['alpha']
        for trial in recorded['alpha_trials']:
            close(f'order{order}/alpha{trial["alpha"]}/tune_loss',path[trial['alpha']][1],trial['tune_credit_mse'])
        beta=path[chosen][0]
        close(f'order{order}/selected_beta',beta,selections[f'beta_order{order}'],atol=2e-10,rtol=2e-8)
        residual=trainY-trainX@beta
        rss=float(residual@residual/24);norm=float(beta@beta)
        close(f'order{order}/rss',rss,recorded['diagnostics']['weighted_rss'])
        close(f'order{order}/penalty_norm',norm,recorded['diagnostics']['penalized_beta_squared_norm'])
        close(f'order{order}/objective',rss+chosen*norm,recorded['diagnostics']['objective'])
        df=float(np.trace(np.linalg.solve(gram+chosen*np.eye(len(beta)),gram)))
        close(f'order{order}/df',df,recorded['diagnostics']['effective_df'])
        close(f'order{order}/trace',float(np.trace(gram)),recorded['diagnostics']['raw_weighted_feature_trace'])
        # Evaluate the immutable frozen coefficients. Selection on exact score
        # ties is verified against the saved scores, with reconstructed-score
        # discrepancies checked separately to disclose numerical near-ties.
        all_credits[f'graph_only_order{order}']=X@selections[f'beta_order{order}']
        fit_replays[str(order)]={'selected_alpha':chosen,'alpha_grid_fully_refitted':True,'alpha_candidates':len(alphas),'solver':'direct_normal_equations','effective_df_reconstructed':df}
    assert frozen['alpha']==fit_replays['2']['selected_alpha']
    # Independently contract each question embedding with the accepted D2
    # coefficient blocks, then use the graph-potential differences.
    oldbeta=d2params['C2_R3'].reshape(9,79)
    contextual=np.asarray([sum((float(a['phi'][q,k])*oldbeta[k] for k in range(9)),start=np.zeros(79)) for q in range(400)])
    all_credits['fixed_old_C2_R3']=np.asarray([X2[q]@contextual[q] for q in range(400)])
    b0=np.asarray([math.fsum(float(v) for v in target[tr,:,e,:].ravel())/(200*2*5) for e in range(12)])
    all_credits['B0']=np.broadcast_to(b0,(400,2,12))
    factor=200/(200+12*fit_replays['1']['selected_alpha'])
    close('order1/closed_form_B0_shrinkage',selections['beta_order1'][1:],factor*b0)
    close('order1/intercept_zero',selections['beta_order1'][0],0.0)
    close('degeneracy/B0_record',b0,result['first_order_degeneracy']['B0'])
    chosen_indices={};tie_disclosures={}
    for name,credit in all_credits.items():
        derived_score=np.abs(credit[:,1]-credit[:,0])
        saved_score=selections[name+'_scores']
        close(name+'/score_reconstruction',derived_score,saved_score)
        ix=top2_scalar(saved_score)
        assert np.array_equal(ix,selections[name+'_indices']),name
        different=np.any(top2_scalar(derived_score)!=ix,axis=1)
        tie_disclosures[name]={'top2_reconstructed_score_near_tie_differences':int(different.sum()),'exact_all_score_ties':int(np.sum(np.ptp(saved_score,axis=1)==0))}
        if np.any(different):
            for q in np.flatnonzero(different):
                own=top2_scalar(derived_score[q:q+1])[0]
                union=set(own.tolist()+ix[q].tolist())
                assert max(saved_score[q,list(union)])-min(saved_score[q,list(union)])<1e-12
        if name in ('B0','graph_only_order1'):assert np.all(saved_score==0)
        chosen_indices[name]=ix
    independent={}
    for part,mask in [('tune',tu),('exposed',ev)]:
        z=scalar_drift(target[mask]);n=len(z)
        uniform=np.asarray([math.fsum(float(x) for x in row)/12 for row in z])
        independent[part]={}
        for name,all_ix in chosen_indices.items():
            ix=all_ix[mask]
            selected=np.asarray([math.fsum(float(z[q,e]) for e in ix[q])/2 for q in range(n)])
            delta=selected-uniform
            for key,value in [('selected_squared_drift',selected),('uniform_expected_squared_drift',uniform),('enrichment',delta)]:
                close(part+'/'+name+'/'+key,value,scored[part+'_'+name+'_'+key])
            observed=result['drift_results'][part][name]
            ci=bootstrap_count_weights(delta)
            close(part+'/'+name+'/selected_mean',math.fsum(selected)/n,observed['selected_squared_drift_mean'])
            close(part+'/'+name+'/uniform_mean',math.fsum(uniform)/n,observed['uniform_top2_expected_squared_drift_mean'])
            close(part+'/'+name+'/enrichment_mean',ci['estimate'],observed['enrichment']['estimate'])
            close(part+'/'+name+'/enrichment_interval',ci['interval95'],observed['enrichment']['interval95'])
            assert observed['negative_squared_drift_estimate_coordinates']==int((z<0).sum())
            assert observed['all_scores_exactly_tied_questions']==int(np.sum(np.ptp(selections[name+'_scores'][mask],axis=1)==0))
            assert not observed['negative_estimates_clipped'] and not observed['same_question_targets_entered_selection']
            independent[part][name]={'selected_squared_drift_mean':math.fsum(selected)/n,'uniform_top2_expected_squared_drift_mean':math.fsum(uniform)/n,'enrichment':ci,'negative_U_coordinates':int((z<0).sum()),'questions':n}
    primary=independent['exposed']['graph_only_order2']
    point=primary['enrichment']['estimate'];low,high=primary['enrichment']['interval95'];selected=primary['selected_squared_drift_mean']
    passed=bool(point>0 and low>0 and selected>0)
    classification='POSITIVE_DEVELOPMENT_SIGNAL' if passed else 'REVERSE_DEVELOPMENT_SIGNAL' if high<0 else 'ZERO_ENRICHMENT_IN_OBSERVED_SAMPLE' if low==high==point==0 else 'ENRICHMENT_WITHOUT_POSITIVE_SELECTED_DRIFT' if low>0 and selected<=0 else 'INCONCLUSIVE'
    status='GO_BOUNDED_INDEPENDENT_PILOT' if passed else 'STOP_CURRENT_CANDIDATE_ROUTE'
    direction='positive' if point>0 else 'negative' if point<0 else 'zero'
    g=result['spending_gate']
    assert passed==g['pass'] and classification==g['evidence_classification'] and status==g['status'] and direction==g['point_direction']
    assert g['checks']['positive_enrichment_point']==(point>0)
    assert g['checks']['positive_enrichment_lower95']==(low>0)
    assert g['checks']['positive_selected_squared_drift_point']==(selected>0)
    assert g['independent_pilot_gpu_hour_cap']==(16 if passed else 0)
    assert not g['fallback_allowed'] and not g['action_gain_gate_used'] and not g['global_MSE_gate_used']
    for path,pin in READS.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==pin
    out={'schema':'mab2_d3_independent_real_replay_v1','status':'PASS_WITH_EXPLORATORY_LIMITATIONS',
        'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':{'accepted_D2_arrays_read':True,'D3_frozen_outputs_read':True,'production_modules_imported':False,'original_questions_gold_or_new_data_read':False,'new_candidates_added':False,'original_results_modified':False},
        'input_sha256':READS,'numerical_checks':len(DIFFS),'maximum_numerical_difference':max(DIFFS.values()),'differences':DIFFS,
        'independent_fit_replay':fit_replays,'tie_disclosures':tie_disclosures,'independent_statistics':independent,
        'gate_reconstruction':{'pass':passed,'status':status,'classification':classification,'point_direction':direction,'independent_pilot_gpu_hour_cap':16 if passed else 0,'fallback_allowed':False},
        'scientific_interpretation':['The unique predeclared graph-only selector did not pass its development spending gate.','A negative cross-seed estimate is retained sampling variation of an estimator; squared population drift is not claimed negative.','An interval spanning zero means the enrichment effect is unresolved, not demonstrated equivalence or universal impossibility.','No auxiliary control can replace the failed candidate; no new GPU pilot is triggered by this gate.','Graph-credit drift enrichment, even if positive, would not itself establish future service or complete-budget benefit.'],
        'costs':{'review_main_process_cpu_seconds':time.process_time(),'review_wall_seconds_from_first_script_timestamp':time.time()-WALL_START,'includes_import_CPU':True,'preparation_commands_CPU_seconds':None,'reviewer_agent_inference_CPU_seconds':None,'child_process_CPU_seconds':0,'new_GPU_calls':0,'new_GPU_hours':0},
        'limitations':['No regrading from original model answers or gold; accepted derived arrays are the audit starting point.','Source-order freeze is checked through markers and hashes plus prior source audit; wall-clock markers alone are not a security proof.','No claim of human author approval, formal venue review, independent new-sample confirmation, or paper readiness.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':out['status'],'checks':out['numerical_checks'],'max_error':out['maximum_numerical_difference'],'primary':primary,'gate':out['gate_reconstruction'],'CPU_seconds':out['costs']['review_main_process_cpu_seconds'],'report_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest()},ensure_ascii=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--d2',type=Path,required=True);p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    with threadpool_limits(limits=4):main(args)
