"""All post-scan analyses read numeric cache and already-exposed D2 arrays only."""
from pathlib import Path
import argparse,hashlib,io,json,time
import numpy as np
# Inputs are supplied explicitly through CLI; no workspace discovery.
def sha(b):return hashlib.sha256(b).hexdigest()
def cross_seed_risk(pred,delta):
    errors=pred[...,None]-delta
    return sum(errors[...,i]*errors[...,j] for i in range(5) for j in range(i+1,5))/10
def group_stats(flag,d,yl,yr):
    selected=flag;num=int(selected.sum());qcount=int(selected.reshape(len(selected),-1).any(1).sum())
    if not num:return {'paired_seed_observations':0,'questions':0,'nonzero_credit_fraction':None}
    return {'paired_seed_observations':num,'questions':qcount,'nonzero_credit_fraction':float((d[selected]!=0).mean()),
            'mean_signed_credit':float(d[selected].mean()),'both_correct_fraction':float(((yl==1)&(yr==1))[selected].mean()),
            'both_wrong_fraction':float(((yl==0)&(yr==0))[selected].mean())}
def bootstrap_difference(flag,d):
    n=len(flag);axes=tuple(range(1,flag.ndim));num1=((d!=0)&flag).sum(axis=axes);den1=flag.sum(axis=axes)
    num0=((d!=0)&~flag).sum(axis=axes);den0=(~flag).sum(axis=axes)
    if den1.sum()==0 or den0.sum()==0:return {'difference':None,'ci95':None,'valid_resamples':0}
    point=float(num1.sum()/den1.sum()-num0.sum()/den0.sum())
    picks=np.random.default_rng(2026111308).integers(0,n,size=(2000,n));a=den1[picks].sum(1);b=den0[picks].sum(1);valid=(a>0)&(b>0)
    values=num1[picks].sum(1)[valid]/a[valid]-num0[picks].sum(1)[valid]/b[valid]
    support_ok=int((den1>0).sum())>=2 and int((den0>0).sum())>=2
    ci=np.quantile(values,[.025,.975]).tolist() if len(values) and support_ok else None
    return {'difference':point,'ci95':ci,'valid_resamples':int(valid.sum()),'resamples':2000,'cluster':'question','seed':2026111308,
            'interval_not_reported_reason':None if support_ok else 'Fewer than two supporting question clusters in a group; conditional nonempty bootstrap cannot estimate exposure-population uncertainty',
            'scope':'exploratory association; multiple tables no familywise claims'}
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','d2-inputs','d2-predictions','pins','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();start=time.time()
    pins=json.loads(args.pins.read_bytes());inv=pins;scan={'cache_sha256':pins['cache_sha256']}
    OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
    raw=args.cache.read_bytes();assert sha(raw)==scan['cache_sha256']
    with np.load(io.BytesIO(raw),allow_pickle=False) as v:c={k:v[k] for k in v.files}
    raw=args.d2_inputs.read_bytes();assert sha(raw)==inv['d2_inputs_sha256']
    with np.load(io.BytesIO(raw),allow_pickle=False) as v:a={k:v[k] for k in v.files}
    raw=args.d2_predictions.read_bytes();assert sha(raw)==inv['d2_prediction_sha256']
    with np.load(io.BytesIO(raw),allow_pickle=False) as v:p={k:v[k] for k in ('B0','C1_R0','C2_R0','C2_R1','C2_R3')}
    keep=np.isin(a['partition'],['train','tune']);left=a['left'].reshape(400,2,12)[keep];right=a['right'].reshape(400,2,12)[keep]
    assert np.array_equal(a['row_question'][left],a['row_question'][right])
    assert np.all(a['row_question'][left]==np.flatnonzero(keep)[:,None,None])
    assert np.all((a['masks'][left]!=a['masks'][right]).sum(-1)==1)
    for edge in range(12):
        assert not a['masks'][left][:,:,edge,edge].any() and a['masks'][right][:,:,edge,edge].all()
    yl=a['quality_by_seed'][left];yr=a['quality_by_seed'][right];delta=yr-yl;target=delta.mean(-1)
    assert c['present'][left].all() and c['present'][right].all()
    edges=[(s,d) for s in range(4) for d in range(4) if s!=d]
    src=np.array([s for s,d in edges])[None,None,:,None];dst=np.array([d for s,d in edges])[None,None,:,None]
    lbad=c['bad'][left];rbad=c['bad'][right]
    source_bad=((lbad>>src)&1).astype(bool)
    assert np.array_equal(source_bad,((rbad>>src)&1).astype(bool))
    source_length=((c['length'][left]>>src)&1).astype(bool);source_trim=((c['trim'][left]>>src)&1).astype(bool)
    assert np.array_equal(source_length,((c['length'][right]>>src)&1).astype(bool))
    assert np.array_equal(source_trim,((c['trim'][right]>>src)&1).astype(bool))
    receiver_bad=((((lbad|rbad)>>(4+dst))&1)>0);agg_bad=(((lbad|rbad)>>8)&1)>0
    receiver_bad_both=((((lbad&rbad)>>(4+dst))&1)>0)
    parse_fail=~(c['parse_ok'][left]&c['parse_ok'][right])
    flags={'source_format_bad':source_bad,'source_length_finish':source_length,'source_message_trimmed':source_trim,
           'receiver_format_bad_either':receiver_bad,'aggregate_format_bad_either':agg_bad,'aggregate_numeric_parse_failure_either':parse_fail}
    receiver_tokens_l=c['received'][left];receiver_tokens_r=c['received'][right]
    for e,(s,d) in enumerate(edges):
        assert np.array_equal(receiver_tokens_r[:,:,e,:,d]-receiver_tokens_l[:,:,e,:,d],c['wire_tokens'][right][:,:,e,:,s])
        assert not (((c['incoming_bad'][left][:,:,e,:]>>e)&1)>0).any()
        assert np.array_equal(((c['incoming_bad'][right][:,:,e,:]>>e)&1)>0,source_bad[:,:,e,:])
    part=a['partition'][keep];tables=[];risks=[];baselines=[];fallback_identity=[];original_metrics=[]
    for name in ('train','tune'):
        select=part==name;d=delta[select];lo=yl[select];hi=yr[select]
        baselines.append({'partition':name,**group_stats(np.ones_like(d,dtype=bool),d,lo,hi)})
        original={model:cross_seed_risk(pred[keep][select],d) for model,pred in p.items()}
        original_metrics.append({'partition':name,'questions':int(select.sum()),'edge_anchor_contexts':int(target[select].size),
            'single_seed_nonzero_fraction':float((d!=0).mean()),'five_seed_empirical_mean_nonzero_fraction':float((target[select]!=0).mean()),
            'original_question_mean_24_context_cross_seed_risk':{m:float(values.mean(axis=(1,2)).mean()) for m,values in original.items()},
            'five_seed_mean_empirical_mse':{m:float(((p[m][keep][select]-target[select])**2).mean()) for m in p},
            'estimated_seed_noise_in_mean_credit':float((d.var(axis=-1,ddof=1)/5).mean()),
            'scope':'Original risk formula and equal question/24-context weights, but historical train/tune not original heldout gate'})
        for fname,flag in flags.items():
            f=flag[select]
            tables.append({'partition':name,'exposure':fname,'exposed':group_stats(f,d,lo,hi),'unexposed':group_stats(~f,d,lo,hi),
                           'nonzero_credit_rate_exposed_minus_unexposed':bootstrap_difference(f,d)})
            anyseed=f.any(-1)
            for status,take in (('any_seed_exposed',anyseed),('all_seeds_unexposed',~anyseed)):
                risk={'partition':name,'exposure':fname,'condition':status,'edge_anchor_contexts':int(take.sum()),'questions':int(take.reshape(len(take),-1).any(1).sum()),'mse':{},
                      'original_cross_seed_risk_contribution':{},'conditional_pooled_cross_seed_risk':{}}
                for model,pred in p.items():
                    r=(pred[keep][select]-target[select])**2;risk['mse'][model]=float(r[take].mean()) if take.any() else None
                    values=original[model]
                    risk['original_cross_seed_risk_contribution'][model]=float((values*take).sum(axis=(1,2)).mean()/24)
                    risk['conditional_pooled_cross_seed_risk'][model]=float(values[take].mean()) if take.any() else None
                risks.append(risk)
        dirty=(source_bad|receiver_bad|agg_bad)[select];clean=~dirty
        zero_count=int((d==0).sum())
        baselines.append({'partition':name,'subset':'all_relevant_source_receiver_aggregate_format_clean',**group_stats(clean,d,lo,hi),
                          'zero_credits_retained_fraction_of_all_zero_credits':float(((d==0)&clean).sum()/zero_count) if zero_count else None})
        graph_clean=((lbad|rbad)==0)[select]
        baselines.append({'partition':name,'subset':'both_graphs_all_nine_nodes_format_clean',**group_stats(graph_clean,d,lo,hi),
                          'zero_credits_retained_fraction_of_all_zero_credits':float(((d==0)&graph_clean).sum()/zero_count) if zero_count else None})
        fallback_identity.append({'partition':name,**group_stats(receiver_bad_both[select],d,lo,hi),
            'receiver_fallback_semantic_basis':'Both have the same checked answer/evidence/uncertainty fallback fields',
            'scope':'Semantic fallback equality only; per-pair wire bytes and receiver/aggregate input or output hashes were not retained',
            'general_downstream_exact_zero_effect_certificate_established':False})
    result={'status':'EXPLORATORY_HISTORICAL_ASSOCIATION_COMPLETE','questions':280,'paired_seed_contrasts':int(delta.size),
            'receiver_added_message_token_checks':int(delta.size),'source_bad_sent_only_when_edge_on_checks':int(delta.size),
            'baselines_and_clean_path':baselines,'associations':tables,'fixed_predictor_risk_strata':risks,
            'original_metric_alignment':original_metrics,
            'late_requested_exploratory_receiver_fallback_identity':fallback_identity,
            'risk_target':'five-seed mean presence credit; exposure split by any seed in paired graphs',
            'metric_definitions':{'association_nonzero':'One seed presence credit d_s=Q_on,s-Q_off,s; not five-seed mean credit',
              'mse_field':'Empirical (prediction-mean_5seed_credit)^2; original tuning/naive diagnostic, not original heldout risk',
              'original_cross_seed_risk':'mean_{s<t}(prediction-d_s)*(prediction-d_t), 10 pairs; negative estimates retained',
              'stratum_risk_contribution':'For each question sum selected-context cross-seed risk /24, then average ALL questions in partition; exposed+unexposed equals original full-partition metric',
              'conditional_pooled_risk':'Cross-seed formula averaged only selected contexts; group weights differ from original whole-question metric; not an unbiased conditional causal/latent risk claim'},
            'fitted_models_or_hyperparameters':0,'raw_execution_rescans':0,'new_R1_access':False,'new_gold_read':False,
            'limitations':['Observational format errors correlate with question/model difficulty and saturation; no repaired-message counterfactual exists',
                          'Receiver/aggregate error flags are post-edge outcomes, not randomized treatments; conditioning may induce collider bias',
                          'Nonzero credit needs final numeric correctness to differ, not merely a valid message',
                          'Train risk is in-sample; tune was historically used for selection; neither is independent confirmation',
                          'Question-cluster bootstrap respects within-question dependence but multiple diagnostic tables are exploratory'],
            'analysis_seconds':time.time()-start,'cache_sha256':scan['cache_sha256']}
    (OUT/'ASSOCIATION_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'questions':280,'paired_seed_contrasts':int(delta.size),'baselines_and_clean_path':baselines},indent=2))
if __name__=='__main__':main()
