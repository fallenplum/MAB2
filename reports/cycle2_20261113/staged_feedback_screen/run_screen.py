"""Historical, CPU-only anonymous finite-bank screen. Never reads R1 or raw text."""
from pathlib import Path
import hashlib
import json
import csv
import time
import numpy as np
from frontier_math import direct_srs, direct_poisson, census_probe_neyman, staged_poisson, lazy_srs_base_cost, self_check

SCRIPT_DIR = Path(__file__).resolve().parent
# CLI-only data/output paths. This post-result revision changes the interface only.
OUT = SRC = D2 = None
FRACTIONS = [.02,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def stats(x):
    x=np.asarray(x)
    return {'n':len(x),'mean':float(x.mean()),'median':float(np.median(x)),'min':float(x.min()),'max':float(x.max())}


def invert_cost(budget, cost_fn, upper):
    lo,hi=0.,float(upper)
    for _ in range(75):
        mid=(lo+hi)/2
        if cost_fn(mid)>budget:
            hi=mid
        else:
            lo=mid
    return lo


def execute():
    start=time.monotonic()
    checks=self_check()
    assert sha(SCRIPT_DIR/'PROTOCOL.json').upper()=='E905D49063DCE0482CEA78D9C79BAA806C42112405FF2367ED98FC9D254016D6'
    assert sha(D2)=='7639dad2581ea52536b7c274d33fd019ad726a9bce9e8e4ec6792c68b77f4963'
    source_results=json.loads((SRC/'RESULTS.json').read_text('utf-8'))
    assert sha(SRC/'HASHED_NODE_CACHE.npz')==source_results['cache_sha256']
    with np.load(D2,allow_pickle=False) as z:
        partitions=z['partition']
        Q=np.flatnonzero(np.isin(partitions,('train','tune')))
        assert len(Q)==280
        L=z['left'].reshape(400,2,12)[Q]
        R=z['right'].reshape(400,2,12)[Q]
        rowq=z['row_question']
    rows=np.load(SRC/'PRIVATE_MEASUREMENT_ROWS.npy',allow_pickle=False)
    assert rows.shape==(33600,9) and np.isin(rows[:,0],Q).all()
    axes=np.indices((280,2,12,5))
    assert np.array_equal(rows[:,:4],np.column_stack([Q[axes[0].ravel()],axes[1].ravel(),axes[2].ravel(),axes[3].ravel()]))
    li=np.broadcast_to(L[...,None],(280,2,12,5)).ravel()
    ri=np.broadcast_to(R[...,None],(280,2,12,5)).ravel()
    seed=rows[:,3].astype(int)
    receivers=np.array([4+dst for src in range(4) for dst in range(4) if src!=dst])
    node=receivers[rows[:,2].astype(int)]
    with np.load(SRC/'HASHED_NODE_CACHE.npz',allow_pickle=False) as z:
        token=z['tokens']
        assert z['present'][li,seed].all() and z['present'][ri,seed].all()
        recv=token[ri,seed,node].astype(float)
        agg=token[ri,seed,8].astype(float)
        base=token[li,seed,:].sum(axis=1).astype(float)
    cat=rows[:,4].astype(int)
    y=rows[:,6]
    assert np.isin(y,(-1,0,1)).all() and (y[cat==0]==0).all()
    assert np.array_equal(recv+agg,rows[:,7])
    assert np.array_equal(np.where(cat==0,agg,0),rows[:,8])
    assert np.array_equal(rowq[li],rows[:,0]) and np.array_equal(rowq[ri],rows[:,0])
    results={'status':'COMPLETE_HISTORICAL_OPTIMISTIC_ORACLE_SCREEN','new_model_calls':0,'new_gpu_hours':0,'new_formal_rounds':0,'R1_read':False,'official_test_read':False,'raw_text_read':False,'quality_by_seed_read':False,'math_checks':checks,'source_sha256':{p.name:sha(p) for p in [SRC/'PRIVATE_MEASUREMENT_ROWS.npy',SRC/'HASHED_NODE_CACHE.npz',D2,SCRIPT_DIR/'PROTOCOL.json',SCRIPT_DIR/'SCHEMA_AND_COST_ADDENDUM.json',SCRIPT_DIR/'frontier_math.py',SCRIPT_DIR/'run_screen.py']},'splits':{}}
    records=[]
    for split in ('train','tune'):
        take=partitions[rows[:,0].astype(int)]==split
        yy,cc,rr,aa,bb=y[take],cat[take],recv[take],agg[take],base[take]
        ll,ss=li[take],seed[take]
        N=len(yy)
        counts=np.array([(cc==h).sum() for h in (1,2)])
        vars_=np.array([yy[cc==h].var(ddof=1) for h in (1,2)])
        y2s=np.array([np.square(yy[cc==h]).sum() for h in (1,2)])
        group_keys=np.column_stack((ll,ss))
        group,first,gcount=np.unique(group_keys,axis=0,return_index=True,return_counts=True)
        base_group=bb[first]
        assert np.all(bb==base_group[np.unique(group_keys,axis=0,return_inverse=True)[1]])
        summary={'N_rows':N,'N_questions':int(np.unique(rows[take,0]).size),'signed_mean':float(yy.mean()),'sample_variance':float(yy.var(ddof=1)),'receiver_tokens':stats(rr),'aggregate_tokens':stats(aa),'ratio_sum_receiver_to_sum_aggregate_tokens':float(rr.sum()/aa.sum()),'mean_per_row_receiver_aggregate_ratio':float(np.mean(rr/aa)),'suffix_no_cache_tokens':int((rr+aa).sum()),'suffix_common_exact_wire_cache_tokens':int((rr+np.where(cc!=0,aa,0)).sum()),'logical_unique_off_graph_seed_baselines':len(group),'logical_base_graph_token_sum':int(base_group.sum()),'logical_base_graph_token_per_pair_sum':int(bb.sum()),'right_graph_also_in_full_left_bank_fraction':float(np.isin(ri[take],np.unique(ll)).mean()),'strata':[],'scenario_summary':{}}
        for h in (0,1,2):
            u=cc==h
            summary['strata'].append({'category':h,'N':int(u.sum()),'credit_mean':float(yy[u].mean()),'credit_sample_variance':float(yy[u].var(ddof=1)),'credit_second_moment':float(np.square(yy[u]).mean()),'receiver_tokens_mean':float(rr[u].mean()),'aggregate_tokens_mean':float(aa[u].mean())})
            assert int(u.sum())==source_results['groups'][split]['cells'][h]['n']
        scenarios=[]
        for ratio in (.5,1.,2.):
            scenarios.append((f'assumed_receiver_aggregate_ratio_{ratio:g}',np.full(N,ratio),np.ones(N),np.zeros(len(group)),False))
        for score_ratio in (0.,.1,1.):
            score_equiv=score_ratio*aa.mean()
            scenarios.append((f'observed_token_score_equivalent_ratio_{score_ratio:g}',rr,aa+score_equiv,base_group+score_equiv,True))
        for scenario,rc,ac,bc,is_observed in scenarios:
            Rtotal=float(rc.sum())
            Atotal=np.array([ac[cc==h].sum() for h in (1,2)])
            Amean=Atotal/counts
            suffix=Rtotal+Atotal.sum()
            B=float(bc.sum())
            modes=('suffix_only','prepaid_full_bank','lazy_baseline') if is_observed else ('suffix_only',)
            bysize=np.unique(gcount)
            bcost_by_size=np.array([bc[gcount==m].sum() for m in bysize])
            def poisson_base(q):
                return float(np.dot(bcost_by_size, -np.expm1(bysize*np.log1p(-q)))) if q<1 else B
            def srs_base(n):
                return lazy_srs_base_cost(bysize,bcost_by_size,N,n)
            for mode in modes:
                allrows=[]
                fractions=sorted(set(FRACTIONS+[min(.999999, Rtotal/suffix+1e-4)]))
                for frac in fractions:
                    budget=frac*suffix if mode=='suffix_only' else (B+frac*suffix if mode=='prepaid_full_bank' else frac*(B+suffix))
                    available=budget-B if mode=='prepaid_full_bank' else budget
                    if mode=='lazy_baseline':
                        n=invert_cost(budget,lambda x:srs_base(x)+x*suffix/N,N)
                        direct=direct_srs(N,float(yy.var(ddof=1)),suffix/N,n*suffix/N)
                        direct['expected_total_cost']=srs_base(n)+direct['expected_suffix_cost']
                        p=invert_cost(budget,lambda q:poisson_base(q)+q*suffix,1.)
                        direct_p=direct_poisson(N,float(np.square(yy).sum()),suffix,p*suffix)
                        staged=staged_poisson(N,y2s,Atotal,Rtotal,budget,poisson_base)
                        staged['expected_total_cost']=staged['expected_suffix_cost']
                        staged['expected_suffix_cost']=float(staged['q']*Rtotal+np.dot(np.asarray(staged['inclusion_h']),Atotal))
                        neyman=census_probe_neyman(N,counts,vars_,Amean,Rtotal,budget-B)
                    else:
                        direct=direct_srs(N,float(yy.var(ddof=1)),suffix/N,available)
                        direct_p=direct_poisson(N,float(np.square(yy).sum()),suffix,available)
                        staged=staged_poisson(N,y2s,Atotal,Rtotal,available)
                        neyman=census_probe_neyman(N,counts,vars_,Amean,Rtotal,available)
                    methods={'direct_uniform_SRS_exact_cache':direct,'direct_uniform_Poisson_exact_cache':direct_p,'staged_Poisson_HT_oracle':staged,'all_receiver_probe_Neyman_SRS_oracle':neyman}
                    for method,fit in methods.items():
                        v=fit.get('variance')
                        if v is not None and v<1e-19:
                            v=0.
                        out={'split':split,'scenario':scenario,'base_mode':mode,'fraction':frac,'budget':float(budget),'method':method,'variance':v,'budget_times_variance':None if v is None else budget*v,'variance_ratio_to_direct_SRS':None if v is None or direct['variance']<1e-19 else v/direct['variance'],'variance_ratio_to_direct_Poisson':None if v is None or direct_p['variance']<1e-19 else v/direct_p['variance'],'feasible':fit.get('feasible',True),'n':fit.get('n'),'q':fit.get('q'),'p1':fit.get('p_h',[None,None])[0],'p2':fit.get('p_h',[None,None])[1]}
                        records.append(out)
                        allrows.append(out)
                candidates=[x for x in allrows if x['method']=='staged_Poisson_HT_oracle' and x['variance_ratio_to_direct_SRS'] is not None and x['fraction']<.999]
                nys=[x for x in allrows if x['method']=='all_receiver_probe_Neyman_SRS_oracle' and x['variance_ratio_to_direct_SRS'] is not None and x['fraction']<.999]
                summary['scenario_summary'][scenario+'/'+mode]={'full_suffix_cost':float(suffix),'full_base_cost':B,'full_direct_census_cost':float(suffix+B),'probe_fraction_of_suffix':Rtotal/suffix,'staged_best_grid_ratio_to_SRS':min(x['variance_ratio_to_direct_SRS'] for x in candidates),'staged_worst_grid_ratio_to_SRS':max(x['variance_ratio_to_direct_SRS'] for x in candidates),'all_probe_best_grid_ratio_to_SRS':min((x['variance_ratio_to_direct_SRS'] for x in nys),default=None),'all_probe_worst_grid_ratio_to_SRS':max((x['variance_ratio_to_direct_SRS'] for x in nys),default=None),'not_global_policy_bound':True}
                if mode=='lazy_baseline':
                    Bprime1=float(bc[gcount==1].sum())
                    margin=[float(y2s[low]*(Rtotal+Bprime1+Atotal[1-low])-y2s[1-low]*Atotal[low]) for low in (0,1)]
                    summary['scenario_summary'][scenario+'/'+mode]['direct_is_global_optimum_in_declared_2strata_Poisson_class']=bool(min(margin)>=0)
                    summary['scenario_summary'][scenario+'/'+mode]['derivative_certificate_margins']=margin
                    summary['scenario_summary'][scenario+'/'+mode]['baseline_cost_derivative_at_q_1']=Bprime1
                    summary['scenario_summary'][scenario+'/'+mode]['staged_best_grid_ratio_to_direct_Poisson']=min(x['variance_ratio_to_direct_Poisson'] for x in candidates)
        results['splits'][split]=summary
    results['elapsed_wall_seconds']=time.monotonic()-start
    (OUT/'RESULTS.json').write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n','utf-8')
    with (OUT/'FRONTIER.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]))
        w.writeheader(); w.writerows(records)
    print(json.dumps({'status':results['status'],'rows':len(records),'splits':{k:{a:v[a] for a in ('N_rows','ratio_sum_receiver_to_sum_aggregate_tokens','logical_base_graph_token_sum','suffix_common_exact_wire_cache_tokens','right_graph_also_in_full_left_bank_fraction')} for k,v in results['splits'].items()},'elapsed_wall_seconds':results['elapsed_wall_seconds']},indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Replay the fixed historical CPU-only oracle screen from authorized anonymous inputs.')
    parser.add_argument('--diagnostic-dir',required=True,type=Path,help='Directory containing the authorized historical diagnostic RESULTS.json, PRIVATE_MEASUREMENT_ROWS.npy and HASHED_NODE_CACHE.npz')
    parser.add_argument('--d2-inputs',required=True,type=Path,help='Authorized frozen D2 inputs.npz; only train/tune mapping metadata is used')
    parser.add_argument('--output',required=True,type=Path,help='New output directory; existing paths are rejected without overwrite')
    args=parser.parse_args()
    SRC=args.diagnostic_dir.expanduser().resolve()
    D2=args.d2_inputs.expanduser().resolve()
    OUT=args.output.expanduser().resolve()
    if OUT.exists():
        parser.error('Output already exists; refusing to overwrite: '+str(OUT))
    if not SRC.is_dir() or not D2.is_file():
        parser.error('Authorized diagnostic directory and D2 input file must exist')
    OUT.mkdir(parents=True,exist_ok=False)
    execute()
