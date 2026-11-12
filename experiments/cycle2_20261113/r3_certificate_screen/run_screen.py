"""No model calls; consume only a validated R2 I/C export after R1 actions freeze."""
from pathlib import Path
import argparse,hashlib,json,time,subprocess,sys
import numpy as np
from certificate_core import run_policy,oracle,structural_cost

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--private-dir',required=True)
    p.add_argument('--out',required=True);p.add_argument('--expected-commit',required=True)
    a=p.parse_args();repo=Path(a.repo);out=Path(a.out);assert not out.exists()
    assert Path(__file__).resolve().is_relative_to(repo.resolve())
    start=time.time();cpu=time.process_time();private_dir=Path(a.private_dir)
    git=['git','-c','safe.directory='+repo.as_posix(),'-C',str(repo)]
    head=subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip()
    assert head==a.expected_commit
    assert not subprocess.check_output(git+['status','--porcelain'],text=True).strip()
    # Reuse the independently frozen, strict input binding and allowlist validator.
    sys.path.insert(0,str(repo/'experiments/cycle2_20261113/r2'))
    import r2_ic_diagnostic as r2
    # Entry point adaptation is deliberately explicit; cannot silently fall back to plain np.load.
    assert sha(r2.__file__)=='edff2218f0d89569de7abbb6a304d64fce138a2f0fa244b8d3ee48bc7ade5665'
    arrays,provenance=r2.load_inputs(private_dir,r2.load_contract())
    r2.validate_arrays(arrays,r2.load_contract())
    rows=[];private=[]
    for ep in range(8):
        old=arrays['C_A_quality'][ep].astype(int);current=arrays['C_B_quality'][ep].astype(int)
        dirty=np.r_[0,arrays['dirty'][ep]]
        expected=int(np.argmax(current.sum(axis=0)))
        rec={'episode':ep,'bank_winner':expected,'full_structural_calls':structural_cost(np.ones_like(old),dirty),
             'old_bank_structural_calls':252,'methods':{}}
        traces={}
        def query(q,arm):return int(current[q,arm])
        for method in ('cheapest','gap_unit','gap_bundle'):
            run=run_policy(old,dirty,method,query)
            assert run['action']==expected
            rec['methods'][method]={k:run[k] for k in ('cost','revealed','action')};traces[method]=run
        random=[]
        for replica in range(256):
            run=run_policy(old,dirty,'random',query,93173+1000*ep+replica)
            assert run['action']==expected
            random.append(run)
        costs=[r['cost'] for r in random]
        rec['methods']['random']={'cost':float(np.mean(costs)),'min':min(costs),'max':max(costs),'replicas':256}
        traces['random']=random
        opt=oracle(current,dirty);rec['oracle']=opt;traces['oracle']=opt
        if opt['status']=='OPTIMAL_CERTIFICATE_VERIFIED':
            assert all(opt['cost']<=rec['methods'][m]['cost']+1e-9 for m in rec['methods'])
        rows.append(rec);private.append(traces)
    all_opt=all(r['oracle']['status']=='OPTIMAL_CERTIFICATE_VERIFIED' for r in rows)
    means={m:float(np.mean([r['methods'][m]['cost'] for r in rows])) for m in rows[0]['methods']}
    denom=min(means[m] for m in ('cheapest','random','gap_unit'))
    om=float(np.mean([r['oracle']['cost'] for r in rows])) if all_opt else None
    total_ratio=(252+means['gap_bundle'])/(252+denom)
    gates={'complete_valid_oracles':all_opt,'nonstay_winners_at_least3':sum(r['bank_winner']!=0 for r in rows)>=3,
        'oracle_ratio_at_most_point80':all_opt and om/denom<=.8,
        'bundle_ratio_at_most_point90':means['gap_bundle']/denom<=.9,
        'bundle_no_worse_than_gap_at_least6':sum(r['methods']['gap_bundle']['cost']<=r['methods']['gap_unit']['cost'] for r in rows)>=6,
        'including_bank_ratio_at_most_point97':total_ratio<=.97}
    out.mkdir(parents=True)
    (out/'PRIVATE_TRACES.json').write_text(json.dumps(private,indent=2)+'\n')
    result={'status':'ELIGIBLE_FOR_REAL_COST_AND_FRESH_SERVICE_DESIGN' if all(gates.values()) else 'NOT_ADMITTED_CURRENT_CERTIFICATE_ROUTE',
        'project_stop':False,'new_gpu_hours':0,'formal_rounds_started':0,'evidence_class':'EXPLORATORY_R1_IC_REUSE_STRUCTURAL_COST_MODEL',
        'mean_costs':means,'oracle_mean_cost':om,'oracle_ratio':om/denom if all_opt else None,
        'bundle_ratio':means['gap_bundle']/denom,'including_old_bank_bundle_ratio':total_ratio,'gates':gates,'episodes':rows,
        'input_export_sha256':sha(private_dir/'R2_IC_INPUTS.npz'),
        'input_receipt_sha256':sha(private_dir/'R2_IC_RECEIPT.json'),'input_provenance':provenance,
        'source_commit':head,'source_sha256':{x.name:sha(x) for x in Path(__file__).parent.glob('*.py')},
        'protocol_sha256':sha(Path(__file__).parent/'PROTOCOL.md'),'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,
        'real_tokens_or_gpu_savings_claimed':False,'future_quality_claimed':False,'S_data_used':False}
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','mean_costs','oracle_ratio','bundle_ratio','including_old_bank_bundle_ratio','gates','cpu_seconds')}))

if __name__=='__main__':main()
